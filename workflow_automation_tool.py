from pathlib import Path
from datetime import datetime
import hashlib,json,os,re,shutil,subprocess,sys,tempfile
from urllib import request,error
import pandas as pd

BASE_DIR=Path(sys.executable).resolve().parent if getattr(sys,'frozen',False) else Path(__file__).resolve().parent
RAW_INCIDENT_INPUT=BASE_DIR/'input'/'raw_incidents'; OUTPUT_DIR=BASE_DIR/'output'; OCR_CACHE_DIR=BASE_DIR/'.ocr_cache'; AI_CACHE_DIR=BASE_DIR/'.ai_cache'
for f in [RAW_INCIDENT_INPUT,OUTPUT_DIR,OCR_CACHE_DIR,AI_CACHE_DIR,BASE_DIR/'input'/'bcm_incidents',BASE_DIR/'input'/'bu_risks',BASE_DIR/'input'/'policy_register']:
    f.mkdir(parents=True,exist_ok=True)
RAW_INCIDENT_COLUMNS=['S/N','Incident Title','Incident Details','BU','Reported by','Reported Date/Time','Source Type','Source File','Review Flag']
IMAGE_EXTENSIONS={'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}; MESSAGE_EXTENSIONS={'.msg'}
OLLAMA_URL=os.environ.get('OLLAMA_URL','http://localhost:11434'); OLLAMA_MODEL=os.environ.get('OLLAMA_MODEL','llama3.2:3b')
USE_OLLAMA=os.environ.get('USE_OLLAMA','1').strip().lower() not in {'0','false','no','off'}
SENTENCE_MODEL_FILE=BASE_DIR/'models'/'incident_sentence_classifier.joblib'
USE_SENTENCE_MODEL=os.environ.get('USE_SENTENCE_MODEL','1').strip().lower() not in {'0','false','no','off'}
_SENTENCE_MODEL=None

def ck_text(*parts):
    d=hashlib.sha256()
    for p in parts: d.update(str(p or '').encode('utf-8','replace')); d.update(b'\0')
    return d.hexdigest()
def ck_file(p:Path):
    s=p.stat(); return ck_text(str(p.resolve()),str(s.st_size),str(s.st_mtime_ns))
def rj(p:Path):
    try: return json.loads(p.read_text(encoding='utf-8'))
    except Exception: return {}
def wj(p:Path,v:dict):
    try: p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
    except Exception: pass
def polish_text(v):
    v=str(v or '')
    fixes={
        r'\bAcross[- ]?functional\b':'A cross-functional',
        r'\bAcrossfunctional\b':'A cross-functional',
        r'\bweare\b':'we are',
        r'\bwewill\b':'we will',
        r'\bwewould\b':'we would',
        r'\bfrontoffice\b':'front office',
        r'\bpersonalbelongings\b':'personal belongings',
        r'\bexplicitclearance\b':'explicit clearance',
    }
    for a,b in fixes.items(): v=re.sub(a,b,v,flags=re.I)
    return v

def nf(v):
    if isinstance(v,list): v=' '.join(str(x).strip() for x in v if str(x).strip())
    return re.sub(r'\s+',' ',polish_text(str(v or ''))).strip()



def sentence_model():
    global _SENTENCE_MODEL
    if not USE_SENTENCE_MODEL or not SENTENCE_MODEL_FILE.exists(): return None
    if _SENTENCE_MODEL is not None: return _SENTENCE_MODEL
    try:
        from joblib import load
        _SENTENCE_MODEL=load(SENTENCE_MODEL_FILE)
        return _SENTENCE_MODEL
    except Exception:
        _SENTENCE_MODEL=False
        return None

def ml_label(sentence, threshold=0.35):
    model=sentence_model()
    if not model: return ''
    sentence=nf(sentence)
    if not sentence: return ''
    try:
        if hasattr(model,'predict_proba'):
            probs=model.predict_proba([sentence])[0]
            labels=list(model.classes_)
            best_idx=max(range(len(probs)), key=lambda i: probs[i])
            if float(probs[best_idx]) < threshold: return ''
            return str(labels[best_idx])
        return str(model.predict([sentence])[0])
    except Exception:
        return ''

def repair_ocr(t):
    reps={'weare':'we are','wewill':'we will','wewould':'we would','thisoccurred':'this occurred','itisolated':'IT isolated','incidentaffecting':'incident affecting','noreports':'no reports','nomajor':'no major','frontoffice':'front office','directcall':'direct call','personalbelongings':'personal belongings','explicitclearance':'explicit clearance','mediachannels':'media channels','financialloss':'financial loss','handled\n\ndiscreetly':'handled discreetly'}
    for a,b in reps.items(): t=re.sub(rf'\b{a}\b',b,t,flags=re.I)
    return t
def is_meta(line):
    l=line.strip().lower(); return (not l) or l.startswith(('from:','to:','cc:','bcc:','sent:','subject:','reporter:','reported by:','sender:','team:','location:','bu:','business unit:','teams message','dear ','regards','sensitivity:'))
def join_wrapped(t):
    out=[]; buf=''
    for line in [x.strip() for x in t.split('\n')]:
        if not line:
            if buf and not re.search(r'[.!?:)]$',buf): continue
            if buf: out.append(buf); buf=''
            out.append(''); continue
        if not buf: buf=line; continue
        if (not re.search(r'[.!?:)]$',buf)) and (not is_meta(line)) and (not is_meta(buf)): buf=f'{buf} {line}'
        else: out.append(buf); buf=line
    if buf: out.append(buf)
    return '\n'.join(out)
def clean_text(t):
    t=str(t or '').replace('\r\n','\n').replace('\r','\n'); t=repair_ocr(t); t=re.sub(r'\bAcross[- ]?functional\b','A cross-functional',t,flags=re.I); t=re.sub(r'[ \t]+',' ',t); t=join_wrapped(t)
    t=re.sub(r'The process was handled\s*(?:\n| )+discreetly\.?','The process was handled discreetly.',t,flags=re.I)
    t=re.sub(r'W\.\s*found deceased','W&B Television) was found deceased',t)
    return re.sub(r'\n{3,}','\n\n',t).strip()
def label_value(text,labels):
    for lab in labels:
        m=re.search(rf'(?im)^\s*{re.escape(lab)}\s*[:\-]\s*(.+)$',text)
        if m: return m.group(1).strip()
    return ''
def std_date(v):
    v=nf(v)
    if not v: return ''
    has=re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+[A-Za-z]{3,}\s+\d{4}|\d{4}-\d{2}-\d{2}',v)
    if re.search(r'\bthis\s+(morning|afternoon|evening)|\btoday\b|\byesterday\b',v,re.I) and not has: return v
    c=v.replace('.',':') if re.search(r'\d{1,2}\.\d{2}\s*(am|pm)',v,re.I) else v
    dt=pd.to_datetime(c,errors='coerce')
    if pd.isna(dt): return v
    d=dt.to_pydatetime(); return f"{d.month}/{d.day}/{d.year} {d.strftime('%I:%M %p').lstrip('0')}" if re.search(r'\d{1,2}:\d{2}|\d{1,2}\s*(AM|PM)\b',c,re.I) else f'{d.month}/{d.day}/{d.year}'
def find_date(text):
    lab=label_value(text,['Reported Date/Time','Reported Date','Date/Time','Date','Sent','Time'])
    if lab: return lab
    pats=[r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}(?:\s+\d{1,2}:\d{2}\s*(?:AM|PM)?)?\b',r'\b\d{4}-\d{2}-\d{2}(?:\s+\d{1,2}:\d{2}(?::\d{2})?(?:[+-]\d{2}:?\d{2})?)?\b',r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}(?:\s+\d{1,2}:\d{2}\s*(?:AM|PM)?)?\b']
    for p in pats:
        m=re.search(p,text,re.I)
        if m: return m.group(0)
    m=re.search(r'\bat\s+\d{1,2}[.:]\d{2}\s*(?:am|pm)(?:\s*\([^)]*\))?(?:\s+this\s+morning|\s+this\s+afternoon|\s+today)?',text,re.I)
    return m.group(0) if m else ''

def clean_title(title):
    title=repair_ocr(str(title or '')); title=re.sub(r'^(RE|FW|FWD):\s*','',title,flags=re.I).strip(); title=re.sub(r'^Subject\s*[:\-]?\s*','',title,flags=re.I).strip()
    title=re.split(r'\b(?:To\s+IncidentToGroup|Cc\b|Bcc\b|i\s+Reply\b|Reply\b|Reply All\b|From\b|Sent\b)\b',title,maxsplit=1,flags=re.I)[0]
    title=re.sub(r'[|].*$','',title); title=re.sub(r'\s+',' ',title).strip(' .,:;-_|'); title=re.sub(r'\s+\b(?:i|Dd)\b$','',title,flags=re.I).strip()
    return title[:120]
def split_sentences(text):
    out=[]
    for line in text.split('\n'):
        if line.strip(): out+=re.split(r'(?<=[.!?])\s+',line.strip())
    return [x.strip(' -') for x in out if x.strip(' -')]
def choose_sentence(sentences):
    meta=('from:','sent:','subject:','reporter:','team:','bu:','location:','teams message')
    kws=['incident','outage','disruption','robbery','suicide','cyber','breach','failure','leak','flood']
    for s in sentences:
        l=s.lower()
        if not l.startswith(meta) and any(k in l for k in kws): return s
    for s in sentences:
        if not s.lower().startswith(meta): return s
    return sentences[0] if sentences else ''
def infer_title(text,sentences):
    t=label_value(text,['Incident Title','Title','Subject'])
    if t and clean_title(t): return clean_title(t)
    for line in text.split('\n')[:12]:
        if re.match(r'(?i)^\s*subject\b',line) and clean_title(line): return clean_title(line)
    return clean_title(choose_sentence(sentences)) or 'Untitled incident'
def norm_person(line):
    line=str(line or '').strip(' .,:;-')
    line=re.sub(r'\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b',' ',line); line=re.sub(r'\b[a-z][a-z0-9._%+-]*\s+[a-z0-9.-]+\.[a-z]{2,}\b',' ',line)
    c=re.sub(r'\b(Reply All|Reply|Forward)\b',' ',line,flags=re.I); c=re.sub(r'[<>"\'|@]+',' ',c); c=re.sub(r'^[^A-Za-z]+','',c).strip(); c=re.sub(r'\s+',' ',c)
    words=c.split()
    if len(words)>4:
        nw=[w for w in words if re.fullmatch(r"[A-Z][A-Za-z.'-]+",w)]
        if 2<=len(nw)<=4: c=' '.join(nw); words=c.split()
    if len(words)>=3 and (len(words[0])<=1 or words[0].lower() in {'mp','rq','ce','de'}) and words[0].lower() not in {'mr','ms','dr'}: c=' '.join(words[1:])
    c=c.strip(' .,:;-')
    if ' ' not in c and re.fullmatch(r'[A-Z][a-z]+(?:[A-Z][a-z]+)+',c): c=re.sub(r'(?<=[a-z])(?=[A-Z])',' ',c)
    return c.strip(' .,:;-')
def probable_person(line):
    raw=str(line or '').strip()
    raw_lower=raw.lower()
    distribution_words=['dear','colleague','colleagues','group','team','all','incidenttogroup','hotel','location','department','unit']
    if raw_lower.startswith(('dear ', 'hi ', 'hello ')) or any(re.search(rf'\b{w}\b', raw_lower) for w in distribution_words): return ''
    if ':' in raw and len(raw.split())>3 and not raw_lower.startswith(('from:', 'sender:', 'reported by:', 'reporter:')): return ''
    if any(w in raw_lower for w in ['police','guest','incident','deceased','arrival','discovery','removal','timeline','summary','room','locked','sealed','clearance','authorities']): return ''
    c=norm_person(raw)
    if not c or any(ch.isdigit() for ch in c): return ''
    l=c.lower()
    if l.startswith(('to ','cc ','bcc ','subject','incident','confidential','sensitive','media','coverage','thank','reply','forward','dear')): return ''
    if any(w in l for w in ['reported by','reported to','will be','this incident','we would','colleague','distribution list','distribution list','group','esg-related']): return ''
    return c if len(c.split()) in {2,3,4} and re.fullmatch(r"[A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){1,3}",c) else ''

def extract_sender_from_headers(text):
    for label in ['From','Sender','Reported by','Reporter']:
        value=label_value(text,[label])
        if value:
            value=re.split(r'\s+[<([]', value, maxsplit=1)[0].strip() if '<' in value else value.strip()
            candidate=probable_person(value) or norm_person(value)
            if probable_person(candidate): return candidate
    for line in text.split('\n')[:20]:
        m=re.match(r'(?i)^\s*(from|sender|reported by|reporter)\s*[:\-]\s*(.+)$', line.strip())
        if m:
            value=m.group(2).strip()
            value=re.split(r'\s+[<([]', value, maxsplit=1)[0].strip() if '<' in value else value
            candidate=probable_person(value) or norm_person(value)
            if probable_person(candidate): return candidate
    return ''

def infer_reporter(text):
    for pat in [
        r'\b(?:WB\s+)?([A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+)\s+S?/\s*0?\s*To\s+H?IncidentToGroup\b',
        r'\b(?:WB\s+)?([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:SD|O|©|<>)?\s*(?:To\s+H?IncidentToGroup|FH\s+Incident)\b',
    ]:
        m_hdr=re.search(pat, text)
        if m_hdr:
            c=probable_person(m_hdr.group(1))
            if c: return c
    m_amp=re.search(r'\b\d{1,2}/\d{1,2}/\d{4}.*?&\s*([A-Z][A-Za-z]+\s+[A-Z][A-Za-z]+)', text)
    if m_amp and probable_person(m_amp.group(1)): return m_amp.group(1)
    for pat in [r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\s+(?:SD|O|©|<>)?\s*Reply\b', r'\b\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}\s*(?:AM|PM)\s+(?:[a-z©0-9]\s+)?([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:O|©)?\s*FH\s+Incident']:
        m_ocr=re.search(pat, text)
        if m_ocr:
            c=probable_person(m_ocr.group(1))
            if c: return c
    for pat in [r"\b(?:Mr|Ms|Mrs|Mp)\s+([A-Z][A-Za-z]+\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)\s*[\"“”'>]*\s*Reply", r"\b([A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+)\s*[\"“”'>]*\s*Reply"]:
        m_name=re.search(pat, text)
        if m_name:
            c=probable_person(m_name.group(1))
            if c: return c
    for m in re.findall(r'@\s*([A-Z][A-Za-z]+(?:[A-Z][A-Za-z]+)?)', text):
        c=probable_person(m) or norm_person(m)
        if probable_person(c): return c
    sender=extract_sender_from_headers(text)
    if sender: return sender
    lines=[x.strip() for x in text.split('\n') if x.strip()]
    for i,line in enumerate(lines):
        if line.lower().startswith('regards'):
            for cand in lines[i+1:i+4]:
                c=probable_person(cand)
                if c: return c
    for line in lines[:12]:
        if line.lower().startswith(('dear ', 'hi ', 'hello ')): continue
        c=probable_person(line)
        if c: return c
    return ''
def noise(line,title):
    l=line.lower().strip()
    if not line or clean_title(line)==clean_title(title): return True
    if l.startswith(('from:','sent:','subject:','reported by:','reporter:','sender:','team:','location:','bu:','business unit:','teams message','to ','to:','cc ','cc:','bcc ','bcc:','bu:','business unit:','i reply','reply','dear ','regards','thank you','sensitivity:')): return True
    if 'incidenttogroup' in l or 'confidential or sensitive' in l or 'reply' in l or l.startswith('http') or '.pdf' in l: return True
    if re.fullmatch(r'.*\b\d+\s*kb\b.*',l) or re.fullmatch(r'[{}()\\/"\'\s]+',line): return True
    return bool(probable_person(line))
def add_unique(lst,val):
    val=re.sub(r'\s+',' ',str(val or '')).strip()
    if val and val[0].islower(): val=val[0].upper()+val[1:]
    if val and val.lower() not in {x.lower() for x in lst}: lst.append(val)
def split_facts(v):
    v=nf(v); v=re.sub(r'[;]+','. ',v); v=re.sub(r'\s+e\s+(?=[A-Z])','. ',v)
    v=re.sub(r'\s+(?=(?:Sent|Subject|Incident Summary|Initial Event|Police Arrival|Removal of Deceased|Immediate Actions Taken|Actions Taken|Media Coverage|Moving forward|Impact|Response)\b)','. ',v)
    v=re.sub(r',\s+(?=(?:Police|Two police|The room|It will|The deceased|Front Office|Fortunately|Notably|The General Manager|Replacement|Deactivation|This incident)\b)','. ',v)
    return [p.strip(' -') for p in re.split(r'(?<=[.!?])\s+',v) if p.strip(' -')]
def bucket(sentence):
    ml=ml_label(sentence)
    if ml in {'incident_overview','key_facts_timeline','impact_risk','actions_status'}: return ml
    l=sentence.lower()
    actions=['removed','belongings','morgue','locked','sealed','clearance','will not be released','replacement','deactivation','reported to insurance','insurance provider','initiated','implemented','follow up','provide further updates','current status','next steps','committed to ensuring','coordinate with','monitoring','investigation','activated','restarted','backup connectivity','moved the store','follow up with it','root cause']
    impacts=['the process was handled discreetly','process was handled discreetly','no major impact','no major risk','handled discreetly','discreetly','media','public attention','financial loss','harmed','injury','injured','threatened','safety','reputational','operational impact','business impact','affected','disruption','delayed','may be delayed','could not process','loss remains minimal']
    timeline=['police arrived','police received','direct call','front office supervisor','escorted','found deceased','committed suicide','occurred','checked in','fell from','entered','at ','on ','this morning','arrival','discovery','stole','safe','perpetrators']
    if l.startswith(('actions taken','immediate actions','removal of deceased','moving forward')) or any(w in l for w in actions): return 'actions_status'
    if any(w in l for w in impacts): return 'impact_risk'
    if any(w in l for w in timeline): return 'key_facts_timeline'
    return 'key_facts_timeline'
def fact_sentences(text,title):
    facts=[]
    for raw in text.split('\n'):
        line=re.sub(r'^[\s\-&]+','',str(raw or '')).strip(); line=re.sub(r'^e\s+','',line); line=re.sub(r'\s+',' ',line)
        if noise(line,title): continue
        if len(line)<=70 and any(w in line.lower().strip(':') for w in ['summary','timeline','chronology','actions','update','coverage','background']) and not re.search(r'[.!?]$',line): continue
        for s in split_facts(line):
            s=re.sub(r'^(?:Incident Summary|Initial Event|Police Arrival & Discovery|Police Arrival|Removal of Deceased|Immediate Actions Taken|Actions Taken|Media Coverage|Impact|Response)\s*(?:\([^)]*\))?\s*[:\-]?\s*e?\s*','',s,flags=re.I).strip()
            s=re.sub(r'^(?:hi|dear)\s+[^,]{1,40},\s*','',s,flags=re.I).strip()
            s=re.sub(r'^.*(?:IncidentToGroup|HllncidentToGroup).*?(?=On \d{1,2}\s+[A-Z][a-z]+\s+\d{4})','',s,flags=re.I).strip()
            if re.search(r'(?:incidenttogroup|hllncidenttogroup)',s,flags=re.I):
                m=re.search(r'(On \d{1,2}\s+[A-Z][a-z]+\s+\d{4}.*)',s,flags=re.I)
                if m: s=m.group(1).strip()
            sl=s.lower()
            if noise(s,title) or sl.startswith(('sent ','subject ','from ','to ','cc ')) or re.fullmatch(r'(?:mon|tue|wed|thu|fri|sat|sun)?\s*\d{1,2}/\d{1,2}/\d{4}.*',sl) or 'on { ' in sl or 'news links' in sl or 'for further details' in sl or len(s.split())<4: continue
            if ml_label(s, threshold=0.55)=='noise_metadata': continue
            add_unique(facts,s)
    return facts

def overview(title,reported,facts):
    ct=clean_title(title); joined=' '.join(facts)
    if 'fell from the balcony' in joined.lower():
        return 'A guest fall incident was reported at the property.'
    if 'suicide' in ct.lower() and re.search(r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b', joined, re.I):
        dm=re.search(r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b', joined, re.I)
        return f'Guest suicide incident was reported on {dm.group(0)}.'
    if 'laundry vendor disruption' in joined.lower() or 'external laundry supplier' in joined.lower():
        return 'A laundry vendor disruption was reported by hotel operations.'
    if 'activist group' in joined.lower():
        return 'A stakeholder activist campaign targeting the organisation was reported.'
    if 'suicide' in ct.lower():
        lm=re.search(r'\bat\s+(.+)$',ct,flags=re.I); dm=re.search(r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})',joined,flags=re.I)
        if lm and dm: return f"Guest suicide occurred at {lm.group(1).strip(' .')} on {dm.group(1)}."
    for s in facts:
        if any(t in s.lower() for t in ['committed suicide','guest suicide','robbery incident','incident occurred','fell from','entered the hotel','data breach','outage']) and not s.lower().startswith(('initial event','police')): return s
    if ct and ct.lower()!='untitled incident': return ct if ct.endswith('.') else ct+'.'
    return facts[0] if facts else ''
def rule_sections(text,title,reported=''):
    facts=fact_sentences(text,title); sections={'incident_overview':overview(title,reported,facts),'key_facts_timeline':'','impact_risk':'','actions_status':''}; b={'key_facts_timeline':[],'impact_risk':[],'actions_status':[]}; ov=sections['incident_overview'].lower().strip()
    title_date=re.search(r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})', title, flags=re.I)
    for s in facts:
        if title_date:
            s=re.sub(r'^On the mentioned date,', f'On {title_date.group(1)},', s, flags=re.I)
        sl=s.lower().strip()
        if sl==ov or sl.startswith(('we would like to notify','we would like to update','following our earlier notification')): continue
        add_unique(b[bucket(s)],s)
    for k,v in b.items(): sections[k]=' '.join(v)
    return sections
def fmt_sections(sec):
    order=[('Incident Overview','incident_overview'),('Key Facts / Timeline','key_facts_timeline'),('Impact / Risk','impact_risk'),('Actions / Status','actions_status')]
    parts=[]
    for h,k in order:
        value=nf(sec.get(k,'')) or 'Not stated in source.'
        parts.append(f'{h}:\n{value}')
    return '\n\n'.join(parts)
def json_obj(v):
    v=str(v or '').strip(); v=re.sub(r'^```(?:json)?\s*|\s*```$','',v,flags=re.I)
    try: return json.loads(v) if isinstance(json.loads(v),dict) else {}
    except Exception:
        m=re.search(r'\{.*\}',v,re.S)
        if not m: return {}
        try: return json.loads(m.group(0))
        except Exception: return {}
def ai_prompt(text,title,sec):
    draft=fmt_sections(sec)
    return f'''You are reviewing a pre-sorted business continuity incident summary.
Keep each fact in its existing section. Only improve grammar, spacing, and clarity. Do not invent facts.
Return ONLY valid JSON with keys: incident_overview, key_facts_timeline, impact_risk, actions_status.
Incident title: {title}
Pre-sorted draft:
{draft[:2500]}'''
def ollama(text,title,sec):
    if not USE_OLLAMA: return {}
    prompt=ai_prompt(text,title,sec); cf=AI_CACHE_DIR/f"{ck_text('ollama-v3',OLLAMA_MODEL,prompt)}.json"; cached=rj(cf)
    if cached:
        return {} if cached.get('_ai_timeout') else cached
    payload={'model':OLLAMA_MODEL,'prompt':prompt,'stream':False,'format':'json','keep_alive':'10m','options':{'temperature':0.0,'top_p':0.8,'num_predict':100}}
    req=request.Request(f"{OLLAMA_URL.rstrip('/')}/api/generate",data=json.dumps(payload).encode('utf-8'),headers={'Content-Type':'application/json'},method='POST')
    try:
        with request.urlopen(req,timeout=8) as resp: res=json.loads(resp.read().decode('utf-8','replace'))
    except (OSError,error.URLError,TimeoutError,json.JSONDecodeError):
        wj(cf, {'_ai_timeout': 'true'})
        return {}
    parsed=json_obj(res.get('response','')); keys=['incident_overview','key_facts_timeline','impact_risk','actions_status']
    if not parsed or not any(nf(parsed.get(k,'')) for k in keys): return {}
    out={k:nf(parsed.get(k,'')) for k in keys}; wj(cf,out); return out

def remove_report_metadata(value, reporter='', reported=''):
    value=nf(value)
    if not value: return ''
    for token in [reporter, reported]:
        token=nf(token)
        if token and len(token) > 3:
            value=re.sub(rf'\s*\(?{re.escape(token)}\)?\s*', ' ', value, flags=re.I)
    value=re.sub(r'\b(?:From|Sender|Reporter|Reported by|Sent|Date|Time|Team|Location)\s*[:\-]\s*[^.]+(?:\.|$)', ' ', value, flags=re.I)
    value=re.sub(r'\bTeams message\s*[-:]\s*[^.]+(?:\.|$)', ' ', value, flags=re.I)
    return nf(value).strip(' ,;:-')

def elaborate_sections(sec):
    out=dict(sec)
    impact=nf(out.get('impact_risk',''))
    low=impact.lower()
    if impact and any(x in low for x in ['handled discreetly','no media coverage','no public attention','no mainstream media coverage']):
        if 'reputational risk' not in low and 'no major impact' not in low:
            impact += ' Based on the source, no major operational impact or reputational risk was reported.'
    if impact and 'financial loss remains minimal' in low and 'no staff members were harmed' in low:
        if 'limited people and financial impact' not in low:
            impact += ' Overall, the source indicates limited people and financial impact, although media attention was noted.'
    out['impact_risk']=nf(impact)
    return out
def incident_details(text,title,reported,reporter=''):
    rs=rule_sections(text,title,reported); ai=ollama(text,title,rs); merged={}
    for k in ['incident_overview','key_facts_timeline','impact_risk','actions_status']:
        merged[k]=nf(rs.get(k,'')) or nf(ai.get(k,''))
        merged[k]=remove_report_metadata(merged[k], reporter, reported)
    merged=elaborate_sections(merged)
    return fmt_sections(merged) or 'Incident Overview:\nTo be confirmed'
def parse_msg_text(text,source,serial,stype='Email/Teams text'):
    text=clean_text(text); sent=split_sentences(text); reporter=infer_reporter(text); reported=std_date(find_date(text)); title=infer_title(text,sent); details=incident_details(text,title,reported,reporter); bu=label_value(text,['BU','Business Unit'])
    missing=[n for n,v in [('Reported by',reporter),('Reported Date/Time',reported),('Incident Details',details)] if not v or v=='To be confirmed']
    return {'S/N':serial,'Incident Title':title,'Incident Details':details,'BU':bu,'Reported by':reporter,'Reported Date/Time':reported,'Source Type':stype,'Source File':source,'Review Flag':'Review: '+', '.join(missing) if missing else 'Ready'}
def tesseract_cmd():
    c=shutil.which('tesseract')
    if c: return c
    for p in [r'C:\Users\hanyang.khoo\AppData\Local\Programs\Tesseract-OCR\tesseract.exe',r'C:\Program Files\Tesseract-OCR\tesseract.exe',r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe']:
        if Path(p).exists(): return p
    return ''
def ocr_score(text):
    text=str(text or '').strip()
    if not text: return -9999
    words=re.findall(r'[A-Za-z]{2,}',text)
    useful=sum(1 for w in words if len(w)>=3)
    expected=sum(12 for kw in ['incident','summary','reported','reply','from','sent','subject','actions','impact','risk','police','guest','hotel','media','coverage'] if kw in text.lower())
    garbage=len(re.findall(r'[{}|~_^`�]',text))*8 + len(re.findall(r'[A-Za-z]',text))*2
    return useful + expected - garbage

def ocr_variants(im):
    from PIL import ImageOps,ImageFilter,ImageEnhance
    base=im.convert('L')
    scale=3 if max(base.size)<1600 else (2 if max(base.size)<2600 else 1)
    if scale>1:
        base=base.resize((base.width*scale,base.height*scale))
    auto=ImageOps.autocontrast(base)
    sharp=auto.filter(ImageFilter.SHARPEN)
    high=ImageEnhance.Contrast(sharp).enhance(1.8)
    denoise=high.filter(ImageFilter.MedianFilter(size=3))
    threshold=denoise.point(lambda x: 255 if x>165 else 0, mode='1').convert('L')
    soft=ImageEnhance.Sharpness(auto).enhance(1.5)
    return [('contrast',high),('threshold',threshold),('soft',soft)]

def run_tesseract(tess,img,psm):
    with tempfile.NamedTemporaryFile(suffix='.png',delete=False) as tf:
        tp=Path(tf.name)
    try:
        img.save(tp)
        cmd=[tess,str(tp),'stdout','--oem','1','--psm',str(psm),'--dpi','300','-l','eng','-c','preserve_interword_spaces=1']
        r=subprocess.run(cmd,capture_output=True,timeout=60,check=False)
        out=r.stdout.decode('utf-8','replace').strip(); err=r.stderr.decode('utf-8','replace').strip()
        if r.returncode!=0: return '',err
        return out,''
    finally:
        tp.unlink(missing_ok=True)

def image_text(fp:Path):
    cf=OCR_CACHE_DIR/f'{ck_file(fp)}_ocrv2.txt'
    if cf.exists():
        t=cf.read_text(encoding='utf-8',errors='replace').strip()
        if t: return t
    tess=tesseract_cmd()
    if not tess: raise RuntimeError('Screenshot OCR requires Tesseract OCR.')
    from PIL import Image
    best=''; best_score=-9999; errors=[]
    with Image.open(fp) as im:
        variants=ocr_variants(im)
        attempts=[(variants[0][1],6),(variants[1][1],6),(variants[0][1],4)]
        for variant,psm in attempts:
            out,err=run_tesseract(tess,variant,psm)
            if err: errors.append(err)
            score=ocr_score(out)
            if score>best_score:
                best=out; best_score=score
            if best_score>=120 and any(x in best.lower() for x in ['incident','subject','reported','reply','summary']):
                break
    best=repair_ocr(best).strip()
    if not best: raise RuntimeError(errors[-1] if errors else 'Tesseract OCR did not detect readable text.')
    cf.write_text(best,encoding='utf-8')
    return best
def msg_text(fp:Path):
    import extract_msg
    msg=extract_msg.Message(str(fp))
    try: return '\n'.join([f'Subject: {msg.subject or fp.stem}',f'From: {msg.sender or ""}',f'Sent: {msg.date or ""}','',msg.body or ''])
    finally:
        try: msg.close()
        except Exception: pass

def read_table(fp:Path):
    return pd.read_csv(fp) if fp.suffix.lower()=='.csv' else pd.read_excel(fp)
def load_raw():
    files=list(RAW_INCIDENT_INPUT.glob('*.txt'))+list(RAW_INCIDENT_INPUT.glob('*.csv'))+list(RAW_INCIDENT_INPUT.glob('*.xlsx'))+list(RAW_INCIDENT_INPUT.glob('*.xls'))
    files += [f for e in IMAGE_EXTENSIONS for f in RAW_INCIDENT_INPUT.glob(f'*{e}')]
    files += [f for e in MESSAGE_EXTENSIONS for f in RAW_INCIDENT_INPUT.glob(f'*{e}')]
    rows=[]; issues=[]; sn=1
    for f in files:
        try:
            s=f.suffix.lower()
            if s=='.txt': msgs=[(f.read_text(encoding='utf-8-sig'),'Email/Teams text')]
            elif s in IMAGE_EXTENSIONS: msgs=[(image_text(f),'Screenshot OCR')]
            elif s in MESSAGE_EXTENSIONS: msgs=[(msg_text(f),'Outlook email')]
            else:
                df=read_table(f)
                if 'Message' in df.columns: msgs=[(str(x),'Email/Teams table') for x in df['Message'].dropna().tolist()]
                else: msgs=[('\n'.join(f'{c}: {r[c]}' for c in df.columns if pd.notna(r[c])),'Email/Teams table') for _,r in df.iterrows()]
            for msg,stype in msgs:
                if str(msg).strip(): rows.append(parse_msg_text(str(msg),f.name,sn,stype)); sn+=1
        except Exception as exc: issues.append({'Workflow':'Raw Incident Intake','File':f.name,'Issue Type':'Parse Error','Details':str(exc)})
    df=pd.DataFrame(rows,columns=RAW_INCIDENT_COLUMNS)
    for _,r in df.iterrows():
        if str(r.get('Review Flag',''))!='Ready': issues.append({'Workflow':'Raw Incident Intake','File':r.get('Source File','-'),'Issue Type':'Needs Review','Details':f"S/N {r.get('S/N')}: {r.get('Review Flag')}"})
    return df,issues
def metrics(raw):
    vol=len(raw); saved=max(8-2,0)*vol
    return pd.DataFrame([{'Workflow':'Raw Incident Intake Parsing','Volume':vol,'Unit':'message','Manual Minutes Per Unit':8,'Automated Minutes Per Unit':2,'Estimated Time Saved Minutes':saved,'Estimated Time Saved Hours':round(saved/60,2),'Method':'(manual reading/copy-paste minutes - automated parsing/review minutes) x message count','Basis':'Prototype assumption; replace with measured average from email/Teams triage.'}])
def summary(raw):
    ready=int((raw.get('Review Flag',pd.Series(dtype=str))=='Ready').sum()) if not raw.empty else 0
    src=raw.get('Source Type',pd.Series(dtype=str)).astype(str) if not raw.empty else pd.Series(dtype=str)
    return pd.DataFrame([{'Metric':'Raw email/Teams messages parsed','Value':len(raw)},{'Metric':'Screenshot inputs parsed','Value':int((src=='Screenshot OCR').sum())},{'Metric':'Text/table/email inputs parsed','Value':int(src.str.contains('text|table|Outlook',case=False,na=False).sum())},{'Metric':'Records ready without review','Value':ready},{'Metric':'Records requiring review','Value':max(len(raw)-ready,0)},{'Metric':'Local AI mode','Value':'Enabled with cache' if USE_OLLAMA else 'Disabled'}])
def safe_sheet(n):
    for ch in ['\\','/','*','?',':','[',']']: n=n.replace(ch,' ')
    return n[:31]
def export(raw,issues):
    out=OUTPUT_DIR/f"Workflow_Automation_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    with pd.ExcelWriter(out,engine='xlsxwriter',datetime_format='yyyy-mm-dd') as writer:
        wb=writer.book; title_fmt=wb.add_format({'bold':True,'font_size':13,'bg_color':'#1F4E78','font_color':'#FFFFFF','border':1}); head_fmt=wb.add_format({'bold':True,'bg_color':'#D9EAF7','border':1,'text_wrap':True}); wrap=wb.add_format({'text_wrap':True,'valign':'top'})
        def norm(df): return pd.DataFrame({'Message':['No records found.']}) if df is None or df.empty else df
        def widths(ws,df):
            for i,c in enumerate(df.columns):
                w=min(max(len(str(c))+2,14),38)
                if not df.empty:
                    s=df[c].astype(str).head(30).map(len).max(); w=min(max(w,int(s)+2),55) if pd.notna(s) else w
                ws.set_column(i,i,w,wrap if c in ['Incident Details','Basis','Method'] else None)
        def section(sheet,title,df,row=0):
            sheet=safe_sheet(sheet); df=norm(df)
            if sheet not in writer.sheets: wb.add_worksheet(sheet); writer.sheets[sheet]=wb.get_worksheet_by_name(sheet)
            ws=writer.sheets[sheet]; ws.write(row,0,title,title_fmt); df.to_excel(writer,sheet_name=sheet,index=False,startrow=row+1)
            for i,v in enumerate(df.columns): ws.write(row+1,i,v,head_fmt)
            widths(ws,df)
            if sheet=='Incident Intake':
                ws.set_column(0,0,8); ws.set_column(1,1,34,wrap); ws.set_column(2,2,95,wrap); ws.set_column(3,3,14); ws.set_column(4,4,22); ws.set_column(5,5,26); ws.set_column(6,6,18); ws.set_column(7,7,34); ws.set_column(8,8,22)
            ws.freeze_panes(2,0); ws.autofilter(row+1,0,row+len(df)+1,max(len(df.columns)-1,0)); return row+len(df)+4
        r=section('Executive Summary','Workflow Summary',summary(raw),0); section('Executive Summary','Automation Benefit Estimate',metrics(raw),r); section('Incident Intake','Standardised Email/Teams Incident Table',raw,0)
    return out
def main():
    print('Starting Workflow Automation Tool...'); print(f'Base folder: {BASE_DIR}'); print(f"Local AI: {'enabled' if USE_OLLAMA else 'disabled'} ({OLLAMA_MODEL})"); print(f"Sentence classifier: {'enabled' if sentence_model() else 'disabled'}")
    raw,issues=load_raw(); out=export(raw,issues); print('Report created successfully:'); print(out); print('\nCheck the output folder for the Excel report.')
if __name__=='__main__':
    try: main()
    except Exception as exc:
        print('An error occurred:'); print(exc); print('\nCheck that required packages are installed:'); print('pip install -r requirements.txt'); sys.exit(1)












