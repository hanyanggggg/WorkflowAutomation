# Workflow Automation Prototype

Python-based workflow automation prototype for consolidating messy BCM incident inputs into a standardised Excel report.

## Overview

Business continuity incident information often arrives through unstructured channels such as emails, Teams messages, screenshots, Outlook message files, and spreadsheets. This prototype reduces manual copy-paste work by extracting incident information, validating key fields, categorising incident details, and generating a clean Excel report.

This repository uses sample/dummy data only. No confidential company data is included.

## Problem Statement

The BCM team may receive incident updates in different formats and wording styles. Manual consolidation creates several issues:

- Time-consuming copy-paste and reformatting work
- Inconsistent incident summaries across submissions
- Missing fields such as reporter, date/time, impact, or action status
- Difficulty reviewing screenshot-based submissions
- Slower preparation of management reports

## Solution

The tool automates the incident intake workflow by:

- Reading raw incident files from an input folder
- Extracting text from screenshots using OCR
- Parsing sender, date/time, title, source type, and source file
- Categorising details into a standard incident format
- Using local AI to polish summaries where available
- Flagging records that require review
- Exporting a standardised Excel report

## Key Features

- Screenshot OCR using Tesseract
- Local AI summarisation using Ollama
- Optional logistic regression sentence classifier
- Standard incident detail template:
  - Incident Overview
  - Key Facts / Timeline
  - Impact / Risk
  - Actions / Status
- Missing-field validation
- Executive summary tab
- Excel report generation

## Technology Used

- Python
- pandas
- openpyxl
- XlsxWriter
- Pillow
- Tesseract OCR
- Ollama local LLM
- scikit-learn / joblib for the optional sentence classifier

## Folder Structure

```text
workflow-automation-github-ready/
├─ workflow_automation_tool.py
├─ train_incident_classifier.py
├─ requirements.txt
├─ README.md
├─ .gitignore
├─ input/
│  ├─ raw_incidents/
│  │  ├─ sample_email_incident.txt
│  │  └─ sample_teams_incident.txt
│  └─ training/
│     └─ sample_labelled_dataset.csv
├─ models/
│  └─ README.md
├─ output/
│  └─ README.md
└─ docs/
   └─ Workflow_Automation_User_Guide.docx
```

## Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Optional dependencies:

- Install Tesseract OCR if processing screenshots.
- Install Ollama if using local AI summaries.
- Pull the local model:

```bash
ollama pull llama3.2:3b
```

## How To Run

Place incident files into:

```text
input/raw_incidents/
```

Run:

```bash
python workflow_automation_tool.py
```

The generated Excel report will appear in:

```text
output/
```

## Input Types Supported

- `.txt` email or Teams message exports
- `.msg` Outlook messages
- `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff` screenshots
- `.csv`, `.xlsx`, `.xls` tables

## Important Notes

- OCR accuracy depends on screenshot quality.
- AI summaries should be reviewed before management use.
- Missing facts cannot be invented by the tool.
- For best accuracy, use `.msg` or `.txt` files instead of screenshots where possible.

## Limitations

- Prototype only; not production-governed.
- Not directly integrated with Outlook, Teams, SharePoint, or a case management platform.
- Local setup requires Tesseract and Ollama for full functionality.
- Classifier accuracy depends on labelled training examples.
- Sensitive or personal data should not be committed to GitHub.

## Future Enhancements

- Direct Outlook/Teams/SharePoint intake
- Web dashboard
- Approval and review workflow
- Audit logs
- Larger labelled training dataset
- Production access controls and retention rules

## Data Privacy

This repository is designed for demonstration only. Use dummy, anonymised, or approved non-confidential data. Do not upload real incident emails, screenshots, names, reports, or internal documents.
