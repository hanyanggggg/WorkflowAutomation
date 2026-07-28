from pathlib import Path
import json
import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR / "input" / "training" / "automation_labelled_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = MODEL_DIR / "incident_sentence_classifier.joblib"
REPORT_FILE = MODEL_DIR / "incident_sentence_classifier_report.txt"
METADATA_FILE = MODEL_DIR / "incident_sentence_classifier_metadata.json"

REQUIRED_COLUMNS = {"text", "label"}
VALID_LABELS = {"incident_overview", "key_facts_timeline", "impact_risk", "actions_status", "noise_metadata"}


def main():
    if not DATASET.exists():
        raise FileNotFoundError(f"Training dataset not found: {DATASET}")

    df = pd.read_csv(DATASET)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["label"] != "")]

    unknown = sorted(set(df["label"]) - VALID_LABELS)
    if unknown:
        raise ValueError(f"Unknown labels found: {unknown}")

    label_counts = df["label"].value_counts().to_dict()
    stratify = df["label"] if df["label"].value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=stratify
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, strip_accents="unicode")),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
    ])
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=0)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dump(model, MODEL_FILE)
    REPORT_FILE.write_text(
        f"Incident sentence classifier training report\n"
        f"Dataset: {DATASET}\n"
        f"Rows used: {len(df)}\n"
        f"Train rows: {len(X_train)}\n"
        f"Test rows: {len(X_test)}\n"
        f"Accuracy: {accuracy:.4f}\n\n"
        f"Label counts:\n{pd.Series(label_counts).to_string()}\n\n"
        f"Classification report:\n{report}\n",
        encoding="utf-8",
    )
    METADATA_FILE.write_text(json.dumps({
        "dataset": str(DATASET),
        "rows_used": int(len(df)),
        "accuracy": float(accuracy),
        "labels": sorted(df["label"].unique().tolist()),
        "label_counts": {k: int(v) for k, v in label_counts.items()},
        "model_file": str(MODEL_FILE),
    }, indent=2), encoding="utf-8")

    print(f"Rows used: {len(df)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Model saved: {MODEL_FILE}")
    print(f"Report saved: {REPORT_FILE}")


if __name__ == "__main__":
    main()
