import argparse
import json
import os
import zipfile
import warnings
import re
from datetime import datetime
from pathlib import Path
import sys
import subprocess

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# ==================== INSTALL PACKAGE ====================
def install_package(package_name):
    try:
        print(f"Sedang menginstal {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"Berhasil menginstal {package_name}!")
    except subprocess.CalledProcessError as e:
        print(f"Gagal menginstal {package_name}. Error: {e}")


# ==================== SETUP ====================
DetectorFactory.seed = 0
warnings.filterwarnings("ignore", category=UserWarning)

# Basic English stopwords
STOPWORDS = set(['a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                 'be', 'been', 'being', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'of', 'about'])


# ==================== HELPER FUNCTIONS ====================
def deteksi_bahasa(teks):
    """Deteksi bahasa dengan penanganan error yang lebih baik"""
    if pd.isna(teks) or not isinstance(teks, str) or not str(teks).strip():
        return "kosong"
    
    try:
        kode_bahasa = detect(teks)
        kamus_bahasa = {
            'id': 'Indonesia',
            'en': 'Inggris',
            'ms': 'Malaysia',
            'id_MS': 'Malaysia'
        }
        return kamus_bahasa.get(kode_bahasa, "kosong")
    except LangDetectException:
        return "kosong"


def count_others_symbol(text):
    if pd.isna(text):
        return 0
    return len(re.findall(r'[^\x20-\x7E\s]', str(text)))


def clean_english_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # Handle negation
    text = re.sub(r'not\s+([a-z]+)', r'not_\1', text)
    # Remove special chars except underscore
    text = re.sub(r'[^a-zA-Z_\s]', '', text)
    # Remove stopwords
    words = text.split()
    text = ' '.join([w for w in words if w not in STOPWORDS])
    # Simple lemmatization
    text = re.sub(r'(ing|ed|s|es|ly)$', '', text, flags=re.IGNORECASE)
    return text.strip()


# ==================== MAIN ANALYSIS ====================
def run_svm_analysis(test_size=0.2):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = Path(f"svm_results_{timestamp}")
    export_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Memuat dataset...")
    df = pd.read_excel("try_only_en_id_dataset_spaylater.xlsx")

    # Pastikan kolom yang diperlukan ada
    if 'full_text' not in df.columns:
        raise KeyError("Kolom 'full_text' tidak ditemukan di dataset!")
    if 'label' not in df.columns:
        raise KeyError("Kolom 'label' tidak ditemukan di dataset!")

    # Undersampling untuk balance class
    print("Melakukan undersampling...")
    min_count = df["label"].value_counts().min()
    df = pd.concat([
        df[df["label"] == label].sample(min_count, random_state=42)
        for label in df["label"].unique()
    ]).reset_index(drop=True)

    # Preprocessing
    print("Preprocessing teks...")
    df["others_symbol_count"] = df["full_text"].apply(count_others_symbol)
    df["text_clean"] = df["full_text"].apply(clean_english_text)   # ← Diperbaiki

    df["bahasa"] = df["text_clean"].apply(deteksi_bahasa)

    X = df["text_clean"].fillna("")
    y = df["label"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Pipeline + GridSearch
    print("Melatih model SVM dengan GridSearch...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 3), max_features=5000)),
        ("svc", SVC(class_weight='balanced'))
    ])

    param_grid = {
        "tfidf__use_idf": [True, False],
        "tfidf__max_df": [0.8, 0.9, 1.0],
        "svc__C": [0.1, 1, 10, 100],
        "svc__kernel": ["linear", "rbf", "poly"],
        "svc__gamma": ["scale", "auto"]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    accuracy = grid.score(X_test, y_test)
    print(f"\nModel Accuracy: {accuracy:.4f}")

    # Predict full dataset
    df["prediksi"] = grid.predict(X)

    # Save results
    df.to_csv(export_dir / "dataset_with_predictions.csv", index=False)

    test_results = pd.DataFrame({
        'text': X_test, 
        'actual': y_test, 
        'prediksi': grid.predict(X_test)
    })
    test_results.to_csv(export_dir / "test_set_predictions.csv", index=False)

    # Classification Report
    with open(export_dir / "classification_report.json", "w") as f:
        json.dump(classification_report(y_test, grid.predict(X_test), output_dict=True), f, indent=4)
    
    with open(export_dir / "classification_report.txt", "w") as f:
        f.write(classification_report(y_test, grid.predict(X_test)))

    # Confusion Matrix
    cm = confusion_matrix(y_test, grid.predict(X_test))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(export_dir / "confusion_matrix.png")
    plt.close()

    # Best Parameters
    with open(export_dir / "best_params.json", "w") as f:
        json.dump(grid.best_params_, f, indent=4)

    # Zip results
    zip_filename = f"svm_results_{timestamp}.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for file in export_dir.iterdir():
            zipf.write(file, file.name)

    print(f"\n✅ Analisis selesai! Hasil disimpan di folder: {export_dir}")
    print(f"📦 File zip: {zip_filename}")


# ==================== CLI ====================
if __name__ == "__main__":
    install_package("openpyxl")   # Hanya dipanggil sekali

    parser = argparse.ArgumentParser(description="SVM Sentiment Analysis CLI Tool")
    parser.add_argument("--test-size", type=float, default=0.2, 
                       help="Test set size (0.1 - 0.5)")
    args = parser.parse_args()

    run_svm_analysis(args.test_size)
