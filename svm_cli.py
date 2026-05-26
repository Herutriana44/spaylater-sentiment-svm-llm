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

def install_package(package_name):
    try:
        print(f"Sedang menginstal {package_name}...")
        # Menggunakan sys.executable memastikan pip berjalan di lingkungan Python yang sama
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"Berhasil menginstal {package_name}!")
    except subprocess.CalledProcessError as e:
        print(f"Gagal menginstal {package_name}. Error: {e}")
install_package("openpyxl")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Memastikan hasil deteksi konsisten dan tidak berubah-ubah
DetectorFactory.seed = 0

def deteksi_bahasa(teks):
    # Validasi input: jika bukan string atau string kosong
    if not teks or not isinstance(teks, str).strip():
        return False
    
    try:
        # Mendeteksi kode bahasa (contoh: 'id', 'en', 'tl')
        kode_bahasa = detect(teks)
        
        # Pemetaan kode bahasa ke nama bahasa yang diinginkan
        kamus_bahasa = {
            'id': 'Indonesia',
            'en': 'Inggris',
            'id_MS': 'Malaysia',  # Beberapa kasus langdetect membaca melayu/malaysia sebagai id atau ms
            'ms': 'Malaysia'
        }
        
        # Kembalikan nama bahasa jika cocok, jika tidak kembalikan False
        return kamus_bahasa.get(kode_bahasa, False)
        
    except LangDetectException:
        # Mengatasi error jika teks tidak mengandung huruf (misal: hanya angka/simbol)
        return False

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Basic list of English stopwords for filtering without NLTK
STOPWORDS = set(['a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'of', 'about'])

def count_others_symbol(text):
    """
    Counts characters that are NOT alphabet, numeric, or common keyboard symbols.
    Common keyboard symbols are generally ASCII printable characters (range 32-126).
    We also exclude common whitespace characters (space, tab, newline, etc.).
    """
    if pd.isna(text):
        return 0
    # [^\x20-\x7E\s] matches any character that is NOT a printable ASCII or a whitespace
    return len(re.findall(r'[^\x20-\x7E\s]', str(text)))

def clean_english_text(text):
    text = str(text).lower()
    # Handle negation
    text = re.sub(r'not\s+([a-z]+)', r'not_\1', text)
    # Remove special chars
    text = re.sub(r'[^a-zA-Z_\s]', '', text)
    # Remove stopwords
    words = text.split()
    text = ' '.join([w for w in words if w not in STOPWORDS])
    # Basic lemmatization (simple truncation for common suffixes)
    text = re.sub(r'(ing|ed|s|es|ly)$', '', text)
    return text.strip()

def run_svm_analysis(test_size):
    # Setup paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = Path(f"svm_results_{timestamp}")
    export_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    # df = pd.read_csv("dataset_dengan_label_20260412_061422.csv")
    df = pd.read_excel("try_only_en_id_dataset_spaylater.xlsx")

    # Undersampling: Balance classes
    min_count = df["label"].value_counts().min()
    df = pd.concat([
        df[df["label"] == label].sample(min_count, random_state=42)
        for label in df["label"].unique()
    ])

    df = df.reset_index(drop=True)

    # Preprocessing
    df["others_symbol_count"] = df["full_text"].apply(count_others_symbol)
    df["text_clean"] = df["text_clean"].apply(clean_english_text)

    df["bahasa"] = df["text_clean"].apply(deteksi_bahasa)

    X = df["text_clean"].fillna("").astype(str)
    y = df["label"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Pipeline with class_weight='balanced'
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 3), max_features=5000)),
        ("svc", SVC(class_weight='balanced'))
    ])


    # Expanded parameter grid
    param_grid = {
        "tfidf__use_idf": [True, False],
        "tfidf__max_df": [0.8, 0.9, 1.0],
        "svc__C": [0.1, 1, 10, 100],
        "svc__kernel": ["linear", "rbf", "poly"],
        "svc__gamma": ["scale", "auto"]
    }

    # Grid Search
    grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    # Accuracy logging
    accuracy = grid.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.4f}")

    # Predict on entire dataset
    df["prediksi"] = grid.predict(df["text_clean"].fillna("").astype(str))
    df.to_csv(export_dir / "dataset_with_predictions.csv", index=False)

    # Predictions
    y_pred = grid.predict(X_test)
    
    # Export test predictions
    test_results = pd.DataFrame({'text': X_test, 'actual': y_test, 'prediksi': y_pred})
    test_results.to_csv(export_dir / "test_set_predictions.csv", index=False)

    # Save Results
    # 1. Classification Report
    report = classification_report(y_test, y_pred, output_dict=True)
    with open(export_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=4)
    
    with open(export_dir / "classification_report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred))

    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(export_dir / "confusion_matrix.png")

    # 3. Best params
    with open(export_dir / "best_params.json", "w") as f:
        json.dump(grid.best_params_, f, indent=4)

    # Create Zip
    zip_filename = f"svm_results_{timestamp}.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for file in export_dir.iterdir():
            zipf.write(file, file.name)
    
    print(f"Analysis complete. Results exported to {export_dir} and zipped as {zip_filename}")

if __name__ == "__main__":
    install_package("openpyxl")
    parser = argparse.ArgumentParser(description="SVM Sentiment Analysis CLI Tool")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size (0.1 - 0.5)")
    args = parser.parse_args()
    
    run_svm_analysis(args.test_size)
