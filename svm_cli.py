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
import unicodedata

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


SLANG_MAP = {
    # Jaksel / Slang Inggris-Indo
    "literally": "benar-benar", "which-is": "yang mana", "prefer": "lebih suka",
    "fyi": "sebagai informasi", "btw": "ngomong-ngomong", "tbh": "sejujurnya",
    # Slang Indonesia / Chat shortened
    "yg": "yang", "gk": "tidak", "ga": "tidak", "gak": "tidak", "tdk": "tidak",
    "bgt": "banget", "mager": "malas gerak", "healing": "liburan",
    "lu": "kamu", "lo": "kamu", "gw": "saya", "gue": "saya", "aq": "saya",
    "dgn": "dengan", "dlm": "dalam", "bisaa": "bisa", "jd": "jadi",
    # Tambahkan padanan lain sesuai temuan di dataset Anda
}

# Gabungan Stopwords (Inggris + Indonesia) yang sudah difilter agar kata sentimen/negasi TIDAK hilang
# Catatan: Kata seperti "tidak", "bukan", "not", "no" HARUS dihapus dari daftar stopwords 
# agar tidak merusak analisis sentimen.
CUSTOM_STOPWORDS = {
    # Indo
    "yang", "untuk", "pada", "ke", "para", "namun", "menurut", "atau", "dan", "bahwa", "di",
    # English
    "the", "a", "an", "and", "is", "are", "was", "were", "to", "of", "for", "in", "on", "at"
}

def clean_mixed_twitter_text(text):
    if pd.isna(text):
        return ""
    
    # 1. Decode HTML & Ubah ke lowercase
    text = str(text).lower()
    
    # 2. Hapus URL, Twitter Username (@user), dan Hashtag (#tag)
    # Catatan: Jika hashtag mengandung makna sentimen (misal #kecewa), 
    # Anda bisa menghapus simbol '#' saja dengan re.sub(r'#([^\s]+)', r'\1', text)
    text = re.sub(re.compile(r"https?://\S+|www\.\S+"), "", text)
    text = re.sub(re.compile(r"@[^\s]+"), "", text)
    text = re.sub(re.compile(r"#[^\s]+"), "", text)
    
    # 3. Bersihkan Emoji dan Karakter Non-ASCII
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # 4. Standarisasi Karakter Berulang (misal: "bangeeeet" -> "banget", "loooove" -> "love")
    # Mengurangi huruf yang berulang lebih dari 2 kali menjadi maksimal 2 huruf
    text = re.sub(r'(.)\1+', r'\1\1', text)
    
    # 5. Hapus Tanda Baca dan Karakter Spesial (Kecuali Huruf dan Spasi)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # 6. Tokenisasi (Pecah jadi kata) untuk pemrosesan kata demi kata
    words = text.split()
    
    # 7. Normalisasi Slang / Bahasa Jaksel
    words = [SLANG_MAP.get(word, word) for word in words]
    
    # 8. Handling Negasi Campuran (Menghubungkan 'not' atau 'tidak' dengan kata setelahnya)
    # Contoh: "not good" -> "not_good", "tidak suka" -> "tidak_suka"
    cleaned_words = []
    skip = False
    for i in range(len(words)):
        if skip:
            skip = False
            continue
        
        # Cek jika kata saat ini adalah kata negasi dan ada kata berikutnya
        if words[i] in ['not', 'tidak', 'tak', 'gak', 'kurang'] and (i + 1) < len(words):
            # Gabungkan dengan kata setelahnya
            cleaned_words.append(f"{words[i]}_{words[i+1]}")
            skip = True  # Lewati kata berikutnya pada iterasi selanjutnya
        else:
            cleaned_words.append(words[i])
            
    # 9. Filter Stopwords (Hanya hapus kata yang benar-benar tidak bermakna)
    final_words = [w for w in cleaned_words if w not in CUSTOM_STOPWORDS]
    
    # 10. Gabungkan kembali dan bersihkan spasi berlebih
    return " ".join(final_words).strip()

# Kamus singkatan umum (bisa diperluas)
SLANG_MAP = {
    "gk": "tidak",
    "ga": "tidak",
    "gak": "tidak",
    "yg": "yang",
    "dgn": "dengan",
    "utk": "untuk",
    "krn": "karena",
    "jd": "jadi",
    "tp": "tapi",
    "tpi": "tapi",
    "sm": "sama",
    "bgt": "banget",
    "banget": "banget",
    "trs": "terus",
    "knp": "kenapa",
    "sy": "saya",
    "gue": "saya",
    "lu": "kamu",
    "udh": "sudah",
    "udah": "sudah",
}

URL_RE = re.compile(r"http\S+|www\.\S+")
EMAIL_RE = re.compile(r"[\w.-]+@[\w.-]+")
MENTION_RE = re.compile(r"@\w+")
MULTISPACE_RE = re.compile(r"\s+")
REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

_factory_sw = StopWordRemoverFactory()
_stopword_remover = _factory_sw.create_stop_word_remover()
_factory_stem = StemmerFactory()
_stemmer = _factory_stem.create_stemmer()


def expand_slang_tokens(text: str) -> str:
    toks = text.split()
    out = [SLANG_MAP.get(t, t) for t in toks]
    return " ".join(out)


def apply_sastrawi_stopword_and_stem(text: str) -> str:
    """Hapus stopword lalu stem dengan Sastrawi (Bahasa Indonesia)."""
    if not text or not str(text).strip():
        return text if isinstance(text, str) else str(text)
    t = _stopword_remover.remove(text)
    t = MULTISPACE_RE.sub(" ", t).strip()
    if not t:
        return ""
    t = _stemmer.stem(t)
    return MULTISPACE_RE.sub(" ", str(t)).strip()


def preprocess_text(text: str, remove_digits: bool = False, use_sastrawi: bool = True) -> str:
    if not isinstance(text, str):
        text = str(text)
    # Unicode NFKC
    t = unicodedata.normalize("NFKC", text)
    t = t.strip()
    # URL & email
    t = URL_RE.sub(" ", t)
    t = EMAIL_RE.sub(" ", t)
    # Mention → token umum (pertahankan konteks bahwa ada mention)
    t = MENTION_RE.sub(" mention ", t)
    # Hashtag: pertahankan kata tanpa #
    t = re.sub(r"#(\w+)", r"\1", t)
    # Huruf kecil
    t = t.lower()
    # Angka opsional
    if remove_digits:
        t = re.sub(r"\d+", " ", t)
    # Tanda baca berlebih → spasi
    t = re.sub(r"[\[\](){}\[\]<>]", " ", t)
    # Ulang karakter berlebihan (mis. "bagusssss" → "bagus")
    t = REPEAT_CHAR_RE.sub(r"\1\1", t)
    t = MULTISPACE_RE.sub(" ", t).strip()
    # Singkatan
    t = expand_slang_tokens(t)
    if use_sastrawi:
        t = apply_sastrawi_stopword_and_stem(t)
    return t


def preprocess_series(series: pd.Series, **kwargs) -> pd.Series:
    return series.astype(str).map(lambda x: preprocess_text(x, **kwargs))


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
    df["text_clean"] = preprocess_series(df["full_text"], remove_digits=False)
    df["text_clean"] = df["text_clean"].apply(clean_mixed_twitter_text)   # ← Diperbaiki
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
    df.to_csv(export_dir / "dataset_with_predictions.csv")

    test_results = pd.DataFrame({
        'text': X_test, 
        'actual': y_test, 
        'prediksi': grid.predict(X_test)
    })
    test_results.to_csv(export_dir / "test_set_predictions.csv)

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
