import argparse
import json
import os
import zipfile
import warnings
import re
from datetime import datetime
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

def clean_english_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def run_svm_analysis(test_size):
    # Setup paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = Path(f"svm_results_{timestamp}")
    export_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv("dataset_dengan_label_20260412_061422.csv")
    
    # Undersampling: Balance classes
    min_count = df["label"].value_counts().min()
    df = pd.concat([
        df[df["label"] == label].sample(min_count, random_state=42)
        for label in df["label"].unique()
    ])
    
    # Preprocessing
    df["text_clean"] = df["text_clean"].apply(clean_english_text)
    X = df["text_clean"].fillna("").astype(str)
    y = df["label"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ("svc", SVC())
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
    parser = argparse.ArgumentParser(description="SVM Sentiment Analysis CLI Tool")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size (0.1 - 0.5)")
    args = parser.parse_args()
    
    run_svm_analysis(args.test_size)
