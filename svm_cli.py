import argparse
import json
import os
import zipfile
import warnings
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

def run_svm_analysis(test_size):
    # Setup paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = Path(f"svm_results_{timestamp}")
    export_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv("dataset_dengan_label_20260412_061422.csv")
    
    # Preprocessing (Vectorizer only)
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

    # Predictions
    y_pred = grid.predict(X_test)

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
