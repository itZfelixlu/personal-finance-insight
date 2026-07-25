"""Self-check script for the transaction categorizer.

Idea: hide the true labels, let the model predict, then compare predictions
against the truth transaction-by-transaction. Reports overall accuracy,
per-class scores, and lists exactly which transactions were misclassified.

For an HONEST estimate we train on a training split and evaluate on a held-out
test split the model has never seen (not on the data it was trained on).

Usage:
    python scripts/check_model.py
    python scripts/check_model.py --show-errors 40   # list up to 40 mistakes
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.pipeline.data_loader import load_transactions_csv
from src.pipeline.preprocessing import preprocess_transactions

DATA_FILE = PROJECT_ROOT / "inputs" / "synthetic_chase_statement_2025.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-errors", type=int, default=25,
                        help="max number of misclassified rows to print")
    parser.add_argument("--data", default=str(DATA_FILE))
    args = parser.parse_args()

    # 1. Load + preprocess (same modules the app uses)
    df = preprocess_transactions(load_transactions_csv(args.data))
    print(f"Loaded {len(df)} labelled transactions from {Path(args.data).name}\n")

    X, y = df["Description"], df["Category"]

    # 2. Split: the test set is held out — the model never sees it during training
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Train the same pipeline used in the notebook
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("clf", LinearSVC(class_weight="balanced")),
    ])
    model.fit(X_train, y_train)

    # 4. Predict on the hidden test set and compare to the truth
    pred = model.predict(X_test)

    print("=" * 60)
    print(f"Overall accuracy : {accuracy_score(y_test, pred):.3f}")
    print(f"Macro-F1         : {f1_score(y_test, pred, average='macro'):.3f}")
    print("=" * 60)
    print("\nPer-category report:\n")
    print(classification_report(y_test, pred))

    # 5. Show the actual mistakes so you can judge if they are reasonable
    check = df.loc[idx_test, ["Description", "Category"]].copy()
    check["Predicted"] = pred
    wrong = check[check["Category"] != check["Predicted"]]

    print(f"Misclassified: {len(wrong)} / {len(check)} "
          f"({len(wrong) / len(check):.1%})\n")
    if len(wrong):
        print(f"{'Description':<40} {'True':<14} {'Predicted':<14}")
        print("-" * 68)
        for _, r in wrong.head(args.show_errors).iterrows():
            print(f"{r['Description'][:39]:<40} {r['Category']:<14} {r['Predicted']:<14}")


if __name__ == "__main__":
    main()
