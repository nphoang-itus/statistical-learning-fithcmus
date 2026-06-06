import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split


ID_COLUMNS = ["file_id", "label", "cleaned_path"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", default="data/processed/features.csv")
    parser.add_argument("--model-path", default="models/random_forest_unknown.joblib")
    parser.add_argument("--report-path", default="reports/random_forest_report.txt")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.features_path)

    X = df.drop(columns=ID_COLUMNS)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=args.random_state,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=args.random_state,
    )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    macro_f1 = f1_score(y_test, y_test_pred, average="macro")

    report = classification_report(y_test, y_test_pred, digits=4)

    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, args.model_path)

    Path(args.report_path).write_text(
        "Random Forest Tree-based Model\n"
        "==============================\n\n"
        f"Train accuracy: {train_accuracy:.4f}\n"
        f"Test accuracy: {test_accuracy:.4f}\n"
        f"Macro F1: {macro_f1:.4f}\n\n"
        f"{report}\n",
        encoding="utf-8",
    )

    print("========== RANDOM FOREST SUMMARY ==========")
    print(f"Model saved to: {args.model_path}")
    print(f"Report saved to: {args.report_path}")
    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print()
    print(report)


if __name__ == "__main__":
    main()