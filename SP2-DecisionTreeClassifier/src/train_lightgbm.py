from pathlib import Path

import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


ID_COLUMNS = ["file_id", "label", "cleaned_path"]


def main():
    df = pd.read_csv("data/processed/features.csv")

    X = df.drop(columns=ID_COLUMNS)
    y = df["label"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        stratify=y_encoded,
        random_state=42,
    )

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(label_encoder.classes_),
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    macro_f1 = f1_score(y_test, y_test_pred, average="macro")

    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_test_pred)

    report = classification_report(
        y_test_labels,
        y_pred_labels,
        digits=4,
    )

    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "label_encoder": label_encoder,
            "feature_names": X.columns.tolist(),
        },
        "models/lightgbm_model.joblib",
    )

    Path("reports/lightgbm_report.txt").write_text(
        "LightGBM Classifier\n"
        "===================\n\n"
        f"Train accuracy: {train_accuracy:.4f}\n"
        f"Test accuracy: {test_accuracy:.4f}\n"
        f"Macro F1: {macro_f1:.4f}\n\n"
        f"{report}\n",
        encoding="utf-8",
    )

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance_df.to_csv("reports/lightgbm_feature_importance.csv", index=False)

    print("========== LIGHTGBM SUMMARY ==========")
    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print()
    print(report)
    print("Top 20 important features:")
    print(importance_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()