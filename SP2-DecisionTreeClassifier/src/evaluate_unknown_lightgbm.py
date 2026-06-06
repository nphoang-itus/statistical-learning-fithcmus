import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


TRAIN_ID_COLUMNS = ["file_id", "label", "cleaned_path"]
UNKNOWN_ID_COLUMNS = ["file_id", "label", "actual_label", "cleaned_path"]


def align_unknown_features(known_features_df, unknown_features_df):
    train_feature_cols = [
        col for col in known_features_df.columns
        if col not in TRAIN_ID_COLUMNS
    ]

    missing_cols = [col for col in train_feature_cols if col not in unknown_features_df.columns]
    extra_cols = [
        col for col in unknown_features_df.columns
        if col not in UNKNOWN_ID_COLUMNS and col not in train_feature_cols
    ]

    if missing_cols:
        raise ValueError(f"Unknown features missing columns: {missing_cols}")

    if extra_cols:
        print(f"[WARN] Unknown features have extra columns ignored: {extra_cols}")

    return unknown_features_df[train_feature_cols]


def predict_with_threshold(model, label_encoder, X, threshold):
    proba = model.predict_proba(X)
    max_proba = proba.max(axis=1)
    class_indices = proba.argmax(axis=1)

    raw_pred = label_encoder.inverse_transform(class_indices)
    final_pred = np.where(max_proba < threshold, "UNKNOWN", raw_pred)

    return final_pred, max_proba, raw_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", default="data/processed/features.csv")
    parser.add_argument("--unknown-features-path", default="data/processed/unknown_features.csv")
    parser.add_argument("--model-path", default="models/lightgbm_model.joblib")
    parser.add_argument("--output-path", default="reports/unknown_threshold_results_lightgbm.csv")
    args = parser.parse_args()

    known_df = pd.read_csv(args.features_path)
    unknown_df = pd.read_csv(args.unknown_features_path)

    bundle = joblib.load(args.model_path)
    model = bundle["model"]
    label_encoder = bundle["label_encoder"]

    X_known = known_df.drop(columns=TRAIN_ID_COLUMNS)
    y_known = known_df["label"]

    _, X_known_test, _, y_known_test = train_test_split(
        X_known,
        y_known,
        test_size=0.2,
        stratify=y_known,
        random_state=42,
    )

    X_unknown = align_unknown_features(known_df, unknown_df)
    y_unknown = pd.Series(["UNKNOWN"] * len(X_unknown))

    X_eval = pd.concat([X_known_test, X_unknown], ignore_index=True)
    y_eval = pd.concat(
        [y_known_test.reset_index(drop=True), y_unknown],
        ignore_index=True,
    )

    rows = []

    for threshold in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99]:
        y_pred, max_proba, raw_pred = predict_with_threshold(
            model=model,
            label_encoder=label_encoder,
            X=X_eval,
            threshold=threshold,
        )

        known_mask = y_eval != "UNKNOWN"
        unknown_mask = y_eval == "UNKNOWN"

        known_accuracy = accuracy_score(y_eval[known_mask], y_pred[known_mask])
        unknown_recall = (y_pred[unknown_mask] == "UNKNOWN").mean()
        overall_accuracy = accuracy_score(y_eval, y_pred)

        known_rejected_rate = (y_pred[known_mask] == "UNKNOWN").mean()
        unknown_accepted_rate = (y_pred[unknown_mask] != "UNKNOWN").mean()

        rows.append({
            "threshold": threshold,
            "overall_accuracy": overall_accuracy,
            "known_accuracy_after_rejection": known_accuracy,
            "unknown_recall": unknown_recall,
            "known_rejected_rate": known_rejected_rate,
            "unknown_accepted_rate": unknown_accepted_rate,
            "known_samples": int(known_mask.sum()),
            "unknown_samples": int(unknown_mask.sum()),
        })

    result_df = pd.DataFrame(rows)

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output_path, index=False)

    print("========== LIGHTGBM UNKNOWN THRESHOLD RESULTS ==========")
    print(result_df.to_string(index=False))
    print(f"\nSaved to: {args.output_path}")

    threshold = 0.90
    y_pred, _, _ = predict_with_threshold(
        model=model,
        label_encoder=label_encoder,
        X=X_eval,
        threshold=threshold,
    )

    print(f"\n========== CLASSIFICATION REPORT @ threshold={threshold} ==========")
    print(classification_report(y_eval, y_pred, digits=4, zero_division=0))


if __name__ == "__main__":
    main()