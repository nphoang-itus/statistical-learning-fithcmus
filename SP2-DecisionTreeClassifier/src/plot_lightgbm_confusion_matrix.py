from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


ID_COLUMNS = ["file_id", "label", "cleaned_path"]


def main():
    df = pd.read_csv("data/processed/features.csv")

    X = df.drop(columns=ID_COLUMNS)
    y = df["label"]

    bundle = joblib.load("models/lightgbm_model.joblib")
    model = bundle["model"]
    label_encoder = bundle["label_encoder"]

    y_encoded = label_encoder.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        stratify=y_encoded,
        random_state=42,
    )

    y_pred = model.predict(X_test)

    class_names = label_encoder.classes_

    cm = confusion_matrix(
        label_encoder.inverse_transform(y_test),
        label_encoder.inverse_transform(y_pred),
        labels=class_names,
    )

    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(18, 18))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    display.plot(
        ax=ax,
        xticks_rotation=90,
        values_format="d",
        colorbar=False,
    )

    ax.set_title("LightGBM Confusion Matrix")
    plt.tight_layout()
    plt.savefig("reports/figures/confusion_matrix_lightgbm.png", dpi=160)
    plt.close()

    print("Saved: reports/figures/confusion_matrix_lightgbm.png")


if __name__ == "__main__":
    main()