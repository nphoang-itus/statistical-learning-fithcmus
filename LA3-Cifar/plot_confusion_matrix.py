from __future__ import annotations

import argparse
import ast
import os
import re
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "xdg-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


DATA_DIR = "data"
MODEL_DIR = "models"
RESULTS_DIR = "results"
FIGURES_DIR = "figures"

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def model_path(model_name: str) -> str:
    return os.path.join(MODEL_DIR, f"{model_name}_model.keras")


def results_path(model_name: str) -> str:
    return os.path.join(RESULTS_DIR, f"{model_name}_results.md")


def load_cifar10_test_data() -> tuple[np.ndarray, np.ndarray]:
    data_path = os.path.join(DATA_DIR, "cifar10.npz")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Run: python main.py configure --dataset cifar10"
        )

    data = np.load(data_path)
    return data["x_test"], data["y_test"]


def load_confusion_matrix_from_results(model_name: str) -> np.ndarray:
    path = results_path(model_name)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Results not found at {path}. "
            f"Run: python main.py test --model {model_name}"
        )

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    match = re.search(r"## Confusion Matrix\s*```(?:\w+)?\s*(.*?)\s*```", text, re.S)
    if match is None:
        raise ValueError(f"Could not find a confusion matrix block in {path}.")

    cm = np.asarray(ast.literal_eval(match.group(1)), dtype=np.int64)
    if cm.shape != (len(CIFAR10_CLASSES), len(CIFAR10_CLASSES)):
        raise ValueError(f"Expected a 10x10 confusion matrix in {path}, got {cm.shape}.")

    return cm


def load_confusion_matrix_from_model(model_name: str) -> np.ndarray:
    import tensorflow as tf

    path = model_path(model_name)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at {path}. "
            f"Run: python main.py train --model {model_name}"
        )

    x_test, y_test = load_cifar10_test_data()
    model = tf.keras.models.load_model(path)

    y_prob = model.predict(x_test, batch_size=128, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    print(f"Predicted label counts: {np.bincount(y_pred, minlength=len(CIFAR10_CLASSES)).tolist()}")

    return confusion_matrix(y_test, y_pred)


def normalize_confusion_matrix(cm: np.ndarray) -> np.ndarray:
    cm = cm.astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)


def plot_confusion_matrix(model_name: str, normalize: bool = False, source: str = "results") -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)

    if source == "results":
        cm = load_confusion_matrix_from_results(model_name)
    elif source == "model":
        cm = load_confusion_matrix_from_model(model_name)
    else:
        raise ValueError(f"Unsupported source: {source}")

    if normalize:
        cm = normalize_confusion_matrix(cm)

    print("Confusion matrix:")
    print(np.array2string(cm, separator=", "))

    fig, ax = plt.subplots(figsize=(10, 8))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CIFAR10_CLASSES,
    )

    display.plot(
        ax=ax,
        values_format=".2f" if normalize else "d",
        xticks_rotation=45,
        colorbar=True,
    )

    title_suffix = "Normalized" if normalize else "Raw Counts"
    ax.set_title(f"{model_name.upper()} Confusion Matrix - {title_suffix}")

    plt.tight_layout()

    suffix = "normalized_confusion_matrix" if normalize else "confusion_matrix"
    output_path = os.path.join(FIGURES_DIR, f"{model_name}_{suffix}.png")

    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved {output_path} from {source}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CIFAR-10 confusion matrix.")
    parser.add_argument(
        "--model",
        required=True,
        choices=["cifar_mlp", "cnn", "cnn_aug"],
        help="Model name.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize confusion matrix by true class.",
    )
    parser.add_argument(
        "--source",
        default="results",
        choices=["cifar_mlp", "cnn", "cnn_aug"],
        help="Use saved results Markdown or recompute predictions from the Keras model.",
    )

    args = parser.parse_args()

    plot_confusion_matrix(
        model_name=args.model,
        normalize=args.normalize,
        source=args.source,
    )


if __name__ == "__main__":
    main()
