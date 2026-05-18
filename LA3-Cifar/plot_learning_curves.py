from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt


RESULTS_DIR = "results"
FIGURES_DIR = "figures"


def load_history(model_name: str) -> dict:
    path = os.path.join(RESULTS_DIR, f"{model_name}_history.json")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"History file not found: {path}. "
            f"Run training first to generate it."
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_metric(
    history: dict,
    model_name: str,
    metric: str,
    val_metric: str,
    ylabel: str,
    output_path: str,
) -> None:
    if metric not in history:
        raise KeyError(f"Missing metric '{metric}' in history.")

    if val_metric not in history:
        raise KeyError(f"Missing validation metric '{val_metric}' in history.")

    epochs = range(1, len(history[metric]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history[metric], marker="o", label=f"train {metric}")
    plt.plot(epochs, history[val_metric], marker="o", label=f"validation {metric}")
    plt.title(f"{model_name.upper()} — {ylabel}")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved {output_path}")


def plot_learning_curves(model_name: str) -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)

    history = load_history(model_name)

    accuracy_path = os.path.join(FIGURES_DIR, f"{model_name}_accuracy_curve.png")
    loss_path = os.path.join(FIGURES_DIR, f"{model_name}_loss_curve.png")

    plot_metric(
        history=history,
        model_name=model_name,
        metric="accuracy",
        val_metric="val_accuracy",
        ylabel="Accuracy",
        output_path=accuracy_path,
    )

    plot_metric(
        history=history,
        model_name=model_name,
        metric="loss",
        val_metric="val_loss",
        ylabel="Loss",
        output_path=loss_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot learning curves from Keras history JSON.")
    parser.add_argument(
        "--model",
        required=True,
        choices=["cifar_mlp", "cnn", "cnn_aug"],
        help="Model name to plot.",
    )

    args = parser.parse_args()
    plot_learning_curves(args.model)


if __name__ == "__main__":
    main()