from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import numpy as np


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


def extract_confusion_matrix(report_path: str) -> np.ndarray:
    content = Path(report_path).read_text(encoding="utf-8")

    match = re.search(
        r"## Confusion Matrix\s*```(?:\n)?(.*?)```",
        content,
        flags=re.DOTALL,
    )

    if not match:
        raise ValueError(f"Could not find confusion matrix in {report_path}")

    matrix_text = match.group(1).strip()
    matrix = ast.literal_eval(matrix_text)

    return np.array(matrix, dtype=int)


def top_directional_confusions(cm: np.ndarray, top_k: int) -> list[tuple[int, str, str]]:
    pairs = []

    for true_idx in range(cm.shape[0]):
        for pred_idx in range(cm.shape[1]):
            if true_idx == pred_idx:
                continue

            pairs.append(
                (
                    int(cm[true_idx, pred_idx]),
                    CIFAR10_CLASSES[true_idx],
                    CIFAR10_CLASSES[pred_idx],
                )
            )

    return sorted(pairs, reverse=True)[:top_k]


def top_pair_confusions(cm: np.ndarray, top_k: int) -> list[tuple[int, str, str, int, int]]:
    pairs = []

    for i in range(cm.shape[0]):
        for j in range(i + 1, cm.shape[1]):
            forward = int(cm[i, j])
            backward = int(cm[j, i])
            total = forward + backward

            pairs.append(
                (
                    total,
                    CIFAR10_CLASSES[i],
                    CIFAR10_CLASSES[j],
                    forward,
                    backward,
                )
            )

    return sorted(pairs, reverse=True)[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, help="Path to *_results.md")
    parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    cm = extract_confusion_matrix(args.report)

    accuracy_from_cm = np.trace(cm) / np.sum(cm)
    print(f"Accuracy from confusion matrix: {accuracy_from_cm:.4f}")

    print("\nTop pair-level confusions:")
    print("| Rank | Pair | Total | Direction detail |")
    print("|---:|---|---:|---|")

    for rank, (total, a, b, a_to_b, b_to_a) in enumerate(
        top_pair_confusions(cm, args.top_k),
        start=1,
    ):
        print(
            f"| {rank} | {a} ↔ {b} | {total} | "
            f"{a} → {b}: {a_to_b}, {b} → {a}: {b_to_a} |"
        )

    print("\nTop directional confusions:")
    print("| Rank | True class | Predicted class | Count |")
    print("|---:|---|---|---:|")

    for rank, (count, true_class, pred_class) in enumerate(
        top_directional_confusions(cm, args.top_k),
        start=1,
    ):
        print(f"| {rank} | {true_class} | {pred_class} | {count} |")


if __name__ == "__main__":
    main()