import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


ID_COLUMNS = ["file_id", "label", "cleaned_path"]


def load_features(features_path: Path):
    df = pd.read_csv(features_path)

    X = df.drop(columns=ID_COLUMNS)
    y = df["label"]

    feature_names = X.columns.tolist()
    class_names = sorted(y.unique().tolist())

    return df, X, y, feature_names, class_names


def save_text_report(report_path: Path, content: str):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content, encoding="utf-8")


def plot_and_save_confusion_matrix(y_test, y_pred, class_names, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_test, y_pred, labels=class_names)

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

    ax.set_title("Decision Tree Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_and_save_tree(model, feature_names, class_names, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(28, 14))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        max_depth=3,
        fontsize=8,
    )
    plt.title("Decision Tree Visualization - Top 3 Levels")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_feature_importance(model, feature_names, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance_df.to_csv(output_path, index=False)

    return importance_df


def train_decision_tree(
    features_path: Path,
    model_path: Path,
    report_path: Path,
    confusion_matrix_path: Path,
    tree_plot_path: Path,
    feature_importance_path: Path,
    max_depth: int,
    min_samples_leaf: int,
    random_state: int,
):
    df, X, y, feature_names, class_names = load_features(features_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state,
    )

    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    report = classification_report(
        y_test,
        y_test_pred,
        digits=4,
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    save_text_report(
        report_path,
        content=(
            "Decision Tree Classifier Baseline\n"
            "=================================\n\n"
            f"Features path: {features_path}\n"
            f"Total samples: {len(df)}\n"
            f"Train samples: {len(X_train)}\n"
            f"Test samples: {len(X_test)}\n"
            f"Number of classes: {len(class_names)}\n"
            f"Number of features: {len(feature_names)}\n\n"
            f"Model params:\n"
            f"- criterion: gini\n"
            f"- max_depth: {max_depth}\n"
            f"- min_samples_leaf: {min_samples_leaf}\n"
            f"- random_state: {random_state}\n\n"
            f"Train accuracy: {train_accuracy:.4f}\n"
            f"Test accuracy: {test_accuracy:.4f}\n\n"
            "Classification report:\n"
            f"{report}\n"
        )
    )

    plot_and_save_confusion_matrix(
        y_test=y_test,
        y_pred=y_test_pred,
        class_names=class_names,
        output_path=confusion_matrix_path,
    )

    plot_and_save_tree(
        model=model,
        feature_names=feature_names,
        class_names=class_names,
        output_path=tree_plot_path,
    )

    importance_df = save_feature_importance(
        model=model,
        feature_names=feature_names,
        output_path=feature_importance_path,
    )

    print("========== TRAINING SUMMARY ==========")
    print(f"Features: {features_path}")
    print(f"Model saved to: {model_path}")
    print(f"Report saved to: {report_path}")
    print(f"Confusion matrix saved to: {confusion_matrix_path}")
    print(f"Tree plot saved to: {tree_plot_path}")
    print(f"Feature importance saved to: {feature_importance_path}")
    print()
    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print()
    print(report)
    print("Top 20 important features:")
    print(importance_df.head(20).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--features-path", default="data/processed/features.csv")
    parser.add_argument("--model-path", default="models/decision_tree_baseline.joblib")
    parser.add_argument("--report-path", default="reports/decision_tree_report.txt")
    parser.add_argument("--confusion-matrix-path", default="reports/figures/confusion_matrix.png")
    parser.add_argument("--tree-plot-path", default="reports/figures/decision_tree_top3.png")
    parser.add_argument("--feature-importance-path", default="reports/feature_importance.csv")

    parser.add_argument("--max-depth", type=int, default=18)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    train_decision_tree(
        features_path=Path(args.features_path),
        model_path=Path(args.model_path),
        report_path=Path(args.report_path),
        confusion_matrix_path=Path(args.confusion_matrix_path),
        tree_plot_path=Path(args.tree_plot_path),
        feature_importance_path=Path(args.feature_importance_path),
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()