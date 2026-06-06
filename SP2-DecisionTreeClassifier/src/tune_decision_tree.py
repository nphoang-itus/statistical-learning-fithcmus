from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


ID_COLUMNS = ["file_id", "label", "cleaned_path"]


def main():
    df = pd.read_csv("data/processed/features.csv")

    X = df.drop(columns=ID_COLUMNS)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    configs = []

    for max_depth in [8, 10, 12, 14, 16, 18, 20, None]:
        for min_samples_leaf in [1, 3, 5, 10, 20]:
            model = DecisionTreeClassifier(
                criterion="gini",
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            configs.append({
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "accuracy": accuracy_score(y_test, y_pred),
                "macro_f1": f1_score(y_test, y_pred, average="macro"),
            })

    result_df = pd.DataFrame(configs).sort_values("macro_f1", ascending=False)

    Path("reports").mkdir(exist_ok=True)
    result_df.to_csv("reports/decision_tree_tuning.csv", index=False)

    print(result_df.head(20).to_string(index=False))

    best = result_df.iloc[0]
    print("\nBest config:")
    print(best)


if __name__ == "__main__":
    main()