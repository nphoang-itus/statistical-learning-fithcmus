# Decision Tree Classifier for Text-based File Format Classification

This project builds a tree-based classifier to classify text-based files by their programming language or file format. The task uses source code files from The Stack dataset, extracts structural/textual features from each file, trains a Decision Tree baseline, and compares it with an advanced LightGBM model.

## 1. Project Overview

The goal is to classify files into at least 32 known classes such as Python, Java, C++, HTML, JSON, YAML, SQL, Dockerfile, Makefile, Markdown, and others.

The project includes:

- Dataset collection from The Stack
- Data cleaning and metadata generation
- Feature extraction from raw file content
- Decision Tree baseline training and tuning
- LightGBM advanced model training
- Unknown format handling using confidence thresholding
- Visualization using confusion matrices, tree plot, and feature importance

## 2. Folder Structure

```text
submission/
├── src/
│   ├── clean_data.py
│   ├── config.py
│   ├── download_stack.py
│   ├── extract_features.py
│   ├── train_decision_tree.py
│   ├── tune_decision_tree.py
│   ├── train_lightgbm.py
│   ├── prepare_unknown_features.py
│   ├── evaluate_unknown_lightgbm.py
│   ├── plot_lightgbm_confusion_matrix.py
│   └── plot_lightgbm_importance.py
│
├── data/
│   ├── features.csv
│   ├── metadata.csv
│   └── unknown_features.csv
│
├── models/
│   ├── decision_tree_tuned.joblib
│   └── lightgbm_model.joblib
│
├── reports/
│   ├── decision_tree_tuned_report.txt
│   ├── decision_tree_tuning.csv
│   ├── feature_importance_tuned.csv
│   ├── lightgbm_report.txt
│   ├── lightgbm_feature_importance.csv
│   ├── unknown_threshold_results_lightgbm.csv
│   └── figures/
│       ├── decision_tree_tuned_top3.png
│       ├── confusion_matrix_tuned.png
│       ├── confusion_matrix_lightgbm.png
│       └── lightgbm_feature_importance.png
│
├── Decision_Tree_Assignment.md
├── README.md
└── requirements.txt
```

## 3. Main Files

### Source code

| File                                    | Description                                         |
| --------------------------------------- | --------------------------------------------------- |
| `src/download_stack.py`                 | Downloads raw files from The Stack dataset by class |
| `src/clean_data.py`                     | Cleans raw files and creates `metadata.csv`         |
| `src/extract_features.py`               | Extracts numeric features from cleaned files        |
| `src/train_decision_tree.py`            | Trains the tuned Decision Tree model                |
| `src/tune_decision_tree.py`             | Runs hyperparameter tuning for Decision Tree        |
| `src/train_lightgbm.py`                 | Trains the LightGBM advanced model                  |
| `src/prepare_unknown_features.py`       | Prepares features for unseen/unknown classes        |
| `src/evaluate_unknown_lightgbm.py`      | Evaluates unknown format handling with LightGBM     |
| `src/plot_lightgbm_confusion_matrix.py` | Exports LightGBM confusion matrix                   |
| `src/plot_lightgbm_importance.py`       | Exports LightGBM feature importance plot            |

### Data files

| File                        | Description                             |
| --------------------------- | --------------------------------------- |
| `data/features.csv`         | Extracted features for 32 known classes |
| `data/metadata.csv`         | Cleaning metadata for raw files         |
| `data/unknown_features.csv` | Extracted features for unseen classes   |

### Model files

| File                                | Description                             |
| ----------------------------------- | --------------------------------------- |
| `models/decision_tree_tuned.joblib` | Final tuned Decision Tree model         |
| `models/lightgbm_model.joblib`      | Final LightGBM model with label encoder |

### Report files

| File                                             | Description                     |
| ------------------------------------------------ | ------------------------------- |
| `reports/decision_tree_tuned_report.txt`         | Decision Tree evaluation report |
| `reports/decision_tree_tuning.csv`               | Hyperparameter tuning results   |
| `reports/lightgbm_report.txt`                    | LightGBM evaluation report      |
| `reports/unknown_threshold_results_lightgbm.csv` | Unknown thresholding results    |
| `reports/figures/`                               | Generated visualizations        |

## 4. Environment Setup

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

For macOS Apple Silicon, LightGBM may require OpenMP:

```bash
brew install libomp
```

If LightGBM tree visualization is needed, Graphviz is required:

```bash
brew install graphviz
```

However, LightGBM tree plotting is optional. The report mainly uses LightGBM feature importance and confusion matrix.

## 5. Dataset

The known dataset contains 32 classes:

```text
C, C#, C++, CSS, CSV, Dockerfile, Go, HTML,
INI, JSON, Java, JavaScript, Kotlin, LaTeX, Lua, Makefile,
Markdown, PHP, Perl, Python, R, Ruby, Rust, SQL,
SVG, Scala, Shell, Swift, TOML, TypeScript, XML, YAML
```

The raw dataset was collected from The Stack with approximately 1,000 files per class.

The unknown dataset contains unseen classes:

```text
Dart, Groovy, Haskell, Julia, Vue
```

These classes are not used during training and are only used for unknown format evaluation.

## 6. Reproduce the Full Pipeline

### Step 1: Download known-class raw data

```bash
python src/download_stack.py --samples-per-class 1000
```

This downloads files into:

```text
data/raw/
```

If Hugging Face access is required, log in first:

```bash
huggingface-cli login
```

### Step 2: Clean raw data

```bash
python src/clean_data.py
```

This creates:

```text
data/processed/metadata.csv
data/processed/cleaned_files/
```

### Step 3: Extract features

```bash
python src/extract_features.py --max-samples-per-class 600
```

This creates:

```text
data/processed/features.csv
```

In the submitted folder, this file is provided as:

```text
data/features.csv
```

### Step 4: Tune Decision Tree

```bash
python src/tune_decision_tree.py
```

This creates:

```text
reports/decision_tree_tuning.csv
```

### Step 5: Train tuned Decision Tree

```bash
python src/train_decision_tree.py \
  --max-depth none \
  --min-samples-leaf 1 \
  --model-path models/decision_tree_tuned.joblib \
  --report-path reports/decision_tree_tuned_report.txt \
  --confusion-matrix-path reports/figures/confusion_matrix_tuned.png \
  --tree-plot-path reports/figures/decision_tree_tuned_top3.png \
  --feature-importance-path reports/feature_importance_tuned.csv
```

Final Decision Tree result:

```text
Test accuracy: 0.8370
Macro F1:      0.8371
```

### Step 6: Train LightGBM

```bash
python src/train_lightgbm.py
```

This creates:

```text
models/lightgbm_model.joblib
reports/lightgbm_report.txt
reports/lightgbm_feature_importance.csv
```

Final LightGBM result:

```text
Test accuracy: 0.9422
Macro F1:      0.9420
```

### Step 7: Prepare unknown features

Download unknown classes first:

```bash
python src/download_stack.py \
  --output-dir data/unknown_raw \
  --samples-per-class 200 \
  --classes Haskell Dart Groovy Vue Julia
```

Then extract unknown features:

```bash
python src/prepare_unknown_features.py
```

This creates:

```text
data/processed/unknown_features.csv
```

In the submitted folder, this file is provided as:

```text
data/unknown_features.csv
```

### Step 8: Evaluate unknown format handling

```bash
python src/evaluate_unknown_lightgbm.py
```

This creates:

```text
reports/unknown_threshold_results_lightgbm.csv
```

Final selected threshold:

```text
Model: LightGBM
Threshold: 0.99
Unknown recall: 0.7048
Known accuracy after rejection: 0.8781
```

### Step 9: Generate LightGBM visualizations

```bash
python src/plot_lightgbm_confusion_matrix.py
python src/plot_lightgbm_importance.py
```

This creates:

```text
reports/figures/confusion_matrix_lightgbm.png
reports/figures/lightgbm_feature_importance.png
```

## 7. Final Results

| Model                        | Test Accuracy | Macro F1 | Role                        |
| ---------------------------- | ------------: | -------: | --------------------------- |
| DecisionTreeClassifier tuned |        0.8370 |   0.8371 | Main Decision Tree baseline |
| LightGBM                     |        0.9422 |   0.9420 | Advanced tree-based model   |

Unknown format handling:

| Model    | Threshold | Unknown Recall | Known Accuracy After Rejection |
| -------- | --------: | -------------: | -----------------------------: |
| LightGBM |      0.99 |         0.7048 |                         0.8781 |

## 8. Notes

The raw dataset is not included in the compact submission because it contains many files and may be large. It can be regenerated using `src/download_stack.py`.

The submitted feature CSV files are enough to reproduce model training and evaluation without downloading the raw dataset again.

If the folder structure differs from the original development project, update the paths in scripts from:

```text
data/processed/features.csv
```

to:

```text
data/features.csv
```

or place the files back into the original `data/processed/` structure before running the scripts.