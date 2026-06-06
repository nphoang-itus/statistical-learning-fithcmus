import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from clean_data import (
    validate_sample,
    normalize_newlines,
    remove_non_printable_noise,
    truncate_text,
    safe_dir_name,
)
from extract_features import extract_features_from_text


LABEL_FROM_DIR = {
    "haskell": "Haskell",
    "dart": "Dart",
    "groovy": "Groovy",
    "vue": "Vue",
    "julia": "Julia",
}


def read_text(path: Path):
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def iter_files(raw_dir: Path):
    for class_dir in sorted(raw_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        actual_label = LABEL_FROM_DIR.get(class_dir.name, class_dir.name)

        for path in sorted(class_dir.iterdir()):
            if path.is_file():
                yield actual_label, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/unknown_raw")
    parser.add_argument("--output-path", default="data/processed/unknown_features.csv")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_path = Path(args.output_path)

    rows = []
    invalid = 0

    for actual_label, path in tqdm(list(iter_files(raw_dir)), desc="Extracting unknown features"):
        text = read_text(path)

        is_valid, reason = validate_sample(actual_label, text)
        if not is_valid:
            invalid += 1
            continue

        text = normalize_newlines(text)
        text = remove_non_printable_noise(text)
        text = truncate_text(text)

        features = extract_features_from_text(text)

        file_id = f"{safe_dir_name(actual_label)}_{len(rows):06d}"

        features["file_id"] = file_id
        features["label"] = "UNKNOWN"
        features["actual_label"] = actual_label
        features["cleaned_path"] = str(path)

        rows.append(features)

    df = pd.DataFrame(rows)

    id_cols = ["file_id", "label", "actual_label", "cleaned_path"]
    feature_cols = [col for col in df.columns if col not in id_cols]
    df = df[id_cols + feature_cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("========== UNKNOWN FEATURE SUMMARY ==========")
    print(f"Raw dir: {raw_dir}")
    print(f"Output: {output_path}")
    print(f"Valid unknown samples: {len(df)}")
    print(f"Invalid skipped: {invalid}")
    print()
    print(df["actual_label"].value_counts().sort_index())


if __name__ == "__main__":
    main()