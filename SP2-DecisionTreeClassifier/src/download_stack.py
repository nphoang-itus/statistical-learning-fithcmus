import argparse
import re
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm


TRAIN_CLASSES = [
    "Python", "Java", "C", "C++", "C#", "JavaScript", "TypeScript",
    "HTML", "CSS", "XML", "SVG", "JSON", "YAML", "TOML", "INI",
    "CSV", "SQL", "Markdown", "Shell", "Dockerfile", "Makefile",
    "PHP", "Ruby", "Go", "Rust", "Kotlin", "Swift", "R", "Lua",
    "Perl", "Scala", "LaTeX"
]


# The Stack v1 language folder candidates.
# Some names may differ depending on HF dataset config, so the script tries candidates.
LANGUAGE_DATA_DIR_CANDIDATES = {
    "Python": ["data/python"],
    "Java": ["data/java"],
    "C": ["data/c"],
    "C++": ["data/c++", "data/cpp"],
    "C#": ["data/c-sharp", "data/csharp", "data/c#"],
    "JavaScript": ["data/javascript"],
    "TypeScript": ["data/typescript"],
    "HTML": ["data/html"],
    "CSS": ["data/css"],
    "XML": ["data/xml"],
    "SVG": ["data/svg"],
    "JSON": ["data/json"],
    "YAML": ["data/yaml"],
    "TOML": ["data/toml"],
    "INI": ["data/ini"],
    "CSV": ["data/csv"],
    "SQL": ["data/sql"],
    "Markdown": ["data/markdown"],
    "Shell": ["data/shell", "data/shellsession", "data/bash"],
    "Dockerfile": ["data/dockerfile"],
    "Makefile": ["data/makefile"],
    "PHP": ["data/php"],
    "Ruby": ["data/ruby"],
    "Go": ["data/go"],
    "Rust": ["data/rust"],
    "Kotlin": ["data/kotlin"],
    "Swift": ["data/swift"],
    "R": ["data/r"],
    "Lua": ["data/lua"],
    "Perl": ["data/perl"],
    "Scala": ["data/scala"],
    "LaTeX": ["data/latex", "data/tex"],
    "Haskell": ["data/haskell"],
    "Dart": ["data/dart"],
    "Groovy": ["data/groovy"],
    "Vue": ["data/vue"],
    "Julia": ["data/julia"],
}


FILE_EXTENSIONS = {
    "Python": ".py",
    "Java": ".java",
    "C": ".c",
    "C++": ".cpp",
    "C#": ".cs",
    "JavaScript": ".js",
    "TypeScript": ".ts",
    "HTML": ".html",
    "CSS": ".css",
    "XML": ".xml",
    "SVG": ".svg",
    "JSON": ".json",
    "YAML": ".yaml",
    "TOML": ".toml",
    "INI": ".ini",
    "CSV": ".csv",
    "SQL": ".sql",
    "Markdown": ".md",
    "Shell": ".sh",
    "Dockerfile": ".Dockerfile",
    "Makefile": ".Makefile",
    "PHP": ".php",
    "Ruby": ".rb",
    "Go": ".go",
    "Rust": ".rs",
    "Kotlin": ".kt",
    "Swift": ".swift",
    "R": ".r",
    "Lua": ".lua",
    "Perl": ".pl",
    "Scala": ".scala",
    "LaTeX": ".tex",
    "Haskell": ".hs",
    "Dart": ".dart",
    "Groovy": ".groovy",
    "Vue": ".vue",
    "Julia": ".jl",
}


def get_content(sample: dict) -> Optional[str]:
    """
    The Stack usually uses 'content', but keep fallback keys
    so the script is more robust.
    """
    for key in ["content", "text", "code"]:
        value = sample.get(key)
        if isinstance(value, str):
            return value
    return None


def is_probably_binary(text: str) -> bool:
    if "\x00" in text:
        return True

    # Too many replacement/control chars usually means non-text or bad decoding.
    control_chars = sum(1 for ch in text if ord(ch) < 32 and ch not in "\n\r\t")
    if len(text) > 0 and control_chars / len(text) > 0.02:
        return True

    return False


def is_probably_minified(text: str) -> bool:
    lines = text.splitlines()
    if not lines:
        return True

    avg_line_length = sum(len(line) for line in lines) / max(len(lines), 1)
    max_line_length = max(len(line) for line in lines)

    # Minified JS/CSS/HTML often has huge one-line content.
    return avg_line_length > 500 or max_line_length > 5000


def normalize_text(text: str, max_lines: int = 120) -> str:
    """
    Keep only first N lines because headers/imports/syntax patterns
    are usually enough for file type classification.
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = lines[:max_lines]
    return "\n".join(lines).strip()


def is_valid_sample(text: Optional[str]) -> bool:
    if text is None:
        return False

    text = text.strip()

    # Assignment explicitly warns to remove empty/near-empty files.
    if len(text) < 10:
        return False

    if is_probably_binary(text):
        return False

    if is_probably_minified(text):
        return False

    return True


def safe_class_dir_name(class_name: str) -> str:
    return (
        class_name.lower()
        .replace("#", "sharp")
        .replace("+", "plus")
        .replace(" ", "_")
    )


def load_stream_for_class(class_name: str):
    candidates = LANGUAGE_DATA_DIR_CANDIDATES[class_name]
    last_error = None

    for data_dir in candidates:
        try:
            print(f"Trying {class_name} from {data_dir} ...")
            ds = load_dataset(
                "bigcode/the-stack",
                data_dir=data_dir,
                split="train",
                streaming=True,
            )
            return ds, data_dir
        except Exception as error:
            last_error = error
            print(f"Failed {data_dir}: {error}")

    raise RuntimeError(f"Cannot load class {class_name}. Last error: {last_error}")


def download_class(class_name: str, samples_per_class: int, output_dir: Path):
    class_dir = output_dir / safe_class_dir_name(class_name)
    class_dir.mkdir(parents=True, exist_ok=True)

    existing_files = list(class_dir.glob("*"))
    if len(existing_files) >= samples_per_class:
        print(f"[SKIP] {class_name}: already has {len(existing_files)} files")
        return

    ds, used_data_dir = load_stream_for_class(class_name)

    ext = FILE_EXTENSIONS.get(class_name, ".txt")
    count = len(existing_files)
    seen_hashes = set()

    progress = tqdm(total=samples_per_class, initial=count, desc=class_name)

    for sample in ds:
        if count >= samples_per_class:
            break

        content = get_content(sample)
        if not is_valid_sample(content):
            continue

        # Type narrowing: is_valid_sample ensures content is str
        assert content is not None
        content = normalize_text(content)

        # Basic exact duplicate filter.
        content_hash = hash(content)
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)

        file_path = class_dir / f"{count:05d}{ext}"
        file_path.write_text(content, encoding="utf-8", errors="ignore")

        count += 1
        progress.update(1)

    progress.close()

    print(f"[DONE] {class_name}: saved {count}/{samples_per_class} files from {used_data_dir}")

    if count < samples_per_class:
        print(f"[WARN] {class_name}: only collected {count} valid samples")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-class", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="data/raw")
    parser.add_argument("--classes", nargs="*", default=TRAIN_CLASSES)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    failed_classes = []

    for class_name in args.classes:
        try:
            download_class(
                class_name=class_name,
                samples_per_class=args.samples_per_class,
                output_dir=output_dir,
            )
        except Exception as error:
            print(f"[ERROR] {class_name}: {error}")
            failed_classes.append(class_name)

    print("\n========== SUMMARY ==========")
    print(f"Output dir: {output_dir}")
    print(f"Requested samples/class: {args.samples_per_class}")

    if failed_classes:
        print("Failed classes:")
        for cls in failed_classes:
            print(f"- {cls}")
    else:
        print("All classes downloaded successfully.")


if __name__ == "__main__":
    main()