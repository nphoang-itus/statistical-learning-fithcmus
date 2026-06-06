import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm


KEYWORD_GROUPS = {
    "python": ["def ", "import ", "from ", "self", "__init__", "class ", "elif ", "None", "True", "False"],
    "java": ["public class", "private ", "protected ", "static void main", "System.out", "package ", "import java"],
    "c": ["#include", "printf", "scanf", "malloc", "free", "struct ", "typedef", "NULL"],
    "cpp": ["#include", "std::", "iostream", "cout", "cin", "namespace", "template", "class "],
    "csharp": ["using System", "namespace ", "public class", "Console.WriteLine", "var ", "string[] args"],
    "javascript": ["function ", "const ", "let ", "var ", "console.log", "=>", "require(", "module.exports"],
    "typescript": ["interface ", "type ", ": string", ": number", ": boolean", "enum ", "implements ", "readonly "],
    "php": ["<?php", "echo ", "$_", "function ", "namespace ", "use ", "->", "::"],
    "ruby": ["def ", "end", "puts ", "require ", "class ", "module ", "attr_", "do |"],
    "go": ["package ", "func ", "import ", "fmt.", "var ", "type ", "struct", "defer "],
    "rust": ["fn ", "let mut", "impl ", "use ", "pub ", "struct ", "enum ", "println!"],
    "kotlin": ["fun ", "val ", "var ", "class ", "object ", "companion object", "package ", "import "],
    "swift": ["func ", "let ", "var ", "import Foundation", "class ", "struct ", "enum ", "print("],
    "r": ["library(", "<-", "data.frame", "function(", "ggplot", "read.csv", "TRUE", "FALSE"],
    "lua": ["local ", "function ", "end", "then", "elseif", "require", "print("],
    "perl": ["use strict", "use warnings", "my $", "sub ", "print ", "$_", "@_"],
    "scala": ["object ", "class ", "def ", "val ", "var ", "extends ", "trait ", "println"],
    "sql": ["select ", "from ", "where ", "insert into", "update ", "delete from", "create table", "join "],
    "shell": ["#!/bin/bash", "#!/bin/sh", "echo ", "export ", "fi", "then", "grep ", "awk ", "sed "],
    "dockerfile": ["from ", "run ", "copy ", "add ", "cmd ", "entrypoint ", "workdir ", "expose "],
    "makefile": ["$(CC)", "$(CXX)", ".PHONY", "all:", "clean:", "install:", "$@", "$<", "$(MAKE)", "Makefile"],
    "latex": ["\\documentclass", "\\begin{document}", "\\section", "\\usepackage", "\\end{document}", "\\cite", "\\ref"],
}


def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace")


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def count_regex(pattern: str, text: str, flags: int = 0) -> int:
    return len(re.findall(pattern, text, flags))


def first_non_empty_line(lines: List[str]) -> str:
    for line in lines:
        if line.strip():
            return line.strip()
    return ""


def classify_first_line_pattern(line: str) -> int:
    """
    Encodes first line into simple pattern category.
    Decision Tree needs numeric values.
    """
    lower = line.lower()

    if not line:
        return 0
    if line.startswith("#!"):
        return 1
    if lower.startswith("<!doctype"):
        return 2
    if lower.startswith("<?xml"):
        return 3
    if lower.startswith("{") or lower.startswith("["):
        return 4
    if line.startswith("#"):
        return 5
    if "=" in line and not line.strip().startswith("//"):
        return 6
    if ":" in line:
        return 7
    if lower.startswith("from "):
        return 8
    if lower.startswith("package "):
        return 9
    if lower.startswith("import "):
        return 10
    return 99


def json_parse_success(text: str) -> int:
    try:
        json.loads(text)
        return 1
    except Exception:
        return 0


def extract_features_from_text(text: str) -> Dict[str, float]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lower = text.lower()
    lines = text.split("\n")
    non_empty_lines = [line for line in lines if line.strip()]

    num_chars = len(text)
    num_lines = len(lines)
    num_non_empty_lines = len(non_empty_lines)

    line_lengths = [len(line) for line in lines]
    avg_line_length = sum(line_lengths) / max(num_lines, 1)
    max_line_length = max(line_lengths, default=0)

    first_line = first_non_empty_line(lines)

    # Basic character counts
    count_lt = text.count("<")
    count_gt = text.count(">")
    count_slash = text.count("/")
    count_backslash = text.count("\\")
    count_brace_open = text.count("{")
    count_brace_close = text.count("}")
    count_bracket_open = text.count("[")
    count_bracket_close = text.count("]")
    count_paren_open = text.count("(")
    count_paren_close = text.count(")")
    count_semicolon = text.count(";")
    count_colon = text.count(":")
    count_comma = text.count(",")
    count_equal = text.count("=")
    count_quote = text.count('"')
    count_single_quote = text.count("'")
    count_hash = text.count("#")
    count_dollar = text.count("$")
    count_at = text.count("@")
    count_underscore = text.count("_")
    count_tab = text.count("\t")

    # Line-level features
    comment_line_count = sum(
        1 for line in non_empty_lines
        if line.strip().startswith(("#", "//", "/*", "*", "--", "%"))
    )

    indentation_lines = sum(
        1 for line in non_empty_lines
        if line.startswith(" ") or line.startswith("\t")
    )

    tag_like_lines = sum(
        1 for line in non_empty_lines
        if "<" in line and ">" in line
    )

    assignment_lines = sum(
        1 for line in non_empty_lines
        if "=" in line
    )

    colon_key_value_lines = sum(
        1 for line in non_empty_lines
        if re.match(r"^\s*[\w\-.\"']+\s*:\s+.+", line)
    )

    equal_key_value_lines = sum(
        1 for line in non_empty_lines
        if re.match(r"^\s*[\w\-\.]+\s*=\s*.+", line)
    )

    csv_like_lines = sum(
        1 for line in non_empty_lines[:30]
        if line.count(",") >= 2 or line.count(";") >= 2 or line.count("\t") >= 2
    )

    makefile_target_lines = sum(
        1 for line in non_empty_lines
        if re.match(r"^[A-Za-z0-9_\-./]+:\s*($|[^/])", line)
        and not re.match(r"^\s*[\w\-.\"']+\s*:\s+.+", line)
    )

    makefile_command_lines = sum(
        1 for line in lines
        if line.startswith("\t") and len(line.strip()) > 0
    )

    makefile_variable_lines = sum(
        1 for line in non_empty_lines
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*[:?+]?=", line)
    )

    features = {
        # Size/shape
        "num_chars": num_chars,
        "num_lines": num_lines,
        "num_non_empty_lines": num_non_empty_lines,
        "avg_line_length": avg_line_length,
        "max_line_length": max_line_length,
        "empty_line_ratio": safe_div(num_lines - num_non_empty_lines, num_lines),

        # First line
        "first_line_pattern": classify_first_line_pattern(first_line),
        "first_line_length": len(first_line),

        # Ratios
        "special_char_ratio": safe_div(count_lt + count_gt + count_slash, num_chars),
        "brace_ratio": safe_div(count_brace_open + count_brace_close, num_chars),
        "bracket_ratio": safe_div(count_bracket_open + count_bracket_close, num_chars),
        "paren_ratio": safe_div(count_paren_open + count_paren_close, num_chars),
        "semicolon_ratio": safe_div(count_semicolon, num_chars),
        "colon_ratio": safe_div(count_colon, num_chars),
        "comma_ratio": safe_div(count_comma, num_chars),
        "equal_ratio": safe_div(count_equal, num_chars),
        "quote_ratio": safe_div(count_quote + count_single_quote, num_chars),
        "hash_ratio": safe_div(count_hash, num_chars),
        "dollar_ratio": safe_div(count_dollar, num_chars),
        "at_ratio": safe_div(count_at, num_chars),
        "underscore_ratio": safe_div(count_underscore, num_chars),
        "tab_ratio": safe_div(count_tab, num_chars),

        # Line ratios
        "comment_line_ratio": safe_div(comment_line_count, num_non_empty_lines),
        "indentation_line_ratio": safe_div(indentation_lines, num_non_empty_lines),
        "html_tag_ratio": safe_div(tag_like_lines, num_non_empty_lines),
        "assignment_line_ratio": safe_div(assignment_lines, num_non_empty_lines),
        "colon_key_value_ratio": safe_div(colon_key_value_lines, num_non_empty_lines),
        "equal_key_value_ratio": safe_div(equal_key_value_lines, num_non_empty_lines),
        "csv_like_line_ratio": safe_div(csv_like_lines, min(num_non_empty_lines, 30)),
        "makefile_target_ratio": safe_div(makefile_target_lines, num_non_empty_lines),
        "makefile_command_ratio": safe_div(makefile_command_lines, num_non_empty_lines),
        "makefile_variable_ratio": safe_div(makefile_variable_lines, num_non_empty_lines),

        # Strong signatures
        "has_doctype": int("<!doctype" in lower),
        "has_html_tag": int("<html" in lower),
        "has_body_tag": int("<body" in lower),
        "has_xml_declaration": int(lower.strip().startswith("<?xml")),
        "has_svg_tag": int("<svg" in lower),
        "has_viewbox": int("viewbox" in lower),
        "has_vcalendar": int("begin:vcalendar" in lower),
        "has_from_header": int("from:" in lower or "to:" in lower),
        "has_mime_boundary": int("content-type: multipart" in lower),
        "json_parse_success": json_parse_success(text),

        # Generic syntax signals
        "has_curly_braces": int("{" in text and "}" in text),
        "has_square_brackets": int("[" in text and "]" in text),
        "has_semicolon": int(";" in text),
        "has_shebang": int(first_line.startswith("#!")),
        "has_import": int("import " in lower),
        "has_package": int("package " in lower),
        "has_class": int("class " in lower),
        "has_function_word": int("function " in lower),
        "has_def_word": int("def " in lower),
        "has_public_word": int("public " in lower),
        "has_include": int("#include" in lower),
        "has_namespace": int("namespace " in lower),
        
        # Markdown
        "markdown_heading_ratio": safe_div(
            sum(1 for line in non_empty_lines if re.match(r"^\s{0,3}#{1,6}\s+", line)),
            num_non_empty_lines
        ),
        "markdown_list_ratio": safe_div(
            sum(1 for line in non_empty_lines if re.match(r"^\s*[-*+]\s+", line)),
            num_non_empty_lines
        ),
        "markdown_code_fence_count": lower.count("```"),
        "markdown_link_count": count_regex(r"\[[^\]]+\]\([^)]+\)", text),
        
        # SQL
        "sql_statement_ratio": safe_div(
            sum(
                1 for line in non_empty_lines
                if re.match(r"^\s*(select|insert|update|delete|create|alter|drop|with)\b", line.lower())
            ),
            num_non_empty_lines
        ),
        "sql_clause_count": count_regex(
            r"\b(select|from|where|join|group by|order by|insert into|create table|primary key|foreign key)\b",
            lower,
        ),
        
        # JavaScript/TypeScript
        "js_arrow_count": lower.count("=>"),
        "js_console_count": lower.count("console."),
        "js_require_count": lower.count("require("),
        "js_export_count": lower.count("export "),
        "js_import_from_count": count_regex(r"import\s+.+\s+from\s+['\"]", lower),
        "ts_type_annotation_count": count_regex(r":\s*(string|number|boolean|any|unknown|void)\b", lower),
        "ts_interface_count": count_regex(r"\binterface\s+\w+", lower),
        "ts_type_alias_count": count_regex(r"\btype\s+\w+\s*=", lower),
    }

    # Keyword group counts/ratios
    for group_name, keywords in KEYWORD_GROUPS.items():
        count = 0
        for keyword in keywords:
            count += lower.count(keyword.lower())
        features[f"kw_{group_name}_count"] = count
        features[f"kw_{group_name}_ratio"] = safe_div(count, num_non_empty_lines)

    return features


def build_features(
    metadata_path: Path,
    output_path: Path,
    max_samples_per_class: int,
    random_state: int,
):
    metadata = pd.read_csv(metadata_path)

    valid_df = metadata[metadata["is_valid"] == True].copy()

    # Balanced sampling per class.
    # Avoid groupby.apply because newer pandas versions may drop grouping columns.
    sampled_parts = []

    for label, group in valid_df.groupby("label"):
        sampled = group.sample(
            n=min(len(group), max_samples_per_class),
            random_state=random_state
        )
        sampled_parts.append(sampled)

    sampled_df = pd.concat(sampled_parts, ignore_index=True)

    rows = []

    for _, row in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="Extracting features"):
        text = read_text(row["cleaned_path"])
        features = extract_features_from_text(text)

        features["file_id"] = row["file_id"]
        features["label"] = row["label"]
        features["cleaned_path"] = row["cleaned_path"]

        rows.append(features)

    feature_df = pd.DataFrame(rows)

    # Put identifiers first.
    id_cols = ["file_id", "label", "cleaned_path"]
    feature_cols = [col for col in feature_df.columns if col not in id_cols]
    feature_df = feature_df[id_cols + feature_cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_path, index=False)

    print("========== FEATURE EXTRACTION SUMMARY ==========")
    print(f"Metadata: {metadata_path}")
    print(f"Output: {output_path}")
    print(f"Samples: {len(feature_df)}")
    print(f"Classes: {feature_df['label'].nunique()}")
    print(f"Features: {len(feature_cols)}")
    print()
    print(feature_df["label"].value_counts().sort_index())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-path", default="data/processed/metadata.csv")
    parser.add_argument("--output-path", default="data/processed/features.csv")
    parser.add_argument("--max-samples-per-class", type=int, default=600)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    build_features(
        metadata_path=Path(args.metadata_path),
        output_path=Path(args.output_path),
        max_samples_per_class=args.max_samples_per_class,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()