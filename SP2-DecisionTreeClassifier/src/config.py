TRAIN_CLASSES = [
    "Python", "Java", "C", "C++", "C#", "JavaScript", "TypeScript",
    "HTML", "CSS", "XML", "SVG", "JSON", "YAML", "TOML", "INI",
    "CSV", "SQL", "Markdown", "Shell", "Dockerfile", "Makefile",
    "PHP", "Ruby", "Go", "Rust", "Kotlin", "Swift", "R", "Lua",
    "Perl", "Scala", "LaTeX"
]

UNKNOWN_CLASSES = [
    "Haskell", "Dart", "Groovy", "Vue", "Svelte"
]

SAMPLES_PER_CLASS = 1000
RANDOM_STATE = 42

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
MODEL_DIR = "models"
FIGURE_DIR = "reports/figures"