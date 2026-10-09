"""Shared, deliberately fixed experiment configuration."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models" / "distilbart-cnn-12-6"
MODEL_ID = "sshleifer/distilbart-cnn-12-6"
DATASET_ID = "abisee/cnn_dailymail"
DATASET_CONFIG = "3.0.0"
SEED = 42
EVALUATION_SIZE = 500
MAX_INPUT_TOKENS = 1024
MAX_ARTICLE_CHARS = 100_000
DEFAULT_MIN_NEW_TOKENS = 30
DEFAULT_MAX_NEW_TOKENS = 128
GENERATION = {
    "num_beams": 4,
    "length_penalty": 2.0,
    "no_repeat_ngram_size": 3,
    "early_stopping": True,
    "do_sample": False,
}
