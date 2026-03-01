from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_BANK_PATH = REPO_ROOT / "data" / "prompt_bank_v1.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"
