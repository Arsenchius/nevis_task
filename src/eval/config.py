from __future__ import annotations

from pathlib import Path

DATA_ROOT = Path("data")
SYNTHETIC_ROOT = DATA_ROOT / "synthetic"
GROUND_TRUTH_DIR = SYNTHETIC_ROOT / "ground_truth"
TRANSCRIPTS_DIR = SYNTHETIC_ROOT / "transcripts"
LABELS_DIR = SYNTHETIC_ROOT / "labels"
VALIDATIONS_DIR = SYNTHETIC_ROOT / "validations"
FAILED_RAW_DIR = SYNTHETIC_ROOT / "failed_raw"
METADATA_PATH = SYNTHETIC_ROOT / "metadata.jsonl"

PROMPT_ROOT = Path("src/prompts/eval")
PROMPT_TASKS = ("ground_truth", "transcript", "validation")

DIFFICULTY_PLAN = {
    "easy": 20,
    "medium": 30,
    "hard": 30,
}

SECTION_COVERAGE_TARGETS = {
    "personal_details": 1.00,
    "household_dependants": 0.70,
    "employment": 0.85,
    "income": 0.90,
    "expenses": 0.60,
    "pensions_retirement": 0.70,
    "savings_investments": 0.65,
    "loans_mortgages": 0.50,
    "other_assets": 0.25,
    "objectives": 0.90,
    "risk_profile_preferences": 0.45,
    "estate_planning": 0.35,
}

CHALLENGE_TAGS = [
    "numeric_exact",
    "numeric_approximate",
    "numeric_range",
    "numeric_shorthand",
    "date_normalization",
    "owner_attribution",
    "client2_present",
    "joint_assets",
    "negation",
    "correction",
    "missing_fields",
    "objectives_free_form",
    "risk_preferences",
    "estate_planning",
    "privacy_reference",
    "multiple_products",
    "advisor_noise",
]

MODEL_CONFIGS = {
    "ground_truth": {
        "model": "gpt-5.4-mini",
        "temperature": 0.1,
        "reasoning": {"effort": "none"},
        "timeout_seconds": 240.0,
    },
    "transcript": {
        "model": "gpt-5.4-mini",
        "temperature": 0.7,
        "reasoning": {"effort": "none"},
        "timeout_seconds": 600.0,
    },
    "validation": {
        "model": "gpt-5.1",
        "temperature": None,
        "reasoning": {"effort": "medium"},
        "timeout_seconds": 600.0,
    },
}
