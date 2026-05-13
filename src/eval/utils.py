from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from src.eval.config import (
    FAILED_RAW_DIR,
    GROUND_TRUTH_DIR,
    LABELS_DIR,
    METADATA_PATH,
    MODEL_CONFIGS,
    PROMPT_ROOT,
    TRANSCRIPTS_DIR,
    VALIDATIONS_DIR,
)
from src.eval.models import LabeledExample
from src.eval.scenarios import DatasetSpec


def load_prompt_yaml(path: Path) -> dict[str, str]:
    """Load the simple prompt YAML format used here: top-level `name: |` blocks."""
    prompts: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []

    for raw_line in path.read_text().splitlines():
        if raw_line and not raw_line.startswith(" ") and raw_line.endswith(": |"):
            if current_key is not None:
                prompts[current_key] = "\n".join(current_lines).strip()
            current_key = raw_line[:-3]
            current_lines = []
            continue

        if current_key is not None:
            current_lines.append(raw_line[2:] if raw_line.startswith("  ") else raw_line)

    if current_key is not None:
        prompts[current_key] = "\n".join(current_lines).strip()

    if not prompts:
        raise ValueError(f"No prompt blocks found in {path}")
    return prompts


def load_prompt_pair(task_name: str) -> dict[str, str]:
    task_dir = PROMPT_ROOT / task_name
    system_prompt = load_prompt_yaml(task_dir / "system.yaml")["prompt"]
    user_prompt = load_prompt_yaml(task_dir / "user.yaml")["prompt"]
    return {"system": system_prompt, "user": user_prompt}


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def save_failed_raw_output(example_id: str, stage: str, raw_text: str) -> None:
    write_text(FAILED_RAW_DIR / stage / f"{example_id}.txt", raw_text)


def save_ground_truth_package(package: dict[str, Any]) -> None:
    example_id = package["example_id"]
    write_json(GROUND_TRUTH_DIR / f"{example_id}.json", package)


def save_transcript_package(package: dict[str, Any]) -> None:
    example_id = package["example_id"]
    write_text(TRANSCRIPTS_DIR / f"{example_id}.txt", package["transcript"])
    write_json(
        LABELS_DIR / f"{example_id}.json",
        {
            "example_id": example_id,
            "difficulty": package["difficulty"],
            "archetype_id": package["archetype_id"],
            "challenge_tags": package["challenge_tags"],
            "section_targets": package["section_targets"],
            "age_band": package.get("age_band"),
            "include_mobile_phone": package.get("include_mobile_phone"),
            "include_email": package.get("include_email"),
            "transcript_path": str(TRANSCRIPTS_DIR / f"{example_id}.txt"),
            "expected": package["expected"],
        },
    )


def save_validated_package(package: dict[str, Any]) -> None:
    example_id = package["example_id"]
    write_json(VALIDATIONS_DIR / f"{example_id}.json", package["validation"])
    metadata_row = {
        "example_id": example_id,
        "difficulty": package["difficulty"],
        "archetype_id": package["archetype_id"],
        "challenge_tags": package["challenge_tags"],
        "section_targets": package["section_targets"],
        "age_band": package.get("age_band"),
        "include_mobile_phone": package.get("include_mobile_phone"),
        "include_email": package.get("include_email"),
        "models": {
            task_name: config["model"] for task_name, config in MODEL_CONFIGS.items()
        },
        "validation": package["validation"],
    }
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing_ids: set[str] = set()
    if METADATA_PATH.exists():
        for line in METADATA_PATH.read_text().splitlines():
            if line.strip():
                try:
                    existing_ids.add(json.loads(line)["example_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    if example_id not in existing_ids:
        with METADATA_PATH.open("a") as f:
            f.write(json.dumps(metadata_row, ensure_ascii=False) + "\n")


save_validate_package = save_validated_package


def save_generated_example(package: dict[str, Any]) -> None:
    save_ground_truth_package(package)
    save_transcript_package(package)
    if "validation" in package:
        save_validated_package(package)


def load_ground_truth_package(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def missing_ground_truth_specs(specs: list[DatasetSpec]) -> list[DatasetSpec]:
    saved_ids = {path.stem for path in GROUND_TRUTH_DIR.glob("*.json")}
    return [spec for spec in specs if spec.example_id not in saved_ids]


def load_failed_raw_output(example_id: str, stage: str = "ground_truth") -> str:
    return (FAILED_RAW_DIR / stage / f"{example_id}.txt").read_text()


def load_transcript_package(label_path: Path) -> dict[str, Any]:
    payload = json.loads(label_path.read_text())
    transcript = Path(payload["transcript_path"]).read_text()
    return {**payload, "transcript": transcript}


def load_labeled_example(label_path: Path) -> LabeledExample:
    payload = json.loads(label_path.read_text())
    return LabeledExample.model_validate(payload)


def reset_synthetic_output_dirs() -> None:
    for directory in [
        GROUND_TRUTH_DIR,
        TRANSCRIPTS_DIR,
        LABELS_DIR,
        VALIDATIONS_DIR,
        FAILED_RAW_DIR,
    ]:
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text("")
