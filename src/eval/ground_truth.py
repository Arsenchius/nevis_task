from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError
from openai import AsyncOpenAI

from src.eval.llm import generate_with_llm
from src.eval.models import CIFExtraction
from src.eval.prompts import GROUND_TRUTH_SYSTEM_PROMPT, make_ground_truth_prompt
from src.eval.scenarios import DatasetSpec
from src.eval.utils import save_failed_raw_output


def spec_to_metadata(spec: DatasetSpec) -> dict[str, Any]:
    return {
        "example_id": spec.example_id,
        "difficulty": spec.difficulty,
        "archetype_id": spec.archetype_id,
        "challenge_tags": spec.challenge_tags,
        "section_targets": spec.section_targets,
        "age_band": spec.age_band,
        "include_mobile_phone": spec.include_mobile_phone,
        "include_email": spec.include_email,
    }


def extract_json_object(text: str) -> str:
    """Extract the first top-level JSON object from raw model text."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError:
        pass

    # Attempt to balance any unclosed braces and/or brackets. Arrays are typically
    # more deeply nested than the top-level object, so close brackets before braces.
    def _balance_and_try(text: str) -> str | None:
        missing_braces = text.count("{") - text.count("}")
        missing_brackets = text.count("[") - text.count("]")
        if missing_braces < 0 or missing_brackets < 0:
            return None
        if missing_braces == 0 and missing_brackets == 0:
            return None
        candidate = text + ("]" * missing_brackets) + ("}" * missing_braces)
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            return None

    balanced = _balance_and_try(cleaned)
    if balanced is not None:
        return balanced

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output")
    candidate = cleaned[start : end + 1]
    try:
        json.loads(candidate)
    except json.JSONDecodeError:
        balanced = _balance_and_try(candidate)
        if balanced is None:
            raise
        candidate = balanced
        json.loads(candidate)
    return candidate


def prefer_non_null_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Resolve duplicate JSON keys by preserving useful non-null values.

    Some generated ground-truth JSONs include a valid object and then repeat the same
    key as null, e.g. {"employment": {...}, "employment": null}. Standard json.loads
    keeps the final null and loses the useful object. For synthetic-data generation,
    preserving the non-null value is the safer recovery.
    """
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result and result[key] is not None and value is None:
            continue
        result[key] = value
    return result


def parse_json_object(text: str) -> dict[str, Any]:
    json_text = extract_json_object(text)
    return json.loads(json_text, object_pairs_hook=prefer_non_null_duplicate_keys)


def normalize_cif_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Repair recoverable schema-shape issues in generated ground-truth JSON."""
    payload = dict(payload)

    # The schema intentionally keeps client2 as an empty ClientProfile object even
    # when has_client2 is false. Models often emit null for absent client2.
    if payload.get("client2") is None:
        payload["client2"] = {}

    for client_key in ("client1", "client2"):
        client = payload.get(client_key)
        if client is None:
            payload[client_key] = {}
            client = payload[client_key]
        if isinstance(client, dict):
            if client.get("personal") is None:
                client["personal"] = {}
            if client.get("employment") is None:
                client["employment"] = {}
            # Household belongs at the CIF root, not inside a client profile.
            client.pop("household", None)

    list_fields = [
        "incomes",
        "expenses",
        "pensions_and_retirement_accounts",
        "savings_and_investments",
        "loans_and_mortgages",
        "other_assets",
        "objectives",
    ]
    for field in list_fields:
        if payload.get(field) is None:
            payload[field] = []

    if payload.get("household") is None:
        payload["household"] = {}
    if payload.get("risk_profile_and_preferences") is None:
        payload["risk_profile_and_preferences"] = {}
    if payload.get("estate_planning") is None:
        payload["estate_planning"] = {}

    return payload


async def generate_ground_truth_for_spec(
    client: AsyncOpenAI,
    spec: DatasetSpec,
    max_schema_retries: int = 3,
) -> dict[str, Any]:
    last_error: Exception | None = None
    last_expected_text = ""
    for attempt in range(1, max_schema_retries + 1):
        expected_text = await generate_with_llm(
            client,
            "ground_truth",
            system_prompt=GROUND_TRUTH_SYSTEM_PROMPT,
            user_prompt=make_ground_truth_prompt(spec),
        )
        last_expected_text = expected_text
        try:
            expected_json = normalize_cif_payload(parse_json_object(expected_text))
            expected = CIFExtraction.model_validate(expected_json)
            break
        except (ValueError, json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            save_failed_raw_output(
                spec.example_id,
                f"ground_truth_attempt_{attempt}",
                expected_text,
            )
    else:
        save_failed_raw_output(spec.example_id, "ground_truth", last_expected_text)
        if last_error is not None:
            raise last_error
        raise ValueError("Ground-truth generation failed without an exception")

    return {
        **spec_to_metadata(spec),
        "expected": expected.model_dump(mode="json"),
    }


def recover_ground_truth_from_raw(spec: DatasetSpec, raw_text: str) -> dict[str, Any]:
    """Build a ground-truth package from a previously saved raw model output."""
    expected_json = normalize_cif_payload(parse_json_object(raw_text))
    expected = CIFExtraction.model_validate(expected_json)
    return {
        **spec_to_metadata(spec),
        "expected": expected.model_dump(mode="json"),
    }
