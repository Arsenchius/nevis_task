from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI

from src.eval.ground_truth import extract_json_object
from src.eval.llm import generate_with_llm
from src.eval.prompts import VALIDATION_SYSTEM_PROMPT, make_validation_prompt
from src.eval.transcripts import package_to_spec


async def validate_transcript_package(
    client: AsyncOpenAI,
    package: dict[str, Any],
) -> dict[str, Any]:
    spec = package_to_spec(package)
    validation_text = await generate_with_llm(
        client,
        "validation",
        system_prompt=VALIDATION_SYSTEM_PROMPT,
        user_prompt=make_validation_prompt(spec, package["transcript"], package["expected"]),
    )
    try:
        validation = json.loads(validation_text)
    except json.JSONDecodeError:
        try:
            validation = json.loads(extract_json_object(validation_text))
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Could not parse validation response as JSON for {package.get('example_id')}"
            ) from exc
    return {**package, "validation": validation}
