from __future__ import annotations

import logging
from typing import Any

from openai import AsyncOpenAI

from src.eval.llm import generate_with_llm
from src.eval.prompts import TRANSCRIPT_SYSTEM_PROMPT, make_transcript_prompt
from src.eval.scenarios import DatasetSpec
from src.eval.transcript_audit import audit_transcript_format

logger = logging.getLogger(__name__)


def package_to_spec(package: dict[str, Any]) -> DatasetSpec:
    return DatasetSpec(
        example_id=package["example_id"],
        difficulty=package["difficulty"],
        archetype_id=package["archetype_id"],
        section_targets=package["section_targets"],
        challenge_tags=package["challenge_tags"],
        age_band=package.get("age_band", "unknown"),
        include_mobile_phone=package.get("include_mobile_phone", False),
        include_email=package.get("include_email", False),
        client1_name=package.get("client1_name", ""),
        client2_name=package.get("client2_name", ""),
    )


async def generate_transcript_for_package(
    client: AsyncOpenAI,
    package: dict[str, Any],
) -> dict[str, Any]:
    spec = package_to_spec(package)
    transcript = await generate_with_llm(
        client,
        "transcript",
        system_prompt=TRANSCRIPT_SYSTEM_PROMPT,
        user_prompt=make_transcript_prompt(spec, package["expected"]),
    )
    audit = audit_transcript_format(transcript)
    if audit.get("non_us_terms"):
        logger.warning(
            "%s: transcript contains non-US terms %s — regeneration recommended",
            package.get("example_id"),
            audit["non_us_terms"],
        )
    if audit.get("speaker_lines_without_timestamps_count", 0) > 0:
        logger.warning(
            "%s: %d speaker lines missing timestamps",
            package.get("example_id"),
            audit["speaker_lines_without_timestamps_count"],
        )
    return {**package, "transcript": transcript}
