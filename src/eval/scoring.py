from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import AsyncOpenAI

from src.eval.models import CIFExtraction
from src.eval.utils import load_prompt_pair

_PROMPTS: dict[str, str] | None = None

_LIST_SECTIONS = [
    "incomes",
    "expenses",
    "pensions_and_retirement_accounts",
    "savings_and_investments",
    "loans_and_mortgages",
    "other_assets",
    "objectives",
]


def _get_prompts() -> dict[str, str]:
    global _PROMPTS
    if _PROMPTS is None:
        _PROMPTS = load_prompt_pair("scoring")
    return _PROMPTS


def _dump(model: Any) -> Any:
    if model is None:
        return None
    return model.model_dump(mode="json")


def _build_sections_payload(gt: CIFExtraction, extracted: CIFExtraction) -> dict[str, Any]:
    payload: dict[str, Any] = {}

    payload["has_client2"] = {"gt": gt.has_client2, "extracted": extracted.has_client2}

    payload["client1_personal"] = {
        "gt": _dump(gt.client1.personal),
        "extracted": _dump(extracted.client1.personal),
    }
    payload["client1_employment"] = {
        "gt": _dump(gt.client1.employment),
        "extracted": _dump(extracted.client1.employment),
    }

    if gt.has_client2:
        payload["client2_personal"] = {
            "gt": _dump(gt.client2.personal),
            "extracted": _dump(extracted.client2.personal),
        }
        payload["client2_employment"] = {
            "gt": _dump(gt.client2.employment),
            "extracted": _dump(extracted.client2.employment),
        }

    payload["household"] = {
        "gt": _dump(gt.household),
        "extracted": _dump(extracted.household),
    }

    for section_field in _LIST_SECTIONS:
        gt_list = getattr(gt, section_field)
        ext_list = getattr(extracted, section_field)
        payload[section_field] = {
            "gt": [_dump(item) for item in gt_list],
            "extracted": [_dump(item) for item in ext_list],
        }

    payload["risk_profile_and_preferences"] = {
        "gt": _dump(gt.risk_profile_and_preferences),
        "extracted": _dump(extracted.risk_profile_and_preferences),
    }
    payload["estate_planning"] = {
        "gt": _dump(gt.estate_planning),
        "extracted": _dump(extracted.estate_planning),
    }

    return payload


@dataclass
class SectionScore:
    section: str
    score: float
    reasoning: str


@dataclass
class ExtractionScoreResult:
    example_id: str
    section_scores: list[SectionScore] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def overall_accuracy(self) -> float:
        if not self.section_scores:
            return 0.0
        return sum(s.score for s in self.section_scores) / len(self.section_scores)

    @property
    def section_summary(self) -> dict[str, float]:
        buckets: dict[str, list[float]] = {}
        for s in self.section_scores:
            base = re.sub(r"\[\d+\]$", "", s.section)
            buckets.setdefault(base, []).append(s.score)
        return {k: sum(v) / len(v) for k, v in buckets.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "overall_accuracy": self.overall_accuracy,
            "section_summary": self.section_summary,
            "section_scores": [
                {"section": s.section, "score": s.score, "reasoning": s.reasoning}
                for s in self.section_scores
            ],
            "error": self.error,
        }


async def score_extraction(
    client: AsyncOpenAI,
    example_id: str,
    gt: CIFExtraction,
    extracted: CIFExtraction,
    *,
    model: str = "gpt-5.4-mini",
    temperature: float = 0.0,
    max_retries: int = 3,
) -> ExtractionScoreResult:
    prompts = _get_prompts()
    sections_payload = _build_sections_payload(gt, extracted)
    sections_json = json.dumps(sections_payload, indent=2, ensure_ascii=False)

    user_prompt = prompts["user"].format(
        example_id=example_id,
        sections_json=sections_json,
    )
    system_prompt = prompts["system"]

    request: dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "timeout": 120.0,
    }

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = await client.responses.create(**request)
            raw = response.output_text.strip()
            parsed = json.loads(raw)
            scores = [
                SectionScore(
                    section=item["section"],
                    score=float(item["score"]),
                    reasoning=item["reasoning"],
                )
                for item in parsed["scores"]
            ]
            return ExtractionScoreResult(example_id=example_id, section_scores=scores)
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                await asyncio.sleep(2**attempt)

    return ExtractionScoreResult(
        example_id=example_id,
        error=f"Failed after {max_retries} attempts: {last_error!r}",
    )


async def score_extractions_async(
    client: AsyncOpenAI,
    examples: list[dict[str, Any]],
    *,
    model: str = "gpt-5.4-mini",
    temperature: float = 0.0,
    max_concurrency: int = 4,
    show_progress: bool = True,
) -> list[ExtractionScoreResult]:
    """Score a batch of extractions concurrently.

    Each element of `examples` must have keys:
      - example_id: str
      - gt: CIFExtraction
      - extracted: CIFExtraction
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    completed = 0
    total = len(examples)

    async def _score_one(ex: dict[str, Any]) -> ExtractionScoreResult:
        nonlocal completed
        async with semaphore:
            result = await score_extraction(
                client,
                ex["example_id"],
                ex["gt"],
                ex["extracted"],
                model=model,
                temperature=temperature,
            )
            completed += 1
            if show_progress:
                acc = f"{result.overall_accuracy:.0%}" if not result.error else "ERROR"
                print(f"  [{completed}/{total}] {ex['example_id']} — {acc}")
            return result

    tasks = [_score_one(ex) for ex in examples]
    return await asyncio.gather(*tasks)
