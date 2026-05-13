from __future__ import annotations

import json
from typing import Any

from src.eval.config import PROMPT_TASKS
from src.eval.models import CIFExtraction
from src.eval.scenarios import DatasetSpec, SCENARIO_ARCHETYPES
from src.eval.utils import load_prompt_pair


PROMPTS = {task_name: load_prompt_pair(task_name) for task_name in PROMPT_TASKS}

GROUND_TRUTH_SYSTEM_PROMPT = PROMPTS["ground_truth"]["system"]
TRANSCRIPT_SYSTEM_PROMPT = PROMPTS["transcript"]["system"]
VALIDATION_SYSTEM_PROMPT = PROMPTS["validation"]["system"]


def render_user_prompt(task_name: str, **kwargs: Any) -> str:
    return PROMPTS[task_name]["user"].format(**kwargs)


def schema_for_prompt() -> dict[str, Any]:
    return CIFExtraction.model_json_schema()


_SECTION_TOPIC_LABELS: dict[str, str] = {
    "employment": "employment status, occupation, employer, start date, retirement age/date",
    "household_dependants": "partner or spouse details, children, dependants",
    "income": "salary, Social Security, pension income, dividends, rental income, any income sources",
    "expenses": "rent, mortgage payments, utilities, insurance premiums, discretionary spending",
    "pensions_retirement": "pension plans, IRA, 401(k), SEP IRA, Roth IRA, annuity, retirement account balances or contributions",
    "savings_investments": "savings accounts, brokerage accounts, cash holdings, investment balances",
    "loans_mortgages": "mortgage, student loans, credit card debt, personal loans, outstanding balances, debt repayments",
    "other_assets": "property, vehicles, business assets, physical assets",
    "objectives": "financial goals, retirement plans, target amounts, planning priorities",
    "risk_profile_preferences": (
        "risk tolerance, attitude to market losses or volatility, investment preferences, "
        "willingness to take risk, how the client reacts to portfolio drops — "
        "the ADVISER must NOT ask any risk or investment-attitude question"
    ),
    "estate_planning": (
        "wills, trusts, powers of attorney, beneficiary designations, estate documents, "
        "inheritance or succession plans — the ADVISER must NOT ask about any of these"
    ),
}


def _suppressed_sections(expected: dict[str, Any]) -> list[str]:
    """Return section names that are empty/null in the expected label.

    These must not be discussed in the transcript so that extractor output
    on those topics is unambiguously a hallucination.
    """

    def _non_empty(value: Any) -> bool:
        if value is None:
            return False
        if value == [] or value == {}:
            return False
        if isinstance(value, dict):
            return any(_non_empty(v) for v in value.values())
        return True

    suppressed: list[str] = []

    emp = expected.get("client1", {}).get("employment", {})
    if not _non_empty(emp):
        suppressed.append("employment")

    hh = expected.get("household", {})
    if not _non_empty(hh):
        suppressed.append("household_dependants")

    if not _non_empty(expected.get("incomes")):
        suppressed.append("income")

    if not _non_empty(expected.get("expenses")):
        suppressed.append("expenses")

    if not _non_empty(expected.get("pensions_and_retirement_accounts")):
        suppressed.append("pensions_retirement")

    if not _non_empty(expected.get("savings_and_investments")):
        suppressed.append("savings_investments")

    if not _non_empty(expected.get("loans_and_mortgages")):
        suppressed.append("loans_mortgages")

    if not _non_empty(expected.get("other_assets")):
        suppressed.append("other_assets")

    if not _non_empty(expected.get("objectives")):
        suppressed.append("objectives")

    rp = expected.get("risk_profile_and_preferences", {})
    if not _non_empty(rp):
        suppressed.append("risk_profile_preferences")

    ep = expected.get("estate_planning", {})
    if not _non_empty(ep):
        suppressed.append("estate_planning")

    return suppressed


def make_ground_truth_prompt(spec: DatasetSpec) -> str:
    archetype = next(a for a in SCENARIO_ARCHETYPES if a["id"] == spec.archetype_id)
    return render_user_prompt(
        "ground_truth",
        example_id=spec.example_id,
        difficulty=spec.difficulty,
        archetype_description=archetype["description"],
        section_targets=spec.section_targets,
        challenge_tags=spec.challenge_tags,
        age_band=spec.age_band,
        include_mobile_phone=spec.include_mobile_phone,
        include_email=spec.include_email,
        client1_name=spec.client1_name or "generate a realistic unique name",
        client2_name=spec.client2_name or "generate a realistic unique name",
        schema_json=json.dumps(schema_for_prompt(), indent=2),
    )


def make_transcript_prompt(spec: DatasetSpec, expected_json: dict[str, Any]) -> str:
    suppressed = _suppressed_sections(expected_json)
    if suppressed:
        lines = [
            f"  - {s}: {_SECTION_TOPIC_LABELS.get(s, s)}"
            for s in suppressed
        ]
        suppressed_str = "\n" + "\n".join(lines)
    else:
        suppressed_str = "none"
    return render_user_prompt(
        "transcript",
        example_id=spec.example_id,
        difficulty=spec.difficulty,
        challenge_tags=spec.challenge_tags,
        suppressed_sections=suppressed_str,
        expected_json=json.dumps(expected_json, indent=2),
    )


def make_validation_prompt(
    spec: DatasetSpec,
    transcript: str,
    expected_json: dict[str, Any],
) -> str:
    suppressed = _suppressed_sections(expected_json)
    if suppressed:
        lines = [
            f"  - {s}: {_SECTION_TOPIC_LABELS.get(s, s)}"
            for s in suppressed
        ]
        suppressed_str = "\n" + "\n".join(lines)
    else:
        suppressed_str = "none"
    return render_user_prompt(
        "validation",
        example_id=spec.example_id,
        difficulty=spec.difficulty,
        challenge_tags=spec.challenge_tags,
        suppressed_sections=suppressed_str,
        transcript=transcript,
        expected_json=json.dumps(expected_json, indent=2),
    )
