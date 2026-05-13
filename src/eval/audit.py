from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import json


SECTION_CONTENT_CHECKS = {
    "personal_details": lambda expected: bool(
        expected.get("client1", {}).get("personal", {}).get("first_name")
        or expected.get("client1", {}).get("personal", {}).get("date_of_birth")
    ),
    # Absence is meaningful for household/dependants: no spouse/partner and no
    # dependants is a valid fact-find outcome represented as null + [].
    "household_dependants": lambda expected: bool(
        expected.get("household", {}).get("partner_or_spouse_name")
        or expected.get("household", {}).get("children_or_dependants")
    ),
    "employment": lambda expected: bool(
        expected.get("client1", {}).get("employment", {}).get("status")
        or expected.get("client1", {}).get("employment", {}).get("occupation")
    ),
    "income": lambda expected: bool(expected.get("incomes")),
    "expenses": lambda expected: bool(expected.get("expenses")),
    "pensions_retirement": lambda expected: bool(
        expected.get("pensions_and_retirement_accounts")
    ),
    "savings_investments": lambda expected: bool(
        expected.get("savings_and_investments")
    ),
    "loans_mortgages": lambda expected: bool(expected.get("loans_and_mortgages")),
    "other_assets": lambda expected: bool(expected.get("other_assets")),
    "objectives": lambda expected: bool(expected.get("objectives")),
    "risk_profile_preferences": lambda expected: bool(
        expected.get("risk_profile_and_preferences", {}).get("risk_score_or_label")
        or expected.get("risk_profile_and_preferences", {}).get("attitude_summary")
        or expected.get("risk_profile_and_preferences", {}).get("preferred_strategy")
        or expected.get("risk_profile_and_preferences", {}).get("key_concerns")
    ),
    "estate_planning": lambda expected: bool(
        expected.get("estate_planning", {}).get("will_status")
        or expected.get("estate_planning", {}).get("power_of_attorney_status")
        or expected.get("estate_planning", {}).get("notes")
    ),
}


def audit_ground_truth_dir(path: str | Path = "data/synthetic/ground_truth") -> dict[str, Any]:
    paths = sorted(Path(path).glob("*.json"))
    years = []
    phones = 0
    emails = 0
    client1_dob_missing = 0
    has_client2 = 0

    for file_path in paths:
        package = json.loads(file_path.read_text())
        expected = package["expected"]
        if expected.get("has_client2"):
            has_client2 += 1

        for client_key in ["client1", "client2"]:
            personal = expected.get(client_key, {}).get("personal", {})
            dob = personal.get("date_of_birth")
            if dob:
                try:
                    years.append(int(str(dob)[:4]))
                except ValueError:
                    pass
            elif client_key == "client1":
                client1_dob_missing += 1

            if personal.get("mobile_phone"):
                phones += 1
            if personal.get("email"):
                emails += 1

    return {
        "total_files": len(paths),
        "mobile_phone_fields": phones,
        "email_fields": emails,
        "has_client2": has_client2,
        "client1_dob_missing": client1_dob_missing,
        "dob_year_min": min(years) if years else None,
        "dob_year_max": max(years) if years else None,
        "dob_year_buckets": dict(sorted(Counter((year // 10) * 10 for year in years).items())),
        "dob_after_2000": sum(year >= 2000 for year in years),
        "dob_before_1960": sum(year < 1960 for year in years),
    }


def find_ground_truth_target_mismatches(
    path: str | Path = "data/synthetic/ground_truth",
) -> list[dict[str, Any]]:
    """Find labels where metadata targets a section but expected JSON is empty."""
    mismatches = []
    for file_path in sorted(Path(path).glob("*.json")):
        package = json.loads(file_path.read_text())
        expected = package["expected"]
        missing_sections = [
            section
            for section in package.get("section_targets", [])
            if section in SECTION_CONTENT_CHECKS
            and not SECTION_CONTENT_CHECKS[section](expected)
        ]
        if missing_sections:
            mismatches.append(
                {
                    "example_id": package["example_id"],
                    "difficulty": package["difficulty"],
                    "archetype_id": package["archetype_id"],
                    "missing_target_sections": missing_sections,
                }
            )
    return mismatches


def find_objective_section_conflicts(
    path: str | Path = "data/synthetic/ground_truth",
) -> list[dict[str, Any]]:
    """Find labels where an objective references a section that is null/empty.

    This catches the case where the GT generator wrote an estate_planning or
    risk objective but left the corresponding structured section unpopulated.
    The transcript generator will discuss those topics to support the objective,
    producing extractable facts that the null ground truth cannot validate.
    """
    OBJECTIVE_KEYWORD_TO_SECTION = {
        "estate": "estate_planning",
    }

    def _section_empty(expected: dict[str, Any], key: str) -> bool:
        val = expected.get(key)
        if not val:
            return True
        if isinstance(val, list):
            return len(val) == 0
        if isinstance(val, dict):
            return not any(v for v in val.values() if v)
        return False

    conflicts = []
    for file_path in sorted(Path(path).glob("*.json")):
        package = json.loads(file_path.read_text())
        expected = package["expected"]
        bad_objectives: list[str] = []
        for obj in expected.get("objectives", []):
            cat = (obj.get("category") or "").lower()
            for keyword, section_key in OBJECTIVE_KEYWORD_TO_SECTION.items():
                if keyword in cat and _section_empty(expected, section_key):
                    bad_objectives.append(
                        f"{obj.get('category')}: {obj.get('status_or_uncertainty')}"
                    )
        if bad_objectives:
            conflicts.append(
                {
                    "example_id": package["example_id"],
                    "difficulty": package["difficulty"],
                    "archetype_id": package["archetype_id"],
                    "conflicting_objectives": bad_objectives,
                }
            )
    return conflicts


def summarise_validations(
    path: str | Path = "data/synthetic/validations",
) -> dict[str, Any]:
    """Aggregate validation results across all saved validation JSONs.

    Returns a summary dict with counts by failure reason, a list of invalid
    cases with their reasons, and overall pass/fail totals.
    """
    paths = sorted(Path(path).glob("*.json"))
    total = len(paths)
    passed = 0
    failed = 0
    reason_counts: Counter = Counter()
    failures: list[dict[str, Any]] = []

    for file_path in paths:
        result = json.loads(file_path.read_text())
        example_id = file_path.stem
        if result.get("is_valid", False):
            passed += 1
        else:
            failed += 1
            reasons = result.get("failure_reasons", ["unknown"])
            reason_counts.update(reasons)
            failures.append(
                {
                    "example_id": example_id,
                    "failure_reasons": reasons,
                    "coverage_issues": result.get("coverage_issues", []),
                    "suppression_leaks": result.get("suppression_leaks", []),
                    "hallucinations": result.get("hallucinations", []),
                    "localization_issues": result.get("localization_issues", []),
                }
            )

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / total, 3) if total else None,
        "failure_reason_counts": dict(reason_counts),
        "failures": sorted(failures, key=lambda x: x["example_id"]),
    }


def find_inverse_section_mismatches(
    path: str | Path = "data/synthetic/ground_truth",
) -> list[dict[str, Any]]:
    """Find labels where a section has content but is NOT in section_targets.

    Useful for catching ground-truth files where the LLM generated data for
    sections that were intentionally excluded from the spec (e.g., estate_planning
    on easy examples, or pensions on archetypes that don't target them).
    """
    mismatches = []
    for file_path in sorted(Path(path).glob("*.json")):
        package = json.loads(file_path.read_text())
        expected = package["expected"]
        targets = set(package.get("section_targets", []))
        extra_sections = [
            section
            for section, check in SECTION_CONTENT_CHECKS.items()
            if section not in targets and check(expected)
        ]
        if extra_sections:
            mismatches.append(
                {
                    "example_id": package["example_id"],
                    "difficulty": package["difficulty"],
                    "archetype_id": package["archetype_id"],
                    "extra_populated_sections": extra_sections,
                }
            )
    return mismatches
