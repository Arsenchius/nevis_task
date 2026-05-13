"""Fix known data-quality issues in generated ground-truth JSON files.

Repairs applied:
  1. MoneyAmount.is_approximate=False with approximate wording in raw_text → set True.
  2. MoneyAmount.is_approximate=False with lower_bound/upper_bound set → set True.
  3. estate_planning.notes containing advisor meta-commentary → stripped to
     factual content only.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.eval.config import GROUND_TRUTH_DIR

_APPROX_WORDS = {"about", "around", "roughly", "approximately", "close to", "maybe", "near"}


def _fix_money_amount(obj: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    raw = (obj.get("raw_text") or "").lower()
    has_approx_word = any(w in raw for w in _APPROX_WORDS)
    has_bounds = obj.get("lower_bound") is not None or obj.get("upper_bound") is not None
    if (has_approx_word or has_bounds) and obj.get("is_approximate") is False:
        return {**obj, "is_approximate": True}, True
    return obj, False


def _fix_values(value: Any) -> tuple[Any, bool]:
    if isinstance(value, dict):
        if "normalized_amount" in value:
            return _fix_money_amount(value)
        changed = False
        result: dict[str, Any] = {}
        for k, v in value.items():
            fixed, c = _fix_values(v)
            result[k] = fixed
            changed = changed or c
        return result, changed
    if isinstance(value, list):
        changed = False
        result_list: list[Any] = []
        for item in value:
            fixed, c = _fix_values(item)
            result_list.append(fixed)
            changed = changed or c
        return result_list, changed
    return value, False


def _clean_estate_notes(notes: str) -> str | None:
    """Strip advisor meta-commentary; keep factual content."""
    if not notes:
        return notes

    # 1. Leading "Advisor noise: <sentence>. " → remove the noise sentence
    notes = re.sub(r'^Advisor noise:\s+[^.]+\.\s*', '', notes, flags=re.IGNORECASE)

    # 2. "Adviser noise <clause>[.;—]" anywhere → remove
    notes = re.sub(r'[Aa]dviser? noise [^.;—]+[.;—]\s*', '', notes)

    # 3. Leading "Advisor <verb> <clause>; " → remove whole leading clause (keep rest)
    notes = re.sub(r'^[Aa]dvisor [^;]+;\s*', '', notes)

    # 4. Transform remaining leading "Advisor <verb> [that] <clause>" → keep clause
    notes = re.sub(
        r'^[Aa]dvisor (mentioned|noted|asked)\s+(?:that\s+)?',
        '',
        notes,
        flags=re.IGNORECASE,
    )

    # 5. "; advisor [also] <verb> <clause>." mid-sentence → replace with ". "
    notes = re.sub(
        r';\s*[Aa]dvisor (?:also )?(mentioned|noted|asked)[^.]+\.',
        '.',
        notes,
        flags=re.IGNORECASE,
    )

    # 6. ", advisor also <verb> <clause>" → remove
    notes = re.sub(
        r',?\s*[Aa]dvisor also (noted|mentioned)[^.]+\.?',
        '',
        notes,
        flags=re.IGNORECASE,
    )

    # 7. " Advisor <verb> <clause>." mid-sentence → remove
    notes = re.sub(
        r'\s+[Aa]dvisor (mentioned|noted)[^.]+\.\s*',
        ' ',
        notes,
        flags=re.IGNORECASE,
    )

    # 8. "X, but client/they corrected that Y" where X began with Advisor → keep Y
    notes = re.sub(
        r'[Aa]dvisor [^,]+, but (?:client|they|she|he) corrected that ',
        '',
        notes,
        flags=re.IGNORECASE,
    )

    # Normalise whitespace and punctuation
    notes = re.sub(r'\s+', ' ', notes)
    notes = re.sub(r'\.[\s\.]+', '. ', notes)

    # Capitalise after sentence boundaries introduced by cleanup
    notes = re.sub(
        r'([.!?]\s+)([a-z])',
        lambda m: m.group(1) + m.group(2).upper(),
        notes,
    )

    notes = notes.strip('. \n').strip()
    if not notes:
        return None
    return notes[0].upper() + notes[1:]


def repair_file(path: Path) -> bool:
    """Repair a single ground-truth JSON file in place. Returns True if changed."""
    pkg = json.loads(path.read_text())
    expected = pkg["expected"]

    fixed_expected, money_changed = _fix_values(expected)

    notes_changed = False
    ep = (fixed_expected.get("estate_planning") or {})
    notes = ep.get("notes")
    if notes:
        cleaned = _clean_estate_notes(notes)
        if cleaned != notes:
            ep = {**ep, "notes": cleaned}
            fixed_expected = {**fixed_expected, "estate_planning": ep}
            notes_changed = True

    if money_changed or notes_changed:
        pkg["expected"] = fixed_expected
        path.write_text(json.dumps(pkg, indent=2, ensure_ascii=False) + "\n")
        return True
    return False


def repair_all(directory: str | Path = GROUND_TRUTH_DIR) -> dict[str, Any]:
    """Repair all ground-truth files and return a summary."""
    paths = sorted(Path(directory).glob("*.json"))
    fixed: list[str] = []
    for p in paths:
        if repair_file(p):
            fixed.append(p.name)
    return {
        "total": len(paths),
        "fixed": len(fixed),
        "unchanged": len(paths) - len(fixed),
        "fixed_files": fixed,
    }


if __name__ == "__main__":
    result = repair_all()
    print(f"Repaired {result['fixed']}/{result['total']} files")
    for name in result["fixed_files"]:
        print(f"  {name}")
