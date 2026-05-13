from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.eval.models import CIFExtraction
from src.eval.scoring import ExtractionScoreResult, SectionScore


def strip_front_matter(text: str) -> str:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return text
    try:
        end = lines.index("---", 1)
        return "\n".join(lines[end + 1:]).lstrip("\n")
    except ValueError:
        return text


def load_passing_cases(
    validations_dir: Path = Path("data/synthetic/validations"),
    labels_dir: Path = Path("data/synthetic/labels"),
) -> list[dict]:
    cases = []
    for vf in sorted(validations_dir.glob("*.json")):
        v = json.loads(vf.read_text())
        if not v.get("is_valid"):
            continue
        label_path = labels_dir / vf.name
        if not label_path.exists():
            continue
        label = json.loads(label_path.read_text())
        transcript = Path(label["transcript_path"]).read_text()
        cases.append({
            "example_id": label["example_id"],
            "difficulty":  label["difficulty"],
            "transcript":  transcript,
            "gt":          CIFExtraction.model_validate(label["expected"]),
        })
    return cases


def cache_load_extractions(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    return [
        {"example_id": r["example_id"],
         "extracted":  CIFExtraction.model_validate(r["extracted"]),
         "error":      r["error"]}
        for r in raw
    ]


def cache_save_extractions(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        [{"example_id": r["example_id"],
          "extracted":  r["extracted"].model_dump(mode="json"),
          "error":      r["error"]}
         for r in results],
        indent=1,
    ))


def cache_load_scores(path: Path) -> list[ExtractionScoreResult] | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    return [
        ExtractionScoreResult(
            example_id=r["example_id"],
            section_scores=[SectionScore(s["section"], s["score"], s["reasoning"])
                            for s in r["section_scores"]],
            error=r["error"],
        )
        for r in raw
    ]


def cache_save_scores(path: Path, scores: list[ExtractionScoreResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.to_dict() for r in scores], indent=1))


def make_scoring_input(results: list[dict], gt_lookup: dict) -> list[dict]:
    return [
        {"example_id": r["example_id"],
         "gt":          gt_lookup[r["example_id"]],
         "extracted":   r["extracted"]}
        for r in results if not r["error"]
    ]


def any_nonnull(obj: Any) -> bool:
    if obj is None:
        return False
    if isinstance(obj, dict):
        return any(any_nonnull(v) for v in obj.values())
    if isinstance(obj, list):
        return any(any_nonnull(item) for item in obj)
    return True


def gt_section_nonempty(gt: CIFExtraction, section: str) -> bool:
    d = gt.model_dump(mode="json")
    m = re.match(r'^(.+)\[(\d+)\]$', section)
    if m:
        field, idx = m.group(1), int(m.group(2))
        lst = d.get(field, [])
        return idx < len(lst) and any_nonnull(lst[idx])
    lookup = {
        "has_client2":                   lambda: d.get("has_client2") is not None,
        "client1_personal":              lambda: any_nonnull(d["client1"]["personal"]),
        "client1_employment":            lambda: any_nonnull(d["client1"]["employment"]),
        "client2_personal":              lambda: bool(d.get("has_client2")) and any_nonnull(d["client2"]["personal"]),
        "client2_employment":            lambda: bool(d.get("has_client2")) and any_nonnull(d["client2"]["employment"]),
        "household":                     lambda: any_nonnull(d["household"]),
        "risk_profile_and_preferences":  lambda: any_nonnull(d["risk_profile_and_preferences"]),
        "estate_planning":               lambda: any_nonnull(d["estate_planning"]),
    }
    fn = lookup.get(section)
    return bool(fn()) if fn else False


def section_scores_from(score_results: list[ExtractionScoreResult]) -> dict[str, list[float]]:
    d: dict[str, list[float]] = defaultdict(list)
    for r in score_results:
        if r.error:
            continue
        for section, score in r.section_summary.items():
            d[section].append(score)
    return d


def prf_metrics(
    score_results: list[ExtractionScoreResult],
    gt_lookup: dict[str, CIFExtraction],
) -> dict[str, float]:
    tp = fp = fn = tn = 0
    for sr in score_results:
        if sr.error:
            continue
        gt = gt_lookup[sr.example_id]
        for ss in sr.section_scores:
            gt_has = gt_section_nonempty(gt, ss.section)
            if   ss.score == 1.0 and     gt_has: tp += 1
            elif ss.score == 0.0 and     gt_has: fn += 1
            elif ss.score == 0.0 and not gt_has: fp += 1
            else:                                tn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    hall = fp / (fp + tn) if (fp + tn) else 0.0
    acc  = [r.overall_accuracy for r in score_results if not r.error]
    return {"acc": statistics.mean(acc) if acc else 0.0,
            "prec": prec, "rec": rec, "f1": f1, "hall": hall,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def count_leaves(obj: Any) -> tuple[int, int]:
    if isinstance(obj, dict):
        t = nn = 0
        for v in obj.values():
            dt, dn = count_leaves(v)
            t += dt; nn += dn
        return t, nn
    if isinstance(obj, list):
        t = nn = 0
        for item in obj:
            dt, dn = count_leaves(item)
            t += dt; nn += dn
        return t, nn
    return 1, (0 if obj is None else 1)


def holdout_metrics(result: dict) -> dict:
    cif: CIFExtraction = result["extracted"]
    d = cif.model_dump(mode="json")
    total, non_null = count_leaves(d)
    list_sections = {s: len(d.get(s, [])) for s in [
        "incomes", "expenses", "pensions_and_retirement_accounts",
        "savings_and_investments", "loans_and_mortgages", "other_assets", "objectives",
    ]}
    def _any_nn(obj: dict) -> bool:
        return any(v is not None for v in obj.values())
    scalar_sections = {
        "has_client2":              d.get("has_client2") is not None,
        "client1.personal":         _any_nn(d["client1"]["personal"]),
        "client1.employment":       _any_nn(d["client1"]["employment"]),
        "client2.personal":         bool(d.get("has_client2")) and _any_nn(d["client2"]["personal"]),
        "client2.employment":       bool(d.get("has_client2")) and _any_nn(d["client2"]["employment"]),
        "household":                bool(d["household"].get("partner_or_spouse_name")
                                        or d["household"].get("children_or_dependants")),
        "risk_profile_and_preferences": (
            _any_nn({k: v for k, v in d["risk_profile_and_preferences"].items() if k != "key_concerns"})
            or bool(d["risk_profile_and_preferences"].get("key_concerns"))
        ),
        "estate_planning":          _any_nn(d["estate_planning"]),
    }
    return {
        "example_id":       result["example_id"],
        "completeness":     non_null / total if total else 0.0,
        "fields_populated": non_null,
        "total_fields":     total,
        "list_sections":    list_sections,
        "scalar_sections":  scalar_sections,
    }
