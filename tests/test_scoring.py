"""Unit tests for src/eval/scoring.py"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.eval.models import (
    CIFExtraction,
    ClientProfile,
    EmploymentDetails,
    EstatePlanning,
    ExpenseItem,
    HouseholdDetails,
    IncomeItem,
    MoneyAmount,
    ObjectiveItem,
    Owner,
    PersonalDetails,
    RiskProfilePreferences,
)
from src.eval.scoring import (
    ExtractionScoreResult,
    SectionScore,
    _build_sections_payload,
    score_extraction,
    score_extractions_async,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(first="Jane", last="Doe", dob="1975-06-01") -> ClientProfile:
    return ClientProfile(
        personal=PersonalDetails(first_name=first, last_name=last, date_of_birth=dob),
        employment=EmploymentDetails(occupation="Engineer", desired_retirement_age=65),
    )


def _make_income(owner=Owner.CLIENT1, amount=90_000.0) -> IncomeItem:
    return IncomeItem(
        owner=owner,
        source_name="salary",
        amount=MoneyAmount(normalized_amount=amount, currency="USD"),
    )


def _make_cif(
    *,
    has_client2: bool = False,
    incomes: list[IncomeItem] | None = None,
    expenses: list[ExpenseItem] | None = None,
    objectives: list[ObjectiveItem] | None = None,
) -> CIFExtraction:
    return CIFExtraction(
        has_client2=has_client2,
        client1=_make_client(),
        household=HouseholdDetails(),
        incomes=incomes or [],
        expenses=expenses or [],
        objectives=objectives or [],
        risk_profile_and_preferences=RiskProfilePreferences(
            risk_score_or_label="moderate",
            key_concerns=["longevity risk"],
        ),
        estate_planning=EstatePlanning(will_status="has will"),
    )


def _mock_llm_response(scores: list[dict[str, Any]]) -> Any:
    """Return a fake OpenAI response object whose output_text is a JSON scores payload."""
    raw = json.dumps({"scores": scores})
    response = MagicMock()
    response.output_text = raw
    return response


# ---------------------------------------------------------------------------
# _build_sections_payload
# ---------------------------------------------------------------------------

class TestBuildSectionsPayload:
    def test_scalar_keys_always_present(self):
        gt = _make_cif()
        ext = _make_cif()
        payload = _build_sections_payload(gt, ext)
        for key in ("has_client2", "client1_personal", "client1_employment", "household",
                    "risk_profile_and_preferences", "estate_planning"):
            assert key in payload, f"missing key: {key}"

    def test_client2_keys_included_when_has_client2(self):
        gt = CIFExtraction(
            has_client2=True,
            client1=_make_client("Alice", "Smith"),
            client2=_make_client("Bob", "Smith"),
        )
        ext = CIFExtraction(
            has_client2=True,
            client1=_make_client("Alice", "Smith"),
            client2=_make_client("Bob", "Smith"),
        )
        payload = _build_sections_payload(gt, ext)
        assert "client2_personal" in payload
        assert "client2_employment" in payload

    def test_client2_keys_absent_when_has_client2_false(self):
        gt = _make_cif(has_client2=False)
        ext = _make_cif(has_client2=False)
        payload = _build_sections_payload(gt, ext)
        assert "client2_personal" not in payload
        assert "client2_employment" not in payload

    def test_list_section_both_populated(self):
        income = _make_income()
        gt = _make_cif(incomes=[income])
        ext = _make_cif(incomes=[income])
        payload = _build_sections_payload(gt, ext)
        assert "incomes" in payload
        assert len(payload["incomes"]["gt"]) == 1
        assert len(payload["incomes"]["extracted"]) == 1

    def test_list_section_gt_empty_extracted_populated(self):
        gt = _make_cif(incomes=[])
        ext = _make_cif(incomes=[_make_income()])
        payload = _build_sections_payload(gt, ext)
        assert payload["incomes"]["gt"] == []
        assert len(payload["incomes"]["extracted"]) == 1

    def test_list_section_both_empty(self):
        gt = _make_cif(incomes=[])
        ext = _make_cif(incomes=[])
        payload = _build_sections_payload(gt, ext)
        assert payload["incomes"]["gt"] == []
        assert payload["incomes"]["extracted"] == []

    def test_gt_values_serialized_as_json_compatible(self):
        gt = _make_cif(incomes=[_make_income()])
        ext = _make_cif(incomes=[])
        payload = _build_sections_payload(gt, ext)
        # Enum values should be plain strings, not Enum instances
        income_gt = payload["incomes"]["gt"][0]
        assert income_gt["owner"] == "client1", "Owner enum should be serialized as string"

    def test_personal_details_values_present(self):
        gt = _make_cif()
        ext = _make_cif()
        payload = _build_sections_payload(gt, ext)
        assert payload["client1_personal"]["gt"]["first_name"] == "Jane"
        assert payload["client1_personal"]["extracted"]["first_name"] == "Jane"


# ---------------------------------------------------------------------------
# ExtractionScoreResult
# ---------------------------------------------------------------------------

class TestExtractionScoreResult:
    def _make_result(self, scores: list[tuple[str, float]]) -> ExtractionScoreResult:
        return ExtractionScoreResult(
            example_id="test_001",
            section_scores=[SectionScore(s, v, "ok") for s, v in scores],
        )

    def test_overall_accuracy_all_correct(self):
        result = self._make_result([("client1_personal", 1.0), ("client1_employment", 1.0)])
        assert result.overall_accuracy == 1.0

    def test_overall_accuracy_all_wrong(self):
        result = self._make_result([("client1_personal", 0.0), ("incomes[0]", 0.0)])
        assert result.overall_accuracy == 0.0

    def test_overall_accuracy_mixed(self):
        result = self._make_result([
            ("client1_personal", 1.0),
            ("incomes[0]", 1.0),
            ("incomes[1]", 0.0),
            ("expenses[0]", 0.0),
        ])
        assert result.overall_accuracy == pytest.approx(0.5)

    def test_overall_accuracy_empty(self):
        result = ExtractionScoreResult(example_id="x")
        assert result.overall_accuracy == 0.0

    def test_section_summary_aggregates_list_indices(self):
        result = self._make_result([
            ("incomes[0]", 1.0),
            ("incomes[1]", 1.0),
            ("incomes[2]", 0.0),
            ("expenses[0]", 1.0),
            ("client1_personal", 1.0),
        ])
        summary = result.section_summary
        assert summary["incomes"] == pytest.approx(2 / 3)
        assert summary["expenses"] == pytest.approx(1.0)
        assert summary["client1_personal"] == pytest.approx(1.0)

    def test_section_summary_single_item(self):
        result = self._make_result([("objectives[0]", 0.0)])
        assert result.section_summary["objectives"] == pytest.approx(0.0)

    def test_to_dict_structure(self):
        result = self._make_result([("client1_personal", 1.0), ("incomes[0]", 0.0)])
        d = result.to_dict()
        assert d["example_id"] == "test_001"
        assert "overall_accuracy" in d
        assert "section_summary" in d
        assert isinstance(d["section_scores"], list)
        assert d["error"] is None

    def test_to_dict_with_error(self):
        result = ExtractionScoreResult(example_id="err_001", error="timeout")
        d = result.to_dict()
        assert d["error"] == "timeout"
        assert d["overall_accuracy"] == 0.0


# ---------------------------------------------------------------------------
# score_extraction (mocked LLM)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestScoreExtraction:
    async def test_successful_call_returns_section_scores(self):
        llm_scores = [
            {"section": "has_client2", "score": 1, "reasoning": "both false"},
            {"section": "client1_personal", "score": 1, "reasoning": "name matches"},
            {"section": "incomes[0]", "score": 0, "reasoning": "amount missing"},
        ]
        mock_client = AsyncMock()
        mock_client.responses.create.return_value = _mock_llm_response(llm_scores)

        gt = _make_cif(incomes=[_make_income()])
        ext = _make_cif(incomes=[])

        with patch("src.eval.scoring._get_prompts", return_value={
            "system": "You are a scorer.",
            "user": "Score this. Example: {example_id}\n{sections_json}",
        }):
            result = await score_extraction(mock_client, "easy_001", gt, ext)

        assert result.example_id == "easy_001"
        assert result.error is None
        assert len(result.section_scores) == 3
        assert result.section_scores[2].section == "incomes[0]"
        assert result.section_scores[2].score == 0.0

    async def test_overall_accuracy_computed_from_scores(self):
        llm_scores = [
            {"section": "client1_personal", "score": 1, "reasoning": "ok"},
            {"section": "client1_employment", "score": 1, "reasoning": "ok"},
            {"section": "incomes[0]", "score": 0, "reasoning": "wrong amount"},
        ]
        mock_client = AsyncMock()
        mock_client.responses.create.return_value = _mock_llm_response(llm_scores)
        gt = _make_cif(incomes=[_make_income()])
        ext = _make_cif(incomes=[_make_income(amount=50_000)])

        with patch("src.eval.scoring._get_prompts", return_value={
            "system": "sys", "user": "{example_id}{sections_json}",
        }):
            result = await score_extraction(mock_client, "easy_002", gt, ext)

        assert result.overall_accuracy == pytest.approx(2 / 3)

    async def test_llm_json_parse_error_returns_error_result(self):
        mock_client = AsyncMock()
        bad_response = MagicMock()
        bad_response.output_text = "not valid json {{{"
        mock_client.responses.create.return_value = bad_response

        gt = _make_cif()
        ext = _make_cif()

        with patch("src.eval.scoring._get_prompts", return_value={
            "system": "sys", "user": "{example_id}{sections_json}",
        }):
            result = await score_extraction(mock_client, "bad_001", gt, ext, max_retries=1)

        assert result.error is not None
        assert result.section_scores == []

    async def test_llm_called_once_on_success(self):
        mock_client = AsyncMock()
        mock_client.responses.create.return_value = _mock_llm_response([
            {"section": "has_client2", "score": 1, "reasoning": "ok"},
        ])

        gt = _make_cif()
        ext = _make_cif()

        with patch("src.eval.scoring._get_prompts", return_value={
            "system": "sys", "user": "{example_id}{sections_json}",
        }):
            await score_extraction(mock_client, "easy_003", gt, ext)

        assert mock_client.responses.create.call_count == 1

    async def test_temperature_forwarded_to_request(self):
        mock_client = AsyncMock()
        mock_client.responses.create.return_value = _mock_llm_response([
            {"section": "has_client2", "score": 1, "reasoning": "ok"},
        ])

        gt = _make_cif()
        ext = _make_cif()

        with patch("src.eval.scoring._get_prompts", return_value={
            "system": "sys", "user": "{example_id}{sections_json}",
        }):
            await score_extraction(mock_client, "easy_005", gt, ext, temperature=0.5)

        call_kwargs = mock_client.responses.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5

    async def test_retry_on_transient_error(self):
        mock_client = AsyncMock()
        good_response = _mock_llm_response([
            {"section": "has_client2", "score": 1, "reasoning": "ok"},
        ])
        mock_client.responses.create.side_effect = [
            RuntimeError("transient"),
            good_response,
        ]

        gt = _make_cif()
        ext = _make_cif()

        with patch("src.eval.scoring._get_prompts", return_value={
            "system": "sys", "user": "{example_id}{sections_json}",
        }), patch("asyncio.sleep", new_callable=AsyncMock):
            result = await score_extraction(mock_client, "easy_004", gt, ext, max_retries=3)

        assert result.error is None
        assert mock_client.responses.create.call_count == 2


# ---------------------------------------------------------------------------
# score_extractions_async (mocked LLM)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestScoreExtractionsAsync:
    async def test_returns_one_result_per_example(self):
        llm_scores = [{"section": "has_client2", "score": 1, "reasoning": "ok"}]
        mock_client = AsyncMock()
        mock_client.responses.create.return_value = _mock_llm_response(llm_scores)

        examples = [
            {"example_id": f"ex_{i:03d}", "gt": _make_cif(), "extracted": _make_cif()}
            for i in range(4)
        ]

        with patch("src.eval.scoring._get_prompts", return_value={
            "system": "sys", "user": "{example_id}{sections_json}",
        }):
            results = await score_extractions_async(
                mock_client, examples, max_concurrency=2, show_progress=False
            )

        assert len(results) == 4
        ids = {r.example_id for r in results}
        assert ids == {f"ex_{i:03d}" for i in range(4)}

    async def test_concurrency_respected(self):
        """Active requests should never exceed max_concurrency."""
        counters = {"active": 0, "peak": 0}

        async def fake_score(client, example_id, gt, extracted, **kwargs):
            counters["active"] += 1
            counters["peak"] = max(counters["peak"], counters["active"])
            await asyncio.sleep(0)
            counters["active"] -= 1
            return ExtractionScoreResult(
                example_id=example_id,
                section_scores=[SectionScore("has_client2", 1.0, "ok")],
            )

        mock_client = AsyncMock()
        examples = [
            {"example_id": f"ex_{i:03d}", "gt": _make_cif(), "extracted": _make_cif()}
            for i in range(8)
        ]

        with patch("src.eval.scoring.score_extraction", side_effect=fake_score):
            await score_extractions_async(
                mock_client, examples, max_concurrency=3, show_progress=False
            )

        assert counters["peak"] <= 3, (
            f"peak concurrency {counters['peak']} exceeded limit of 3"
        )

    async def test_error_in_one_does_not_stop_others(self):
        # With max_concurrency=1, examples run sequentially. ex_001 should exhaust
        # all 3 retries (calls 2, 3, 4) and return an error; ex_000 and ex_002 succeed.
        call_count = [0]

        async def fake_create(**kwargs):
            call_count[0] += 1
            if 2 <= call_count[0] <= 4:  # all 3 retry attempts of ex_001
                raise RuntimeError("boom")
            return _mock_llm_response([
                {"section": "has_client2", "score": 1, "reasoning": "ok"}
            ])

        mock_client = AsyncMock()
        mock_client.responses.create.side_effect = fake_create

        examples = [
            {"example_id": f"ex_{i:03d}", "gt": _make_cif(), "extracted": _make_cif()}
            for i in range(3)
        ]

        with patch("src.eval.scoring._get_prompts", return_value={
            "system": "sys", "user": "{example_id}{sections_json}",
        }), patch("asyncio.sleep", new_callable=AsyncMock):
            results = await score_extractions_async(
                mock_client, examples, max_concurrency=1,
                show_progress=False,
            )

        assert len(results) == 3
        errored = [r for r in results if r.error is not None]
        # the second example exhausted retries and returned an error result
        assert len(errored) == 1
