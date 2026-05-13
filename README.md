# CIF Extraction Pipeline — Nevis AI Engineer Task

End-to-end system for extracting structured Customer Information Form (CIF) data from financial planning transcripts using LLMs, with a fully synthetic evaluation dataset and LLM-as-judge scoring.

---

## Overview

Financial advisors conduct fact-find calls that produce unstructured transcripts. This project builds a pipeline that reads those transcripts and outputs a structured CIF — a normalised JSON document covering 12 sections of client information — and rigorously evaluates extraction quality against ground-truth labels.

---

## Repository structure

```
eval.ipynb                  — dataset generation, schema design, and audit
solution.ipynb              — extraction pipeline and evaluation
src/
  eval/
    models.py               — Pydantic schema for CIF, all models, enums
    scoring.py              — async LLM-as-judge scorer
    config.py               — model configs, dataset plan, section coverage targets
    scenarios.py            — archetype and difficulty definitions
    ground_truth.py         — ground-truth generation (LLM)
    transcripts.py          — synthetic transcript generation (LLM)
    evaluation.py           — transcript validation against ground truth
    pipeline.py             — end-to-end staged generation
    audit.py                — quality checks and conflict detection
    utils.py                — shared helpers (prompt loading, data IO)
  extraction/
    utils.py                — pipeline utilities (caching, metrics, scoring helpers)
  prompts/
    eval/                   — ground_truth / transcript / validation prompt YAMLs
    extraction/
      v1/                   — single-shot extraction prompts
      v2/                   — 12 per-section extraction prompts
      validate/             — LLM validation stage prompts
tests/
  test_scoring.py           — 25 unit tests for scoring module
```

---

## Evaluation dataset

A fully synthetic dataset of **68 validated fact-find transcripts** is used for evaluation. 80 cases were generated in total; 12 were excluded after failing the transcript validation stage (coverage gaps or transcript defects that could not be resolved through regeneration). All metrics below are computed on the 68 passing cases.

Generation follows a three-stage pipeline:

1. **Ground truth** — an LLM generates a structured `CIFExtraction` label for a given archetype and difficulty spec.
2. **Transcript** — a second LLM call turns the label into a realistic advisor–client dialogue that faithfully covers the labelled fields and withholds everything else.
3. **Validation** — a third LLM call checks the transcript against the label and flags coverage gaps; failures are regenerated until they pass.

### Dataset composition

| Difficulty | Count |
|---|---|
| Easy | 19 |
| Medium | 27 |
| Hard | 22 |

| Archetype | Count |
|---|---|
| pre\_retirement\_couple | 12 |
| inheritance\_windfall | 10 |
| dual\_income\_mortgage\_household | 9 |
| high\_debt\_low\_savings | 8 |
| retired\_widowed\_client | 10 |
| messy\_corrections\_privacy | 8 |
| young\_high\_earner\_family | 6 |
| self\_employed\_single | 5 |

Challenge tags injected across the dataset include: `numeric_exact`, `numeric_approximate`, `numeric_range`, `owner_attribution`, `client2_present`, `joint_assets`, `negation`, `correction`, `missing_fields`, `objectives_free_form`, `risk_preferences`, `estate_planning`, `privacy_reference`, `advisor_noise`.

---

## Extraction pipeline

Two extraction strategies were implemented and compared.

### V1 — Single-shot extraction

One LLM call per transcript. The full `CIFExtraction` JSON Schema is embedded in the system prompt; the model returns the complete object in a single response constrained by `json_object` mode, then validated by Pydantic.

### V2 — Per-section extraction

12 parallel LLM calls per transcript, one per CIF section. Each call receives a focused, section-specific system prompt that includes only the relevant JSON Schema fragment. Results are merged into a single `CIFExtraction`. Per-section prompts encode domain rules — for example, that a monthly payment alone is not sufficient evidence for a loan record, or that annuities belong in pensions rather than other assets.

### Validation stage

An optional second-pass LLM call (using a different model) reviews the extraction against the original transcript and removes any items or fields not directly supported by what was said. This is applied independently to V1 and V2 outputs, giving four variants in total.

**Models used:**
- Extraction: `gpt-5.1`
- Scoring / validation: `gpt-5.4-mini`
- All runs at temperature 0.

---

## Evaluation method

Scoring uses an **LLM-as-judge** approach: for each extracted CIF, the scorer receives ground-truth and extracted values side by side and assigns a binary score (0 or 1) per section or list item. Metrics are computed from these scores:

- **Accuracy** — fraction of scored sections that match ground truth (TP + TN / all).
- **Precision** — TP / (TP + FP); how often extracted items are correct.
- **Recall** — TP / (TP + FN); how much of the ground truth was captured.
- **F1** — harmonic mean of precision and recall.
- **Hallucination rate** — FP / (FP + TN); how often the model adds items that have no ground-truth counterpart.

---

## Results

| Metric | V1 | V1+val | V2 | V2+val |
|---|---|---|---|---|
| Accuracy | 90.0% | 91.3% | 89.5% | **91.6%** |
| Precision | 96.1% | **97.2%** | 94.8% | 96.3% |
| Recall | 92.8% | 93.4% | 93.5% | **94.6%** |
| F1 | 94.4% | 95.3% | 94.1% | **95.5%** |
| Hallucination rate | 44.8% | **36.4%** | 49.4% | 44.2% |

**V2+val** is the best overall variant by accuracy, recall, and F1. **V1+val** has the lowest hallucination rate and highest precision, making it preferable when false positives are more costly than missed fields.

### Per-section breakdown (V1+val vs V2+val)

| Section | V1+val | V2+val | Winner |
|---|---|---|---|
| household | 67.6% | **88.2%** | V2 +21 pp |
| loans\_and\_mortgages | 87.1% | **92.7%** | V2 +6 pp |
| incomes | 94.4% | **98.5%** | V2 +4 pp |
| risk\_profile\_and\_preferences | 88.7% | **93.1%** | V2 +4 pp |
| objectives | 88.3% | **90.4%** | V2 +2 pp |
| client2\_employment | **88.5%** | 82.7% | V1 +6 pp |
| expenses | **96.5%** | 92.7% | V1 +4 pp |
| client1\_personal | **97.1%** | 94.1% | V1 +3 pp |
| other\_assets | **76.9%** | 48.1% | V1 +29 pp |

The `household` section is the clearest V2 win — isolated prompting eliminates confusion with other sections. `other_assets` is V2's largest failure: section-isolated calls lose cross-section context and misroute items (annuities, inheritance, brokerage accounts) that have already been captured elsewhere.

---

## Directions for improvement

### Prompt engineering
- **`other_assets` in V2**: the isolated prompt still hallucinates or misroutes items captured in other sections. Injecting a summary of what V2 has already extracted into the `other_assets` call (sequential rather than fully parallel) would give the model necessary context.
- **`client2_employment`**: V1 handles this better because the full-transcript context makes it easier to identify which speaker is the second client. V2 could include the output of the `personal` section call as additional context before running `employment`.
- **Tighter list section rules**: hallucination rates remain high in absolute terms (36–49%), driven mainly by `client2_employment` (100% hallucination on single-client cases), `loans_and_mortgages` (~93%), and `objectives` (~92%). Confidence-gating — requiring at least one explicit quote before accepting a list item — could reduce these.

### Architecture
- **Hybrid V1+V2**: use V1 for sections where it outperforms (personal, expenses, employment) and V2 for sections where focused prompting wins (household, incomes, loans). A routing layer could select strategy per section based on transcript length and section coverage estimate.
- **Cross-section awareness in V2**: pass a lightweight "already extracted" summary as context to each subsequent section call to prevent double-counting and misrouting.
- **Stronger validator**: the current validation model (`gpt-5.4-mini`) is the same model used for scoring. Using a larger or differently-instructed model as the validator may catch more subtle hallucinations.
- **Retrieval-augmented extraction**: for long transcripts (the real Nevis transcript is ~24k tokens), split the transcript into turns and retrieve the most relevant passages per section before extraction, reducing irrelevant context.

### Evaluation
- **Ground-truth coverage**: 12 of the 80 generated cases were discarded due to transcript defects. A more robust generation loop (or human review of edge cases) would increase the effective dataset size.
- **Real-data validation**: all evaluation is on synthetic transcripts. Measuring agreement between model extractions and human-annotated labels on real calls would validate that synthetic performance transfers.
- **Confidence scoring**: the current scorer returns binary 0/1 per section. A partial-credit scorer (e.g. field-level rather than section-level) would give finer signal and distinguish "mostly correct" from "completely wrong" extractions.
