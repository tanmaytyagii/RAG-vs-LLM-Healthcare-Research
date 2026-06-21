# Comparison Report: RAG vs. LLM (Healthcare QA)

_Generated: 2026-06-21T20:05:36.994844+00:00_

Total evaluated records: **10** (LLM: 5, RAG: 5)

## Aggregated Metric Comparison

| Metric | LLM (Avg.) | RAG (Avg.) | Relative Gain (%) |
|---|---|---|---|
| Factual Accuracy | 0.726 | 0.870 | +19.8% |
| Faithfulness / Grounding | 0.000 | 0.844 | Δ +0.844 (baseline=0) |
| Hallucination Rate (↓) | 0.262 | 0.093 | +64.5% |
| Clinical Safety | 0.722 | 0.919 | +27.4% |
| Fluency / Coherence | 0.892 | 0.847 | -5.1% |
| Response Diversity | 0.744 | 0.792 | +6.5% |
| Latency (s, ↓ better) | 1.840 | 2.461 | -33.8% |
| Token Usage (↓ better) | 176.400 | 373.200 | -111.6% |

## Interpretation

Consistent with the source survey's findings (Table III), RAG is expected to show substantial gains in factual accuracy, faithfulness, and clinical safety, with a corresponding large reduction in hallucination rate, at the cost of increased latency and token usage from the added retrieval step. Fluency may be marginally lower for RAG due to the constraint of grounding generation in retrieved context rather than unconstrained decoding.

## Notes on Methodology

- Factual accuracy / faithfulness / hallucination / clinical safety are scored either via lexical-overlap and keyword heuristics, or via LLM-as-judge scoring (see `evaluation.llm_judge_score`) when `use_llm_judge=True`.
- Retrieval precision/recall are only defined for the RAG system and require ground-truth relevant-source labels in `data/healthcare_dataset.csv`.
- All scores are normalized to a [0, 1] scale, matching Section III-D of the source paper.