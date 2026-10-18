# aeroreg

Retrieval and evaluation toolkit for aviation regulations (14 CFR Parts 23, 25, 27, 29), built **eval-first**: every retrieval method has to beat a measured baseline on a hand-verified golden set before it earns a place in the pipeline.

> **Status:** Phase 1, step 1. Ingestion, chunking, a BM25 baseline and the evaluation harness are implemented and unit-tested. Hybrid retrieval, reranking and cited LLM answers are next. No benchmark numbers are published yet because the golden set has not been written.

## Problem

Certification engineers spend a lot of time answering "which requirements apply to this design, and where is that written?". A useful assistant must point to the exact section, not paraphrase from memory. That makes retrieval quality, and the ability to *measure* it, the core problem.

## Design decisions

- **Section-level relevance.** Ground truth is "which 14 CFR section answers this question", because that is how engineers cite regulations. Chunks are collapsed to sections before computing metrics.
- **Citation metadata on every chunk.** Each chunk carries part, subpart, section number and heading, and its indexed text starts with a contextual header (`14 CFR § 25.1309 Equipment, systems, and installations.`).
- **Paragraph-preserving chunking.** Sub-paragraph markers like `(a)(1)` stay attached to their text.
- **Dependency-free BM25 baseline.** Small enough to read in one sitting, and the thing every later method is compared against.
- **Offline tests.** The test suite uses a small XML fixture, so CI never touches the network.

## Quickstart

```bash
pip install -e ".[dev]"
make test

make data                      # downloads Parts 23/25/27/29 from the public eCFR API
aeroreg search "fatigue and damage tolerance of structure"
```

## Evaluating

1. Copy `data/eval/golden_set.example.jsonl` to `data/eval/golden_set.jsonl`.
2. Write about 30 questions, each with the section(s) that actually answer it. Verify every answer against the regulation text yourself. Do not generate ground truth with an LLM.
3. Run:

```bash
make eval
```

Reports are written to `reports/<name>.json` with recall@k, hit@k, MRR and the list of failed questions.

Golden set format (one JSON object per line):

```json
{"id": "q001", "question": "...", "relevant_sections": ["25.1309"]}
```

## Layout

```
src/aeroreg/
  ingest.py      eCFR download and XML parsing into sections
  chunking.py    paragraph-preserving chunks with citation metadata
  retriever.py   BM25 baseline
  evaluation.py  metrics, golden set loading, eval runner
  cli.py         aeroreg ingest | search | eval
tests/           offline unit tests with an XML fixture
data/eval/       golden set (you write this)
```

## Roadmap

- [ ] Golden set of ~30 verified questions
- [ ] Published BM25 baseline numbers
- [ ] Dense embeddings and hybrid retrieval (BM25 + vectors), compared on the same golden set
- [ ] Reranking
- [ ] Cited answer generation with an LLM, plus faithfulness and hallucination-rate evals
- [ ] FastAPI service, Docker image, deployed demo with cost and latency numbers
- [ ] Write-up: what failed and why

## Data

Regulation text comes from the [eCFR API](https://www.ecfr.gov/developer/documentation/api/v1), a public U.S. government source. This project is not legal or certification advice.
