import json

import pytest

from aeroreg.chunking import Chunk
from aeroreg.evaluation import (
    GoldenItem,
    dedupe_preserving_order,
    hit_at_k,
    load_golden,
    recall_at_k,
    reciprocal_rank,
    run_eval,
)
from aeroreg.retriever import BM25Retriever, tokenize


def make_chunk(section: str, heading: str, text: str, idx: int = 0) -> Chunk:
    return Chunk(
        chunk_id=f"{section}#{idx}",
        section=section,
        part=section.split(".")[0],
        heading=heading,
        subpart="",
        citation=f"14 CFR § {section}",
        text=text,
    )


CORPUS = [
    make_chunk("25.1309", "Equipment, systems, and installations",
               "Catastrophic failure conditions must be extremely improbable."),
    make_chunk("25.571", "Damage-tolerance and fatigue evaluation of structure",
               "Fatigue and damage tolerance of airplane structure must be evaluated."),
    make_chunk("25.1353", "Electrical equipment and installations",
               "Electrical equipment must be installed so it does not cause hazards."),
]


def test_tokenize_keeps_section_numbers_and_drops_stopwords():
    assert tokenize("The failure of § 25.1309 shall be rare") == ["failure", "25.1309", "rare"]


def test_bm25_ranks_most_relevant_chunk_first():
    retriever = BM25Retriever(CORPUS)
    hits = retriever.search("fatigue damage tolerance structure", k=3)

    assert hits[0].chunk.section == "25.571"
    assert hits[0].score > 0


def test_bm25_finds_chunk_by_section_number():
    retriever = BM25Retriever(CORPUS)
    assert retriever.search("25.1353", k=1)[0].chunk.section == "25.1353"


def test_bm25_returns_nothing_for_unknown_terms():
    assert BM25Retriever(CORPUS).search("zzzz qqqq", k=5) == []


def test_bm25_handles_empty_corpus():
    assert BM25Retriever([]).search("anything") == []


def test_metrics():
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, {"b", "z"}, 2) == 0.5
    assert recall_at_k(ranked, {"b", "z"}, 1) == 0.0
    assert hit_at_k(ranked, {"c"}, 3) == 1.0
    assert hit_at_k(ranked, {"c"}, 2) == 0.0
    assert reciprocal_rank(ranked, {"c", "d"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, {"zzz"}) == 0.0


def test_recall_rejects_empty_relevant_set():
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), 1)


def test_dedupe_preserves_first_appearance_order():
    assert dedupe_preserving_order(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]


def test_load_golden_validates_rows(tmp_path):
    good = tmp_path / "good.jsonl"
    good.write_text(
        json.dumps({"id": "q1", "question": "Q?", "relevant_sections": ["25.571"]}) + "\n",
        encoding="utf-8",
    )
    assert load_golden(good) == [GoldenItem("q1", "Q?", frozenset({"25.571"}))]

    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps({"id": "q1", "question": "Q?", "relevant_sections": []}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_golden(bad)


def test_run_eval_end_to_end_reports_metrics_and_failures():
    retriever = BM25Retriever(CORPUS)
    golden = [
        GoldenItem("q1", "fatigue damage tolerance of structure", frozenset({"25.571"})),
        GoldenItem("q2", "catastrophic failure conditions improbable", frozenset({"25.1309"})),
        GoldenItem("q3", "completely unrelated gibberish zzzz", frozenset({"25.1353"})),
    ]

    report = run_eval(golden, lambda q, k: retriever.search(q, k=k))

    assert report.n_questions == 3
    assert report.metrics["hit@1"] == pytest.approx(2 / 3)
    assert report.metrics["mrr"] == pytest.approx(2 / 3)
    assert [f["id"] for f in report.failures] == ["q3"]
    assert "| mrr |" in report.to_markdown()


def test_run_eval_rejects_empty_golden_set():
    with pytest.raises(ValueError):
        run_eval([], lambda q, k: [])
