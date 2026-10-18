"""Retrieval evaluation: metrics, golden-set loading and a reproducible runner.

Relevance is judged at the *section* level: a retrieved chunk counts as a hit if it
comes from a section the golden set marks as relevant. Chunks are collapsed to
sections (keeping first-appearance rank) before metrics are computed.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from aeroreg.retriever import Hit

KS = (1, 3, 5, 10)


@dataclass(frozen=True)
class GoldenItem:
    id: str
    question: str
    relevant_sections: frozenset[str]


def dedupe_preserving_order(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def recall_at_k(ranked: Sequence[str], relevant: set[str] | frozenset[str], k: int) -> float:
    if not relevant:
        raise ValueError("relevant set must not be empty")
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def hit_at_k(ranked: Sequence[str], relevant: set[str] | frozenset[str], k: int) -> float:
    return 1.0 if set(ranked[:k]) & set(relevant) else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: set[str] | frozenset[str]) -> float:
    for rank, section in enumerate(ranked, start=1):
        if section in relevant:
            return 1.0 / rank
    return 0.0


def load_golden(path: Path) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            relevant = frozenset(raw["relevant_sections"])
            if not relevant:
                raise ValueError(f"{path}:{line_no}: relevant_sections must not be empty")
            items.append(GoldenItem(raw["id"], raw["question"], relevant))
    return items


@dataclass
class EvalReport:
    n_questions: int
    metrics: dict[str, float]
    failures: list[dict] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [f"Questions: {self.n_questions}", "", "| Metric | Value |", "|---|---|"]
        lines += [f"| {name} | {value:.3f} |" for name, value in self.metrics.items()]
        return "\n".join(lines)


def run_eval(
    golden: Sequence[GoldenItem],
    search: Callable[[str, int], list[Hit]],
    ks: Sequence[int] = KS,
    pool: int = 50,
) -> EvalReport:
    """Run every golden question through `search` and aggregate section-level metrics."""
    if not golden:
        raise ValueError("golden set is empty")

    totals: dict[str, float] = {f"recall@{k}": 0.0 for k in ks}
    totals.update({f"hit@{k}": 0.0 for k in ks})
    totals["mrr"] = 0.0
    failures: list[dict] = []

    for item in golden:
        hits = search(item.question, pool)
        ranked = dedupe_preserving_order([h.chunk.section for h in hits])
        for k in ks:
            totals[f"recall@{k}"] += recall_at_k(ranked, item.relevant_sections, k)
            totals[f"hit@{k}"] += hit_at_k(ranked, item.relevant_sections, k)
        rr = reciprocal_rank(ranked, item.relevant_sections)
        totals["mrr"] += rr
        if hit_at_k(ranked, item.relevant_sections, max(ks)) == 0.0:
            failures.append(
                {
                    "id": item.id,
                    "question": item.question,
                    "expected": sorted(item.relevant_sections),
                    "top_retrieved": ranked[:5],
                }
            )

    n = len(golden)
    return EvalReport(
        n_questions=n,
        metrics={name: total / n for name, total in totals.items()},
        failures=failures,
    )
