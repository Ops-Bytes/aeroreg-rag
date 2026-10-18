"""A small, dependency-free BM25 retriever. This is the baseline every later method must beat."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from aeroreg.chunking import Chunk

# Keeps section numbers such as "25.1309" as a single token.
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)*")

_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was "
    "were will with shall must may".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b

        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._doc_len: list[int] = []
        for idx, chunk in enumerate(chunks):
            terms = Counter(tokenize(chunk.search_text))
            self._doc_len.append(sum(terms.values()))
            for term, tf in terms.items():
                self._postings[term].append((idx, tf))

        self._n = len(chunks)
        self._avgdl = (sum(self._doc_len) / self._n) if self._n else 0.0

    def _idf(self, term: str) -> float:
        df = len(self._postings.get(term, ()))
        return math.log(1 + (self._n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 10) -> list[Hit]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf(term)
            for idx, tf in postings:
                norm = 1 - self.b + self.b * self._doc_len[idx] / self._avgdl
                scores[idx] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [Hit(chunk=self.chunks[idx], score=score) for idx, score in ranked]
