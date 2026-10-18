"""Split sections into retrievable chunks that always carry their citation metadata."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from aeroreg.ingest import Section


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    section: str  # "25.1309"
    part: str
    heading: str
    subpart: str
    citation: str
    text: str

    @property
    def search_text(self) -> str:
        """Text used for indexing: a contextual header plus the chunk body."""
        return f"{self.citation} {self.heading}. {self.text}"


def _split_long_paragraph(paragraph: str, max_words: int) -> list[str]:
    words = paragraph.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def chunk_section(section: Section, max_words: int = 220) -> list[Chunk]:
    """Pack whole paragraphs into chunks of at most ~max_words words.

    Paragraph boundaries are preserved so sub-paragraph markers like (a)(1) stay
    attached to their text. A single paragraph longer than max_words is split by words.
    """
    pieces: list[str] = []
    for paragraph in section.text.split("\n"):
        if len(paragraph.split()) > max_words:
            pieces.extend(_split_long_paragraph(paragraph, max_words))
        else:
            pieces.append(paragraph)

    groups: list[list[str]] = []
    current: list[str] = []
    current_words = 0
    for piece in pieces:
        n = len(piece.split())
        if current and current_words + n > max_words:
            groups.append(current)
            current, current_words = [], 0
        current.append(piece)
        current_words += n
    if current:
        groups.append(current)

    return [
        Chunk(
            chunk_id=f"{section.number}#{i}",
            section=section.number,
            part=section.part,
            heading=section.heading,
            subpart=section.subpart,
            citation=section.citation,
            text="\n".join(group),
        )
        for i, group in enumerate(groups)
    ]


def chunk_sections(sections: list[Section], max_words: int = 220) -> list[Chunk]:
    return [c for s in sections for c in chunk_section(s, max_words=max_words)]


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]
