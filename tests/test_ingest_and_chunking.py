from pathlib import Path

from aeroreg.chunking import Chunk, chunk_section, chunk_sections, load_chunks, save_chunks
from aeroreg.ingest import Section, parse_sections

FIXTURE = Path(__file__).parent / "fixtures" / "sample_part.xml"


def test_parse_sections_extracts_metadata_and_skips_reserved():
    sections = parse_sections(FIXTURE.read_bytes())

    assert [s.number for s in sections] == ["25.101", "25.1309"]

    s = sections[1]
    assert s.part == "25"
    assert s.heading == "Equipment, systems, and installations"
    assert s.subpart == "Subpart F—Equipment"
    assert s.citation == "14 CFR § 25.1309"
    assert "catastrophic failure condition" in s.text


def test_chunk_section_keeps_short_section_whole():
    section = parse_sections(FIXTURE.read_bytes())[0]
    chunks = chunk_section(section, max_words=220)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "25.101#0"
    assert chunks[0].text.startswith("(a) Each performance requirement")


def test_chunk_section_splits_on_paragraph_boundaries():
    section = Section("25", "25.1", "Test", "Subpart A", "\n".join(["word " * 60] * 4))
    chunks = chunk_section(section, max_words=130)

    assert len(chunks) == 2
    assert all(len(c.text.split()) <= 130 for c in chunks)
    assert [c.chunk_id for c in chunks] == ["25.1#0", "25.1#1"]


def test_chunk_section_splits_oversized_paragraph_by_words():
    section = Section("25", "25.2", "Test", "Subpart A", "word " * 500)
    chunks = chunk_section(section, max_words=200)

    assert len(chunks) == 3
    assert sum(len(c.text.split()) for c in chunks) == 500


def test_search_text_includes_citation_and_heading():
    chunk = chunk_sections(parse_sections(FIXTURE.read_bytes()))[-1]
    assert chunk.search_text.startswith("14 CFR § 25.1309 Equipment, systems, and installations.")


def test_save_and_load_roundtrip(tmp_path):
    chunks = chunk_sections(parse_sections(FIXTURE.read_bytes()))
    path = tmp_path / "out" / "chunks.jsonl"
    save_chunks(chunks, path)

    loaded = load_chunks(path)
    assert loaded == chunks
    assert all(isinstance(c, Chunk) for c in loaded)
