"""Command line interface: ingest, search, eval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aeroreg.chunking import chunk_sections, load_chunks, save_chunks
from aeroreg.evaluation import load_golden, run_eval
from aeroreg.ingest import download_part, parse_sections
from aeroreg.retriever import BM25Retriever

RAW_DIR = Path("data/raw")
CHUNKS_PATH = Path("data/processed/chunks.jsonl")
REPORT_DIR = Path("reports")


def cmd_ingest(args: argparse.Namespace) -> int:
    sections = []
    for part in args.parts:
        path = RAW_DIR / f"title14_part{part}.xml"
        if args.refresh or not path.exists():
            print(f"Downloading 14 CFR Part {part} ...")
            path = download_part(part, RAW_DIR)
        parsed = parse_sections(path.read_bytes())
        print(f"  Part {part}: {len(parsed)} sections")
        sections.extend(parsed)

    chunks = chunk_sections(sections, max_words=args.max_words)
    save_chunks(chunks, CHUNKS_PATH)
    print(f"Wrote {len(chunks)} chunks from {len(sections)} sections to {CHUNKS_PATH}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    retriever = BM25Retriever(load_chunks(CHUNKS_PATH))
    for rank, hit in enumerate(retriever.search(args.query, k=args.k), start=1):
        c = hit.chunk
        print(f"{rank:>2}. {hit.score:6.2f}  {c.citation}  {c.heading}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    retriever = BM25Retriever(load_chunks(CHUNKS_PATH))
    golden = load_golden(Path(args.golden))
    report = run_eval(golden, lambda q, k: retriever.search(q, k=k))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"{args.name}.json"
    out.write_text(
        json.dumps(
            {"n_questions": report.n_questions, "metrics": report.metrics,
             "failures": report.failures},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(report.to_markdown())
    print(f"\nFailures (no relevant section in top 10): {len(report.failures)}")
    print(f"Full report: {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aeroreg", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Download 14 CFR parts and build chunks")
    p_ingest.add_argument("--parts", nargs="+", default=["23", "25", "27", "29"])
    p_ingest.add_argument("--max-words", type=int, default=220)
    p_ingest.add_argument("--refresh", action="store_true", help="Re-download even if cached")
    p_ingest.set_defaults(func=cmd_ingest)

    p_search = sub.add_parser("search", help="Search the chunk index")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=5)
    p_search.set_defaults(func=cmd_search)

    p_eval = sub.add_parser("eval", help="Evaluate retrieval against a golden set")
    p_eval.add_argument("--golden", default="data/eval/golden_set.jsonl")
    p_eval.add_argument("--name", default="bm25_baseline", help="Report file name (no extension)")
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
