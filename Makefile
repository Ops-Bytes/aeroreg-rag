.PHONY: install lint test data eval

install:
	pip install -e ".[dev]"

lint:
	ruff check .

test:
	pytest -q

# Download 14 CFR Parts 23, 25, 27, 29 from eCFR and build chunks
data:
	aeroreg ingest --parts 23 25 27 29

# Run retrieval evals against data/eval/golden_set.jsonl
eval:
	aeroreg eval --golden data/eval/golden_set.jsonl
