.PHONY: install test demo real
install:
	pip install -r requirements.txt && pip install -e .
test:
	pytest -q
demo:      # synthetic demonstration data (NOT eMERGE) -> results/
	python -m rwe_pgx_ml.pipeline --config config/config.yaml --out results --synthetic
real:      # your approved eMERGE-PGx extract, schema in data/README.md
	python -m rwe_pgx_ml.pipeline --config config/config.yaml --out results_real
