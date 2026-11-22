PY ?= python

.PHONY: install data train evaluate monitor serve benchmark test all

install:   ## create the environment
	$(PY) -m pip install -r requirements.txt
data:      ## download + leak-free split + leakage audit
	$(PY) -m src.data_pipeline
train:     ## baselines + LightGBM, export artifacts
	$(PY) -m src.train
evaluate:  ## one-time test-set evaluation + figures
	$(PY) -m src.evaluate
monitor:   ## drift report: healthy data, then two synthetic stress scenarios
	$(PY) -m src.monitoring --input data/test.csv
	$(PY) -m src.monitoring --input data/test.csv --simulate-drift moderate
	$(PY) -m src.monitoring --input data/test.csv --simulate-drift severe
serve:     ## REST API on http://localhost:8000 (docs at /docs)
	$(PY) -m uvicorn serving.app:app --host 0.0.0.0 --port 8000
benchmark: ## latency percentiles, throughput, robustness probes
	$(PY) -m serving.benchmark
test:
	$(PY) -m pytest -q
all: data train evaluate monitor benchmark test
