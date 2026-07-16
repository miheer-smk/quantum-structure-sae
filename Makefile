# Local CI-style checks. `make smoke` is the Phase-0 acceptance gate.
PY ?= .venv/bin/python

.PHONY: test test-fast lint smoke check

test:            ## full pytest suite (slow tests included, ~10 min)
	$(PY) -m pytest -q

test-fast:       ## fast subset only (excludes @pytest.mark.slow)
	$(PY) -m pytest -q -m "not slow"

lint:            ## ruff, same selection as CI
	$(PY) -m ruff check src/ tests/ scripts/ experiments/ --select=E,F,W --ignore=E501,W291,W293

smoke:           ## tiny end-to-end run: config -> data -> train -> SAE -> logs
	$(PY) experiments/smoke.py --config configs/smoke.yaml

check: lint test-fast smoke   ## everything a commit should pass locally
