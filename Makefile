SHELL := /bin/bash

DOCS_PORT ?= 8000

.PHONY: help format serve-docs

help:
	@echo "Targets:"
	@echo "  make format        - run ruff (autofix) on src/"
	@echo "  make serve-docs    - serve docs/ at http://localhost:$(DOCS_PORT)"

format:
	uv run ruff check src --fix

serve-docs:
	@echo "Serving docs/ at http://localhost:$(DOCS_PORT)"
	@cd docs && python3 -m http.server $(DOCS_PORT)

# Prove the GS-HOTA plumbing: score ground truth against itself, which must give 1.0.
# Pitch space only -- SoccerTrack v2 has no ground-truth detections.
gs-hota-test:
	.venv/bin/python tests/test_gs_hota_identity.py

# Prove the BAS plumbing: score ground truth against itself, which must give mAP 1.0.
bas-map-test:
	.venv/bin/python tests/test_bas_map_identity.py
