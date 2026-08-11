SHELL := /bin/bash

DOCS_PORT ?= 8000

DATA ?= /data/share/SoccerTrack-v2/data
PYTHON ?= .venv/bin/python
CALIB_OUT ?= outputs/calibration_all

.PHONY: help format serve-docs calibration calibration-repro calibration-sweep calibrated-videos tracking-projection

help:
	@echo "Targets:"
	@echo "  make format             - run ruff (autofix) on src/"
	@echo "  make serve-docs         - serve docs/ at http://localhost:$(DOCS_PORT)"
	@echo "  make calibration        - calibrate all 10 matches from pitch keypoints + validate"
	@echo "  make calibration-repro  - re-run into a temp dir and assert the numbers are unchanged"
	@echo "  make calibration-sweep  - the diagnostic strategy sweep (see docs/calibration-findings.md)"
	@echo "  make calibrated-videos  - render calibrated per-half panoramas (FORCE=1 to re-render)"
	@echo "  make tracking-projection- project BePro tracking onto frames to validate the chain"
	@echo ""
	@echo "Vars: DATA=$(DATA)  PYTHON=$(PYTHON)  CALIB_OUT=$(CALIB_OUT)"

# Fit cv2.fisheye from each match's 65 pitch keypoints and render every result with the
# undistorted keypoints overlaid, so correctness is judged by eye. Deterministic: the same
# inputs give byte-identical maps. See docs/calibration-findings.md.
calibration:
	$(PYTHON) scripts/calibration/calibrate_all_from_keypoints.py --data $(DATA) --out $(CALIB_OUT)
	@echo ""
	@echo "Now open $(CALIB_OUT)/index.html"

# Guards against silent drift: recalibrates into a scratch dir and diffs the summary.
calibration-repro:
	@rm -rf /tmp/soccertrack-calib-repro
	@$(PYTHON) scripts/calibration/calibrate_all_from_keypoints.py --data $(DATA) \
		--out /tmp/soccertrack-calib-repro >/dev/null
	@if diff -q <(cut -f1-4 $(CALIB_OUT)/summary.tsv) \
	            <(cut -f1-4 /tmp/soccertrack-calib-repro/summary.tsv) >/dev/null; then \
		echo "REPRODUCIBLE: match/status/rms/straightness identical to $(CALIB_OUT)"; \
	else \
		echo "DRIFT DETECTED:"; \
		diff <(cut -f1-4 $(CALIB_OUT)/summary.tsv) <(cut -f1-4 /tmp/soccertrack-calib-repro/summary.tsv); \
		exit 1; \
	fi

calibration-sweep:
	./scripts/calibration/test_calibration_strategies.sh

# Render the calibrated per-half panoramas the GSR baseline consumes. Resumable: existing
# outputs are skipped unless FORCE=1. Roughly 10-15 min per half on the local GPU.
calibrated-videos:
	$(PYTHON) scripts/calibration/render_calibrated_videos.py --data $(DATA) $(if $(FORCE),--force,)

# Project BePro tracking positions onto each frame to validate the coordinate chain.
tracking-projection:
	$(PYTHON) scripts/calibration/project_tracking_to_image.py --data $(DATA)
	@echo ""
	@echo "Now open outputs/tracking_projection/index.html"

format:
	uv run ruff check src --fix

serve-docs:
	@echo "Serving docs/ at http://localhost:$(DOCS_PORT)"
	@cd docs && python3 -m http.server $(DOCS_PORT)
