# Dataset & Website TODO — SoccerTrack v2

Live checklist for the dataset itself, the docs landing page, and anything user-facing that isn't the paper. Tick items as they land. New items welcome — keep them concrete and one-line.

Paper tasks live in the private companion repo [`AtomScott/soccertrack-v2-paper`](https://github.com/AtomScott/soccertrack-v2-paper). Index: [`../TODO.md`](../TODO.md).

## Dataset release

- [ ] Confirm Hugging Face dataset at <https://huggingface.co/datasets/atomscott/soccertrack-v2> is live, with a populated dataset card linking back to the GitHub repo and the paper.
- [x] Decide and apply licences:
  - [x] **Data**: CC BY 4.0 — see [`LICENSE-DATA`](../LICENSE-DATA).
  - [x] **Code**: MIT — see [`LICENSE`](../LICENSE).
- [x] Add a top-level `LICENSE` file in the repo ([`LICENSE`](../LICENSE) + [`LICENSE-DATA`](../LICENSE-DATA)).
- [x] Provide a `download.sh` (or HF snapshot script) that fetches all matches — [`scripts/download.sh`](../scripts/download.sh) (wraps `hf` / `huggingface-cli`, supports `--match` filter + `--revision` pin).
- [ ] Publish per-match SHA256 checksums (`docs/checksums.txt` or in the dataset card).
- [ ] Document the canonical train/val/test split (or note "no canonical split"). *Note: baseline starter kits currently use `[91..97] / 98 / 99,100]` — promote this into prose once confirmed. Provisional split now committed in the paper's `02_dataset.tex`.*
- [x] Add a per-match metadata table skeleton ([`docs/matches.json`](matches.json)) — fields still to be filled in for each match.

## Annotation format docs

- [x] Spec the GSR JSON schema (one example + field reference) — [`docs/format-gsr.md`](format-gsr.md).
- [x] Spec the BAS JSON schema (one example + field reference) — [`docs/format-bas.md`](format-bas.md).
- [x] Document pitch coordinate system (origin, axes, units, pitch dimensions) — in [`format-gsr.md`](format-gsr.md).
- [x] Document time alignment (frame-rate, tolerance with BAS timestamps) — in [`format-gsr.md`](format-gsr.md) and [`format-bas.md`](format-bas.md).
- [ ] Add a tiny "demo subset" (1 short sequence) to the GitHub repo so users can validate parsers before downloading full matches.

## Loader, baselines, evaluation

- [x] Pure-Python dataset loader — [`src/data_utils/soccertrack_v2.py`](../src/data_utils/soccertrack_v2.py).
- [x] Quickstart notebook — [`notebooks/quickstart.ipynb`](../notebooks/quickstart.ipynb).
- [x] Baseline starter kits — [`baselines/{gsr,bas,mot}/`](../baselines/).
- [x] Evaluation CLIs — `python -m src.evaluation.{gs_hota,bas_map,mot_hota}`.
- [~] Hugging Face snapshot helper — [`src/data_utils/load_from_hf.py`](../src/data_utils/load_from_hf.py) (wraps `snapshot_download`; good enough for most users). A full `datasets.load_dataset(...)` loading script is still wanted.
- [ ] Fill in real GSR baseline numbers (train + eval + commit to [`docs/leaderboards/gsr.json`](leaderboards/gsr.json)).
- [ ] Fill in real BAS baseline numbers (commit to [`docs/leaderboards/bas.json`](leaderboards/bas.json)).
- [ ] Fill in a real MOT baseline (commit to [`docs/leaderboards/mot.json`](leaderboards/mot.json)).

## Landing page (`docs/index.html`, `docs/index-ja.html`)

- [ ] Confirm GitHub Pages is enabled on the **public** `AtomScott/SoccerTrack-v2` repo and that `https://atomscott.github.io/SoccerTrack-v2/` serves the current `docs/`.
- [x] Promote Hugging Face as primary, Google Drive as mirror; added `scripts/download.sh` snippet (EN + JA).
- [ ] Replace arXiv badge link with the published DOI badge once npj Artificial Intelligence accepts (arXiv v2 with the journal-version title submitted 2026-10-05).
- [ ] Confirm `assets/og-image.jpg` matches the v2 brand (currently a 117 KB JPG).
- [ ] Sanity-check JA translations against the latest EN copy (drift will accumulate as we update).
- [x] Add a "Cite" widget with copy-to-clipboard BibTeX (both EN + JA pages).
- [x] Add per-task leaderboard placeholders — [`docs/leaderboard.html`](leaderboard.html).
- [x] Cross-link developer markdown docs and issue templates from both EN + JA landing pages.

## Task pages (`docs/task-{gsr,bas,mot}.html`)

- [x] Verify each task page links back to `index.html` and to the relevant external challenge.
- [x] Add a "Download starter kit" link per task — point to `baselines/{gsr,bas,mot}/`. Also added a "Reference CLI" eval snippet on each.
- [x] Fix stale schema examples on `task-gsr.html` and `task-bas.html` (now match `format-{gsr,bas}.md`; file layout corrected to per-half GSR JSONs).
- [ ] Mirror task pages in Japanese (currently EN-only).

## Markdown docs (`docs/*.md`)

- [x] Update `docs/README.md` to list landing-page entry points — EN/JA index, task pages, leaderboard, format specs.
- [~] Accuracy audit of dev-docs against current code:
  - [x] `setup.md` rewritten to match uv + pyproject.toml workflow; removed references to non-existent `requirements.txt`, `.env.example`, `download_sample_data.py`, `verify_setup.py`.
  - [x] `cli.md`: replaced stale `create_ground_truth_fixed_bboxes.sh` reference with real `create_ground_truth.sh`; added `download.sh` entry.
  - [ ] `configuration.md`, `data_processing.md`, `ground_truth_creation.md`, `visualization.md`, `calibration.md` — still need a spot-check pass against `src/` and `scripts/`.
- [x] Cross-link the markdown docs from `index.html` and `index-ja.html` (dedicated "Developer Docs" / "開発者向けドキュメント" section).

## Issue / community handling

- [x] Add issue templates under `.github/ISSUE_TEMPLATE/` (bug, dataset issue, docs fix, feature request).
- [x] Add a `CONTRIBUTING.md` linking to the relevant TODO files.
- [x] Add a "report a problem with the dataset" channel — dedicated "Report a problem" cards (EN + JA) on the landing page deep-link to the `dataset_issue.yml` template with the `dataset` label pre-applied.

## Local dev / CI

- [x] `make serve-docs` works locally — keep the README pointing to it.
- [x] GitHub Actions job to lint `src/` with ruff on every push — [`.github/workflows/lint.yml`](../.github/workflows/lint.yml).
- [x] GitHub Actions job to link-check markdown docs — [`.github/workflows/docs.yml`](../.github/workflows/docs.yml).
- [ ] Optional: extend link-check to HTML pages once rate-limit-friendly allowlists are tuned.

## Suspected dataset issues flagged by agents

*Use this section to log suspected annotation / metadata problems without editing them. Maintainers batch-fix in a dataset re-release.*

- _(none yet)_
