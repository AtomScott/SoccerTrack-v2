# BAS tooling

Ball action spotting for SoccerTrack v2: the data audit that had to come first, and
Experiment B of [`docs/experiment-design-bas.md`](../../docs/experiment-design-bas.md) —
spotting ball events from **player trajectories alone**, no pixels and no ball.

Findings and measured numbers: [`docs/bas-findings.md`](../../docs/bas-findings.md).

## Order of operations

```bash
# 1. Stream the released GSR GameState files into compact per-half position caches.
#    They are 2.7 GB each (54 GB total) and json.loads peaks around 25 GB, so the
#    obvious reader cannot build a training set. ~2 min with 5 workers.
python scripts/bas/extract_tracks.py --all --workers 5

# 2. Check the extractor against an INDEPENDENT source (the interim pitch-plane CSVs).
#    Pins the field extraction, the frame convention and the metre scaling at once.
python scripts/bas/extract_tracks.py --validate

# 3. Audit the annotations and write the period table everything else depends on.
python scripts/bas/audit_annotations.py --write-periods --check-alignment --paper-stats

# 4. Pair features with event targets, one .npz per half. ~4 min with 6 workers.
python scripts/bas/build_dataset.py --workers 6

# 5. Train, tune the decoder on validation, write predictions.
python scripts/bas/train_trajectory.py --epochs 24 --hidden 128 --dropout 0.2

# 6. Chance floors, without which a mAP figure cannot be read.
python scripts/bas/baseline_priors.py --kind uniform --out outputs/bas/pred_uniform

# 7. Score, and emit the paper's two tables.
python scripts/bas/make_tables.py --pred outputs/bas/pred_test \
    --baseline outputs/bas/pred_uniform --matches 128057 132831
```

Sweeps, both validation-only:

```bash
EPOCHS=24 bash scripts/bas/sweep.sh                  # capacity / regularisation
HIDDEN=128 DROPOUT=0.2 bash scripts/bas/ablate.sh    # which feature groups matter
```

## What each script is for

| script | does |
|---|---|
| `extract_tracks.py` | streams GSR GameState → per-half `.npz` of pitch positions. `--validate` checks against the interim CSVs. **Never reads `bbox_image`** — those boxes are auto-generated and are not ground truth. |
| `audit_annotations.py` | resolves the period structure, writes `configs/bas_periods.json`, checks event↔track alignment against a randomised null, and prints the dataset statistics the paper asks for. |
| `build_dataset.py` | features + soft per-class targets per half. Drops third-period events and says how many. |
| `train_trajectory.py` | trains the TCN, selects on **validation mAP** (not loss), grid-searches the decoder on validation, writes SoccerNet-schema predictions with per-event `score`. |
| `baseline_priors.py` | `uniform` / `random` chance floors from training-split rates, plus a deliberately illegitimate `oracle-rate` variant for context. |
| `make_tables.py` | scores model and chance, prints a per-class model-minus-chance table, emits `tab:bas_results` and `tab:bas_per_class` as `.tex`. |
| `sweep.sh`, `ablate.sh` | validation-only searches over capacity/regularisation and over feature groups. |

## Three things that will bite anyone who skips the audit

1. **Three matches have a third 45-minute period with no video and no GSR file.** 2,231
   events (9.4%) have no input data. Scoring a perfect period-1-and-2 prediction against
   the released files gives mAP@1s **0.8409 instead of 1.0000**, all of the loss on test
   match 132831. `src/data_utils/bas_periods.py` is the one place that decides this;
   `tests/test_bas_map_periods.py` fails if the filter is removed.

2. **`position` is absolute match time, and periods overlap on that clock.** A first half
   can run to 48:29 while the second half starts at 45:00, so `position // 45min` is wrong.
   The `gameTime` prefix is also wrong on 103 events. An event's period is decided by whether
   its frame lands inside that period's annotated GSR range.

3. **Pitch coordinates are quantised to 1.05 m.** A one-frame velocity is 26 m/s of pure
   quantisation noise — the median measured "speed" over 0.08 s is exactly 0.00 m/s while
   the 95th percentile is 13.13 m/s. Velocity is taken at 0.48 s and 1.52 s. The same
   quantisation made an `argmin`-based ball proxy tie on 6.7% of frames; it is now a soft
   minimum.

## The test split is scored once

`128057` and `132831` are the SoccerTrack Challenge 2025 test matches. Every choice —
capacity, dropout, learning rate, decoding floor, NMS radius, which epoch — is made on
`117093` and `132877`, which are held out from training for that purpose and chosen to
mirror the test split's composition (one two-period match, one three-period match).
