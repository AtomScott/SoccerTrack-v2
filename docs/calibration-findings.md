# Calibration: defects found, and how to reproduce the fix

Investigated 2026-08-10/11. This document exists so nobody has to re-derive any of it.

## TL;DR

All ten matches now calibrate from their pitch keypoints and are accepted by
`CALIB_CHECK_COND`. One match (**132831**) was catastrophically broken by **two transposed
keypoint labels**; the correction is committed as data under `data_corrections/`.

```bash
make calibration          # calibrate all ten + build a validation page
make calibration-repro    # assert the numbers have not drifted
```

Then open `outputs/calibration_all/index.html`. Each panel is one match's undistorted frame
with its undistorted keypoints drawn on top — a correct calibration makes both 21-point
touchlines straight and lands them on the painted lines.

| match | RMS (px) | touchline straightness (px) | keypoints inside canvas |
|---|---|---|---|
| 117093 | 8.64 | 3.04 | 100% |
| 128058 | 9.35 | 3.16 | 100% |
| 128057 | 9.27 | 3.17 | 100% |
| 118577 | 9.40 | 3.10 | 100% |
| 118578 | 9.73 | 3.29 | 100% |
| 132877 | 9.78 | 3.30 | 100% |
| 118576 | 9.88 | 3.26 | 100% |
| 118575 | 10.04 | 3.28 | 100% |
| **132831** | **18.07** | **5.76** | 100% |
| 117092 | 30.63 | 9.43 | 100% |

Deterministic: identical inputs produce byte-identical `mapx.npy`/`mapy.npy`. Verified.

## Defect 1 — two transposed keypoint labels in 132831

| pitch label | as shipped | corrected |
|---|---|---|
| `(88.5,13.84)` | `[3356.0, 781.0]` | `[2640.0, 379.0]` |
| `(105,54.16)` | `[2640.0, 379.0]` | `[3356.0, 781.0]` |

```
as shipped   RMS = 1260.95 px  -> CALIB_CHECK_COND rejects the fit
corrected    RMS =   18.07 px  -> accepted
```

Both points sit on genuine pitch features, so the annotation looks right by eye — 63 of the
65 are correct. Isolation: 132831's 42 touchline points alone fit at RMS 13.22 (healthy),
its 23 penalty-area and centre-circle points alone at 1437.84. Projecting all 65 through
the touchline-only fit leaves 21 points at 13–44 px and exactly these two at ~795 px, each
sitting where the other belongs. Across all ten matches, 132831 is the only one with any
gross outlier.

**Consequences.** The degenerate maps reached `$DATA/raw/132831/`,
`$DATA/interim/calibrated_keypoints/132831/` and `$DATA/production/raw/132831/`, and
132831's GSR labels were generated through them. 132831 is in the **test split**. Its GSR
labels still need regenerating.

**Do not merge the `for_soccernet` branch's calibration change.** It removes
`CALIB_CHECK_COND`, converting a correct refusal into silent corruption. Running that code
reproduces the corrupt `mapx.npy` on disk byte-for-byte, which is how its provenance was
established. It is not a fix; it is what suppressed the error report about this two-line
data-entry mistake.

## Defect 2 — the undistortion canvas is inherited from the input frame

`src/calibration/generate_calibration_mappings.py` builds the map with `balance=1` and an
output canvas equal to the input size. Measured share of frame width occupied by the
undistorted pitch at `balance=1`:

| | pitch / frame width |
|---|---|
| the eight 4096-wide matches | ~0.71 |
| 117092 (3840×1906) | 0.52 |
| 132831 (3840×1504) | **0.22** |

At 0.22 the pitch is a small patch and most of the canvas maps outside the source image,
producing a large black hourglass. **This is not a fold** — checking that source-x increases
monotonically along every output row confirms the warp is valid. Sizing the canvas to
contain the pitch instead puts 100% of keypoints inside for every match.

## Defect 3 — GSR files declare the wrong image dimensions

Every released `production/gsr/<match>/<match>_{1st,2nd}.json` declares `3840×1504`, which
is **132831's** geometry. Correct for that match only.

The pitch `lines` inside those files are the `_keypoints.json` pixels divided by 3840 (x)
and 1504 (y) — verified: denormalising by those constants reproduces each match's keypoint
pixel maxima exactly. So the eight 4096-wide matches report max normalised x of 1.025
(= 3935/3840, i.e. above 1.0), and 117092 is wrong in y.

Two things follow:

- **The keypoints are the clean, authoritative source**, in true pixels. The normalised copy
  in the GSR files is derived and mis-scaled for nine of the ten matches.
- Pixels are recoverable exactly (`norm × 3840`, `norm × 1504`), so this is a deterministic
  rescale, not lost data. But `bbox_image` in the same files is in *true* pixel space, so a
  single uniform correction does not apply to the whole file.

## Defect 4 — documented paths that do not exist

`docs/calibration.md` instructs:

```
./scripts/calibration/generate_calibration_mappings.sh <match_id>
./scripts/calibration/calibrate_camera.sh <match_id>
```

Neither path exists — the scripts are at `scripts/generate_calibration_mappings.sh` and
`scripts/calibrate_camera.sh`. Two further mismatches in that flow:

1. `scripts/generate_calibration_mappings.sh` passes `output_dir=data/interim/<match>`, but
   `scripts/calibrate_camera.sh` reads the maps from
   `data/interim/calibrated_keypoints/<match>`. The Python module's own default is the
   correct location, so **invoke the module directly and omit `--output_dir`**.
2. That wrapper points `video_path` at `data/raw/<match>/<match>_panorama_1st_half.mp4`,
   which exists for no match — the halves live under `interim/`, and `raw/` holds the
   untrimmed `<match>_panorama.mp4`. Harmless in effect provided the substitute has the same
   dimensions, since the video is read *only* for frame width and height.

## A metric trap worth knowing

A collinearity or straightness residual computed on undistorted keypoints reports a
**perfect ~0.00** for a fully collapsed calibration, because collapsed points are trivially
collinear. 132831's broken calibration scored 0.00 — better than the healthy 1.56–1.92
band — while rendering as an unusable smear. Several rounds of investigation were lost to
that false pass.

**Always pair the residual with the bounding-box spread of the undistorted points.** Both
scripts here do, and both say so inline.

## Files

| path | purpose |
|---|---|
| `data_corrections/132831_keypoints.json` | the corrected annotation, authoritative over `$DATA` |
| `scripts/calibration/calibrate_all_from_keypoints.py` | calibrate all ten + validation page (`make calibration`) |
| `scripts/calibration/fix_132831_keypoints.py` | the single-match fix; `--apply` writes into `$DATA` after prompting, with backups |
| `scripts/calibration/test_calibration_strategies.sh` | the diagnostic sweep that found defects 1 and 2 |

## Environment

Requires OpenCV with the `fisheye` module (developed against **cv2 4.10.0**) and numpy.
Both are in the project venv, so `make calibration` uses `.venv/bin/python` directly rather
than `uv run` — `uv run` reinstalls the project's editable package as a side effect, which
is undesirable in a read-only analysis.

## Still open

- Regenerate 132831's GSR labels from the corrected calibration.
- Decide whether the GSR pitch `lines` get renormalised per match, or whether documenting
  the 3840×1504 normaliser is sufficient for the release.
- Decide whether the paper should record that a test-split match's calibration was
  corrected post-hoc.
