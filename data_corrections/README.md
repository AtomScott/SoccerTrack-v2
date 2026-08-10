# Data corrections

Corrected copies of annotation files from the SoccerTrack v2 dataset, kept here so that
a fix is stored as **data** rather than only as code that patches data at runtime.

Anything in this directory is authoritative over the corresponding file under
`$DATA/raw/<match>/`. `scripts/calibration/calibrate_all_from_keypoints.py` loads from
here automatically when a file is present.

These are small text files, deliberately committed so the correction is reviewable as a
diff and survives independently of whoever has shell access to the shared data mount.

## `132831_keypoints.json`

Two entries held **each other's** image coordinates:

| pitch label | as shipped | corrected |
|---|---|---|
| `(88.5,13.84)` | `[3356.0, 781.0]` | `[2640.0, 379.0]` |
| `(105,54.16)` | `[2640.0, 379.0]` | `[3356.0, 781.0]` |

Nothing else in the file differs — `diff` against the original is exactly these two pairs.

**Effect on calibration:**

```
as shipped   cv2.fisheye.calibrate RMS = 1260.95 px  -> CALIB_CHECK_COND rejects the fit
corrected                          RMS =   18.07 px  -> accepted
```

For scale, the other nine matches fit between 8.6 and 30.6 px.

**How it was found.** Both points sit on real pitch features, so the annotation looks
correct by eye and 63 of the 65 points are fine. Fitting 132831's 42 touchline points
alone gives RMS 13.22 (healthy) while its 23 penalty-area and centre-circle points alone
give 1437.84. Projecting all 65 through the touchline-only fit leaves 21 points at
13–44 px and exactly these two at ~795 px, each sitting where the other belongs. Of all
ten matches, 132831 is the only one with any gross outlier.

**Why it mattered.** At RMS 1260.95 the fit is meaningless, and `CALIB_CHECK_COND` in
`src/calibration/generate_calibration_mappings.py` correctly refuses it. The unmerged
`for_soccernet` branch removes that flag, which converted the refusal into a silently
degenerate calibration; applying the resulting remap to 132831's own footage produces a
radial smear with no recognisable pitch. Those maps were written into
`$DATA/raw/132831/`, `$DATA/interim/calibrated_keypoints/132831/` and
`$DATA/production/raw/132831/`, and 132831's GSR labels were generated through them.
132831 is in the test split.

**Still outstanding:** the shared dataset copy has not been overwritten, and 132831's GSR
labels still need regenerating from the corrected calibration. To write the correction
into the dataset itself, use `scripts/calibration/fix_132831_keypoints.py --apply`, which
prompts and takes `.broken-backup` copies first.

See `docs/calibration-findings.md` for the full account and reproduction commands.
