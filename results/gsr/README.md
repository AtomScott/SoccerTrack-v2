# GSR baseline results — accuracy against sequence length

Match **128057, 1st half** (test split). Nested prefixes of identical footage, so every point sees
the same content and the only variable is how much of it the global modules must reconcile at once.
One configuration throughout: jersey gates `min_roi_area=100`, `min_obb_aspect_ratio=0.6`, and
detection-level team clustering.

| length | frames | detections | GS-HOTA | attrs off | tracklets | vs 23 real | wall |
|---|---|---|---|---|---|---|---|
| 30 s | 750 | 16.5k | 37.155 | — | 26 | 1.1× | 4 min |
| 1 min | 1,500 | 33k | **47.589** | — | 35 | 1.5× | 12 min |
| 2 min | 3,000 | 66k | 39.882 | — | 42 | 1.8× | 37 min |
| 5 min | 7,500 | 165k | 30.974 | — | 60 | 2.6× | 2.6 h |
| 10 min | 15,000 | 301k | 25.604 | 33.908 | 99 | 4.3× | 5.5 h |
| **45 min (full half)** | **67,625** | **1,389k** | **18.094** | 26.036 | **345** | **15.0×** | **~37 h** |

## What the curve says

Accuracy **peaks at one minute and falls monotonically after**, with no sign of levelling out. The
45-minute figure is 38% of the one-minute figure on identical footage.

**Fragmentation is the mechanism.** Predicted tracklet count runs from 1.1× the true player count
at 30 seconds to **15.0× at a full half**. GS-HOTA partitions detections into classes by
`(role, team, jersey)`, so a tracklet split in two cannot carry one identity — and the per-tracklet
votes for role, team and jersey each get a smaller, noisier sample to work from.

The attributes-off column separates the two failure modes. At 45 minutes geometry and detection are
still respectable (DetA 51.2, LocA 84.7) — the pipeline still *finds* players and puts them in the
right place. What collapses is association and identity (AssA 13.4). Any work aimed at the headline
number should target fragmentation, not detection.

## Caveats a reader needs

- **Single match, single half.** No error bars. 128057 is a test-split match; do not tune on it.
- **Non-finite pitch coordinates grow with length**: 0.16% of predictions at 10 minutes, **0.435%
  at 45**. Calibration near the horizon degrades as the sequence lengthens. Those detections are
  dropped before scoring (they cannot be matched in pitch space at all) and the count is recorded
  in each `score_*.json`.
- **The 45-minute run was two-stage** (detector+pose → state → remaining twelve modules) for
  resumability. The split sits before reid so `embeddings` are never written to disk, where
  `forget_columns` would have discarded them.
- **The tracklet-level team CSV crosses over** with the detection-level one: stock clustering is
  better below 2 minutes, worse by 5. Neither is a universal default, and a sweep mixing them
  measures the config rather than the length.

## Files

- `length_sweep_128057_1st.csv` — the curve above (detection-level team)
- `length_sweep_128057_1st_tracklet_team.csv` — the same lengths with stock tracklet-level team
- `score_10min_128057_1st.json`, `score_45min_128057_1st.json` — full metric breakdowns

Reproduce a score with `python scripts/gsr/score_one.py --pred <pred.json> --gt <Labels-GameState.json>`.
