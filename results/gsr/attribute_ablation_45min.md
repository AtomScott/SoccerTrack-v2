# Where the 45-minute points actually go

Match 128057, 1st half, full 67,625 frames. Each attribute switched on in turn, so the drop is
that attribute's marginal cost. GS-HOTA partitions detections by `(role, team, jersey)`, so a wrong
attribute is **no match at all**.

| configuration | GS-HOTA | DetA | AssA |
|---|---|---|---|
| nothing (geometry + association only) | **26.036** | 51.209 | 13.409 |
| role only | 25.637 | 49.358 | 13.480 |
| role + team | 22.364 | 35.116 | 14.374 |
| role + team + jersey — **official** | **18.094** | 14.672 | 22.319 |

**Marginal cost:** role **−0.40**, team **−3.27**, jersey **−4.27**. Total attribute cost **7.94**.

## The key structural fact

The attributes-off score, **26.036, is the ceiling for the official metric.** Perfect role, team and
jersey would score exactly that, because perfect attributes partition without losing matches. So all
attribute work put together is worth **at most 7.94 points**.

The ceiling itself is set by `HOTA = sqrt(DetA × AssA)` = `sqrt(51.209 × 13.409)` = 26.2. And:

| lever | current | if perfect, ceiling becomes |
|---|---|---|
| detection (DetA) | 51.2 | 36.6 |
| **association (AssA)** | **13.4** | **71.6** |

**Association is the dominant lever by a wide margin.** Detection and localisation are already
respectable (DetA 51.2, LocA 84.7) — the pipeline finds players and puts them in the right place.
Identity over time is what fails.

AssA of 13.4 is worse than fragmentation alone explains. Roughly 65 substantial tracklets for 23
players is ~2.8× inflation, which would cost far less. The rest is identity *swapping*: 41 tracklets
span more than 3× as many frames as they contain detections, i.e. ids are re-used across gaps rather
than following one player.

## Measured dead end

Dropping the junk-tracklet tail is **worth 0.014 points** and is not worth doing:

| minimum tracklet length | GS-HOTA | tracklets kept |
|---|---|---|
| 0 | 18.094 | 345 |
| 10 | 18.097 | 155 |
| 100 | 18.104 | 65 |
| 500 | 18.108 | 59 |

Removing 81% of the tracklets changes nothing, which confirms they hold no meaningful detection
mass. Tracklet *count* is a bad proxy for tracking quality here — use AssA.

## Ranked work list

1. **`use_spatial_connect` thresholds are in pixels but applied to `bbox_pitch` in metres**
   (`sn_gamestate/gta/connect_track.py`). The module runs; its thresholds are meaningless. This
   targets association, the dominant lever, and it is a bug rather than a tuning exercise.
2. **Team clustering per time window** rather than globally over all 1.39M detections. Team accuracy
   fell 86.6% at 5 min to 76.17% at 45, which is appearance drift over three quarters of an hour.
   Worth up to 3.27 points.
3. **Jersey recognition** — worth up to 4.27 points, but it is a recognition-quality problem
   (45.95% accurate, only 3.4% null) rather than anything configurable. Re-tuning the gates is
   cheap to try: the current 100/0.6 was selected on a *one-minute* clip.
4. **Non-finite pitch coordinates** (0.435% at 45 min, up from 0.16% at 10 min) — a correctness bug
   near the horizon that grows with length. Negligible score impact.
5. ~~Junk-tracklet filter~~ — measured at +0.014, dead.
