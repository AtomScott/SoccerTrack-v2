# BAS annotation format — SoccerTrack v2

This document specifies the Ball Action Spotting (BAS) annotation format shipped with SoccerTrack v2. BAS annotations record the timestamps and classes of 12 ball-related events per match, aligned to the global video timeline.

The format is deliberately close to the [SoccerNet Action Spotting](https://www.soccer-net.org/tasks/ball-action-spotting) convention — `gameTime` + `position` + `label` — so existing evaluation code should port with minimal changes.

## File layout

```
bas/
├── 117092/
│   └── 117092_12_class_events.json
├── 117093/
│   └── ...
└── ...
```

One JSON file per match. Unlike GSR (one file per half), BAS events for both halves are concatenated into a single file — the `gameTime` field encodes the half.

## File schema

Top-level object:

```json
{
  "UrlLocal": "117093",
  "UrlYoutube": null,
  "annotations": [
    {
      "gameTime": "1 - 12:30",
      "position": "750000",
      "label": "Pass",
      "team": "left",
      "player_id": "117093_L_9",
      "visibility": "visible"
    },
    ...
  ]
}
```

| Top-level field | Type | Description |
|---|---|---|
| `UrlLocal` | string | Match ID, matches the folder name. |
| `UrlYoutube` | string or null | External link if the match is mirrored publicly. |
| `annotations` | array | Ordered list of events, sorted by `position`. |

## Event schema

| Field | Type | Required | Description |
|---|---|---|---|
| `gameTime` | string | yes | Format `"<half> - <mm:ss>"`. `<half>` is `1` or `2`. `<mm:ss>` is the match clock within that half, starting at `00:00` at kickoff of the half. |
| `position` | string of integer ms | yes | Milliseconds from kickoff **of the half in `gameTime`**. Prefer this over parsing `gameTime` for precise alignment. |
| `label` | string | yes | One of the 12 classes below. |
| `team` | string | yes | `"left"` or `"right"`, the team performing the action. Sides match the `team_side` convention in `format-gsr.md` for that half. |
| `player_id` | string or integer or null | no | Actor's player ID when identifiable; `null` otherwise. Same convention as GSR's `player_id`. |
| `visibility` | string | no | `"visible"` or `"not shown"`. `"not shown"` marks events whose outcome is known from context but whose ball contact is not clearly observable in the panoramic frame. |

## Label set (12 classes)

| Class | Semantic |
|---|---|
| `Pass` | Intentional ground pass between teammates. |
| `Drive` | Player carries the ball (dribble / run with the ball). |
| `Header` | Ball struck with the head. |
| `High Pass` | Lofted pass (ground not touched between origin and target). |
| `Out` | Ball leaves the field of play across a side/goal line. |
| `Cross` | Pass from wide area towards the penalty area. |
| `Throw In` | Throw-in after `Out` on a sideline. |
| `Shot` | Attempt on goal. |
| `Ball Player Block` | Defender blocks a pass or shot. |
| `Player Successful Tackle` | Defensive tackle that wins possession. |
| `Free Kick` | Restart by free kick (direct or indirect). |
| `Goal` | Goal scored. |

Labels are strings matching this table exactly (case and spacing). Consumers must treat the class list as fixed and fail loudly on unknown labels — adding a class is a dataset-wide change.

## Time alignment

> **The three struck-through bullets below are WRONG about the released data.** They describe
> the format as designed; the files diverge from it. Which side gets corrected is a dataset
> decision that has not been taken. Until it is, follow "What the released files actually do"
> and use `src.data_utils.bas_periods`, which implements it.

- ~~`position` is in **milliseconds from the kickoff of the half specified in `gameTime`**, not from the start of the match.~~
- ~~Frame index within the corresponding half video is `round(int(position) / 40)`.~~
- ~~Cross-referencing a BAS event to GSR: look up GSR records with `image_id == round(int(position) / 40)`.~~

Frame rate is indeed **25 fps**, and ball contact does sit within ±1 frame of `position` for
headers and blocks.

## What the released files actually do

Measured across all ten matches. Evidence in [`bas-findings.md`](bas-findings.md);
reproduce with [`../scripts/bas/audit_annotations.py`](../scripts/bas/audit_annotations.py).

**1. `position` is ABSOLUTE from the start of the match, not from the half's kickoff.**
Verified on 117093, whose half-2 events run 2,700,760–5,506,280 ms rather than restarting
near zero. `round(position / 40)` therefore lands **67,500 frames — 45 minutes — late for
every second-half event.** `gameTime`'s `mm:ss` is on the same absolute clock, so a
second-half event reads `"2 - 45:00"` at kickoff, not `"2 - 00:00"`.

**2. The event array is keyed `actions`, not `annotations`.**

**3. Labels are UPPER CASE in the files** (`"HIGH PASS"`), Title Case in this document. The
label *set* is identical; `src.data_utils.soccertrack_v2` canonicalises on read.

**4. Three matches have a THIRD 45-minute period, and nothing was filmed or tracked for it.**
117092, 132831 and 132877 were played as three periods — 132831 and 132877 declare
`matchFullTime="8100000"`, and 117092 carries an `EXTRA_FIRST_HALF` period element with a
real frame range. **2,225 events (9.4% of the 23,663 annotated) fall in that period, and the
release contains only `_1st` and `_2nd` videos and GSR files**, so those events have no
imagery and no tracks. Test match 132831 alone has 721 of them, 22.8% of its annotations.

**5. The `gameTime` period prefix is unreliable, and `position` alone cannot replace it.**
Of those 2,225 events, 2,129 carry no prefix at all and **96 carry a prefix of `1` or `2`
beside a clock past 90 minutes** — e.g. `{"gameTime": "1 - 135:27", "position": "8127120"}`.
And `position` cannot simply be divided into periods, because periods **overlap on the
nominal clock**: 118576's first half runs to 48:29 while its second half starts at 45:00.

Take the prefix as a hypothesis and accept it only when the resulting frame lands inside
that period's annotated GSR frame range:

```python
from src.data_utils.bas_periods import load_table, period_and_frame

periods = load_table()["128057"]["periods"]        # configs/bas_periods.json
period, frame = period_and_frame(ev["gameTime"], int(ev["position"]), periods)
# period == 3 means "no imagery and no tracks"; frame is then None
```

**6. The event-to-frame mapping is `frame = 1 + (position - t0_ms) / 40`**, with `t0_ms` the
match-clock time of GSR GameState frame 1, tabulated per match and period in
`configs/bas_periods.json`. GameState frame 1 corresponds to raw tracking frame 251 — which
was measured against an independent source, not derived: the obvious "frame 1 is the period's
first frame" mapping is off by exactly one frame.

**7. `visibility` is never populated.** The field documented below as `"visible"` /
`"not shown"` is absent from all 23,663 events, so a pipeline that filters on it filters
nothing.

## Evaluation

Report **mAP@1s** (tight) and **mAP@5s** (loose) temporal mean average precision using the SoccerNet BAS protocol. Per-class breakdown is expected in the paper's BAS results table.

See [`src.evaluation.bas_map`](cli.md) for a CLI wrapper, or use the SoccerNet implementation directly.

## Parsing example

```python
import json
from pathlib import Path

data = json.loads(Path("bas/117093/117093_12_class_events.json").read_text())
for ev in data["actions"]:                     # NOT "annotations" -- see above
    half, clock = ev["gameTime"].split(" - ")  # raises on the 2,129 prefix-less events
    t_ms = int(ev["position"])                 # ABSOLUTE match time, not per-half
    print(half, clock, t_ms, ev["label"].title(), ev["team"])
```

That snippet still crashes on 117092, 132831 and 132877, whose third-period events have no
`" - "` in `gameTime`. Prefer the loader, which handles all of it:

```python
from src.data_utils.soccertrack_v2 import load_match

for ev in load_match("/data/share/SoccerTrack-v2/data/production", "117093").bas_events():
    print(ev.half, ev.clock, ev.t_ms, ev.label, ev.team)
```

## Known edge cases

- **Dual actions at the same timestamp.** A shot that becomes a goal is two annotations: one `Shot` and one `Goal`, same `position`. Do not collapse.
- **`Throw In` after `Out`.** `Throw In` timestamp is the restart, not the moment the ball left play — pair with the preceding `Out` using time proximity.
- **`Free Kick` covers direct and indirect.** No sub-class distinction in this release.
- **No `Penalty` class.** Penalties are annotated as `Shot` (from the spot) and optionally `Goal`.

## See also

- [`format-gsr.md`](format-gsr.md) — Game State Reconstruction format.
- [`task-bas.html`](task-bas.html) — task description and benchmark pointers.
- [`docs/TODO.md`](TODO.md) — ongoing release checklist.
