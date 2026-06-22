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

- `position` is in **milliseconds from the kickoff of the half specified in `gameTime`**, not from the start of the match.
- Video frame rate for alignment: **25 fps**. Frame index within the corresponding half video is `round(int(position) / 40)`.
- Cross-referencing a BAS event to GSR: pick the half file indicated by `gameTime`, then look up GSR records with `image_id == round(int(position) / 40)`. Expect some events (headers, blocks) to show ball contact ±1 frame from the annotated `position`.

## Evaluation

Report **mAP@1s** (tight) and **mAP@5s** (loose) temporal mean average precision using the SoccerNet BAS protocol. Per-class breakdown is expected in the paper's BAS results table.

See [`src.evaluation.bas_map`](cli.md) for a CLI wrapper, or use the SoccerNet implementation directly.

## Parsing example

```python
import json
from pathlib import Path

data = json.loads(Path("bas/117093/117093_12_class_events.json").read_text())
for ev in data["annotations"]:
    half, clock = ev["gameTime"].split(" - ")
    t_ms = int(ev["position"])
    print(half, clock, t_ms, ev["label"], ev["team"])
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
