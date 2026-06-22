# GSR annotation format — SoccerTrack v2

This document specifies the Game State Reconstruction (GSR) annotation format shipped with SoccerTrack v2. GSR annotations record, for every annotated frame, where each entity (player, goalkeeper, referee) is on the pitch in 2D metric coordinates, plus persistent identifying attributes (track ID, jersey number, role, team side).

The format is deliberately close to the [SoccerNet GSR](https://www.soccer-net.org/tasks/game-state-reconstruction) convention so that tooling ports across.

## File layout

```
gsr/
├── 117092/
│   ├── 117092_1st.json      # 1st half GSR annotations
│   └── 117092_2nd.json      # 2nd half GSR annotations
├── 117093/
│   └── ...
└── ...
```

One JSON file per half per match. Each file is a JSON array of per-entity-per-frame annotation records.

## Record schema

Each annotation is a flat JSON object:

```json
{
  "image_id": 12480,
  "track_id": 7,
  "player_id": "117092_L_9",
  "role": "player",
  "jersey_number": 9,
  "team_side": "left",
  "x": 48.21,
  "y": 34.07,
  "bbox_image": [1840, 710, 108, 242],
  "bbox_pitch":  [46.8, 33.1, 1.4, 2.0]
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `image_id` | integer | yes | Zero-indexed frame number in the half's panoramic video. |
| `track_id` | integer | yes | Per-half track identifier. Persistent within a half; **not** guaranteed across halves (re-link across halves via `player_id` if available, or via jersey + team). |
| `player_id` | string or integer or null | no | Stable identifier for the real-world player across halves and matches when known; `null` for referees, "other", or when the player could not be identified. Format is maintainer-internal; treat as opaque. |
| `role` | string | yes | One of `"player"`, `"goalkeeper"`, `"referee"`, `"other"`. |
| `jersey_number` | integer or null | yes | `0`–`99` when visible; `null` when not observable (e.g. back turned) or not applicable (referee). |
| `team_side` | string or null | yes | `"left"` or `"right"` for players/goalkeepers; `null` for referees and `"other"`. Sides are fixed **per half** from the attacking direction at kickoff; note left/right may flip between halves for the same physical team. |
| `x`, `y` | float | yes | Player position on the pitch, in **metres**. See [pitch coordinate system](#pitch-coordinate-system). Corresponds to the bottom-centre of `bbox_pitch` (approximately the player's feet). |
| `bbox_image` | `[x, y, w, h]` of ints | no | Bounding box in image pixels on the panoramic video (top-left origin, x right, y down). Omitted for entities derived from pitch-plane-only annotations. |
| `bbox_pitch` | `[x, y, w, h]` of floats | no | Bounding box projected to the pitch plane, in metres. |

Any field not listed above should be treated as forward-compatible metadata and ignored by loaders that don't know about it.

## Pitch coordinate system

- **Units**: metres.
- **Origin**: centre of the pitch (`x = 0, y = 0`).
- **Axes**: `x` grows towards the right side of the pitch as seen from the main broadcast camera; `y` grows towards the top of the pitch in the same view. Right-handed.
- **Pitch dimensions**: assumed `105 m × 68 m` (FIFA standard). Any per-match override, when one is recorded, lives in the per-match metadata table [`docs/matches.json`](matches.json); the current skeleton does not yet populate pitch-dimension fields, so treat `105 × 68` as authoritative until then. The four corners of the pitch rectangle lie at `(±52.5, ±34.0)`.
- **Goal lines**: `x = ±52.5`. **Sidelines**: `y = ±34.0`. **Halfway line**: `x = 0`.

Positions outside the rectangle are legal (balls / players can leave the field of play). Clip only for visualisation, never before metric computation.

## Time alignment

- GSR annotation frame rate matches the source panoramic video — **25 fps** for all SoccerTrack v2 matches.
- `image_id = 0` is the first frame of the half video, which has already been trimmed to start at kickoff by the pipeline (`scripts/trim_video_into_halves.sh`).
- To cross-reference a GSR frame against a BAS event at global timestamp `t_ms`, convert BAS `position` (milliseconds from kickoff of the **half encoded in `gameTime`**) to `image_id = round(t_ms / 40)` on the matching half file. Tolerance for alignment is `±1` frame (`±40 ms`).

## Parsing example

Minimal Python:

```python
import json
from pathlib import Path
from collections import defaultdict

path = Path("gsr/117093/117093_1st.json")
records = json.loads(path.read_text())

by_frame: dict[int, list] = defaultdict(list)
for r in records:
    by_frame[r["image_id"]].append(r)

# Entities visible at frame 12480
for r in by_frame[12480]:
    print(r["track_id"], r["role"], r["jersey_number"], r["x"], r["y"])
```

A richer loader returning `Match` / `Frame` / `Player` objects is available as `src.data_utils.soccertrack_v2.load_match` (see [`cli.md`](cli.md)).

## Evaluation

The reference metric is **GS-HOTA** (the SoccerNet GSR variant of HOTA). See [`src.evaluation.gs_hota`](cli.md) for a thin CLI wrapper around the SoccerNet implementation, or use that implementation directly. Report `GS-HOTA`, `DetA`, `AssA`, and `LocA` as a standard breakdown.

## Known edge cases

- **Track ID non-uniqueness across halves.** A player can have different `track_id` in the 1st and 2nd halves; re-link via `player_id` (where present) or jersey + team.
- **Jersey occlusion.** `jersey_number = null` is common — don't drop these rows; they still contribute to pitch-space tracking.
- **Goalkeeper role flips.** `role` can change from `"player"` to `"goalkeeper"` if a substitution swaps GK duties; this is reflected per frame.
- **Referees lack team side.** Expect `team_side = null` for `role = "referee"` and `"other"`.

## See also

- [`format-bas.md`](format-bas.md) — Ball Action Spotting format.
- [`task-gsr.html`](task-gsr.html) — task description and benchmark pointers.
- [`docs/TODO.md`](TODO.md) — ongoing release checklist.
