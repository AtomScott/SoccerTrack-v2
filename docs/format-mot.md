# MOT annotation format — SoccerTrack v2

This document specifies the multi-object tracking (MOT) annotation format shipped
with SoccerTrack v2. MOT annotations record, for every annotated frame, the
**image-plane** bounding box and persistent track ID of each tracked entity
(player, goalkeeper, referee). Unlike [GSR](format-gsr.md), MOT lives entirely in
pixel space — there is no pitch projection.

The format is the standard **MOTChallenge** comma-separated `gt.txt` schema, so it
plugs directly into [TrackEval](https://github.com/JonathonLuiten/TrackEval). The
SoccerTrack v2 evaluator [`src.evaluation.mot_hota`](cli.md) is a thin wrapper over
TrackEval's `MotChallenge2DBox` dataset class — it reads exactly the layout and
columns described below. **If you change the column order or units, the evaluator
silently mis-scores; keep this format byte-for-byte MOTChallenge.**

## File layout

Ground truth, as consumed by `src.evaluation.mot_hota`:

```
mot/
├── 117092/
│   ├── gt/
│   │   └── gt.txt          # ground-truth tracks (this document)
│   └── seqinfo.ini         # sequence metadata (see below)
├── 117093/
│   └── ...
└── ...
```

Predictions, in the **same** column format, one file per sequence, under the
native MOTChallenge `trackers/<tracker>/data/<seq>.txt` layout (one `<tracker>`
directory per tracking run):

```
<pred_root>/
├── bytetrack/              # one tracker run (any name; auto-discovered)
│   └── data/
│       ├── 117092.txt      # your tracker output for sequence 117092
│       ├── 117093.txt
│       └── ...
└── ...
```

The sequence name is the `<seq>` (e.g. `117092`, or `117092_1st` for a single
half — see the half note below). It is passed to the evaluator with
`--matches 117092 117093 ...`, and is the key used to pair a
`mot/<seq>/gt/gt.txt` against the matching `<pred_root>/<tracker>/data/<seq>.txt`.

> **Why the `<tracker>/data/` levels?** `MotChallenge2DBox` reads tracker files at
> `<pred_root>/<tracker>/<TRACKER_SUB_FOLDER>/<seq>.txt` (default
> `TRACKER_SUB_FOLDER=data`) and discovers each `<tracker>` by listing
> `<pred_root>`. `src.evaluation.mot_hota` also sets `SKIP_SPLIT_FOL=True` so
> neither ground truth nor predictions get nested under an extra
> `MOT17-custom` split folder. Both the GT tree above and the baseline writer
> (`baselines/mot/train.py`) follow exactly these resolved paths.

> **Half vs. full match.** GSR/BAS files are keyed per half; MOT sequences here are
> keyed by `<match_id>` only. Treat each `<match_id>` as one MOT sequence — if you
> evaluate halves separately, give them distinct sequence names (e.g.
> `117092_1st`, `117092_2nd`) and list each as its own `--matches` entry, keeping
> the `gt/gt.txt` + `seqinfo.ini` layout per sequence.

## Row schema (`gt.txt` and `data.txt`)

Each line is one detection of one track in one frame. Fields are **comma-separated**
with **no header row**. Ten columns, in this exact order:

```
<frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<class>,<visibility>,<unused>
```

Example (`mot/117092/gt/gt.txt`):

```
1,7,1840,710,108,242,1,1,1
1,8,612,540,96,228,1,1,1
2,7,1844,708,108,243,1,1,1
2,8,615,541,97,229,1,1,0.6
```

| # | Column | Type | Description |
|---|---|---|---|
| 1 | `frame` | integer | **1-indexed** frame number within the sequence video. (MOTChallenge convention — note this differs from GSR's 0-indexed `image_id`; add 1 when converting GSR → MOT.) |
| 2 | `id` | integer | Track identifier, persistent across frames within the sequence. `-1` is reserved for "no id" in detection-only files; ground truth always has a real id. |
| 3 | `bb_left` | float | Bounding-box top-left **x** in image pixels (origin top-left, x right). |
| 4 | `bb_top` | float | Bounding-box top-left **y** in image pixels (y down). |
| 5 | `bb_width` | float | Bounding-box width in pixels. |
| 6 | `bb_height` | float | Bounding-box height in pixels. |
| 7 | `conf` | float | Confidence / flag. In **ground truth**, `1` = consider this box, `0` = ignore it (the box is excluded from scoring). In **predictions**, the tracker's confidence score (any float; TrackEval may threshold it). |
| 8 | `class` | integer | Object class id. SoccerTrack v2 MOT uses class `1` ("pedestrian"/person) for all tracked entities; the role split (player / goalkeeper / referee) is **not** encoded here — use [GSR](format-gsr.md) for role. |
| 9 | `visibility` | float | Visibility ratio in `[0, 1]` (fraction of the box not occluded). Ground truth only; predictions may set `1`. |
| 10 | `<unused>` | — | MOTChallenge reserves a 10th column (3D x in MOT16/20). It is unused for 2D box tracking and may be omitted. The evaluator reads the first nine columns. |

Notes:

- **Box coordinates are pixels in the panoramic video frame**, the same image
  plane as GSR's `bbox_image` (`[x, y, w, h]`). A GSR `bbox_image` row maps to MOT
  columns 3–6 directly.
- `conf = 0` rows in ground truth are **ignore regions / don't-care boxes**: do not
  delete them, TrackEval needs them to suppress false-positive penalties there.
- The class column is present for MOTChallenge compatibility but is effectively
  constant (`1`) in this dataset. Do not rely on it to recover role.

## `seqinfo.ini`

`MotChallenge2DBox` reads a `seqinfo.ini` per sequence to learn the frame count and
resolution. Minimal example (`mot/117092/seqinfo.ini`):

```ini
[Sequence]
name=117092
imDir=img1
frameRate=25
seqLength=135000
imWidth=3840
imHeight=2160
imExt=.jpg
```

| Key | Meaning |
|---|---|
| `name` | Sequence id; must match the `<match_id>` directory and the `--matches` argument. |
| `frameRate` | Frames per second — **25 fps** for all SoccerTrack v2 matches. |
| `seqLength` | Number of frames in the sequence (1-indexed `frame` ranges `1 .. seqLength`). |
| `imWidth`, `imHeight` | Panoramic frame resolution in pixels; the reference space for all box coordinates. |
| `imDir`, `imExt` | Image directory / extension. Required by TrackEval's config even when scoring from `gt.txt` alone; values are conventional and not consumed by the box-only metrics. |

> The exact `seqLength`, `imWidth`, and `imHeight` are per-match and recorded with
> the released data — do not assume the example values above. See the per-match
> metadata table [`docs/matches.json`](matches.json) when those fields are
> populated.

## Time alignment

- Frame rate is **25 fps** (`1` frame = `40 ms`), matching GSR and BAS.
- MOT `frame` is **1-indexed**; GSR `image_id` is **0-indexed**. To cross-reference,
  use `image_id = frame - 1` on the same half/sequence video.
- To map a BAS event (`position` ms from kickoff of its half) onto a MOT frame of
  that half: `frame = round(int(position) / 40) + 1`.

## Evaluation

The reference metrics are **HOTA**, **IDF1**, and **MOTA**, computed via TrackEval.
Run:

```bash
python -m src.evaluation.mot_hota \
    --pred PRED_ROOT --gt GT_ROOT \
    --matches 128057 132831 \
    --metrics HOTA IDF1 MOTA
```

where `GT_ROOT` contains `<seq>/gt/gt.txt` (+ `<seq>/seqinfo.ini`) per sequence
(point `--gt` at the directory that holds the `<seq>` folders, e.g. `data/mot`)
and `PRED_ROOT` contains `<tracker>/data/<seq>.txt`. Pass `--tracker NAME` to
score one specific tracker run, or omit it to auto-discover every sub-dir of
`PRED_ROOT`. TrackEval must be installed separately (`pip install` from the
[TrackEval repo](https://github.com/JonathonLuiten/TrackEval));
`src.evaluation.mot_hota` raises a clear error if it is missing.

> The evaluator sets `SKIP_SPLIT_FOL=True` on `MotChallenge2DBox`. Without it,
> TrackEval (whose `BENCHMARK`/`SPLIT_TO_EVAL` default to `MOT17`/`custom`) would
> prepend a `MOT17-custom/` folder to **both** the GT and prediction paths above,
> and scoring the documented un-nested layout would fail with
> `ini file does not exist: <seq>/seqinfo.ini`. Do not create a `MOT17-custom`
> directory; the evaluator reads the un-nested paths shown here.

Report `HOTA`, `DetA`, `AssA`, `IDF1`, and `MOTA` as the standard breakdown in the
paper's MOT results table.

## Known edge cases

- **1-indexed frames.** The single most common conversion bug is forgetting that
  MOT frames start at `1` while GSR `image_id` starts at `0`. Off-by-one shifts
  every box by one frame and tanks HOTA.
- **Ignore boxes (`conf = 0`).** Present in ground truth for regions where tracking
  is not scored (e.g. crowd, bench). Keep them; TrackEval uses them to avoid
  penalising detections there.
- **Track ID scope.** `id` is unique and persistent **within a sequence**, not
  across matches or across halves if you split a match into two sequences.
- **Class is not role.** All entities share `class = 1`; recover player vs.
  goalkeeper vs. referee from [GSR](format-gsr.md), not from MOT.

## See also

- [`format-gsr.md`](format-gsr.md) — Game State Reconstruction format (pitch space, roles).
- [`format-bas.md`](format-bas.md) — Ball Action Spotting format.
- [`task-mot.html`](task-mot.html) — task description and benchmark pointers.
- [`docs/TODO.md`](TODO.md) — ongoing release checklist.
</content>
</invoke>
