"""SoccerTrack v2 dataset loader.

Thin, dependency-free reader for GSR / BAS JSON annotations. See docs/format-gsr.md
and docs/format-bas.md for the on-disk schema.

    from src.data_utils.soccertrack_v2 import load_match

    m = load_match("path/to/dataset", match_id="117093")
    for frame in m.gsr_frames(half=1):
        for p in frame.entities:
            print(frame.image_id, p.track_id, p.role, p.x, p.y)

    for ev in m.bas_events():
        print(ev.half, ev.t_ms, ev.label, ev.team)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Literal, Optional

Role = Literal["player", "goalkeeper", "referee", "other"]
TeamSide = Literal["left", "right"]

FPS: int = 25

# Nominal length of one half in milliseconds. BAS `position` values and the `gameTime`
# clock are ABSOLUTE from the start of the match, not relative to the half -- verified on
# 117093, whose half-2 events run 2,700,760..5,506,280 ms with gameTime "2 - 45:00" rather
# than restarting near zero. docs/format-bas.md and the paper both state the opposite
# ("milliseconds from kickoff of the half indicated in gameTime") and give a cross-reference
# formula of floor(position/40), which misaligns every half-2 event by 45 minutes. Whichever
# of the two is made authoritative, the other must change.
HALF_MS: int = 45 * 60 * 1000
BAS_LABELS: tuple[str, ...] = (
    "Pass",
    "Drive",
    "Header",
    "High Pass",
    "Out",
    "Cross",
    "Throw In",
    "Shot",
    "Ball Player Block",
    "Player Successful Tackle",
    "Free Kick",
    "Goal",
)


@dataclass(frozen=True)
class Player:
    """Per-frame GSR record for one entity."""

    track_id: int
    role: Role
    jersey_number: Optional[int]
    team_side: Optional[TeamSide]
    x: float
    y: float
    player_id: Optional[str | int] = None
    bbox_image: Optional[tuple[int, int, int, int]] = None
    bbox_pitch: Optional[tuple[float, float, float, float]] = None


@dataclass(frozen=True)
class Frame:
    """All GSR entities observed in a single panoramic frame of a half."""

    half: int
    image_id: int
    entities: tuple[Player, ...]

    @property
    def t_ms(self) -> int:
        """Milliseconds since kickoff of this half (assuming FPS = 25)."""
        return int(round(self.image_id * 1000 / FPS))


@dataclass(frozen=True)
class Event:
    """One BAS annotation."""

    half: int
    clock: str  # "mm:ss" within the half
    t_ms: int  # ms since kickoff of `half`
    label: str
    team: Optional[TeamSide] = None
    player_id: Optional[str | int] = None
    visibility: Optional[str] = None
    # Detector confidence, for PREDICTIONS only; None in ground truth. Average precision is
    # defined over a confidence-ranked list, so an evaluator that ignores this is not computing
    # AP. src/evaluation/bas_map.py ranks on it.
    score: Optional[float] = None

    @property
    def t_ms_in_half(self) -> int:
        """Milliseconds since kickoff *of this half*.

        `t_ms` is absolute from the start of the match (see HALF_MS), so the per-half video
        offset needs the whole halves before it subtracted.

        APPROXIMATE. This subtracts a NOMINAL 45-minute half, but real halves run over: with
        stoppage, half-2 events reach ~48 minutes of in-half time and a few sit slightly
        before the nominal boundary, giving small negative values. For frame-exact alignment
        use the per-period frameStart in <match>_tracker_box_metadata.xml instead of this.
        """
        return self.t_ms - (self.half - 1) * HALF_MS

    @property
    def image_id(self) -> int:
        """Frame index within this half's video (assuming FPS = 25).

        Uses t_ms_in_half, not t_ms. Using the absolute time here -- as this property did
        before -- put every half-2 event 67,500 frames late.
        """
        return int(round(self.t_ms_in_half * FPS / 1000))


@dataclass
class Match:
    """One SoccerTrack v2 match. Lazy: JSONs are parsed on first access per half."""

    root: Path
    match_id: str
    _gsr_cache: dict[int, tuple[Frame, ...]] = field(default_factory=dict, repr=False)
    _bas_cache: Optional[tuple[Event, ...]] = field(default=None, repr=False)

    # ---- GSR -----------------------------------------------------------------

    def gsr_path(self, half: int) -> Path:
        return self.root / "gsr" / self.match_id / f"{self.match_id}_{_half_suffix(half)}.json"

    def gsr_frames(self, half: int) -> tuple[Frame, ...]:
        if half not in self._gsr_cache:
            self._gsr_cache[half] = tuple(_parse_gsr(self.gsr_path(half), half))
        return self._gsr_cache[half]

    def gsr_frame(self, half: int, image_id: int) -> Optional[Frame]:
        """O(log n) lookup of the Frame at `image_id` (None if absent)."""
        frames = self.gsr_frames(half)
        lo, hi = 0, len(frames)
        while lo < hi:
            mid = (lo + hi) // 2
            if frames[mid].image_id < image_id:
                lo = mid + 1
            else:
                hi = mid
        if lo < len(frames) and frames[lo].image_id == image_id:
            return frames[lo]
        return None

    # ---- BAS -----------------------------------------------------------------

    def bas_path(self) -> Path:
        return self.root / "bas" / self.match_id / f"{self.match_id}_12_class_events.json"

    def bas_events(self) -> tuple[Event, ...]:
        if self._bas_cache is None:
            self._bas_cache = tuple(_parse_bas(self.bas_path()))
        return self._bas_cache

    def events_of(self, label: str) -> tuple[Event, ...]:
        if label not in BAS_LABELS:
            raise ValueError(f"Unknown BAS label: {label!r}. Expected one of {BAS_LABELS}.")
        return tuple(e for e in self.bas_events() if e.label == label)


# ---- Parsers -----------------------------------------------------------------


def _half_suffix(half: int) -> str:
    if half == 1:
        return "1st"
    if half == 2:
        return "2nd"
    raise ValueError(f"half must be 1 or 2, got {half!r}")


def _parse_gsr(path: Path, half: int) -> Iterator[Frame]:
    """Parse one half's GSR annotations, in either on-disk layout.

    THE RELEASED FILES ARE SoccerNet GameState, not the flat record list this function
    originally assumed. All 20 of them are a JSON OBJECT with ``info``/``images``/
    ``annotations``/``categories``. Iterating that as a list yields the top-level KEY STRINGS,
    so ``r["image_id"]`` raised ``TypeError: string indices must be integers`` on every real
    file -- this function could not read a single one. See docs/format-gsr.md.

    Both layouts are accepted, dispatching on the parsed type rather than on a filename:

      * GameState object -- the released ground truth. ``image_id`` is a STRING whose numeric
        suffix is a 1-BASED frame number ("3000001" is frame 1); ``bbox_image`` and
        ``bbox_pitch`` are dicts.
      * flat list of records -- what predictions and derived files use. ``image_id`` is a
        0-based integer frame index; the bboxes are 4-element sequences.

    FRAME INDEX CONVENTION: ``Frame.image_id`` is always the value as stored, so a GameState
    file yields 1-based indices and a flat-record file yields 0-based ones. This function does
    not silently reconcile them, because the correct offset between annotations and video is a
    per-half property that must be measured, not assumed -- see
    scripts/gsr/measure_frame_offset.py. ``Frame.t_ms`` is derived from ``image_id`` and
    inherits the same convention.
    """
    if not path.exists():
        raise FileNotFoundError(f"GSR annotation not found: {path}")
    data = json.loads(path.read_text())

    grouped: dict[int, list[Player]] = {}
    if isinstance(data, dict):
        if "annotations" not in data:
            raise KeyError(
                f"{path.name} is a JSON object without an 'annotations' key; top-level keys "
                f"are {sorted(data)}. Expected SoccerNet GameState or a flat record list."
            )
        for a in data["annotations"]:
            if a.get("supercategory") not in (None, "object"):
                continue                      # skip the pitch/camera annotations
            role = (a.get("attributes") or {}).get("role")
            if role == "ball":
                continue                      # not an athlete; GS-HOTA ignores it too
            image_id = _frame_number(a["image_id"])
            grouped.setdefault(image_id, []).append(_player_from_gamestate(a))
    else:
        for r in data:
            grouped.setdefault(int(r["image_id"]), []).append(_player_from(r))

    for image_id in sorted(grouped):
        yield Frame(half=half, image_id=image_id, entities=tuple(grouped[image_id]))


def _frame_number(image_id) -> int:
    """Numeric frame index from a GameState ``image_id``.

    Released files use "3" + a 6-digit 1-based frame ("3000001"). Sequences staged for TrackLab
    use the 10-character SoccerNet form (split id + 3-digit sequence id + 6-digit frame). In
    both cases the last six digits are the frame number.
    """
    s = str(image_id)
    tail = s.split("_")[-1]
    return int(tail[-6:]) if len(tail) > 6 else int(tail)


def _player_from_gamestate(a: dict) -> Player:
    """Build a Player from a GameState object annotation."""
    attrs = a.get("attributes") or {}
    bp = a.get("bbox_pitch") or {}
    bi = a.get("bbox_image") or {}
    return Player(
        track_id=int(a["track_id"]),
        role=attrs.get("role", "player"),
        jersey_number=_maybe_int(attrs.get("jersey")),
        team_side=attrs.get("team"),
        # The pitch position is the bottom-middle point, in centre-origin metres. In the
        # released ground truth the left/middle/right points are degenerate -- identical
        # values -- so only the middle one carries information.
        x=float(bp["x_bottom_middle"]),
        y=float(bp["y_bottom_middle"]),
        player_id=attrs.get("player_id"),
        bbox_image=(
            (float(bi["x"]), float(bi["y"]), float(bi["w"]), float(bi["h"]))
            if {"x", "y", "w", "h"} <= set(bi) else None
        ),
        bbox_pitch=None,   # the six-key dict does not fit the 4-tuple field; x/y carry it
    )


def _player_from(r: dict) -> Player:
    bbox_image = tuple(r["bbox_image"]) if "bbox_image" in r and r["bbox_image"] is not None else None
    bbox_pitch = tuple(r["bbox_pitch"]) if "bbox_pitch" in r and r["bbox_pitch"] is not None else None
    return Player(
        track_id=int(r["track_id"]),
        role=r["role"],
        jersey_number=_maybe_int(r.get("jersey_number")),
        team_side=r.get("team_side"),
        x=float(r["x"]),
        y=float(r["y"]),
        player_id=r.get("player_id"),
        bbox_image=bbox_image,  # type: ignore[arg-type]
        bbox_pitch=bbox_pitch,  # type: ignore[arg-type]
    )


# The released BAS files spell labels in UPPER CASE ("HIGH PASS") while BAS_LABELS -- and
# the paper -- use Title Case ("High Pass"). The label *set* is identical, so this is purely
# a casing mismatch. Canonicalise on read so the public API keeps the documented spelling.
_BAS_LABEL_BY_CASEFOLD = {label.casefold(): label for label in BAS_LABELS}

# Likewise the event array is called "actions" in the released files, not "annotations".
_BAS_EVENT_KEYS = ("actions", "annotations")


def _parse_bas(path: Path) -> Iterator[Event]:
    """Parse one match's BAS events.

    Tolerates two divergences between the released files and what docs/format-bas.md and
    the paper describe, because as shipped this function could not read a single real file:

      * the event array is keyed "actions", not "annotations" (KeyError on every file);
      * labels are UPPER CASE, not Title Case (ValueError on every event, since unknown
        labels raise rather than being skipped).

    Both are accepted; labels are canonicalised to the documented Title Case spelling. An
    unrecognised label still raises -- silently dropping events would understate recall.
    """
    if not path.exists():
        raise FileNotFoundError(f"BAS annotation not found: {path}")
    data = json.loads(path.read_text())
    for key in _BAS_EVENT_KEYS:
        if key in data:
            events = data[key]
            break
    else:
        raise KeyError(
            f"{path.name} has none of {_BAS_EVENT_KEYS}; top-level keys are {sorted(data)}"
        )
    for a in events:
        gt = str(a["gameTime"])
        if " - " in gt:
            half_str, clock = gt.split(" - ", 1)
            half = int(half_str)
        else:
            # Three matches (117092, 132831, 132877) carry a third block of events whose
            # gameTime omits the half prefix entirely ("90:01"). Their `position` continues
            # past 5,400,000 ms, i.e. beyond 90 minutes, so they are a third period. Infer
            # the period from the clock rather than dropping the events silently.
            clock = gt
            half = int(int(a["position"]) // HALF_MS) + 1
        raw = a["label"]
        label = _BAS_LABEL_BY_CASEFOLD.get(str(raw).casefold())
        if label is None:
            raise ValueError(
                f"Unknown BAS label in {path.name}: {raw!r}. Expected one of {BAS_LABELS} "
                "(case-insensitive)."
            )
        # Predictions may carry a detector confidence under either key; ground truth has none.
        raw_score = a.get("score", a.get("confidence"))
        yield Event(
            half=half,
            clock=clock,
            t_ms=int(a["position"]),
            label=label,
            team=a.get("team"),
            player_id=a.get("player_id"),
            visibility=a.get("visibility"),
            score=None if raw_score is None else float(raw_score),
        )


def _maybe_int(v) -> Optional[int]:
    if v is None or v == "":
        return None
    return int(v)


# ---- Public entry points -----------------------------------------------------


def load_match(root: str | Path, match_id: str) -> Match:
    """Return a lazy Match handle rooted at `root` for `match_id`.

    `root` is the top-level folder with `gsr/`, `bas/`, `mot/`, `videos/` subdirectories.
    """
    root = Path(root)
    if not (root / "gsr").is_dir():
        raise FileNotFoundError(f"Expected `gsr/` under {root}. Is this a SoccerTrack v2 root?")
    return Match(root=root, match_id=str(match_id))


def list_matches(root: str | Path) -> list[str]:
    """Return available match IDs (sorted) under a SoccerTrack v2 root."""
    root = Path(root)
    gsr_root = root / "gsr"
    if not gsr_root.is_dir():
        return []
    return sorted(p.name for p in gsr_root.iterdir() if p.is_dir())


def iter_matches(root: str | Path) -> Iterable[Match]:
    for mid in list_matches(root):
        yield load_match(root, mid)
