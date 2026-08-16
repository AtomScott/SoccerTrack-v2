"""Which period a BAS event belongs to, and whether it is inside the benchmark.

THE PROBLEM THIS SOLVES
    Three of the ten matches (117092, 132831, 132877) were played as THREE 45-minute
    periods. 2,231 events -- 9.4% of the 23,663 annotated, and 22.8% of test match 132831 --
    fall in a third period for which no video and no GSR file exist. Scoring against events
    that have no input data depresses every model's recall by a per-match amount, so they
    are outside the benchmark. The benchmark is 21,432 events.

WHY NEITHER FIELD IN THE FILE IS ENOUGH ON ITS OWN
    ``gameTime`` is ``"<period> - <ABSOLUTE mm:ss>"``. Its prefix is unreliable on exactly
    those three matches: 2,129 third-period events carry no prefix and 102 carry "1" or "2"
    next to a clock past 90 minutes, e.g. ``"1 - 135:27"``.

    ``position`` cannot replace it, because the periods OVERLAP on the nominal clock --
    118576's first half runs to 48:29 while its second half starts at 45:00 -- so a rule
    of the form ``position // 45min`` misassigns genuine stoppage-time events.

THE RULE
    Take the prefix as a hypothesis and accept it only when the resulting frame index lands
    inside that period's annotated GSR frame range. That is exactly the condition "this
    event has input data", which is the property the benchmark actually depends on, and it
    keeps the 19 prefix-"2" events past 90 minutes that really are second-half stoppage.

    frame = 1 + (position - t0_ms) / 40,  with t0_ms from configs/bas_periods.json.

    t0 was measured, not computed: GameState frame 1 is raw tracking frame 251, verified to
    0.0000 m against the independent pitch-plane CSVs. See scripts/bas/audit_annotations.py.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

FRAME_MS = 40.0
EVALUABLE_PERIODS = (1, 2)
DEFAULT_TABLE = Path(__file__).resolve().parents[2] / "configs" / "bas_periods.json"


@lru_cache(maxsize=8)
def load_table(path: str | Path | None = None) -> dict:
    p = Path(path) if path is not None else DEFAULT_TABLE
    if not p.exists():
        raise FileNotFoundError(
            f"period table not found at {p}. Generate it with "
            "`python scripts/bas/audit_annotations.py --write-periods`.")
    return json.loads(p.read_text())["matches"]


def period_and_frame(game_time: str, position: int,
                     match_periods: dict) -> tuple[int, Optional[int]]:
    """Return (period, frame). Period 3 means "no tracks and no imagery"; frame is None."""
    gt = str(game_time)
    prefix = None
    if " - " in gt:
        head = gt.split(" - ", 1)[0].strip()
        if head.isdigit():
            prefix = int(head)
    if prefix in EVALUABLE_PERIODS:
        spec = match_periods[str(prefix)]
        if spec.get("n_frames") is not None and spec.get("t0_ms") is not None:
            frame = int(round((int(position) - spec["t0_ms"]) / FRAME_MS)) + 1
            if 1 <= frame <= spec["n_frames"]:
                return prefix, frame
    return 3, None


def is_evaluable(game_time: str, position: int, match_periods: dict) -> bool:
    return period_and_frame(game_time, position, match_periods)[0] in EVALUABLE_PERIODS


def frame_to_ms(frame: int, period: int, match_periods: dict) -> int:
    """Inverse of the mapping above: the match-clock position of a GSR frame."""
    return int(round(match_periods[str(period)]["t0_ms"] + (frame - 1) * FRAME_MS))
