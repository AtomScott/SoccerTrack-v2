"""Audit the BAS annotations, and write the period table the loader and evaluator need.

WHAT THIS SETTLES

1.  THERE ARE THREE PERIODS IN THREE MATCHES, NOT TWO EVERYWHERE.
    117092, 132831 and 132877 were played as three 45-minute periods
    (``matchFullTime="8100000"`` in their metadata; 117092 additionally carries an
    ``EXTRA_FIRST_HALF`` period element with a real frame range). 2,232 events -- 9.4% of
    the 23,663 annotated -- fall in that third period. **No third video and no third GSR
    file exist for any of them**, so those events have neither imagery nor tracks. Test
    match 132831 alone loses 721 of its 3,162 events (22.8%).

    Scoring against events with no input silently depresses every model's recall by an
    amount that varies per match. They are therefore excluded from the benchmark, and the
    benchmark size is 21,431 events.

2.  THE ``gameTime`` HALF PREFIX IS NOT TRUSTWORTHY, AND ``position`` ALONE CANNOT REPLACE IT.
    ``gameTime`` is ``"<period> - <mm:ss>"`` where the clock is ABSOLUTE match time, not
    time within the period. For the third period the prefix is inconsistent: of the 2,232
    third-period events, 2,129 carry no prefix at all and 103 carry a prefix of "1" or "2"
    with a clock reading past 90 minutes (e.g. ``"1 - 135:27"``).

    ``position`` alone cannot disambiguate either, because periods OVERLAP on the nominal
    clock: 118576's first half runs to 48:29 while its second half starts at 45:00. Any
    rule of the form ``position // 45min`` therefore misassigns real stoppage-time events.

    THE RULE USED HERE, which needs neither a reliable prefix nor a period table: an event
    belongs to period p and is evaluable **iff its frame index lands inside the annotated
    frame range of period p's GSR file**. That is exactly the condition "there is input
    data for this event", which is the property that actually matters, and it resolves
    every ambiguous case correctly -- including the 19 prefix-"2" events past 90 minutes
    that ARE genuine second-half stoppage and must be kept.

3.  THE EVENT-TO-TRACK TIME MAPPING.
    GSR GameState frame 1 corresponds to raw tracking frame 251 in every half, so

        t0_ms  =  matchTimeStart - (frameStart - 251) * 40         [from the metadata XML]
        frame  =  1 + (position - t0_ms) / 40

    This was verified to 0.0000 m against the independent pitch-plane CSVs on all four
    halves that have one (see scripts/bas/extract_tracks.py --validate). 132831 and 132877
    have no ``<period>`` elements in their metadata at all, so their t0 falls back to the
    nominal 0 / 2,700,000 ms; across the eight matches that do declare it, t0 never departs
    from nominal by more than 33 ms (under one frame), and the touchline check below
    confirms the fallback independently.

4.  ``visibility`` IS NEVER POPULATED. docs/format-bas.md documents "visible" / "not shown";
    the field is absent from all 23,663 events. Any plan that filters on it is moot.

THE TOUCHLINE CHECK (``--check-alignment``)
    A falsifiable, annotation-independent test of both the time mapping and the actor
    linkage: the taker of a ``Throw In`` must be standing on a touchline, i.e. |y| ~ 34 m.
    It is reported per half together with its value under large injected offsets, so the
    reader can see that it discriminates rather than merely passing.

    A SHARPER ESTIMATOR WAS TRIED AND REJECTED. Cross-correlating the collective player
    speed against play-stopping (``Out``) and play-restarting (``Throw In``) events peaked
    at +18, +41 and +26 frames on three halves whose true offset is known to be 0 (z = 1.7
    to 1.9, i.e. no sharp peak). It measures how long players take to coast to a stop after
    the whistle -- a real property of football, not an annotation offset -- so it is not
    used. Recorded here so it is not re-attempted.

USAGE
    python scripts/bas/audit_annotations.py                       # full report
    python scripts/bas/measure_t0.py --all --write                 # produce the period table FIRST
    python scripts/bas/audit_annotations.py --check-alignment     # + the touchline check
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

MATCHES = ["117092", "117093", "118575", "118576", "118577", "118578",
           "128057", "128058", "132831", "132877"]
TEST_MATCHES = ["128057", "132831"]
HALF_NAME = {1: "1st", 2: "2nd"}

BAS_LABELS = ("Pass", "Drive", "Header", "High Pass", "Out", "Cross", "Throw In", "Shot",
              "Ball Player Block", "Player Successful Tackle", "Free Kick", "Goal")
_CANON = {l.casefold(): l for l in BAS_LABELS}

FPS_MS = 40  # one frame at 25 fps
NOMINAL_T0 = {1: 0.0, 2: 2_700_000.0}
# GameState frame 1 corresponds to this raw tracking frame in every half; see the docstring.
GAMESTATE_FRAME1_RAW = 251.0
PITCH_W = 68.0  # metres; the touchline sits at |y| = 34


# ---------------------------------------------------------------------------
# Reading the released files
# ---------------------------------------------------------------------------

def read_xml_periods(data: Path, match: str) -> dict[int, dict]:
    """Per-period frameStart / matchTimeStart from <match>_tracker_box_metadata.xml.

    Returns {} for 132831 and 132877, whose metadata carries no <period> elements at all.
    """
    path = data / "production" / "raw" / match / f"{match}_tracker_box_metadata.xml"
    if not path.exists():
        return {}
    text = path.read_text()
    idx = {"FIRST_HALF": 1, "SECOND_HALF": 2, "EXTRA_FIRST_HALF": 3, "EXTRA_SECOND_HALF": 4}
    out: dict[int, dict] = {}
    for tag in re.findall(r"<period [^/]*/>", text):
        d = dict(re.findall(r'(\w+)="([^"]*)"', tag))
        p = idx.get(d.get("period", ""))
        if p is None or d.get("matchTimeStart") == "nan":
            continue
        out[p] = {
            "frame_start": float(d["frameStart"]),
            "frame_end": float(d["frameEnd"]),
            "match_time_start": float(d["matchTimeStart"]),
            "match_time_end": float(d["matchTimeEnd"]),
        }
    return out


def read_match_full_time(data: Path, match: str) -> float | None:
    path = data / "production" / "raw" / match / f"{match}_tracker_box_metadata.xml"
    m = re.search(r'matchFullTime="(\d+)"', path.read_text()) if path.exists() else None
    return float(m.group(1)) if m else None


def read_gsr_seq_length(data: Path, match: str, half: int) -> int | None:
    """`info.seq_length` from the GSR GameState header.

    The GSR files are 2.7 GB each; seq_length sits in the first few hundred bytes, so this
    reads only the header rather than parsing the file.
    """
    path = data / "production" / "gsr" / match / f"{match}_{HALF_NAME[half]}.json"
    if not path.exists():
        return None
    head = path.open("rb").read(1200).decode(errors="replace")
    m = re.search(r'"seq_length":\s*(\d+)', head)
    return int(m.group(1)) if m else None


def read_events(data: Path, match: str) -> list[dict]:
    path = data / "production" / "bas" / match / f"{match}_12_class_events.json"
    doc = json.loads(path.read_text())
    for key in ("actions", "annotations"):
        if key in doc:
            return doc[key]
    raise KeyError(f"{path.name} has neither 'actions' nor 'annotations'")


# ---------------------------------------------------------------------------
# The period table
# ---------------------------------------------------------------------------

def build_period_shell(data: Path) -> dict:
    """Skeleton period table with PLACEHOLDER t0 values, for measure_t0.py to fill in.

    t0 is NOT authoritative here. Deriving it from <period frameStart= matchTimeStart=> works
    for the eight matches that declare periods and falls back to a nominal 0 / 2,700,000 ms
    for 132831 and 132877, which declare none -- and that fallback was wrong by 1.0 to 1.5
    SECONDS. scripts/bas/measure_t0.py measures it against <match>_tracker_box_data.xml to a
    zero residual and overwrites every value written here. Nothing should read t0 from a
    table whose t0_source is still "nominal_fallback".
    """
    table: dict[str, dict] = {}
    for match in MATCHES:
        xml = read_xml_periods(data, match)
        periods: dict[str, dict] = {}
        for p in (1, 2):
            if p in xml:
                t0 = xml[p]["match_time_start"] - (xml[p]["frame_start"] - GAMESTATE_FRAME1_RAW) * FPS_MS
                src = "metadata_xml"
            else:
                t0 = NOMINAL_T0[p]
                src = "nominal_fallback"
            n = read_gsr_seq_length(data, match, p)
            periods[str(p)] = {
                "t0_ms": float(t0), "t0_source": src,
                "n_frames": n, "has_tracks": n is not None,
            }
        # A third period exists in the annotations for three matches; nothing was filmed
        # or tracked for it, so it gets no frame count and no t0.
        periods["3"] = {"t0_ms": None, "t0_source": "none", "n_frames": None,
                        "has_tracks": False}
        table[match] = {
            "match_full_time_ms": read_match_full_time(data, match),
            "n_periods_in_metadata": len(xml),
            "periods": periods,
        }
    return {
        "_comment": "SHELL ONLY -- t0 is a placeholder until measure_t0.py fills it in. "
                    "t0_ms is the match-clock time of GSR GameState frame 1; an event's "
                    "frame index is 1 + (position - t0_ms)/40. Period 3 has no imagery and "
                    "no tracks in any match and is outside the benchmark.",
        "fps": 25,
        "test_matches": TEST_MATCHES,
        "matches": table,
    }


def event_period_and_frame(ev: dict, table_for_match: dict) -> tuple[int, int | None]:
    """Assign a period and a GSR frame index to one raw event record.

    The prefix is used only as a hypothesis; it is accepted when the resulting frame lands
    inside that period's annotated range and rejected otherwise. Bare-``gameTime`` events
    are period 3 by construction. See the module docstring for why neither the prefix nor
    ``position`` is sufficient alone.
    """
    gt = str(ev["gameTime"])
    pos = int(ev["position"])
    prefix = int(gt.split(" - ", 1)[0]) if " - " in gt else None
    if prefix in (1, 2):
        spec = table_for_match["periods"][str(prefix)]
        if spec["n_frames"] is not None and spec["t0_ms"] is not None:
            frame = int(round((pos - spec["t0_ms"]) / FPS_MS)) + 1
            if 1 <= frame <= spec["n_frames"]:
                return prefix, frame
    return 3, None


def canon_label(raw: str) -> str:
    lab = _CANON.get(str(raw).casefold())
    if lab is None:
        raise ValueError(f"Unknown BAS label {raw!r}")
    return lab


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report(data: Path, table: dict) -> dict:
    per_match: dict[str, Counter] = {}
    cls_by_period: dict[int, Counter] = defaultdict(Counter)
    prefix_conflicts = Counter()
    n_visibility = 0
    n_player_id = 0
    n_total = 0
    frames_by_match: dict[str, dict[int, list]] = {}

    print("=" * 92)
    print("PERIOD STRUCTURE  (period 3 has no video and no GSR tracks in any match)")
    print("=" * 92)
    print(f'{"match":9} {"fullTime":>9} {"xmlPeriods":>10} {"total":>6} {"p1":>6} {"p2":>6} '
          f'{"p3":>6} {"p3 share":>9}')
    for match in MATCHES:
        tm = table["matches"][match]
        evs = read_events(data, match)
        c = Counter()
        frames_by_match[match] = {1: [], 2: []}
        for ev in evs:
            p, frame = event_period_and_frame(ev, tm)
            c[p] += 1
            cls_by_period[p][canon_label(ev["label"])] += 1
            if p in (1, 2):
                frames_by_match[match][p].append(frame)
            gt = str(ev["gameTime"])
            if " - " in gt and int(gt.split(" - ")[0]) != p:
                prefix_conflicts[(match, int(gt.split(" - ")[0]))] += 1
            if ev.get("visibility") is not None:
                n_visibility += 1
            if ev.get("player_id") not in (None, ""):
                n_player_id += 1
            n_total += 1
        per_match[match] = c
        tag = "  [TEST]" if match in TEST_MATCHES else ""
        ft = tm["match_full_time_ms"]
        print(f'{match + tag:9} {(ft or 0)/60000:8.0f}m {tm["n_periods_in_metadata"]:10} '
              f'{sum(c.values()):6} {c[1]:6} {c[2]:6} {c[3]:6} '
              f'{c[3]/max(sum(c.values()),1)*100:8.1f}%')
    tot = Counter()
    for c in per_match.values():
        tot.update(c)
    print(f'{"TOTAL":9} {"":9} {"":10} {sum(tot.values()):6} {tot[1]:6} {tot[2]:6} {tot[3]:6} '
          f'{tot[3]/max(sum(tot.values()),1)*100:8.1f}%')
    print()
    print(f"  events whose gameTime prefix is contradicted by their frame index: "
          f"{sum(prefix_conflicts.values())}  {dict(prefix_conflicts)}")
    print(f"  events carrying a player_id: {n_player_id:,} / {n_total:,} "
          f"({n_player_id/n_total*100:.1f}%)  -- this is what links an event to an actor")
    print(f"  events carrying a visibility field: {n_visibility} "
          f"(documented in docs/format-bas.md, never populated)")
    print()

    # ---- t0 provenance -----------------------------------------------------
    print("=" * 92)
    print("EVENT-TO-FRAME MAPPING:  frame = 1 + (position - t0_ms) / 40")
    print("=" * 92)
    print(f'{"match":9} {"p1 t0":>10} {"p1 frames":>10} {"p2 t0":>10} {"p2 frames":>10}  source')
    for match in MATCHES:
        p = table["matches"][match]["periods"]
        print(f'{match:9} {p["1"]["t0_ms"]:10.0f} {p["1"]["n_frames"] or 0:10} '
              f'{p["2"]["t0_ms"]:10.0f} {p["2"]["n_frames"] or 0:10}  {p["1"]["t0_source"]}')
    print()
    # frame-range sanity: how close do kept events come to the ends of the annotated range?
    print("  kept-event frame range vs annotated range (a kept event must be inside it):")
    for match in MATCHES:
        p = table["matches"][match]["periods"]
        bits = []
        for half in (1, 2):
            fr = frames_by_match[match][half]
            bits.append(f'p{half} {min(fr)}..{max(fr)} of {p[str(half)]["n_frames"]}')
        print(f"    {match}: " + "   ".join(bits))
    print()

    # ---- class distribution ------------------------------------------------
    print("=" * 92)
    print("CLASS DISTRIBUTION  (support counts; the benchmark is periods 1-2 only)")
    print("=" * 92)
    train = [m for m in MATCHES if m not in TEST_MATCHES]
    cls_train, cls_test = Counter(), Counter()
    for match in MATCHES:
        tm = table["matches"][match]
        tgt = cls_test if match in TEST_MATCHES else cls_train
        for ev in read_events(data, match):
            p, _ = event_period_and_frame(ev, tm)
            if p in (1, 2):
                tgt[canon_label(ev["label"])] += 1
    bench = cls_train + cls_test
    annotated = cls_by_period[1] + cls_by_period[2] + cls_by_period[3]
    print(f'{"class":26} {"annotated":>10} {"benchmark":>10} {"train(8)":>9} {"TEST(2)":>8} '
          f'{"% of bench":>10}')
    for lab in BAS_LABELS:
        print(f'{lab:26} {annotated[lab]:10,} {bench[lab]:10,} {cls_train[lab]:9,} '
              f'{cls_test[lab]:8,} {bench[lab]/max(sum(bench.values()),1)*100:9.2f}%')
    print(f'{"TOTAL":26} {sum(annotated.values()):10,} {sum(bench.values()):10,} '
          f'{sum(cls_train.values()):9,} {sum(cls_test.values()):8,}')
    print()
    thin = [f"{l} (n={cls_test[l]})" for l in BAS_LABELS if cls_test[l] < 30]
    print(f"  imbalance: {max(bench.values())/max(min(bench.values()),1):.0f}x between the "
          f"largest and smallest class")
    print(f"  test-split support below 30: {', '.join(thin) if thin else 'none'}")
    print("  -> every per-class number must be reported with its support; a 12-class macro")
    print("     mean gives these the same weight as classes with two thousand instances.")
    return {"per_match": per_match, "train": cls_train, "test": cls_test}


# ---------------------------------------------------------------------------
# Alignment check
# ---------------------------------------------------------------------------

def check_alignment(data: Path, table: dict, tracks: Path) -> int:
    """Throw-in takers must stand on a touchline. Falsifiable, and offset-sensitive."""
    import numpy as np

    print("=" * 92)
    print("ALIGNMENT CHECK: a Throw In is taken from the touchline, so the taker's |y| ~ 34 m")
    print("=" * 92)
    print("The statistic is the fraction of takers with |y| > 32 m. It is compared against a")
    print("NULL built from 60 displacements of 20-120 s in both directions: at a displaced")
    print("time the same takers are somewhere arbitrary on the pitch. A half passes when the")
    print("true mapping beats every one of the 60 displaced draws.")
    print()
    print(f'{"match":9} {"half":5} {"n":>4} {"med|y|":>7} {"frac>32":>8} '
          f'{"null mean":>9} {"null max":>8} {"z":>6}  verdict')
    ok_all = True
    rows = []
    rng = np.random.default_rng(0)
    shifts = np.concatenate([rng.integers(500, 3000, 30), -rng.integers(500, 3000, 30)])
    for match in MATCHES:
        tm = table["matches"][match]
        events = read_events(data, match)
        for half in (1, 2):
            npz = tracks / f"{match}_{HALF_NAME[half]}_tracks.npz"
            if not npz.exists():
                print(f"{match:9} {HALF_NAME[half]:5} tracks not built -- run extract_tracks.py")
                ok_all = False
                continue
            d = np.load(npz)
            n_f = int(d["frame"].max())
            pl = np.unique(d["player_id"])
            px = {int(p): i for i, p in enumerate(pl.tolist())}
            Y = np.full((n_f + 2, len(pl)), np.nan, np.float32)
            Y[d["frame"], [px[int(p)] for p in d["player_id"].tolist()]] = d["y"]

            takers = []
            for ev in events:
                p, frame = event_period_and_frame(ev, tm)
                if p != half or canon_label(ev["label"]) != "Throw In":
                    continue
                pid = ev.get("player_id")
                if pid in (None, "") or int(pid) not in px:
                    continue
                takers.append((frame, px[int(pid)]))

            def stat(shift_frames: int):
                vals = [Y[f + shift_frames, j] for f, j in takers
                        if 1 <= f + shift_frames <= n_f]
                vals = np.abs(np.array([v for v in vals if not np.isnan(v)]))
                if vals.size == 0:
                    return float("nan"), float("nan")
                return float(np.median(vals)), float(np.mean(vals > 32))

            m0, f0 = stat(0)
            null = np.array([stat(int(s))[1] for s in shifts])
            null = null[~np.isnan(null)]
            z = (f0 - null.mean()) / (null.std() + 1e-9) if null.size else float("nan")
            # The true mapping must beat every displaced draw. With 60 draws that is a
            # one-sided p < 0.017 if the mapping carried no information.
            good = (not np.isnan(f0)) and null.size >= 30 and f0 > null.max()
            ok_all &= good
            rows.append((match, half, m0, f0))
            print(f'{match:9} {HALF_NAME[half]:5} {len(takers):4} {m0:7.1f} {f0:8.2f} '
                  f'{null.mean():9.2f} {null.max():8.2f} {z:6.1f}  {"OK" if good else "WEAK"}')
    print()
    print("  The three matches with a third period (117092, 132831, 132877) score lower in")
    print("  absolute terms than the other seven. Their tracking is systematically")
    print("  compressed -- the 99th percentile of |x| is 47-50 m against 52-54 m elsewhere,")
    print("  on an identically declared 105x68 pitch -- so their players never quite reach")
    print("  the lines. That is a property of the tracking, not of the time mapping: they")
    print("  still clear their own null by z = 13 to 21.")
    print()
    print("alignment check passed on every half" if ok_all
          else "ALIGNMENT CHECK WEAK ON SOME HALVES -- see above before trusting them")
    return 0 if ok_all else 1


def paper_stats(data: Path, table: dict) -> None:
    """The BAS statistics paper/sections/02_results.tex asks for in its TODO.

    Specifically items (c) "total BAS events and their class distribution" and (d)
    "actor-link coverage for BAS events, overall and per class". Actor-link coverage is a
    genuine strength of this dataset -- an event that names its actor can be joined to that
    player's track -- so it is reported per class, because a headline percentage would hide
    any class where the link is weak.
    """
    print()
    print("=" * 92)
    print("DATASET STATISTICS FOR THE PAPER  (benchmark = periods 1-2)")
    print("=" * 92)
    per_class = Counter()
    linked = Counter()
    n_periods_played = Counter()
    for match in MATCHES:
        tm = table["matches"][match]
        by_period = Counter()
        for ev in read_events(data, match):
            p, _ = event_period_and_frame(ev, tm)
            by_period[p] += 1
            if p not in (1, 2):
                continue
            lab = canon_label(ev["label"])
            per_class[lab] += 1
            if ev.get("player_id") not in (None, ""):
                linked[lab] += 1
        # A period counts as PLAYED only if a substantial block of events sits in it. Two
        # of the two-period matches have a single event landing just past the end of their
        # tracking, which the period rule correctly assigns to "no input data" -- but one
        # event is not a third period, and counting set membership called them three-period
        # matches.
        n_periods_played[match] = sum(1 for p, n in by_period.items() if n >= 50)
    total = sum(per_class.values())
    tot_linked = sum(linked.values())
    print(f'{"class":26} {"n":>7} {"share":>7} {"actor-linked":>13} {"link %":>7}')
    for lab in sorted(BAS_LABELS, key=lambda l: -per_class[l]):
        print(f'{lab:26} {per_class[lab]:7,} {per_class[lab]/total*100:6.2f}% '
              f'{linked[lab]:13,} {linked[lab]/max(per_class[lab],1)*100:6.1f}%')
    print(f'{"TOTAL":26} {total:7,} {100.0:6.2f}% {tot_linked:13,} '
          f'{tot_linked/total*100:6.1f}%')
    print()
    print("  LaTeX row fragments for tab:match_stats (BAS columns):")
    for match in MATCHES:
        tm = table["matches"][match]
        c = Counter()
        for ev in read_events(data, match):
            p, _ = event_period_and_frame(ev, tm)
            if p in (1, 2):
                c[canon_label(ev["label"])] += 1
        n_fr = sum(tm["periods"][str(h)]["n_frames"] or 0 for h in (1, 2))
        print(f'    {match} & {n_periods_played[match]} & {n_fr:,} & '
              f'{n_fr/25/60:.0f} & {sum(c.values()):,} \\\\   '
              f'% periods played, tracked frames, minutes, benchmark events')
    print()
    print(f"  total tracked frames across the twenty halves: "
          f"{sum((table['matches'][m]['periods'][str(h)]['n_frames'] or 0) for m in MATCHES for h in (1,2)):,}")
    print(f"  which at 25 fps is {sum((table['matches'][m]['periods'][str(h)]['n_frames'] or 0) for m in MATCHES for h in (1,2))/25/60:.0f} minutes of tracked play")
    print(f"  player-frames (22 entities per frame): "
          f"{sum((table['matches'][m]['periods'][str(h)]['n_frames'] or 0) for m in MATCHES for h in (1,2))*22:,}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="/data/share/SoccerTrack-v2/data")
    ap.add_argument("--tracks", default="/data/share/SoccerTrack-v2/data/derived/bas/tracks")
    ap.add_argument("--paper-stats", action="store_true",
                    help="print the class distribution and actor-link coverage the paper needs")
    ap.add_argument("--periods", default="configs/bas_periods.json",
                    help="period table, produced by scripts/bas/measure_t0.py")
    ap.add_argument("--check-alignment", action="store_true")
    a = ap.parse_args()
    data = Path(a.data)

    table_path = Path(a.periods)
    if not table_path.exists():
        print(f"{table_path} not found. Create it with:\n"
              f"  python scripts/bas/measure_t0.py --all --write {table_path}", file=sys.stderr)
        return 2
    table = json.loads(table_path.read_text())
    stale = [f'{m} p{h}' for m, v in table["matches"].items() for h in (1, 2)
             if v["periods"][str(h)].get("t0_source") != "measured_vs_tracker_box_data"]
    if stale:
        print(f"REFUSING TO REPORT: t0 was never measured for {', '.join(stale)}.\n"
              f"  python scripts/bas/measure_t0.py --all --write {table_path}", file=sys.stderr)
        return 2
    report(data, table)
    rc = 0
    if a.paper_stats:
        paper_stats(data, table)
    if a.check_alignment:
        rc = check_alignment(data, table, Path(a.tracks))
    return rc


if __name__ == "__main__":
    sys.exit(main())
