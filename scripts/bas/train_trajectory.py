"""Train the trajectory spotter, decode its output to spots, and write BAS prediction files.

Experiment B of docs/experiment-design-bas.md, on GROUND-TRUTH tracks. This measures
whether complete game state determines ball events. It is NOT a deployable pipeline: it
assumes perfect GSR, so it is an upper bound on any system that has to estimate the tracks
first. That distinction must survive into the paper.

SPLIT
    test        128057, 132831        the SoccerTrack Challenge 2025 test split; touched
                                      exactly once, at the end, and never for tuning
    validation  117093, 132877        one two-period and one three-period match, mirroring
                                      the composition of the test split
    train       the remaining six

DECODING
    Per class, the predicted probability series is scanned for local maxima above a floor,
    then thinned by within-class non-maximum suppression. Each surviving peak becomes one
    event whose confidence is the peak height -- which is what makes average precision
    meaningful, since AP is defined over a confidence-ranked list.

    The decoding floor and NMS radius are chosen on the VALIDATION matches only.

USAGE
    python scripts/bas/train_trajectory.py --epochs 40
    python scripts/bas/train_trajectory.py --eval-only --ckpt outputs/bas/spotter.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.bas.augment import mirror  # noqa: E402
from src.bas.model import TrajectorySpotter  # noqa: E402
from src.evaluation.bas_map import ap_tolerant  # noqa: E402

BAS_LABELS = ("Pass", "Drive", "Header", "High Pass", "Out", "Cross", "Throw In", "Shot",
              "Ball Player Block", "Player Successful Tackle", "Free Kick", "Goal")
HALF_NAME = {1: "1st", 2: "2nd"}

TEST = ["128057", "132831"]
VAL = ["117093", "132877"]
TRAIN = ["117092", "118575", "118576", "118577", "118578", "128058"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_split(ds_dir: Path, matches: list[str]) -> list[dict]:
    out = []
    for m in matches:
        for h in (1, 2):
            p = ds_dir / f"{m}_{HALF_NAME[h]}_dataset.npz"
            if not p.exists():
                raise FileNotFoundError(f"{p} -- run scripts/bas/build_dataset.py first")
            d = np.load(p)
            out.append({"match": m, "half": h, "feat": d["feat"], "target": d["target"],
                        "frames": d["frames"], "stride": int(d["stride"]),
                        "ev_frame": d["ev_frame"], "ev_class": d["ev_class"],
                        "ev_t_ms": d["ev_t_ms"]})
    return out


def fit_normaliser(halves: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X = np.concatenate([h["feat"] for h in halves])
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-6] = 1.0   # constant columns (e.g. squad size) must not divide by zero
    return mu.astype(np.float32), sd.astype(np.float32)


def sample_windows(halves, n, win, rng, augment: bool = True):
    """Random crops, optionally reflected.

    Football is symmetric under reflection and none of the twelve labels is handed, so
    mirroring the pitch end-to-end or top-to-bottom gives three extra valid views of every
    window with the targets untouched. With only twelve training halves this matters: the
    unaugmented model's validation loss doubled by epoch 6 while its training loss halved.
    See src/bas/augment.py, whose transform is checked against a full feature rebuild in
    tests/test_bas_augment.py.
    """
    F, T = [], []
    for _ in range(n):
        h = halves[rng.integers(len(halves))]
        L = h["feat"].shape[0]
        s = int(rng.integers(0, max(1, L - win)))
        f = h["feat"][s:s + win]
        if augment:
            f = mirror(f, bool(rng.integers(2)), bool(rng.integers(2)))
        F.append(f)
        T.append(h["target"][s:s + win])
    return np.stack(F), np.stack(T)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(halves, mu, sd, args, val_halves):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = TrajectorySpotter(n_features=halves[0]["feat"].shape[1],
                              n_classes=len(BAS_LABELS), hidden=args.hidden,
                              dropout=args.dropout)
    print(f"  receptive field {model.receptive_field} rows "
          f"(+/-{model.receptive_field // 2 * halves[0]['stride'] / 25:.0f} s)")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * args.steps, pct_start=0.2)

    # Positive rows are rare and rarer still for the rare classes, so the BCE is weighted
    # per class by (negatives / positives), capped so Header (26 training events in the
    # whole set) cannot dominate the gradient.
    tgt = np.concatenate([h["target"] for h in halves])
    pos = tgt.sum(0)
    pos_weight = np.clip((tgt.shape[0] - pos) / np.maximum(pos, 1.0), 1.0, args.max_pos_weight)
    print("  per-class pos_weight: " +
          ", ".join(f"{l.split()[0]}={w:.0f}" for l, w in zip(BAS_LABELS, pos_weight)))
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))

    # MODEL SELECTION IS ON VALIDATION mAP, NOT VALIDATION LOSS. With pos_weight up to 50
    # the BCE is dominated by confident false positives on the rare classes and is not
    # monotone in average precision -- the first run's val loss rose from epoch 2 onwards
    # while the model was still improving as a detector. Selecting on the metric the paper
    # reports removes that mismatch. The decode setting used here is fixed and provisional;
    # the real one is grid-searched afterwards, also on validation.
    best = (-1.0, None, 0)
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        t0 = time.time()
        for _ in range(args.steps):
            f, t = sample_windows(halves, args.batch, args.window, rng,
                                  augment=not args.no_augment)
            f = torch.from_numpy(np.ascontiguousarray((f - mu) / sd))
            t = torch.from_numpy(t)
            opt.zero_grad()
            loss = lossf(model(f), t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss)
        vl = validation_loss(model, val_halves, mu, sd, lossf)
        vm = val_map(model, val_halves, mu, sd, args.periods_table,
                     floor=args.sel_floor, nms=args.sel_nms)[1]
        flag = ""
        if vm > best[0]:
            best = (vm, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep)
            flag = "  <- best"
        print(f"  epoch {ep:3}/{args.epochs}  train {tot/args.steps:.4f}  "
              f"val loss {vl:.4f}  val mAP@1s {vm:.4f}  {time.time()-t0:5.1f}s{flag}",
              flush=True)
    model.load_state_dict(best[1])
    print(f"  restored epoch {best[2]} (val mAP@1s {best[0]:.4f})")
    return model


def val_map(model, halves, mu, sd, periods, floor: float, nms: int,
            tols=(1, 5), preds=None) -> dict[int, float]:
    """Macro mAP over the validation halves at one decode setting."""
    if preds is None:
        preds = {(h["match"], h["half"]): predict_half(model, h, mu, sd) for h in halves}
    scores = {c: {t: ([], []) for t in tols} for c in range(12)}
    n_gt = {c: 0 for c in range(12)}
    n_pred = 0
    for h in halves:
        t0 = periods[h["match"]]["periods"][str(h["half"])]["t0_ms"]
        sp = decode(preds[(h["match"], h["half"])], h["frames"], t0, h["stride"], floor, nms)
        n_pred += len(sp)
        for c in range(12):
            gt = np.sort(h["ev_t_ms"][h["ev_class"] == c])
            n_gt[c] += gt.size
            pr = sorted([s for s in sp if s["cls"] == c], key=lambda s: -s["score"])
            pt = np.array([s["t_ms"] for s in pr])
            ps = [s["score"] for s in pr]
            for t in tols:
                fl = match_flags(pt, gt, t * 1000)
                scores[c][t][0].extend(ps)
                scores[c][t][1].extend(fl.tolist())
    out = {}
    for t in tols:
        aps = [ap_tolerant(scores[c][t][0], scores[c][t][1], n_gt[c])
               for c in range(12) if n_gt[c] > 0]
        out[t] = float(np.mean(aps)) if aps else 0.0
    out["n_pred"] = n_pred
    return out


@torch.no_grad()
def validation_loss(model, halves, mu, sd, lossf) -> float:
    model.eval()
    tot, n = 0.0, 0
    for h in halves:
        p = predict_half(model, h, mu, sd, logits=True)
        tot += float(lossf(torch.from_numpy(p)[None], torch.from_numpy(h["target"])[None]))
        n += 1
    return tot / max(n, 1)


@torch.no_grad()
def predict_half(model, half, mu, sd, chunk: int = 4096, pad: int = 256,
                 logits: bool = False) -> np.ndarray:
    """Dense per-row prediction over a whole half, in overlapping chunks.

    The chunks overlap by `pad` on each side and the overlap is discarded, so no row is
    scored with a truncated receptive field -- which would otherwise put a seam artefact
    every `chunk` rows and invent peaks there.
    """
    model.eval()
    X = ((half["feat"] - mu) / sd).astype(np.float32)
    T = X.shape[0]
    out = np.zeros((T, len(BAS_LABELS)), np.float32)
    s = 0
    while s < T:
        e = min(T, s + chunk)
        a, b = max(0, s - pad), min(T, e + pad)
        z = model(torch.from_numpy(X[a:b])[None])[0].numpy()
        out[s:e] = z[s - a:s - a + (e - s)]
        s = e
    return out if logits else 1.0 / (1.0 + np.exp(-out))


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def decode(prob: np.ndarray, frames: np.ndarray, t0_ms: float, stride: int,
           floor: float, nms_rows: int) -> list[dict]:
    """Peak-pick each class independently, then within-class NMS. Returns scored spots."""
    spots = []
    for c in range(prob.shape[1]):
        p = prob[:, c]
        # strict local maximum against the immediate neighbours
        cand = np.where((p[1:-1] >= p[:-2]) & (p[1:-1] > p[2:]) & (p[1:-1] >= floor))[0] + 1
        if cand.size == 0:
            continue
        order = cand[np.argsort(-p[cand])]
        taken = np.zeros(prob.shape[0], bool)
        for r in order:
            lo, hi = max(0, r - nms_rows), min(prob.shape[0], r + nms_rows + 1)
            if taken[lo:hi].any():
                continue
            taken[r] = True
            spots.append({"row": int(r), "cls": c, "score": float(p[r]),
                          "t_ms": int(round(t0_ms + (int(frames[r]) - 1) * 40.0))})
    return spots


def ms_to_clock(t_ms: int) -> str:
    s = max(0, t_ms) // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


def write_predictions(out_root: Path, match: str, spots_by_half: dict, periods: dict):
    """One SoccerNet-schema JSON per match, ranked by descending confidence."""
    actions = []
    for half, spots in spots_by_half.items():
        for s in spots:
            actions.append({
                "gameTime": f"{half} - {ms_to_clock(s['t_ms'])}",
                "label": BAS_LABELS[s["cls"]],
                "position": str(s["t_ms"]),
                "team": "left",
                "score": round(s["score"], 6),
            })
    actions.sort(key=lambda a: -a["score"])
    d = out_root / match
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{match}_12_class_events.json").write_text(
        json.dumps({"match_id": match, "fps": 25.0, "actions": actions}, indent=1))
    return len(actions)


# ---------------------------------------------------------------------------
# Tuning the decoder on validation
# ---------------------------------------------------------------------------

def tune_decoder(model, halves, mu, sd, periods, floors, nms_list) -> tuple[float, float, int]:
    """Grid-search the decoding floor and NMS radius on the VALIDATION halves only.

    Note on the floor: 11-point interpolated AP takes the maximum precision at each recall
    level, so appending lower-ranked predictions can only ever raise recall and can never
    lower the reported AP. The floor will therefore always be driven to its minimum. That
    is a property of the metric, not a trick -- the SoccerNet protocol places no cap on the
    number of predictions -- but it means the emitted spot count must be reported alongside
    the score, which format_report does.
    """
    preds = {(h["match"], h["half"]): predict_half(model, h, mu, sd) for h in halves}
    best = (-1.0, floors[0], nms_list[0])
    print(f'{"floor":>7} {"nms_rows":>9} {"mAP@1s":>8} {"mAP@5s":>8} {"n_pred":>9}')
    for fl in floors:
        for nms in nms_list:
            r = val_map(model, halves, mu, sd, periods, fl, nms, preds=preds)
            print(f'{fl:7.3f} {nms:9} {r[1]:8.4f} {r[5]:8.4f} {r["n_pred"]:9,}')
            if r[1] > best[0]:
                best = (r[1], fl, nms)
    print(f"  chosen on validation: floor={best[1]}, nms_rows={best[2]} "
          f"(mAP@1s {best[0]:.4f})")
    return best


def match_flags(pred_t: np.ndarray, gt_t: np.ndarray, tol_ms: int) -> np.ndarray:
    """1 for a true positive, 0 for a false positive, greedily in confidence order."""
    used = np.zeros(gt_t.size, bool)
    out = np.zeros(pred_t.size, np.int8)
    for i, t in enumerate(pred_t):
        if gt_t.size == 0:
            break
        d = np.abs(gt_t - t)
        d[used] = tol_ms + 1
        j = int(np.argmin(d)) if d.size else -1
        if j >= 0 and d[j] <= tol_ms:
            used[j] = True
            out[i] = 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="/data/share/SoccerTrack-v2/data/derived/bas/dataset")
    ap.add_argument("--periods", default="configs/bas_periods.json")
    ap.add_argument("--out", default="outputs/bas")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--max-pos-weight", type=float, default=50.0)
    ap.add_argument("--no-augment", action="store_true",
                    help="disable reflection augmentation (for the ablation)")
    ap.add_argument("--sel-floor", type=float, default=0.02,
                    help="provisional decode floor used for per-epoch model selection")
    ap.add_argument("--sel-nms", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    ds = Path(a.dataset)
    periods = json.loads(Path(a.periods).read_text())["matches"]
    a.periods_table = periods   # train() needs it for per-epoch validation mAP
    out_root = Path(a.out)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"train  {TRAIN}\nval    {VAL}\ntest   {TEST}  (scored once, never tuned on)\n")
    tr = load_split(ds, TRAIN)
    va = load_split(ds, VAL)
    mu, sd = fit_normaliser(tr)

    if a.eval_only:
        ck = torch.load(a.ckpt or (out_root / "spotter.pt"), map_location="cpu")
        model = TrajectorySpotter(n_features=tr[0]["feat"].shape[1], hidden=ck["hidden"])
        model.load_state_dict(ck["state"])
        mu, sd = ck["mu"], ck["sd"]
        floor, nms = ck["floor"], ck["nms_rows"]
    else:
        print("training")
        model = train(tr, mu, sd, a, va)
        print("\ntuning the decoder on the validation matches")
        _, floor, nms = tune_decoder(model, va, mu, sd, periods,
                                     floors=[0.02, 0.05, 0.10, 0.20, 0.30, 0.45],
                                     nms_list=[2, 5, 8, 12])
        torch.save({"state": model.state_dict(), "mu": mu, "sd": sd, "hidden": a.hidden,
                    "floor": floor, "nms_rows": nms}, out_root / "spotter.pt")
        print(f"  wrote {out_root/'spotter.pt'}")

    print("\nwriting predictions")
    for name, matches in (("val", VAL), ("test", TEST)):
        root = out_root / f"pred_{name}"
        for m in matches:
            halves = load_split(ds, [m])
            spots = {}
            for h in halves:
                t0 = periods[m]["periods"][str(h["half"])]["t0_ms"]
                spots[h["half"]] = decode(predict_half(model, h, mu, sd), h["frames"],
                                          t0, h["stride"], floor, nms)
            n = write_predictions(root, m, spots, periods)
            print(f"  {name} {m}: {n:,} spots -> {root/m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
