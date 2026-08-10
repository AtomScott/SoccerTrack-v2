"""BAS baseline: MViT-B clip features + a compact T-DEED-style spotter.

This module is a *real* (CPU-importable) ball-action-spotting pipeline for
SoccerTrack v2. It is organised as small, individually unit-testable functions
plus a thin ``argparse`` ``main``. All heavy imports (torch / torchvision /
video decoders) are deferred inside the functions that need them, so

    python -m baselines.bas.train --help

works with nothing but the stdlib + PyYAML installed.

Pipeline
--------
1. **Features** — ~2 s clips are encoded with **torchvision MViT-V1-B**
   (Kinetics-400 weights, 768-d pooled embedding). Clips are sampled with a
   ``stride_s`` hop, giving one feature vector per hop with a known centre
   timestamp (ms, per-half).
2. **Spotter** — a compact **T-DEED-style** temporal model: a small 1-D
   temporal-conv encoder over a sliding window of clip features followed by a
   per-clip classification head over the 12 BAS classes + background. This is a
   deliberately small re-implementation in the spirit of T-DEED
   (Xarles et al., CVPR-W 2024; https://github.com/arturxe2/T-DEED, MIT licence)
   — namely dense per-frame logits + local-maximum decoding (Soft-NMS-free,
   within-class non-max suppression) — *not* a vendored copy.
3. **Prediction** — per-clip class probabilities are decoded into events
   (within-class local-max + NMS), then written in the **SoccerNet BAS** JSON
   schema that :mod:`src.evaluation.bas_map` parses.

CRITICAL (bas-5): the SoccerNet BAS prediction schema has **no score field**, so
the *order* of the ``annotations`` list is the ranking consumed by mAP. We sort
every emitted event by **descending confidence** (see :func:`write_predictions`).
``src.evaluation.bas_map`` currently re-sorts predictions by time internally, so
this is harmless today and correct if/when the evaluator honours input order.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, Sequence

import yaml

if TYPE_CHECKING:  # import only for type checkers, never at runtime
    import numpy as np
    import torch

# The 12 BAS classes, in the canonical order defined by the dataset loader.
# We import lazily-friendly: this is a pure-python tuple with no heavy deps.
from src.data_utils.soccertrack_v2 import BAS_LABELS

# Per-class default decoding parameters. ``BACKGROUND`` is the implicit 0-th
# logit class used during training; it is never emitted as an event.
BACKGROUND: str = "__background__"
NUM_CLASSES: int = len(BAS_LABELS)  # 12
FPS: int = 25  # SoccerTrack v2 alignment frame rate (see docs/format-bas.md)


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------


@dataclass
class FeatureCfg:
    name: str = "mvit_b_16x4"
    clip_s: float = 2.0  # temporal extent of one clip (seconds)
    stride_s: float = 1.0  # hop between consecutive clip centres (seconds)
    num_frames: int = 16  # frames sampled per clip fed to MViT (T dim)
    crop_size: int = 224
    embed_dim: int = 768  # MViT-V1-B pooled feature dim


@dataclass
class SpotterCfg:
    name: str = "t-deed"
    hidden: int = 256
    window: int = 15  # number of consecutive clip-features per training window
    lr: float = 1.0e-3
    batch_size: int = 32
    epochs: int = 20
    weight_decay: float = 1.0e-4
    # Decoding
    nms_window_s: float = 1.0  # within-class NMS radius (seconds)
    # Drop peaks below this softmax prob. Over the 13-way softmax (12 classes +
    # background) the residual mass on near-flat foreground tracks sits around
    # ~0.03, so a 0.01 floor floods the output with spurious peaks. A higher
    # floor keeps only confident peaks; tune on the val set.
    min_confidence: float = 0.3
    # Require a foreground peak to also beat the background prob at that clip
    # before it is emitted (suppresses peaks on tracks the model thinks are
    # background). Set False to recover the old background-agnostic behaviour.
    require_above_background: bool = True
    # Optional cap on emitted peaks per class per half (0 = no cap).
    max_peaks_per_class: int = 0


@dataclass
class EvalCfg:
    tolerances_s: tuple[int, ...] = (1, 5)
    metric: str = "map"
    out: Path = Path("./outputs/bas_baseline/scores.json")
    pred_root: Path = Path("./outputs/bas_baseline/preds")


@dataclass
class DataCfg:
    root: Path = Path("./data")
    train_matches: list[str] = field(default_factory=list)
    val_matches: list[str] = field(default_factory=list)
    test_matches: list[str] = field(default_factory=list)


@dataclass
class BasConfig:
    data: DataCfg
    features: FeatureCfg
    spotter: SpotterCfg
    eval: EvalCfg

    @staticmethod
    def from_dict(cfg: dict) -> "BasConfig":
        d = cfg.get("data", {})
        f = cfg.get("features", {})
        s = cfg.get("spotter", {})
        e = cfg.get("eval", {})

        def _ids(key: str) -> list[str]:
            return [str(m) for m in d.get(key, [])]

        return BasConfig(
            data=DataCfg(
                root=Path(d.get("root", "./data")),
                train_matches=_ids("train_matches"),
                val_matches=_ids("val_matches"),
                test_matches=_ids("test_matches"),
            ),
            features=FeatureCfg(
                name=f.get("name", "mvit_b_16x4"),
                clip_s=float(f.get("clip_s", 2.0)),
                stride_s=float(f.get("stride_s", 1.0)),
                num_frames=int(f.get("num_frames", 16)),
                crop_size=int(f.get("crop_size", 224)),
                embed_dim=int(f.get("embed_dim", 768)),
            ),
            spotter=SpotterCfg(
                name=s.get("name", "t-deed"),
                hidden=int(s.get("hidden", 256)),
                window=int(s.get("window", 15)),
                lr=float(s.get("lr", 1.0e-3)),
                batch_size=int(s.get("batch_size", 32)),
                epochs=int(s.get("epochs", 20)),
                weight_decay=float(s.get("weight_decay", 1.0e-4)),
                nms_window_s=float(s.get("nms_window_s", 1.0)),
                min_confidence=float(s.get("min_confidence", 0.3)),
                require_above_background=bool(s.get("require_above_background", True)),
                max_peaks_per_class=int(s.get("max_peaks_per_class", 0)),
            ),
            eval=EvalCfg(
                tolerances_s=tuple(int(t) for t in e.get("tolerances_s", (1, 5))),
                metric=e.get("metric", "map"),
                out=Path(e.get("out", "./outputs/bas_baseline/scores.json")),
                pred_root=Path(e.get("pred_root", "./outputs/bas_baseline/preds")),
            ),
        )

    @staticmethod
    def load(path: Path) -> "BasConfig":
        return BasConfig.from_dict(yaml.safe_load(Path(path).read_text()))


# ----------------------------------------------------------------------------
# (1) MViT-B clip feature extraction
# ----------------------------------------------------------------------------


def build_mvit_extractor(cfg: FeatureCfg, pretrained: bool = True):
    """Build an MViT-V1-B feature extractor (classifier head stripped).

    Returns ``(model, transform)`` where ``model(clip)`` maps a clip batch of
    shape ``(B, C=3, T, H, W)`` to a pooled ``(B, embed_dim)`` feature, and
    ``transform`` is the matching torchvision ``VideoClassification`` transform.

    IMPORTANT layout note: the torchvision video transform consumes a
    ``(..., T, C, H, W)`` clip (it normalises over the ``C`` axis) and *returns*
    a ``(C, T, H, W)`` tensor ready for MViT. Therefore raw clips fed to the
    transform must be ``(T, C, H, W)`` — see :func:`decode_video_to_clips` and
    :func:`extract_features_from_clips`. Feeding ``(C, T, H, W)`` to the
    transform raises ``RuntimeError`` (size mismatch on the normalise step).

    Heavy imports are deferred to here so module import stays light.
    """
    import torch.nn as nn
    import torchvision.models.video as video_models

    weights = video_models.MViT_V1_B_Weights.KINETICS400_V1 if pretrained else None
    model = video_models.mvit_v1_b(weights=weights)
    # Strip the 400-way classifier; keep the 768-d pooled embedding.
    model.head = nn.Identity()
    model.eval()
    transform = (
        weights.transforms()
        if weights is not None
        else video_models.MViT_V1_B_Weights.KINETICS400_V1.transforms()
    )
    return model, transform


def clip_center_times_ms(
    num_clips: int, stride_s: float, clip_s: float, offset_ms: int = 0
) -> list[int]:
    """Centre timestamp (ms, per-half) of each sampled clip.

    Clip ``i`` spans ``[i*stride, i*stride + clip_s)`` seconds; its centre is
    ``i*stride + clip_s/2``. ``offset_ms`` shifts the origin (kept at 0 for
    per-half kickoff alignment).
    """
    half = clip_s / 2.0
    return [int(round((i * stride_s + half) * 1000)) + offset_ms for i in range(num_clips)]


def num_clips_for_duration(duration_s: float, stride_s: float, clip_s: float) -> int:
    """How many clips of length ``clip_s`` fit at hop ``stride_s`` in a half.

    The last clip must fully fit inside ``duration_s``.
    """
    if duration_s < clip_s:
        return 0
    return int(math.floor((duration_s - clip_s) / stride_s)) + 1


def extract_features_from_clips(model, transform, clips) -> "np.ndarray":
    """Encode an iterable/tensor of raw clips into an ``(N, embed_dim)`` array.

    ``clips`` may be a single tensor ``(N, T, C, H, W)`` or an iterable of
    per-clip tensors ``(T, C, H, W)``. The torchvision ``VideoClassification``
    ``transform`` is applied per clip; it consumes ``(T, C, H, W)`` and yields
    ``(C, T, H, W)``, which ``model`` then maps to ``(embed_dim,)``. Runs under
    ``torch.no_grad`` on whatever device ``model`` lives on.
    """
    import numpy as np
    import torch

    device = next(model.parameters()).device
    if isinstance(clips, torch.Tensor) and clips.dim() == 5:
        clip_list = [clips[i] for i in range(clips.shape[0])]
    else:
        clip_list = list(clips)
    if not clip_list:
        return np.zeros((0, _model_embed_dim(model)), dtype=np.float32)

    feats: list["np.ndarray"] = []
    model.eval()
    with torch.no_grad():
        for clip in clip_list:
            x = transform(clip) if transform is not None else clip
            if x.dim() == 4:
                x = x.unsqueeze(0)  # -> (1, C, T, H, W)
            x = x.to(device)
            out = model(x)  # (1, embed_dim)
            feats.append(out.squeeze(0).cpu().numpy().astype(np.float32))
    return np.stack(feats, axis=0)


def _model_embed_dim(model) -> int:
    # MViT-V1-B pooled embedding is 768; fall back gracefully.
    return getattr(model, "_bas_embed_dim", 768)


def decode_video_to_clips(
    video_path: Path, cfg: FeatureCfg
) -> tuple["torch.Tensor", list[int]]:
    """Decode a half video into ``(N, T, C, H, W)`` raw clips + centre times (ms).

    Clips are laid out ``(N, T, C, H, W)`` (per-clip ``(T, C, H, W)``) because
    that is exactly what the torchvision ``VideoClassification`` transform in
    :func:`build_mvit_extractor` consumes; the transform then emits ``(C, T, H,
    W)`` for MViT. (Feeding ``(C, T, H, W)`` to the transform raises a size
    mismatch on its per-channel normalise step.)

    Uses OpenCV (already a project dependency) to read frames. This touches the
    filesystem and is *not* exercised by the CPU unit tests; it is the glue used
    on a real machine with the dataset present.
    """
    import cv2
    import numpy as np
    import torch

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = n_frames / fps if fps else 0.0

    n_clips = num_clips_for_duration(duration_s, cfg.stride_s, cfg.clip_s)
    centers = clip_center_times_ms(n_clips, cfg.stride_s, cfg.clip_s)

    clips: list["np.ndarray"] = []
    clip_frames = int(round(cfg.clip_s * fps))
    for i in range(n_clips):
        start_f = int(round(i * cfg.stride_s * fps))
        # Evenly sample cfg.num_frames frames from this clip's span.
        idxs = np.linspace(start_f, start_f + clip_frames - 1, cfg.num_frames).round().astype(int)
        frames: list["np.ndarray"] = []
        for fi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok:
                frame = np.zeros((cfg.crop_size, cfg.crop_size, 3), dtype=np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        # (T, H, W, C) -> (T, C, H, W) — the layout the torchvision transform wants.
        clip = torch.from_numpy(np.stack(frames, axis=0)).permute(0, 3, 1, 2).contiguous()
        clips.append(clip.numpy())
    cap.release()
    if not clips:
        return torch.zeros((0, cfg.num_frames, 3, cfg.crop_size, cfg.crop_size)), centers
    tensor = torch.from_numpy(np.stack(clips, axis=0))
    return tensor, centers


# ----------------------------------------------------------------------------
# (2) Compact T-DEED-style spotter
# ----------------------------------------------------------------------------


def build_spotter(feat_cfg: FeatureCfg, spot_cfg: SpotterCfg):
    """Construct the compact T-DEED-style spotter module (deferred torch import)."""
    import torch.nn as nn

    class TDEEDSpotter(nn.Module):
        """Per-clip dense classifier over BAS classes + background.

        Architecture (compact, in the spirit of T-DEED):
          - input projection of MViT features to ``hidden`` dims,
          - a small stack of 1-D temporal residual conv blocks (captures local
            temporal context across neighbouring clips),
          - a linear head producing ``NUM_CLASSES + 1`` logits per clip (index 0
            is background).

        ``forward`` accepts ``(B, T, embed_dim)`` and returns ``(B, T, C+1)``
        logits, where ``T`` is the number of clip-features in the window.
        """

        def __init__(self, embed_dim: int, hidden: int, n_classes: int):
            super().__init__()
            self.n_classes = n_classes
            self.proj = nn.Linear(embed_dim, hidden)
            self.act = nn.GELU()
            self.blocks = nn.ModuleList(
                [self._temporal_block(hidden) for _ in range(2)]
            )
            self.norm = nn.LayerNorm(hidden)
            self.head = nn.Linear(hidden, n_classes + 1)  # +1 background

        @staticmethod
        def _temporal_block(hidden: int) -> nn.Module:
            return nn.Sequential(
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            )

        def forward(self, x):  # x: (B, T, embed_dim)
            h = self.act(self.proj(x))  # (B, T, hidden)
            ht = h.transpose(1, 2)  # (B, hidden, T) for Conv1d
            for blk in self.blocks:
                ht = ht + blk(ht)  # residual temporal conv
            h = ht.transpose(1, 2)  # (B, T, hidden)
            h = self.norm(h)
            return self.head(h)  # (B, T, n_classes + 1)

    model = TDEEDSpotter(feat_cfg.embed_dim, spot_cfg.hidden, NUM_CLASSES)
    return model


def labels_to_clip_targets(
    events: Sequence,
    centers_ms: Sequence[int],
    half: int,
    tol_ms: int = 500,
) -> "np.ndarray":
    """Build per-clip integer class targets for one half.

    Each clip centre is assigned the class of the nearest same-half event within
    ``tol_ms``; clips with no nearby event are background (0). Class indices are
    ``label_index + 1`` (0 reserved for background). On ties / overlaps the
    closest event wins; dual events (same position, e.g. Shot+Goal) are handled
    by :func:`decode_predictions` at inference, not here (a single soft target
    per clip is sufficient for the compact baseline).
    """
    import numpy as np

    targets = np.zeros(len(centers_ms), dtype=np.int64)
    half_events = [e for e in events if getattr(e, "half", None) == half]
    for ci, c in enumerate(centers_ms):
        best_cls = 0
        best_dt = tol_ms + 1
        for e in half_events:
            dt = abs(int(e.t_ms) - int(c))
            if dt <= tol_ms and dt < best_dt:
                best_dt = dt
                best_cls = label_index(e.label) + 1
        targets[ci] = best_cls
    return targets


def label_index(label: str) -> int:
    """Index of ``label`` within :data:`BAS_LABELS` (raises on unknown)."""
    return BAS_LABELS.index(label)


def make_windows(
    features: "np.ndarray", targets: "np.ndarray", window: int, stride: int = 1
) -> tuple["np.ndarray", "np.ndarray"]:
    """Slice ``(N, D)`` features + ``(N,)`` targets into windows of length ``window``.

    Returns ``(W, window, D)`` feature windows and ``(W, window)`` target windows.
    Sequences shorter than ``window`` are right-padded with zeros (features) and
    background (targets), yielding a single window so short halves still train.
    """
    import numpy as np

    n, d = features.shape
    if n < window:
        pad_f = np.zeros((window - n, d), dtype=features.dtype)
        pad_t = np.zeros((window - n,), dtype=targets.dtype)
        return (
            np.concatenate([features, pad_f], axis=0)[None, ...],
            np.concatenate([targets, pad_t], axis=0)[None, ...],
        )
    starts = list(range(0, n - window + 1, max(1, stride)))
    if starts[-1] != n - window:
        starts.append(n - window)
    fw = np.stack([features[s : s + window] for s in starts], axis=0)
    tw = np.stack([targets[s : s + window] for s in starts], axis=0)
    return fw, tw


def train_spotter(
    model,
    feature_windows: "np.ndarray",
    target_windows: "np.ndarray",
    spot_cfg: SpotterCfg,
    device: str = "cpu",
    verbose: bool = False,
):
    """Train the spotter with cross-entropy over per-clip targets.

    Plain PyTorch loop (no Lightning) so it runs unchanged on CPU for the tiny
    synthetic tests. ``feature_windows`` is ``(W, T, D)``, ``target_windows`` is
    ``(W, T)`` of integer class ids. Returns the trained model.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    model = model.to(device)
    model.train()
    opt = torch.optim.AdamW(
        model.parameters(), lr=spot_cfg.lr, weight_decay=spot_cfg.weight_decay
    )

    feats = torch.as_tensor(np.asarray(feature_windows), dtype=torch.float32)
    tgts = torch.as_tensor(np.asarray(target_windows), dtype=torch.long)
    n = feats.shape[0]

    # Class weights: down-weight background which dominates the per-clip targets.
    counts = torch.bincount(tgts.reshape(-1), minlength=NUM_CLASSES + 1).float()
    weights = 1.0 / torch.clamp(counts, min=1.0)
    weights = weights / weights.sum() * (NUM_CLASSES + 1)
    weights = weights.to(device)

    bs = max(1, spot_cfg.batch_size)
    for epoch in range(spot_cfg.epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            xb = feats[idx].to(device)  # (b, T, D)
            yb = tgts[idx].to(device)  # (b, T)
            logits = model(xb)  # (b, T, C+1)
            loss = F.cross_entropy(
                logits.reshape(-1, NUM_CLASSES + 1), yb.reshape(-1), weight=weights
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * xb.shape[0]
        if verbose:
            print(f"[bas] epoch {epoch + 1}/{spot_cfg.epochs} loss={total / max(1, n):.4f}")
    model.eval()
    return model


def predict_clip_probs(model, features: "np.ndarray", device: str = "cpu") -> "np.ndarray":
    """Run the spotter over a full half's features → ``(N, C+1)`` probabilities.

    The whole half is fed as a single sequence (``B=1``) so the temporal convs
    see global context, then softmax over the class dimension.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    if features.shape[0] == 0:
        return np.zeros((0, NUM_CLASSES + 1), dtype=np.float32)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
        logits = model(x)  # (1, N, C+1)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy().astype(np.float32)
    return probs


# ----------------------------------------------------------------------------
# (3) Decode → events → SoccerNet JSON
# ----------------------------------------------------------------------------


@dataclass
class PredEvent:
    """An emitted prediction prior to JSON serialisation."""

    half: int
    t_ms: int
    label: str
    confidence: float


def decode_predictions(
    probs: "np.ndarray",
    centers_ms: Sequence[int],
    half: int,
    nms_window_s: float = 1.0,
    min_confidence: float = 0.3,
    require_above_background: bool = True,
    max_peaks_per_class: int = 0,
) -> list[PredEvent]:
    """Decode per-clip class probabilities into events for one half.

    For each foreground class independently:
      1. take that class's probability track over clips,
      2. keep clips that are a **local maximum** (T-DEED-style peak picking),
      3. drop peaks below ``min_confidence``,
      4. (default) drop peaks whose foreground prob does not exceed the
         background prob at that clip — the model considers that clip background,
      5. apply within-class NMS so no two kept peaks are closer than
         ``nms_window_s`` (keep the higher-confidence one),
      6. (optional) cap the number of kept peaks per class to
         ``max_peaks_per_class`` (0 = no cap), keeping the most confident.

    Over a 13-way softmax (12 classes + background) the residual probability
    mass on near-flat foreground tracks sits around ~0.03, so a tiny
    ``min_confidence`` floor (e.g. 0.01) floods the output with spurious peaks
    on every class and tanks precision/mAP. The defaults here (raised floor +
    background-dominance guard) keep only confident peaks; tune on the val set.

    Returns events with their softmax confidence (used purely for ranking — the
    SoccerNet schema stores no score). Classes are decoded independently so a
    dual event (e.g. Shot + Goal at the same instant) can be emitted from the
    same clip on different class tracks.
    """
    out: list[PredEvent] = []
    if probs.shape[0] == 0:
        return out
    n = probs.shape[0]
    nms_ms = int(round(nms_window_s * 1000))
    centers = list(centers_ms)
    bg = probs[:, 0]  # background prob per clip

    for cls in range(NUM_CLASSES):
        track = probs[:, cls + 1]  # +1 to skip background
        # Local maxima (strict on the rising side, >= on falling to keep plateaus once).
        peaks: list[tuple[float, int]] = []  # (confidence, clip_index)
        for i in range(n):
            left = track[i - 1] if i > 0 else -1.0
            right = track[i + 1] if i + 1 < n else -1.0
            if not (track[i] >= left and track[i] >= right):
                continue
            if track[i] < min_confidence:
                continue
            if require_above_background and not (track[i] > bg[i]):
                continue
            peaks.append((float(track[i]), i))
        if not peaks:
            continue
        # NMS within class: greedily keep highest-confidence peaks, suppress
        # neighbours within nms_ms.
        peaks.sort(key=lambda p: p[0], reverse=True)
        kept_times: list[int] = []
        kept_for_class = 0
        for conf, i in peaks:
            if max_peaks_per_class and kept_for_class >= max_peaks_per_class:
                break
            t = centers[i]
            if all(abs(t - kt) > nms_ms for kt in kept_times):
                kept_times.append(t)
                kept_for_class += 1
                out.append(
                    PredEvent(half=half, t_ms=int(t), label=BAS_LABELS[cls], confidence=conf)
                )
    return out


def events_to_soccernet(
    events: Iterable[PredEvent], match_id: str
) -> dict:
    """Serialise predicted events into the SoccerNet BAS JSON object.

    Output matches exactly what :func:`src.data_utils.soccertrack_v2._parse_bas`
    reads: a top-level object with ``UrlLocal`` / ``UrlYoutube`` / ``annotations``,
    each annotation carrying ``gameTime`` (``"<half> - <mm:ss>"``), ``position``
    (ms-since-half-kickoff, as a *string of an integer*), ``label``, and ``team``.

    bas-5: events are emitted **sorted by descending confidence** so the list
    order is the mAP ranking.
    """
    ranked = sorted(events, key=lambda e: e.confidence, reverse=True)
    annotations = []
    for e in ranked:
        annotations.append(
            {
                "gameTime": f"{e.half} - {_ms_to_clock(e.t_ms)}",
                "position": str(int(e.t_ms)),  # string of integer ms (per spec)
                "label": e.label,
                "team": None,  # baseline does not predict team
                "confidence": round(float(e.confidence), 6),  # ignored by parser; debugging aid
            }
        )
    return {"UrlLocal": str(match_id), "UrlYoutube": None, "annotations": annotations}


def _ms_to_clock(t_ms: int) -> str:
    """Convert ms-since-half-kickoff to ``"mm:ss"`` within the half."""
    total_s = int(round(t_ms / 1000))
    return f"{total_s // 60:02d}:{total_s % 60:02d}"


def write_predictions(
    events: Iterable[PredEvent], pred_root: Path, match_id: str
) -> Path:
    """Write ``<pred_root>/<mid>/<mid>_12_class_events.json`` and return its path."""
    pred_root = Path(pred_root)
    out_dir = pred_root / str(match_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{match_id}_12_class_events.json"
    obj = events_to_soccernet(events, match_id)
    out_file.write_text(json.dumps(obj, indent=2))
    return out_file


# ----------------------------------------------------------------------------
# Orchestration helpers
# ----------------------------------------------------------------------------


def _half_video_paths(data_root: Path, match_id: str) -> list[tuple[int, Path]]:
    """Best-effort discovery of per-half video files for a match.

    The dataset layout for videos is not pinned in this repo yet; we probe a few
    plausible names under ``<root>/videos/<mid>/``. Returns ``(half, path)`` only
    for files that exist. Empty if none found (so the caller degrades to writing
    empty predictions rather than crashing).
    """
    vid_dir = data_root / "videos" / str(match_id)
    found: list[tuple[int, Path]] = []
    candidates = {
        1: [f"{match_id}_1st.mp4", f"{match_id}_1.mp4", "1.mp4", "1st_half.mp4"],
        2: [f"{match_id}_2nd.mp4", f"{match_id}_2.mp4", "2.mp4", "2nd_half.mp4"],
    }
    for half, names in candidates.items():
        for name in names:
            p = vid_dir / name
            if p.exists():
                found.append((half, p))
                break
    return found


def extract_match_features(
    extractor, transform, data_root: Path, match_id: str, feat_cfg: FeatureCfg
) -> dict[int, tuple["np.ndarray", list[int]]]:
    """Extract MViT features for each available half of a match.

    Returns ``{half: (features (N,D), centers_ms [N])}``. Halves without a video
    file are skipped. Requires the dataset + torch; not exercised by CPU tests.
    """
    out: dict[int, tuple["np.ndarray", list[int]]] = {}
    for half, vpath in _half_video_paths(Path(data_root), match_id):
        clips, centers = decode_video_to_clips(vpath, feat_cfg)
        feats = extract_features_from_clips(extractor, transform, clips)
        out[half] = (feats, centers)
    return out


def run_training(cfg: BasConfig, verbose: bool = True):
    """Full training run: extract features for train matches, build per-clip
    targets, and fit the spotter. Returns the trained spotter model.

    Requires the dataset + a torch install. Raises ``FileNotFoundError`` if no
    train videos are discoverable (so failures are loud, not silent).
    """
    import numpy as np

    from src.data_utils.soccertrack_v2 import load_match

    extractor, transform = build_mvit_extractor(cfg.features, pretrained=True)
    spotter = build_spotter(cfg.features, cfg.spotter)

    all_fw: list["np.ndarray"] = []
    all_tw: list["np.ndarray"] = []
    found_any = False
    for mid in cfg.data.train_matches:
        per_half = extract_match_features(
            extractor, transform, cfg.data.root, mid, cfg.features
        )
        if not per_half:
            if verbose:
                print(f"[bas] WARNING: no videos found for train match {mid}; skipping")
            continue
        found_any = True
        match = load_match(cfg.data.root, mid)
        events = match.bas_events()
        for half, (feats, centers) in per_half.items():
            targets = labels_to_clip_targets(events, centers, half)
            fw, tw = make_windows(feats, targets, cfg.spotter.window)
            all_fw.append(fw)
            all_tw.append(tw)

    if not found_any:
        raise FileNotFoundError(
            f"No train videos found under {cfg.data.root}/videos/. Cannot train."
        )

    feature_windows = np.concatenate(all_fw, axis=0)
    target_windows = np.concatenate(all_tw, axis=0)
    spotter = train_spotter(
        spotter, feature_windows, target_windows, cfg.spotter, verbose=verbose
    )
    return spotter, extractor, transform


def run_inference(
    cfg: BasConfig, spotter, extractor, transform, match_ids: Sequence[str], verbose: bool = True
) -> list[Path]:
    """Predict + write SoccerNet JSON for each ``match_id``. Returns written paths.

    Matches with no discoverable video get an empty (but well-formed) prediction
    file so the evaluator can still run over the full test set.
    """
    written: list[Path] = []
    for mid in match_ids:
        per_half = extract_match_features(
            extractor, transform, cfg.data.root, mid, cfg.features
        )
        events: list[PredEvent] = []
        for half, (feats, centers) in per_half.items():
            probs = predict_clip_probs(spotter, feats)
            events.extend(
                decode_predictions(
                    probs,
                    centers,
                    half,
                    nms_window_s=cfg.spotter.nms_window_s,
                    min_confidence=cfg.spotter.min_confidence,
                    require_above_background=cfg.spotter.require_above_background,
                    max_peaks_per_class=cfg.spotter.max_peaks_per_class,
                )
            )
        path = write_predictions(events, cfg.eval.pred_root, mid)
        written.append(path)
        if verbose:
            print(f"[bas] wrote {len(events)} events -> {path}")
    return written


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="BAS baseline: MViT-B features + compact T-DEED-style spotter."
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to config.yaml")
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Skip training; only run inference (requires a checkpoint path via --ckpt).",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Optional spotter checkpoint to load/save (state_dict).",
    )
    args = parser.parse_args(argv)

    cfg = BasConfig.load(args.config)
    print(f"[bas] features:      {cfg.features.name} (clip={cfg.features.clip_s}s, stride={cfg.features.stride_s}s)")
    print(f"[bas] spotter:       {cfg.spotter.name} (hidden={cfg.spotter.hidden}, window={cfg.spotter.window})")
    print(f"[bas] train matches: {cfg.data.train_matches}")
    print(f"[bas] test matches:  {cfg.data.test_matches}")
    print(f"[bas] pred_root:     {cfg.eval.pred_root}")

    import torch

    if args.predict_only:
        if args.ckpt is None or not args.ckpt.exists():
            raise SystemExit("--predict-only requires an existing --ckpt to load the spotter.")
        extractor, transform = build_mvit_extractor(cfg.features, pretrained=True)
        spotter = build_spotter(cfg.features, cfg.spotter)
        spotter.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
        spotter.eval()
    else:
        spotter, extractor, transform = run_training(cfg, verbose=True)
        if args.ckpt is not None:
            args.ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(spotter.state_dict(), args.ckpt)
            print(f"[bas] saved spotter -> {args.ckpt}")

    run_inference(cfg, spotter, extractor, transform, cfg.data.test_matches, verbose=True)
    print("[bas] done. Score with: python -m baselines.bas.eval --config <config>")


if __name__ == "__main__":
    main()
