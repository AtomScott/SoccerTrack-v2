"""GS-HOTA plumbing test: scoring ground truth against itself must give HOTA == 1.0.

This is the only test that can prove the SoccerNetGS wiring is correct without any
predictions. If the layout, the config keys, or the identity attributes are wrong, a
perfect "prediction" will not score 1.0.

The fixture is a real slice of a released production GSR file rather than synthetic data,
so it exercises the actual schema. Slicing keeps it fast: the full files are ~2.7 GB per
half, of which a couple of hundred frames is plenty to validate plumbing.

Evaluation is PITCH SPACE ONLY. SoccerTrack v2 has no ground-truth detections, so
bounding boxes are never read: EVAL_SPACE stays 'pitch' (the scorer's default) and the
similarity comes from bbox_pitch's bottom-middle point under a gaussian at
EVAL_DIST_TOL=5 m.

Run:
    .venv/bin/python -m pytest tests/test_gs_hota_identity.py -q -s
    .venv/bin/python tests/test_gs_hota_identity.py          # standalone, prints detail
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DATA = Path(os.environ.get("DATA", "/data/share/SoccerTrack-v2/data"))
MATCH = os.environ.get("GS_TEST_MATCH", "117093")
HALF = "1st"
N_FRAMES = int(os.environ.get("GS_TEST_FRAMES", "150"))


def _slice_production_gsr(path: Path, n_frames: int):
    """Stream the first *n_frames* worth of images+annotations out of a huge GSR file.

    Brace-balanced scanning, never json.load on the whole file -- these are gigabytes.
    """
    text_images: list[dict] = []
    text_anns: list[dict] = []
    categories: list[dict] = []

    with open(path, "rb") as fh:
        head = fh.read(64_000_000).decode("utf8", "replace")

    def objects_after(marker: str, src: str, limit: int | None = None):
        idx = src.find(marker)
        if idx < 0:
            return []
        out, depth, start = [], 0, None
        for i in range(src.find("[", idx), len(src)):
            ch = src[i]
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        out.append(json.loads(src[start:i + 1]))
                    except Exception:
                        pass
                    start = None
                    if limit and len(out) >= limit:
                        break
            elif ch == "]" and depth == 0:
                break
        return out

    text_images = objects_after('"images"', head, limit=n_frames)
    keep_ids = {im["image_id"] for im in text_images}

    # annotations sit after images; scan from there and keep only the frames we kept
    ann_idx = head.find('"annotations"')
    if ann_idx >= 0:
        for ann in objects_after('"annotations"', head[ann_idx:], limit=n_frames * 40):
            if ann.get("image_id") in keep_ids and ann.get("supercategory") == "object":
                text_anns.append(ann)

    # categories live at the tail of the file
    with open(path, "rb") as fh:
        fh.seek(max(0, path.stat().st_size - 8000))
        tail = fh.read().decode("utf8", "replace")
    categories = objects_after('"categories"', tail)

    return text_images, text_anns, categories


def build_fixture(tmp: Path):
    src = DATA / "production" / "gsr" / MATCH / f"{MATCH}_{HALF}.json"
    if not src.exists():
        raise FileNotFoundError(src)
    images, anns, cats = _slice_production_gsr(src, N_FRAMES)
    if not images or not anns:
        raise AssertionError(f"fixture slice empty: {len(images)} images, {len(anns)} annotations")

    seq = f"{MATCH}-{HALF}"
    gt_root = tmp / "gt"
    trk_root = tmp / "trackers"
    (gt_root / seq).mkdir(parents=True, exist_ok=True)
    (trk_root / "identity" / "data").mkdir(parents=True, exist_ok=True)

    info = {"version": "1.3", "name": seq, "source": "soccertrack-v2-test"}
    gt = {"info": info, "images": images, "annotations": anns, "categories": cats}
    (gt_root / seq / "Labels-GameState.json").write_text(json.dumps(gt))

    # The scorer reads GT under "annotations" and predictions under "predictions".
    # A perfect tracker is literally the same annotations under the other key.
    pred = {"info": info, "images": images, "predictions": anns, "categories": cats}
    (trk_root / "identity" / "data" / f"{seq}.json").write_text(json.dumps(pred))
    return seq, gt_root, trk_root, len(images), len(anns)


def score(seq: str, gt_root: Path, trk_root: Path) -> dict:
    import trackeval as te
    from trackeval.datasets import SoccerNetGS

    eval_cfg = te.Evaluator.get_default_eval_config()
    eval_cfg.update({"PRINT_CONFIG": False, "TIME_PROGRESS": False,
                     "DISPLAY_LESS_PROGRESS": True, "PRINT_RESULTS": False,
                     "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False,
                     "PLOT_CURVES": False, "USE_PARALLEL": False})
    ds_cfg = SoccerNetGS.get_default_dataset_config()
    ds_cfg.update({
        "GT_FOLDER": str(gt_root),
        "TRACKERS_FOLDER": str(trk_root),
        "TRACKERS_TO_EVAL": ["identity"],
        "TRACKER_SUB_FOLDER": "data",
        "SKIP_SPLIT_FOL": True,       # our layout is <gt>/<seq>/..., not <gt>/<split>/<seq>/...
        "SEQ_INFO": {seq: None},      # discovery would look under <GT_FOLDER>/<split> otherwise
        "OUTPUT_FOLDER": str(gt_root.parent / "out"),
        "PRINT_CONFIG": False,
        "EVAL_SPACE": "pitch",        # NO bounding boxes: SoccerTrack v2 has no GT detections
    })
    evaluator = te.Evaluator(eval_cfg)
    results, _ = evaluator.evaluate([SoccerNetGS(ds_cfg)], [te.metrics.HOTA()])
    return results


def extract_hota(results: dict) -> float:
    """Pull the aggregate HOTA out of TrackEval's nested result dict."""
    import numpy as np

    def walk(node):
        if isinstance(node, dict):
            if "HOTA" in node and not isinstance(node["HOTA"], dict):
                v = node["HOTA"]
                arr = np.atleast_1d(np.asarray(v, dtype=float))
                yield float(np.mean(arr))
            for v in node.values():
                yield from walk(v)
    vals = list(walk(results))
    if not vals:
        raise AssertionError("no HOTA value found in results")
    return max(vals)


def test_gt_against_itself_scores_one(tmp_path=None):
    import tempfile
    tmp = Path(tmp_path) if tmp_path else Path(tempfile.mkdtemp(prefix="gshota-"))
    seq, gt_root, trk_root, n_img, n_ann = build_fixture(tmp)
    print(f"fixture: {seq}  {n_img} frames, {n_ann} annotations  ->  {tmp}")
    results = score(seq, gt_root, trk_root)
    hota = extract_hota(results)
    print(f"HOTA (ground truth vs itself) = {hota:.6f}")
    assert hota > 0.999, f"expected ~1.0 scoring GT against itself, got {hota}"


if __name__ == "__main__":
    test_gt_against_itself_scores_one()
    print("PASS")
