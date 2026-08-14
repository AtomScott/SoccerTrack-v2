"""Stage nested-prefix sequences of one half, to test how accuracy varies with LENGTH.

The hypothesis being tested: GSR accuracy degrades as a sequence gets longer. The mechanism
is specific -- most of the pipeline is causal (detector, pose, reid, BoT-SORT) and cannot be
affected by future frames, but FOUR modules are global over the whole video:
    GTALink              global tracklet association
    MajorityVoteTracklet per-tracklet role/jersey vote
    TrackletTeamClustering   k-means over ALL tracklet embeddings
    TrackletTeamSideLabeling mean pitch position per team
More minutes means more tracklets, so global association and clustering get harder. That
predicts AssA and the attribute accuracies degrade with length while LocA stays flat.

ZERO EXTRA DISK: every prefix points img1 at the full sequence's img1 via symlink and only
ships a truncated Labels-GameState.json. TrackLab builds each frame path as
    <video_dir>/<info.im_dir>/<images[i].file_name>
so a sequence that declares N images only ever touches the first N JPEGs.

Each prefix gets a unique info.id and info.name, because the released files all say id "1"
and would collide (see docs/format-gsr.md defects section).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

# label -> (frames, sequence id). 25 fps.
PREFIXES = [
    ("30s",   750,   "921"),
    ("1min",  1500,  "922"),
    ("2min",  3000,  "926"),
    ("5min",  7500,  "923"),
    ("15min", 22500, "924"),
    ("30min", 45000, "925"),
]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/storage/SoccerTrack-v2/SoccerNetGS")
    ap.add_argument("--split", default="test")
    ap.add_argument("--source", default="CLPD-128057-1st",
                    help="already-staged full sequence to take prefixes of")
    ap.add_argument("--only", nargs="*", default=None, help="subset of labels to stage")
    a = ap.parse_args()

    src_dir = Path(a.root) / a.split / a.source
    labels = src_dir / "Labels-GameState.json"
    if not labels.exists():
        print(f"MISSING {labels}"); return 1
    print(f"loading {labels.name} ({labels.stat().st_size/1e6:.0f} MB) ...", flush=True)
    d = json.load(open(labels))
    images_all = sorted(d["images"], key=lambda im: int(str(im["image_id"]).split("_")[-1]))
    print(f"source: {len(images_all)} images, {len(d['annotations'])} annotations", flush=True)

    by_image = {}
    for x in d["annotations"]:
        by_image.setdefault(x["image_id"], []).append(x)

    print(f"\n{'label':7} {'frames':>7} {'seq_id':>7} {'anns':>10}  dir")
    print("-" * 76)
    for label, n, seq_id in PREFIXES:
        if a.only and label not in a.only:
            continue
        if n > len(images_all):
            print(f"{label:7} {n:>7} -- source has only {len(images_all)} frames, skipped")
            continue
        name = f"{a.source}-{label}"
        dst = Path(a.root) / a.split / name
        dst.mkdir(parents=True, exist_ok=True)

        link = dst / "img1"
        if link.is_symlink() or link.exists():
            if not link.is_symlink():
                print(f"{label}: {link} exists and is not a symlink, refusing"); continue
        else:
            link.symlink_to(src_dir / "img1")

        imgs = [dict(im) for im in images_all[:n]]
        anns = []
        for im in imgs:
            anns.extend(by_image.get(im["image_id"], []))
        info = dict(d["info"])
        info.update({"id": seq_id, "name": name, "seq_length": n,
                     "clip_start": "0", "clip_stop": str(n * 40)})
        out = {"info": info, "images": imgs, "annotations": anns,
               "categories": d["categories"]}
        f = dst / "Labels-GameState.json"
        with open(f, "w") as fh:
            json.dump(out, fh)
        print(f"{label:7} {n:>7} {seq_id:>7} {len(anns):>10}  {dst.name} "
              f"({f.stat().st_size/1e6:.0f} MB labels, img1 -> symlink)")
    print("\nDisk added for frames: 0 bytes (img1 is a symlink in every prefix).")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
