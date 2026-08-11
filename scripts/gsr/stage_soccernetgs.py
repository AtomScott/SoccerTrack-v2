"""Stage a SoccerTrack v2 half as a SoccerNetGS sequence TrackLab can read.

This is DATA STAGING ONLY -- it does not touch the GSR pipeline. The released
<match>_<half>.json files are already SoccerNet GameState, so nothing is converted;
we only (a) extract img1/*.jpg from the panorama video and (b) patch metadata that
is stale or collides between sequences.

Everything here follows the known-good precedent already on disk at
/data/share/SoccerNetGS/test/CLPD-117093, which was staged for the earlier 30s run:
  * frames come from the ORIGINAL panorama (4096x1080), not the calibrated render
  * images[].width/height are corrected to the real frame size
  * info.id is made unique (the released files all say "1")

Verified facts this relies on:
  * image_id "3{N:06d}" is 1-based frame N of <match>_panorama_<half>_half.mp4.
    Established by matching CLPD-117093's clip_start=969000ms (frame 24226) against
    production image_id "3024226": identical bbox_image for 21 of 22 tracks.
  * the released width/height (3840x1504) is STALE. bbox_image is really in
    4096x1080 -- byte-identical to the staged clip, x-ratio 1.0000.
  * the GT annotates ~25 frames PAST the end of the video, so seq_length must be
    clamped to the video's real frame count or TrackLab will request missing JPEGs.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

SPLIT_ID = {"train": 1, "valid": 2, "test": 3, "challenge": 4}
# Short 3-digit sequence ids keep image_id at SoccerNet's 10 characters
# (split_id + seq_id + 6-digit frame), which the scorer's fallback path expects.
SEQ_ID = {
    ("117092","1st"):"901", ("117092","2nd"):"902", ("117093","1st"):"903", ("117093","2nd"):"904",
    ("118575","1st"):"905", ("118575","2nd"):"906", ("118576","1st"):"907", ("118576","2nd"):"908",
    ("118577","1st"):"909", ("118577","2nd"):"910", ("118578","1st"):"911", ("118578","2nd"):"912",
    ("128057","1st"):"913", ("128057","2nd"):"914", ("128058","1st"):"915", ("128058","2nd"):"916",
    ("132831","1st"):"917", ("132831","2nd"):"918", ("132877","1st"):"919", ("132877","2nd"):"920",
}

def video_frames(p: Path) -> tuple[int,int,int]:
    out = subprocess.run(["ffprobe","-v","error","-select_streams","v:0",
        "-show_entries","stream=width,height,nb_frames","-of","csv=p=0",str(p)],
        capture_output=True, text=True, check=True).stdout.strip()
    w,h,n = out.split(",")[:3]
    return int(w), int(h), int(n)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/share/SoccerTrack-v2/data")
    ap.add_argument("--out-root", default="/mnt/storage/SoccerTrack-v2/SoccerNetGS")
    ap.add_argument("--match", required=True)
    ap.add_argument("--half", required=True, choices=["1st","2nd"])
    ap.add_argument("--split", default="test", choices=list(SPLIT_ID))
    ap.add_argument("--start", type=int, default=1, help="1-based first frame to stage")
    ap.add_argument("--nframes", type=int, default=-1, help="-1 = to end of video")
    ap.add_argument("--quality", type=int, default=2, help="ffmpeg -q:v (2 = high)")
    ap.add_argument("--skip-frames", action="store_true", help="labels only")
    a = ap.parse_args()

    data = Path(a.data)
    gt_path = data/"production"/"gsr"/a.match/f"{a.match}_{a.half}.json"
    vid = data/"interim"/a.match/f"{a.match}_panorama_{a.half}_half.mp4"
    for p in (gt_path, vid):
        if not p.exists():
            print(f"MISSING: {p}"); return 1

    vw, vh, vn = video_frames(vid)
    seq_id = SEQ_ID[(a.match, a.half)]
    split_id = SPLIT_ID[a.split]
    name = f"CLPD-{a.match}-{a.half}"
    dst = Path(a.out_root)/a.split/name
    (dst/"img1").mkdir(parents=True, exist_ok=True)

    print(f"{name}: video {vw}x{vh} {vn} frames", flush=True)
    print("loading GT...", flush=True)
    d = json.load(open(gt_path))
    n_img_gt = len(d["images"])
    declared = (d["images"][0].get("width"), d["images"][0].get("height"))

    start = a.start
    end = vn if a.nframes < 0 else min(vn, start + a.nframes - 1)
    end = min(end, n_img_gt)                      # clamp: GT runs past the video
    count = end - start + 1
    print(f"GT images {n_img_gt}  declared {declared[0]}x{declared[1]} (stale)  "
          f"-> staging frames {start}..{end} ({count})", flush=True)
    if n_img_gt > vn:
        print(f"  NOTE: GT annotates {n_img_gt - vn} frames beyond the video; trimmed.", flush=True)

    # Old image_id -> new. Frames are renumbered so the sequence starts at 000001.jpg.
    remap, images = {}, []
    for i in d["images"]:
        n = int(i["image_id"][1:])                # "3024226" -> 24226 (1-based frame)
        if not (start <= n <= end): continue
        k = n - start + 1
        new_id = f"{split_id}{seq_id}{k:06d}"
        remap[i["image_id"]] = new_id
        j = dict(i)
        j["image_id"] = new_id
        j["file_name"] = f"{k:06d}.jpg"
        j["width"], j["height"] = vw, vh         # the released value is stale
        images.append(j)

    anns = []
    for x in d["annotations"]:
        nid = remap.get(x.get("image_id"))
        if nid is None: continue
        y = dict(x); y["image_id"] = nid
        if "id" in y: y["id"] = f"{nid}{int(y.get('track_id') or 0):04d}"
        anns.append(y)

    info = dict(d["info"])
    info.update({"id": seq_id, "name": name, "seq_length": count,
                 "im_dir": "img1", "im_ext": ".jpg",
                 "clip_start": str((start-1)*40), "clip_stop": str(end*40)})
    out = {"info": info, "images": images, "annotations": anns, "categories": d["categories"]}
    f = dst/"Labels-GameState.json"
    with open(f,"w") as fh: json.dump(out, fh)
    print(f"wrote {f.name}: {len(images)} images, {len(anns)} annotations, "
          f"{f.stat().st_size/1e6:.0f} MB", flush=True)

    if a.skip_frames:
        print("--skip-frames: not extracting JPEGs"); return 0

    print(f"extracting {count} frames -> {dst/'img1'} ...", flush=True)
    cmd = ["ffmpeg","-v","error","-y"]
    if start > 1:
        cmd += ["-ss", f"{(start-1)/25.0:.6f}"]
    cmd += ["-i", str(vid), "-frames:v", str(count), "-q:v", str(a.quality),
            "-vsync","0","-start_number","1", str(dst/"img1"/"%06d.jpg")]
    subprocess.run(cmd, check=True)
    got = len(list((dst/"img1").glob("*.jpg")))
    size = sum(p.stat().st_size for p in (dst/"img1").glob("*.jpg"))/1e9
    print(f"extracted {got} JPEGs, {size:.1f} GB  ({1e3*size/max(got,1):.2f} MB/frame)")
    if got != count:
        print(f"  !! expected {count} frames, got {got}"); return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
