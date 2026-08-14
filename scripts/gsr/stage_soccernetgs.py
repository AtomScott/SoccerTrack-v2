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
  * the released width/height (3840x1504) is STALE. bbox_image is really in
    4096x1080 -- byte-identical to the staged clip, x-ratio 1.0000.
  * THE VIDEO IS MISSING FRAMES FROM THE START OF THE ANNOTATED PERIOD, so the ground
    truth's frame N is NOT video frame N. See the START OFFSET comment in main(). For
    128057's first half the gap is 25 frames (1.0 s) and ignoring it cost ~36 GS-HOTA
    points. The offset is derived per half from <match>_padding_info.csv, which exists
    for all ten matches.

A RETRACTED CLAIM, kept here as a warning:
    An earlier version of this file asserted that image_id "3{N:06d}" is 1-based frame N
    of the video, "established by matching CLPD-117093's clip_start=969000ms (frame 24226)
    against production image_id 3024226: identical bbox_image for 21 of 22 tracks."
    That check compared a staged LABEL file against the production LABEL file. Both are
    annotations, so it could only ever confirm that the two label files agree with each
    other -- it was structurally incapable of detecting a label-to-PIXEL offset, which is
    the defect that was actually present. To verify alignment you must compare labels
    against something derived from the imagery.
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


def declared_frames(data: Path, match: str, half: str, fps: int = 25) -> int | None:
    """Frames the annotation period declares, from <match>_padding_info.csv.

    That file gives, per half, Start/End Match Time in milliseconds:
        Event Period,Period Order,Video,Padding,Start Match Time,End Match Time
        FIRST_HALF,0,128057,3689000,0,2706000
    (End - Start) * fps / 1000 is the number of annotated frames, and for 128057's first half
    that is exactly 67,650 -- the ground truth's own seq_length.
    """
    p = data/"raw"/match/f"{match}_padding_info.csv"
    if not p.exists():
        return None
    want = "FIRST_HALF" if half == "1st" else "SECOND_HALF"
    import csv as _csv
    with open(p) as fh:
        for row in _csv.DictReader(fh):
            if (row.get("Event Period") or "").strip().upper() == want:
                try:
                    ms = int(row["End Match Time"]) - int(row["Start Match Time"])
                except (KeyError, ValueError):
                    return None
                return int(round(ms * fps / 1000.0))
    return None

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
    ap.add_argument("--frame-offset", type=int, default=None,
                    help="GT frames missing from the START of the video. Default: derived from "
                         "<match>_padding_info.csv as declared_frames - video_frames.")
    a = ap.parse_args()

    data = Path(a.data)
    gt_path = data/"production"/"gsr"/a.match/f"{a.match}_{a.half}.json"
    vid = data/"interim"/a.match/f"{a.match}_panorama_{a.half}_half.mp4"
    for p in (gt_path, vid):
        if not p.exists():
            print(f"MISSING: {p}"); return 1

    vw, vh, vn = video_frames(vid)
    decl = declared_frames(data, a.match, a.half)
    if a.frame_offset is not None:
        offset = a.frame_offset
        osrc = "--frame-offset"
    elif decl is not None:
        offset = max(0, decl - vn)
        osrc = f"padding_info.csv ({decl} declared - {vn} in video)"
    else:
        offset = 0
        osrc = "no padding_info.csv; assuming 0"
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
    print(f"GT images {n_img_gt}  declared {declared[0]}x{declared[1]} (stale)", flush=True)
    print(f"START OFFSET {offset} frames ({offset/25.0:.2f} s) from {osrc}", flush=True)
    print(f"  -> video frame k carries GT frame k+{offset}; "
          f"staging video frames {start}..{end} ({count})", flush=True)
    if n_img_gt > vn:
        print(f"  NOTE: GT annotates {n_img_gt - vn} frames beyond the video; trimmed.", flush=True)

    # Old image_id -> new, WITH THE START OFFSET APPLIED.
    #
    # The video is missing `offset` frames from the START of the annotated period, so video
    # frame k shows what the ground truth calls frame k + offset. Attaching GT frame k to
    # video frame k -- which this script originally did -- misaligns labels from imagery by
    # offset/fps seconds. For 128057 that is 25 frames = 1.0 s, and it cost roughly 36 GS-HOTA
    # points: scoring frames 27-750 went 40.87 -> 77.04 once the shift was applied.
    #
    # Two independent lines fix the value at ~25: exact arithmetic from padding_info.csv
    # (67,650 declared - 67,625 in the video), and an assignment-free cross-correlation of
    # detector output against labels, whose correlation peaks at lag 26-27 (0.9670 at lag 0 ->
    # 0.9981 at the peak, flat across 25-28). The arithmetic value is used because it comes
    # from the data's declared timing rather than from maximising a similarity.
    remap, images = {}, []
    for i in d["images"]:
        n = int(i["image_id"][1:])                # "3024226" -> 24226 (1-based GT frame)
        if not (start + offset <= n <= end + offset): continue
        k = n - start - offset + 1               # -> video frame / file number
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
