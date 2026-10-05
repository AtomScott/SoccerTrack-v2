"""Validate generated calibrated keypoints through the module's REAL chain.

The chain in manual_calib_distorted.py is:
    distorted px --TPS--> calibrated px --inv(H_pc)--> pitch (corner) --> centre-origin
An earlier check of mine compared inv(H_pc) applied straight to distorted pixels, which
skips the TPS and is meaningless. _fit_tps / _tps_warp_points below are copied verbatim
from that module so this exercises the same arithmetic.

Also normalises the calibrated canvas. Undistorting 128057 with balance=0 puts control
points at x in [-7973, 11234] for a 4096-wide image; TPS fit on that extrapolates wildly.
A similarity transform composes cleanly with H_pc (so the pitch mapping is unchanged) and
keeps the fit conditioned, so we map the keypoint box to 117093's proportions.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import cv2
import numpy as np

FLAGS = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
         + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_K3 + cv2.fisheye.CALIB_FIX_K4)
CRITERIA = (cv2.TermCriteria_COUNT + cv2.TermCriteria_EPS, 100, 1e-6)
W, H_IMG = 4096, 1080
XSPAN_FRAC = 2925.0/4096.0          # 117093's shipped x-span, as a fraction of width

def _tps_kernel(r2, eps=1e-9):
    with np.errstate(divide='ignore', invalid='ignore'):
        return r2*np.log(r2+eps)

def _fit_tps(src_xy, dst_xy, reg=1e-3):
    N=src_xy.shape[0]; C=src_xy.astype(np.float64); Y=dst_xy.astype(np.float64)
    diff=C[:,None,:]-C[None,:,:]; r2=np.sum(diff**2,axis=2); K=_tps_kernel(r2)
    P=np.concatenate([np.ones((N,1)),C],axis=1)
    L=np.zeros((N+3,N+3)); L[:N,:N]=K+reg*np.eye(N); L[:N,N:]=P; L[N:,:N]=P.T
    cx=np.linalg.solve(L,np.concatenate([Y[:,0],np.zeros(3)]))
    cy=np.linalg.solve(L,np.concatenate([Y[:,1],np.zeros(3)]))
    return {"ctrl":C,"w_x":cx[:N],"a_x":cx[N:],"w_y":cy[:N],"a_y":cy[N:]}

def _tps_warp_points(m, pts):
    C=m["ctrl"]; P=pts.astype(np.float64)
    diff=P[:,None,:]-C[None,:,:]; r2=np.sum(diff**2,axis=2); K=_tps_kernel(r2)
    ones=np.ones((P.shape[0],1)); A=np.concatenate([ones,P],axis=1)
    x=K@m["w_x"]+A@m["a_x"]; y=K@m["w_y"]+A@m["a_y"]
    return np.stack([x,y],axis=1)

def load_kp(match, corrections, data):
    p=corrections/f"{match}_keypoints.json"
    src="corrections" if p.exists() else "dataset"
    if not p.exists(): p=data/"raw"/match/f"{match}_keypoints.json"
    d=json.load(open(p)); keys=list(d)
    pitch=np.array([[float(v) for v in re.match(r"\(([-\d.]+),\s*([-\d.]+)\)",k).groups()] for k in keys])
    return keys, pitch, np.array(list(d.values()),float), src

def undistort(pitch, image, normalise=True):
    objp=np.concatenate([pitch,np.zeros((len(pitch),1))],1).astype(np.float32).reshape(-1,1,3)
    imgp=image.astype(np.float32).reshape(-1,1,2)
    rms,K,D,_,_=cv2.fisheye.calibrate([objp],[imgp],(W,H_IMG),np.zeros((3,3)),np.zeros((4,1)),
                                      None,None,flags=FLAGS,criteria=CRITERIA)
    nk=cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K,D,(W,H_IMG),np.eye(3),balance=0.0)
    und=cv2.fisheye.undistortPoints(imgp,K,D,P=nk).reshape(-1,2)
    if normalise:
        span=und[:,0].max()-und[:,0].min()
        s=(XSPAN_FRAC*W)/span
        cx=(und[:,0].min()+und[:,0].max())/2; cy=(und[:,1].min()+und[:,1].max())/2
        und=np.stack([(und[:,0]-cx)*s + W/2, (und[:,1]-cy)*s + H_IMG/2],1)
    return rms, und

def chain(distorted, calibrated, pitch_corner):
    """Return f(distorted px) -> pitch centre-origin, exactly as the module composes it."""
    tps=_fit_tps(distorted, calibrated, reg=1e-3)
    H_pc,_=cv2.findHomography(pitch_corner.astype(np.float64), calibrated.astype(np.float64), 0)
    Hi=np.linalg.inv(H_pc)
    def f(pts):
        cal=_tps_warp_points(tps, pts)
        pit=cv2.perspectiveTransform(cal.reshape(-1,1,2), Hi).reshape(-1,2)
        return pit - np.array([52.5, 34.0])
    return f

ap = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data", default="/data/share/SoccerTrack-v2/data")
ap.add_argument("--corrections", default="data_corrections",
                help="dir of corrected keypoint files that override the dataset (e.g. 132831)")
ap.add_argument("--matches", nargs="*", default=["128057", "132831"])
ap.add_argument("--verify", default="117093",
                help="match that ships a calibrated set, used as the equivalence reference")
ap.add_argument("--out", default=None,
                help="output dir; default writes beside the distorted file under <data>/raw/<match>")
args = ap.parse_args()
DATA = Path(args.data); CORR = Path(args.corrections)

print("="*76)
print(f"VALIDATION on {args.verify} -- the only match shipping a calibrated set")
print("="*76)
keys,pitch,image,src=load_kp(args.verify, CORR, DATA)
sh=json.load(open(DATA/"raw"/args.verify/f"{args.verify}_calibrated_keypoints.json"))
common=[k for k in keys if k in sh]; idx=[keys.index(k) for k in common]
cal_sh=np.array([sh[k] for k in common])
rms,und=undistort(pitch,image)
f_sh=chain(image[idx], cal_sh,      pitch[idx])
f_gn=chain(image[idx], und[idx],    pitch[idx])
tgt=pitch[idx]-np.array([52.5,34.0])
for nm,f in (("shipped",f_sh),("generated",f_gn)):
    e=np.linalg.norm(f(image[idx])-tgt,axis=1)
    print(f"  {nm:10} keypoints -> pitch : median {np.median(e):6.3f} m   p95 {np.percentile(e,95):6.3f} m")
gx,gy=np.meshgrid(np.linspace(150,W-150,50), np.linspace(140,780,26))
pts=np.stack([gx.ravel(),gy.ravel()],1)
a,b=f_sh(pts),f_gn(pts)
on=(a[:,0]>-52.5)&(a[:,0]<52.5)&(a[:,1]>-34)&(a[:,1]<34)
d=np.linalg.norm(a[on]-b[on],axis=1)
print(f"  full-chain agreement over {on.sum()} on-pitch grid points:")
print(f"    median {np.median(d):.4f} m   p95 {np.percentile(d,95):.4f} m   max {d.max():.4f} m")

print()
print("="*76)
print("GENERATED SETS")
print("="*76)
out=Path(args.out) if args.out else None
if out: out.mkdir(parents=True, exist_ok=True)
for m in args.matches:
    keys,pitch,image,src=load_kp(m, CORR, DATA)
    try: rms,und=undistort(pitch,image)
    except cv2.error: print(f"{m}: CALIBRATION REFUSED"); continue
    f=chain(image,und,pitch)
    e=np.linalg.norm(f(image)-(pitch-np.array([52.5,34.0])),axis=1)
    dst = (out / f"{m}_calibrated_keypoints.json") if out else \
          (DATA / "raw" / m / f"{m}_calibrated_keypoints.json")
    if dst.exists():
        bak = dst.with_suffix(".json.backup")
        if not bak.exists():
            bak.write_bytes(dst.read_bytes())
            print(f"    backed up existing -> {bak.name}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    json.dump({k:[float(u[0]),float(u[1])] for k,u in zip(keys,und)}, open(dst,"w"), indent=2)
    print(f"{m}: rms={rms:6.2f}px  src={src}")
    print(f"    calibrated x [{und[:,0].min():7.1f}..{und[:,0].max():7.1f}] y [{und[:,1].min():6.1f}..{und[:,1].max():6.1f}]")
    print(f"    keypoints -> pitch : median {np.median(e):6.3f} m   p95 {np.percentile(e,95):6.3f} m")

    print(f"    -> {dst}")
