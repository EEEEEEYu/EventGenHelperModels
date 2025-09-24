#!/usr/bin/env python3
"""
Extract events & RGB from M3ED .h5 files, undistort using calibration, and
optionally render a 3-up visualization video (events-only, RGB-only, overlay).

This version is **memory-safe**: it avoids loading full arrays by streaming
from the HDF5 datasets (lazy per-frame/per-slice reads) and computing
frame↔event windows without materializing the entire event timestamp array.

Assumptions (from M3ED docs):
- Event cameras at /prophesee/{left,right} with datasets x,y,t,p and /calib/*
- RGB (AR0144 on OVC) at /ovc/rgb with datasets data (N,H,W,3) and ts
- For imagers, ts_map_prophesee_{left,right}_t maps each frame index to an
  event index. We treat the mapping as an inclusive start index for events for
  a frame and use the next frame's index as the exclusive end.

Outputs (mirrors input sequence folders):
  <out_root>/<sequence>/
      rgb/
      rgb_undistorted/
      events_rgb_left/                # per-frame event images aligned to RGB
      events_rgb_left_undistorted/
      overlay_rgb_left_undistorted/
      meta.json

Usage examples:
  python extract_m3ed.py --in_root dataset/m3ed --out_root output_m3ed

Notes:
- If time-maps are absent, we compute per-frame event ranges by *streaming*
  through the event timestamps and aligning to RGB timestamps.
- Distortion models supported: "radtan" (plumb bob) via cv2.undistort,
  "equidistant" (fisheye) via cv2.fisheye.
- Event undistortion is done by undistorting points with cv2.(fisheye.)undistortPoints
  then reprojecting to pixel grid.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm

import h5py
import numpy as np
import cv2

# ---------------------------
# Utilities
# ---------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def read_scalar_str(dset) -> str:
    """Safely read a scalar string/bytes dataset from h5."""
    val = dset[()]
    if isinstance(val, bytes):
        return val.decode("utf-8")
    if isinstance(val, np.ndarray) and val.dtype.kind in {"S", "O", "U"}:
        return str(val.tolist())
    return str(val)


# ---------------------------
# Calibration handling
# ---------------------------

def load_calib(g: h5py.Group) -> Dict:
    """Load intrinsics + distortion from a group's /calib.
    Expects datasets:
      /calib/intrinsics -> [fx, fy, cx, cy]
      /calib/distortion_coeffs -> k parameters (usually 4)
      /calib/distortion_model -> scalar string, e.g., 'radtan' or 'equidistant'
      /calib/resolution -> [W, H]
    """
    calib = {}
    calib_root = g["calib"]
    intr = np.array(calib_root["intrinsics"][()], dtype=np.float64).reshape(-1)
    fx, fy, cx, cy = intr.tolist()
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    D = np.array(calib_root["distortion_coeffs"][()], dtype=np.float64).reshape(-1)
    model = read_scalar_str(calib_root["distortion_model"]).lower()
    res = np.array(calib_root["resolution"][()], dtype=int).reshape(-1)
    W, H = int(res[0]), int(res[1])
    calib.update(dict(K=K, D=D, model=model, W=W, H=H))
    return calib


def undistort_image(img: np.ndarray, K: np.ndarray, D: np.ndarray, model: str,
                    out_size: Tuple[int, int]) -> np.ndarray:
    """Undistort an image with OpenCV. out_size: (W, H)."""
    W, H = out_size
    if model in ("equidistant", "fisheye"):
        D_use = (D[:4] if D.size >= 4 else np.pad(D, (0, 4 - D.size))).astype(np.float64)
        Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D_use, (W, H), np.eye(3), balance=0.0
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D_use, np.eye(3), Knew, (W, H), cv2.CV_16SC2
        )
        return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
    else:
        newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (W, H), alpha=0)
        return cv2.undistort(img, K, D, None, newK)


def undistort_event_points(x: np.ndarray, y: np.ndarray, K: np.ndarray,
                           D: np.ndarray, model: str) -> Tuple[np.ndarray, np.ndarray]:
    """Undistort event (x,y) pixels. Returns (xu, yu)."""
    pts = np.stack([x, y], axis=1).astype(np.float64).reshape(-1, 1, 2)
    if model in ("equidistant", "fisheye"):
        D_use = (D[:4] if D.size >= 4 else np.pad(D, (0, 4 - D.size))).astype(np.float64)
        pts_ud = cv2.fisheye.undistortPoints(pts, K, D_use)
    else:
        pts_ud = cv2.undistortPoints(pts, K, D)
    pts_ud = pts_ud.reshape(-1, 2)
    xu = K[0, 0] * pts_ud[:, 0] + K[0, 2]
    yu = K[1, 1] * pts_ud[:, 1] + K[1, 2]
    return xu, yu


# ---------------------------
# Event rendering
# ---------------------------

def render_events_to_image(x: np.ndarray, y: np.ndarray, p: np.ndarray,
                           res: Tuple[int, int], polarity_mode: str = "bipolar",
                           point_size: int = 1) -> np.ndarray:
    """Rasterize events into an image. 'bipolar' uses G for pos, R for neg."""
    W, H = res
    if polarity_mode == "mono":
        img = np.zeros((H, W), dtype=np.float32)
        for xi, yi in zip(x, y):
            xi_i = int(round(xi)); yi_i = int(round(yi))
            if 0 <= xi_i < W and 0 <= yi_i < H:
                img[yi_i, xi_i] += 1.0
        img = np.clip(img / (img.max() + 1e-6), 0, 1)
        img8 = (img * 255).astype(np.uint8)
        return cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    else:
        img = np.zeros((H, W, 3), dtype=np.uint8)
        for xi, yi, pi in zip(x, y, p):
            xi_i = int(round(xi)); yi_i = int(round(yi))
            if 0 <= xi_i < W and 0 <= yi_i < H:
                if pi > 0: img[yi_i, xi_i, 1] = 255
                else:      img[yi_i, xi_i, 2] = 255
        if point_size > 1:
            img = cv2.dilate(img, np.ones((point_size, point_size), np.uint8))
        return img


# ---------------------------
# Core extraction/sync (streaming)
# ---------------------------

def get_group(f: h5py.File, path: str) -> Optional[h5py.Group]:
    return f[path] if path in f else None


def extract_rgb_group(f: h5py.File, rgb_path: str) -> Dict:
    g = get_group(f, rgb_path)
    if g is None:
        raise FileNotFoundError(f"RGB group '{rgb_path}' not found in file")
    data_ds = g["data"]             # h5py.Dataset, do NOT materialize
    ts = g["ts"][()] if "ts" in g else None  # ts is small (Nf,)
    calib = load_calib(g)
    tm_left = g.get("ts_map_prophesee_left_t")
    tm_right = g.get("ts_map_prophesee_right_t")
    ts_map = tm_left[()] if tm_left is not None else (tm_right[()] if tm_right is not None else None)
    return dict(data=data_ds, ts=ts, calib=calib, ts_map=ts_map)


def extract_events_group(f: h5py.File, ev_path: str) -> Dict:
    g = get_group(f, ev_path)
    if g is None:
        raise FileNotFoundError(f"Events group '{ev_path}' not found in file")
    # Keep dataset handles; do NOT load into RAM
    x_ds = g["x"]; y_ds = g["y"]; t_ds = g["t"]; p_ds = g["p"]
    mm = g.get("ms_map_idx")
    ms_map_idx = mm[()] if mm is not None else None
    calib = load_calib(g)
    return dict(x=x_ds, y=y_ds, t=t_ds, p=p_ds, ms_map_idx=ms_map_idx, calib=calib)


def frame_event_ranges_from_ts_map(ts_map: np.ndarray, Ne: int, Nf: int) -> List[Tuple[int, int]]:
    """Per-frame (start,end) event index ranges from ts_map (stream-friendly)."""
    starts = ts_map.astype(np.int64)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = Ne
    return [(int(starts[i]), int(ends[i])) for i in range(Nf)]


def compute_event_ranges_stream(rgb_ts: np.ndarray, ev_t_ds: h5py.Dataset,
                                chunk_size: int = 5_000_000) -> List[Tuple[int, int]]:
    """Compute per-frame [start,end) event index ranges by STREAMING over ev_t.
    Avoids loading the entire event timestamp array. Assumes rgb_ts is sorted.
    Returns a list of length len(rgb_ts) with event index boundaries.
    """
    if rgb_ts is None or len(rgb_ts) == 0:
        return [(0, ev_t_ds.shape[0])]

    Nf = len(rgb_ts)
    Ne = ev_t_ds.shape[0]
    # We will find, for each frame time, the first event index >= that time.
    # Two-pointer sweep through ev_t in chunks.
    ranges_starts = np.zeros(Nf, dtype=np.int64)
    cur_event_idx = 0
    rgb_i = 0

    while rgb_i < Nf and cur_event_idx < Ne:
        to = min(cur_event_idx + chunk_size, Ne)
        t_chunk = ev_t_ds[cur_event_idx:to]  # loads a slice only
        # advance within this chunk
        # For each remaining rgb time, find first idx >= time using searchsorted on the chunk
        while rgb_i < Nf:
            t_target = rgb_ts[rgb_i]
            # Position within current chunk
            pos = np.searchsorted(t_chunk, t_target, side="left")
            if pos < t_chunk.size:
                # Found within this chunk
                ranges_starts[rgb_i] = cur_event_idx + pos
                rgb_i += 1
            else:
                # Need to move to next chunk
                break
        cur_event_idx = to

    # For any remaining rgb times beyond last event, set start = Ne
    while rgb_i < Nf:
        ranges_starts[rgb_i] = Ne
        rgb_i += 1

    # Convert starts to [start,end) per frame
    ranges = []
    for i in range(Nf):
        start = int(ranges_starts[i])
        end = int(ranges_starts[i+1]) if i+1 < Nf else Ne
        ranges.append((start, end))
    return ranges


# ---------------------------
# Pipeline per file (streaming)
# ---------------------------

def process_file(h5_path: Path, out_root: Path, ev_cam: str = "left", make_viz: bool = True,
                 ev_chunk: int = 5_000_000, stage: str = "both"):
    """Two-pass, low-memory processing:
    Pass 1 (RGB-only): stream RGB frames, save raw + undistorted to disk.
    Pass 2 (events-only): stream event slices per frame window, render events & overlays by
    reloading the saved undistorted RGB from disk. This avoids holding both modalities in RAM.
    """
    seq_name = h5_path.parent.name
    out_dir = out_root / seq_name
    rgb_dir = out_dir / "rgb"
    rgbu_dir = out_dir / "rgb_undistorted"
    ev_dir = out_dir / f"events_rgb_{ev_cam}"
    evu_dir = out_dir / f"events_rgb_{ev_cam}_undistorted"
    ovl_dir = out_dir / f"overlay_rgb_{ev_cam}_undistorted"
    for d in [rgb_dir, rgbu_dir, ev_dir, evu_dir, ovl_dir]:
        ensure_dir(d)

    with h5py.File(str(h5_path), "r") as f:
        rgb = extract_rgb_group(f, "/ovc/rgb")
        ev = extract_events_group(f, f"/prophesee/{ev_cam}")

        Nf = int(rgb["data"].shape[0])
        Ne = int(ev["t"].shape[0])
        print(f"[INFO] {seq_name}: {Nf} RGB frames; events: {Ne}")

        # Calib
        K_rgb, D_rgb, mdl_rgb = rgb["calib"]["K"], rgb["calib"]["D"], rgb["calib"]["model"]
        K_ev, D_ev, mdl_ev = ev["calib"]["K"], ev["calib"]["D"], ev["calib"]["model"]
        res_rgb = (rgb["calib"]["W"], rgb["calib"]["H"])  # (W,H)

        # -----------------
        # PASS 1: RGB-only
        # -----------------
        if stage in ("rgb", "both"):
            print("[PASS 1] RGB → saving raw and undistorted frames")
            for i in tqdm(range(Nf)):
                img = rgb["data"][i]
                if img.ndim == 3 and img.shape[-1] == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.ndim == 3 and img.shape[-1] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                elif img.ndim == 2:
                    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                # Uncomment this line to write raw RGB frames
                # cv2.imwrite(str(rgb_dir / f"{i:06d}.png"), img)
                img_ud = undistort_image(img, K_rgb, D_rgb, mdl_rgb, res_rgb)
                cv2.imwrite(str(rgbu_dir / f"{i:06d}.png"), img_ud)
                del img, img_ud
        
        # Determine event windows per frame (ts_map preferred, else streaming by timestamps)
        if stage in ("events", "both"):
            if rgb["ts_map"] is not None:
                ranges = frame_event_ranges_from_ts_map(rgb["ts_map"], Ne, Nf)
            else:
                ranges = compute_event_ranges_stream(rgb_ts=rgb["ts"], ev_t_ds=ev["t"], chunk_size=ev_chunk)
        
        # -----------------
        # PASS 2: Events-only
        # -----------------
        if stage in ("events", "both"):
            print("[PASS 2] Events → rendering event images and overlays")
            for i in range(Nf):
                s, e = ranges[i]
                if e > s:
                    xw = ev["x"][s:e].astype(np.float64)
                    yw = ev["y"][s:e].astype(np.float64)
                    pw = ev["p"][s:e].astype(np.int8)
                else:
                    xw = yw = np.empty((0,), dtype=np.float64)
                    pw = np.empty((0,), dtype=np.int8)
                ev_img = render_events_to_image(xw, yw, pw, (res_rgb[0], res_rgb[1]), polarity_mode="bipolar")
                cv2.imwrite(str(ev_dir / f"{i:06d}.png"), ev_img)
                if xw.size:
                    xu, yu = undistort_event_points(xw, yw, K_ev, D_ev, mdl_ev)
                else:
                    xu = yu = np.empty((0,), dtype=np.float64)
                evu_img = render_events_to_image(xu, yu, pw, (res_rgb[0], res_rgb[1]), polarity_mode="bipolar")
                cv2.imwrite(str(evu_dir / f"{i:06d}.png"), evu_img)
                img_ud = cv2.imread(str(rgbu_dir / f"{i:06d}.png"), cv2.IMREAD_COLOR)
                if img_ud is not None:
                    overlay = img_ud
                    mask = evu_img > 0
                    overlay[mask] = evu_img[mask]
                    cv2.imwrite(str(ovl_dir / f"{i:06d}.png"), overlay)
                    del img_ud, overlay
                del xw, yw, pw, ev_img, evu_img
        
        # Save meta (convert numpy to lists)
        meta = dict(
            sequence=seq_name,
            h5=str(h5_path),
            rgb_calib={k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in rgb["calib"].items()},
            ev_calib={k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in ev["calib"].items()},
            rgb_frames=Nf,
            events=Ne,
            ev_cam=ev_cam,
            notes="Two-pass processing: Pass1 RGB-only → disk; Pass2 events-only with streaming + on-disk overlays.",
        )
        with open(out_dir / "meta.json", "w") as fp:
            json.dump(meta, fp, indent=2)

    # Assemble a 3-up visualization video
    if make_viz and stage in ("events", "both"):
        # Require that RGB undistorted frames exist for the middle panel
        if any(rgbu_dir.glob("*.png")) and any(evu_dir.glob("*.png")) and any(ovl_dir.glob("*.png")):
            make_three_up_video(rgbu_dir, evu_dir, ovl_dir, out_dir / f"three_up_{ev_cam}.mp4", fps=20)
        else:
            print("[WARN] Skipping 3-up video (missing one of: rgbu, evu, ovl frames)")


# ---------------------------
# Visualization (3-up)
# ---------------------------

def make_three_up_video(rgbu_dir: Path, evu_dir: Path, ovl_dir: Path, out_path: Path, fps: int = 20):
    imgs = sorted([p for p in rgbu_dir.glob("*.png")])
    if not imgs:
        print(f"[WARN] No frames in {rgbu_dir}, skipping 3-up video")
        return
    first = cv2.imread(str(imgs[0]))
    H, W = first.shape[:2]
    canvas_h, canvas_w = H, W * 3
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (canvas_w, canvas_h))

    for p in imgs:
        rgb = cv2.imread(str(p))
        ev = cv2.imread(str(evu_dir / p.name))
        ov = cv2.imread(str(ovl_dir / p.name))
        if rgb is None or ev is None or ov is None:
            continue
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:, 0:W] = ev
        canvas[:, W:2*W] = rgb
        canvas[:, 2*W:3*W] = ov
        cv2.putText(canvas, "Events", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(canvas, "RGB (undistorted)", (W+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(canvas, "Overlay", (2*W+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        vw.write(canvas)
    vw.release()
    print(f"[OK] Wrote 3-up video: {out_path} (left: events, mid: RGB-undistorted, right: overlay)")


# ---------------------------
# Batch over directory tree
# ---------------------------

def find_h5s(in_root: Path) -> List[Path]:
    return sorted(Path(in_root).glob("**/*.h5"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", type=str, required=True, help="Root containing <seq>/<seq>.h5 structure")
    ap.add_argument("--out_root", type=str, required=True, help="Output root")
    ap.add_argument("--ev_cam", type=str, default="left", choices=["left", "right"], help="Event camera to align with RGB")
    ap.add_argument("--no_viz", action="store_true", help="Disable 3-up visualization video")
    ap.add_argument("--events_chunk", type=int, default=5_000_000, help="Event timestamp chunk size for streaming alignment")
    ap.add_argument("--stage", type=str, choices=["rgb", "events", "both"], default="both",
                    help="Which pass to run: RGB-only, Events-only, or Both")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    ensure_dir(out_root)

    h5s = find_h5s(in_root)
    if not h5s:
        print(f"[ERR] No .h5 found under {in_root}")
        return

    for h5p in h5s:
        try:
            process_file(h5p, out_root, ev_cam=args.ev_cam, make_viz=not args.no_viz, ev_chunk=args.events_chunk, stage=args.stage)
        except Exception as e:
            print(f"[ERR] Failed {h5p}: {e}")


if __name__ == "__main__":
    main()
