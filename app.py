import io
import re
import warnings
from pathlib import Path
from functools import lru_cache

import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_file

warnings.filterwarnings("ignore")

app = Flask(__name__)
BASE_DIR = Path(__file__).parent

# ──────────────────────────────────────────────────────────────────────────────
# Dataset discovery
# ──────────────────────────────────────────────────────────────────────────────

def _discover_dataset():
    """Find the IgG-1D Cryo-EM dataset or fall back to a generic layout."""
    # Walk up to 2 levels looking for images/ + vols/ siblings
    for root in [BASE_DIR, *BASE_DIR.rglob("*")]:
        if not root.is_dir():
            continue
        if (root / "images").is_dir() and (root / "vols").is_dir():
            return _parse_igg_dataset(root)
    return _parse_generic_dataset(BASE_DIR)


def _parse_igg_dataset(root):
    images_dir = root / "images"
    vols_dir   = root / "vols"

    # SNR sub-directories
    snr_dirs = sorted(d for d in images_dir.iterdir()
                      if d.is_dir() and d.name.lower().startswith("snr"))

    # Volume sub-directory (e.g. 128_org)
    vol_subdirs = [d for d in vols_dir.iterdir() if d.is_dir()]
    vol_dir = vol_subdirs[0] if vol_subdirs else vols_dir

    gt_by_num = {f.stem: f for f in vol_dir.iterdir()
                 if f.suffix.lower() == ".mrc"}

    if not snr_dirs or not gt_by_num:
        return None

    # Collect .mrcs files from the first SNR dir to enumerate samples
    first_snr = snr_dirs[0]
    samples = []
    for mrcs in sorted(first_snr.glob("*.mrcs")):
        m = re.match(r"^(\d+)", mrcs.name)
        if not m:
            continue
        num = m.group(1)
        if num not in gt_by_num:
            continue

        noisy_paths = {}
        for snr_dir in snr_dirs:
            candidates = {re.match(r"^(\d+)", f.name).group(1): f
                          for f in snr_dir.glob("*.mrcs")
                          if re.match(r"^(\d+)", f.name)}
            if num in candidates:
                noisy_paths[snr_dir.name] = str(candidates[num])

        samples.append({
            "id":          num,
            "name":        f"Sample {num}",
            "type":        "mrcs_stack",
            "noisy_paths": noisy_paths,
            "gt_path":     str(gt_by_num[num]),
            "snr_levels":  sorted(noisy_paths.keys()),
        })

    return {
        "type":       "igg",
        "samples":    samples,
        "snr_levels": sorted(snr_dirs, key=lambda d: d.name),
    }


def _parse_generic_dataset(root):
    """Pair images named *noisy*/*gt* in the dataset/ subfolder or root."""
    NOISY = re.compile(r"(noisy|noise|raw|input)", re.I)
    GT    = re.compile(r"(gt|ground.truth|clean|target|ref)", re.I)
    EXTS  = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".mrc"}

    search_dir = root / "dataset" if (root / "dataset").is_dir() else root
    files = [f for f in search_dir.iterdir() if f.suffix.lower() in EXTS]

    noisy_files = [f for f in files if NOISY.search(f.stem)]
    gt_files    = [f for f in files if GT.search(f.stem)]

    # Try to pair by stripping the pattern suffix and matching stems
    pairs = []
    for nf in noisy_files:
        base = NOISY.sub("", nf.stem).strip("_- ")
        match = next((g for g in gt_files if GT.sub("", g.stem).strip("_- ") == base), None)
        if match:
            pairs.append({
                "id":          nf.stem,
                "name":        base or nf.stem,
                "type":        "image_pair",
                "noisy_paths": {"default": str(nf)},
                "gt_path":     str(match),
                "snr_levels":  ["default"],
            })

    return {"type": "generic", "samples": pairs, "snr_levels": []}


_dataset_cache = None

def get_dataset():
    global _dataset_cache
    if _dataset_cache is None:
        _dataset_cache = _discover_dataset()
    return _dataset_cache


def _find_sample(sample_id):
    ds = get_dataset()
    if not ds:
        return None
    return next((s for s in ds["samples"] if s["id"] == sample_id), None)


# ──────────────────────────────────────────────────────────────────────────────
# Image helpers
# ──────────────────────────────────────────────────────────────────────────────

def _normalize(data):
    """Percentile-stretch float array → uint8."""
    data = data.astype(np.float32)
    lo, hi = np.percentile(data, [1, 99])
    data = np.clip((data - lo) / max(hi - lo, 1e-8), 0, 1)
    return (data * 255).astype(np.uint8)


def _to_png(arr2d):
    from PIL import Image
    img = Image.fromarray(arr2d, mode="L")
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def _to_png_thumb(arr2d, size=128):
    from PIL import Image
    img = Image.fromarray(arr2d, mode="L")
    img.thumbnail((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


# ── Particle from .mrcs stack ─────────────────────────────────────────────────

@lru_cache(maxsize=120)
def _load_particle(path, idx):
    import mrcfile
    with mrcfile.mmap(path, mode="r", permissive=True) as mrc:
        data = mrc.data[idx].copy()
    return _normalize(data)


def particle_png(path, idx):
    return _to_png(_load_particle(path, idx))


def particle_thumb_png(path, idx=0):
    return _to_png_thumb(_load_particle(path, idx))


def mrcs_info(path):
    import mrcfile
    with mrcfile.mmap(path, mode="r", permissive=True) as mrc:
        shape = mrc.data.shape       # (N, H, W)
        vsize = float(mrc.voxel_size.x)
    return {"n_particles": int(shape[0]),
            "particle_shape": [int(shape[1]), int(shape[2])],
            "voxel_size_angstrom": round(vsize, 3)}


# ── Slice / MIP from 3-D .mrc volume ─────────────────────────────────────────

@lru_cache(maxsize=4)
def _load_volume(path):
    """Cache the full volume array (shared across slice/MIP requests)."""
    import mrcfile
    with mrcfile.mmap(path, mode="r", permissive=True) as mrc:
        return mrc.data[:].copy()   # (Z, Y, X) float32


@lru_cache(maxsize=60)
def _load_volume_slice(path, axis, idx):
    """idx=None → MIP (maximum intensity projection) along axis."""
    vol = _load_volume(path)
    n   = vol.shape[axis]

    if idx is None:
        # Maximum Intensity Projection — shows the full molecule in 2D
        data = np.max(vol, axis=axis)
    else:
        sidx = int(np.clip(idx, 0, n - 1))
        if   axis == 0: data = vol[sidx, :, :]
        elif axis == 1: data = vol[:, sidx, :]
        else:           data = vol[:, :, sidx]
        data = data.copy()

    return _normalize(data), int(n)


def volume_slice_png(path, axis=0, idx=None):
    arr, n = _load_volume_slice(path, axis, idx)
    return _to_png(arr), n


def volume_thumb_png(path):
    arr, _ = _load_volume_slice(path, 0, None)   # thumbnail = MIP
    return _to_png_thumb(arr)


@lru_cache(maxsize=20)
def _volume_pointcloud(path, threshold_pct=0.15, max_points=8000):
    """Extract 3-D point cloud for Three.js visualisation."""
    vol = _load_volume(path)   # (Z, Y, X) float32
    vmax = float(vol.max())
    thresh = vmax * threshold_pct
    if thresh == 0:
        return {"x": [], "y": [], "z": [], "intensity": [], "shape": list(vol.shape), "n_total": 0}

    z_idx, y_idx, x_idx = np.where(vol > thresh)
    vals = vol[z_idx, y_idx, x_idx]
    n_total = int(len(x_idx))

    if n_total > max_points:
        rng = np.random.default_rng(42)
        sel = rng.choice(n_total, max_points, replace=False)
        x_idx, y_idx, z_idx, vals = x_idx[sel], y_idx[sel], z_idx[sel], vals[sel]

    intensity = ((vals - thresh) / max(vmax - thresh, 1e-8)).clip(0, 1)

    # Centre around origin
    cx, cy, cz = vol.shape[2] // 2, vol.shape[1] // 2, vol.shape[0] // 2
    return {
        "x": (x_idx - cx).tolist(),
        "y": (y_idx - cy).tolist(),
        "z": (z_idx - cz).tolist(),
        "intensity": intensity.tolist(),
        "shape": list(vol.shape),
        "n_total": n_total,
    }


def volume_info(path):
    import mrcfile
    with mrcfile.open(path, mode="r", permissive=True) as mrc:
        shape = list(mrc.data.shape)
        vs    = mrc.voxel_size
    return {"shape": shape,
            "voxel_size_angstrom": {"x": float(vs.x), "y": float(vs.y), "z": float(vs.z)}}


# ──────────────────────────────────────────────────────────────────────────────
# Orientation data  (RELION .star files)
# ──────────────────────────────────────────────────────────────────────────────

import math as _math

@lru_cache(maxsize=8)
def _parse_star_angles(star_path):
    """Parse a RELION .star file and return {sample_id: [(rot,tilt,psi), ...]}."""
    result   = {}   # {sample_id: {particle_0based_idx: (rot,tilt,psi)}}
    col_img  = col_rot = col_tilt = col_psi = None
    col_idx  = 0
    in_ptcl  = False   # inside data_particles block
    in_loop  = False   # inside the loop_ header

    with open(star_path, "r") as fh:
        for raw in fh:
            line = raw.strip()
            if line == "data_particles":
                in_ptcl = True;  col_idx = 0;  continue
            if not in_ptcl:
                continue
            if line == "loop_":
                in_loop = True;  continue
            if in_loop and line.startswith("_rln"):
                name = line.split()[0]
                if name == "_rlnImageName":  col_img  = col_idx
                if name == "_rlnAngleRot":   col_rot  = col_idx
                if name == "_rlnAngleTilt":  col_tilt = col_idx
                if name == "_rlnAnglePsi":   col_psi  = col_idx
                col_idx += 1
                continue
            if in_loop and col_img is not None and line and not line.startswith("_") and not line.startswith("#"):
                parts = line.split()
                need  = max(col_img, col_rot, col_tilt, col_psi) + 1
                if len(parts) < need:
                    continue
                img_name = parts[col_img]          # e.g. "1@000_particles_128.mrcs"
                if "@" not in img_name:
                    continue
                pidx_str, fname = img_name.split("@", 1)
                sample_id = fname.split("_")[0]    # "000"
                pidx      = int(pidx_str) - 1      # 0-based

                if sample_id not in result:
                    result[sample_id] = {}
                result[sample_id][pidx] = (
                    float(parts[col_rot]),
                    float(parts[col_tilt]),
                    float(parts[col_psi]),
                )

    # sort each sample's list by particle index
    return {
        sid: [angles[i] for i in sorted(angles)]
        for sid, angles in result.items()
    }


def _angles_to_xyz(rot_deg, tilt_deg):
    """Convert RELION Euler angles to unit-sphere viewing direction (x,y,z)."""
    t = _math.radians(tilt_deg)
    r = _math.radians(rot_deg)
    return (
        _math.sin(t) * _math.cos(r),   # x
        _math.cos(t),                   # y  (north pole = MIP direction)
        _math.sin(t) * _math.sin(r),   # z
    )


# ──────────────────────────────────────────────────────────────────────────────
# Flask routes
# ──────────────────────────────────────────────────────────────────────────────

def _send_png(data: bytes):
    return send_file(io.BytesIO(data), mimetype="image/png")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/config")
def api_config():
    ds = get_dataset()
    if not ds:
        return jsonify({"error": "No dataset found"}), 404
    return jsonify({
        "type":         ds["type"],
        "snr_levels":   [d.name if hasattr(d, "name") else d
                         for d in ds.get("snr_levels", [])],
        "sample_count": len(ds["samples"]),
    })


@app.route("/api/samples")
def api_samples():
    ds = get_dataset()
    if not ds:
        return jsonify([])
    return jsonify([{
        "id":         s["id"],
        "name":       s["name"],
        "snr_levels": s.get("snr_levels", []),
    } for s in ds["samples"]])


@app.route("/api/image/noisy/<sample_id>")
def api_noisy_image(sample_id):
    s = _find_sample(sample_id)
    if not s:
        abort(404)
    snr      = request.args.get("snr") or (s["snr_levels"][0] if s["snr_levels"] else "default")
    particle = int(request.args.get("particle", 0))
    path     = s["noisy_paths"].get(snr)
    if not path:
        abort(404)

    if s["type"] == "mrcs_stack":
        return _send_png(particle_png(path, particle))
    else:
        from PIL import Image
        img = Image.open(path).convert("L")
        buf = io.BytesIO(); img.save(buf, "PNG"); buf.seek(0)
        return send_file(buf, mimetype="image/png")


@app.route("/api/image/gt/<sample_id>")
def api_gt_image(sample_id):
    s = _find_sample(sample_id)
    if not s:
        abort(404)
    axis  = int(request.args.get("axis", 0))
    idx_q = request.args.get("slice")
    idx   = int(idx_q) if idx_q is not None else None

    gt_path = s["gt_path"]
    ext     = Path(gt_path).suffix.lower()

    if ext in (".mrc", ".mrcs"):
        png, n_slices = volume_slice_png(gt_path, axis, idx)
        resp = _send_png(png)
        resp.headers["X-N-Slices"] = str(n_slices)
        return resp
    else:
        from PIL import Image
        img = Image.open(gt_path).convert("L")
        buf = io.BytesIO(); img.save(buf, "PNG"); buf.seek(0)
        resp = send_file(buf, mimetype="image/png")
        resp.headers["X-N-Slices"] = "1"
        return resp


@app.route("/api/thumbnail/noisy/<sample_id>")
def api_noisy_thumb(sample_id):
    s = _find_sample(sample_id)
    if not s:
        abort(404)
    snr  = request.args.get("snr") or (s["snr_levels"][0] if s["snr_levels"] else "default")
    path = s["noisy_paths"].get(snr)
    if not path:
        abort(404)
    return _send_png(particle_thumb_png(path, 0))


@app.route("/api/thumbnail/gt/<sample_id>")
def api_gt_thumb(sample_id):
    s = _find_sample(sample_id)
    if not s:
        abort(404)
    return _send_png(volume_thumb_png(s["gt_path"]))


@app.route("/api/orientations/<sample_id>")
def api_orientations(sample_id):
    s = _find_sample(sample_id)
    if not s:
        abort(404)

    snr  = request.args.get("snr") or (s["snr_levels"][0] if s["snr_levels"] else None)
    path = s["noisy_paths"].get(snr) if snr else None
    if not path:
        return jsonify({"orientations": []})

    star_path = Path(path).parent / f"{snr}.star"
    if not star_path.exists():
        return jsonify({"orientations": []})

    by_sample = _parse_star_angles(str(star_path))
    angles    = by_sample.get(sample_id, [])

    orientations = []
    for rot, tilt, psi in angles:
        x, y, z = _angles_to_xyz(rot, tilt)
        orientations.append({
            "x": round(x, 5), "y": round(y, 5), "z": round(z, 5),
            "rot": round(rot, 2), "tilt": round(tilt, 2), "psi": round(psi, 2),
        })

    return jsonify({"orientations": orientations})


@app.route("/api/volume3d/<sample_id>")
def api_volume3d(sample_id):
    s = _find_sample(sample_id)
    if not s:
        abort(404)
    gt_path = s.get("gt_path", "")
    if Path(gt_path).suffix.lower() not in (".mrc", ".mrcs"):
        abort(404)
    threshold = float(request.args.get("threshold", 0.15))
    return jsonify(_volume_pointcloud(gt_path, threshold))


@app.route("/api/metadata/<sample_id>")
def api_metadata(sample_id):
    s = _find_sample(sample_id)
    if not s:
        abort(404)

    snr  = request.args.get("snr") or (s["snr_levels"][0] if s["snr_levels"] else "default")
    path = s["noisy_paths"].get(snr)

    meta = {"sample_id": sample_id, "name": s["name"]}

    if path and s["type"] == "mrcs_stack":
        info = mrcs_info(path)
        meta["noisy"] = {
            "file":           Path(path).name,
            "n_particles":    info["n_particles"],
            "particle_shape": info["particle_shape"],
            "size_mb":        round(Path(path).stat().st_size / 1e6, 1),
            "voxel_size_angstrom": info["voxel_size_angstrom"],
        }

    gt_path = s.get("gt_path")
    if gt_path and Path(gt_path).suffix.lower() in (".mrc", ".mrcs"):
        info = volume_info(gt_path)
        meta["gt"] = {
            "file":     Path(gt_path).name,
            "shape":    info["shape"],
            "voxel_size": info["voxel_size_angstrom"],
            "size_mb":  round(Path(gt_path).stat().st_size / 1e6, 1),
        }

    return jsonify(meta)


# ──────────────────────────────────────────────────────────────────────────────
# U-Net denoising  (lazy-loaded on first request)
# ──────────────────────────────────────────────────────────────────────────────

_unet = {"model": None, "device": None}


def _percentile_norm(arr: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(arr, [1, 99])
    arr = np.clip(arr, lo, hi)
    mean, std = arr.mean(), arr.std()
    return ((arr - mean) / (std + 1e-8)).astype(np.float32)


def _load_unet() -> bool:
    if _unet["model"] is not None:
        return True
    try:
        import sys as _sys, torch
        _sys.path.insert(0, str(BASE_DIR / "pipeline"))
        from config import CFG
        from model import UNet
        ckpt = Path(CFG["ckpt_best"])
        if not ckpt.exists():
            return False
        device = torch.device(CFG["device"])
        m = UNet(base_channels=CFG["base_channels"],
                 depth=CFG["depth"], dropout_p=0.0).to(device)
        m.load_state_dict(torch.load(str(ckpt), map_location=device))
        m.eval()
        _unet["model"] = m
        _unet["device"] = device
        print(f"[denoise] U-Net loaded — {device}")
        return True
    except Exception as exc:
        print(f"[denoise] Load failed: {exc}")
        return False


@app.route("/api/denoise/status")
def api_denoise_status():
    try:
        import sys as _sys
        _sys.path.insert(0, str(BASE_DIR / "pipeline"))
        from config import CFG
        available = Path(CFG["ckpt_best"]).exists()
    except Exception:
        available = False
    return jsonify({"available": available})


@lru_cache(maxsize=4)
def _load_clean_npy(path: str) -> np.ndarray:
    return np.load(path)   # (1000, 1, 128, 128) or (1000, 128, 128)


@app.route("/api/image/clean2d/<sample_id>")
def api_clean2d_image(sample_id):
    s = _find_sample(sample_id)
    if not s: abort(404)
    particle  = int(request.args.get("particle", 0))
    any_noisy = next(iter(s["noisy_paths"].values()), None)
    if not any_noisy: abort(404)
    try:
        conf_num = int(sample_id)
    except ValueError:
        abort(404)
    clean_dir = Path(any_noisy).parent.parent.parent / "clean_projections"
    npy_path  = clean_dir / f"clean_2d_targets_{conf_num:03d}.npy"
    if not npy_path.exists(): abort(404)
    arr   = _load_clean_npy(str(npy_path))
    frame = arr[particle]
    if frame.ndim == 3: frame = frame[0]
    return _send_png(_to_png(_normalize(frame.astype(np.float32))))


@app.route("/api/image/denoised/<sample_id>")
def api_denoised_image(sample_id):
    s = _find_sample(sample_id)
    if not s:
        abort(404)
    if not _load_unet():
        abort(503)

    snr      = request.args.get("snr") or (s["snr_levels"][0] if s["snr_levels"] else "default")
    particle = int(request.args.get("particle", 0))
    path     = s["noisy_paths"].get(snr)
    if not path:
        abort(404)

    import mrcfile as _mrc, torch
    with _mrc.mmap(path, mode="r", permissive=True) as mrc:
        raw = mrc.data[particle].copy()

    tensor = torch.from_numpy(_percentile_norm(raw)).unsqueeze(0).unsqueeze(0)
    tensor = tensor.to(_unet["device"])
    with torch.no_grad():
        pred, _ = _unet["model"](tensor)
    result = pred.squeeze().cpu().numpy()
    return _send_png(_to_png(_normalize(result)))


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ds = get_dataset()
    sep = "=" * 52
    print(f"\n{sep}")
    print("   Cryo-EM Dataset Viewer")
    print(sep)
    if ds:
        snrs = [d.name if hasattr(d, "name") else d for d in ds.get("snr_levels", [])]
        print(f"   Dataset : {ds['type']}")
        print(f"   Samples : {len(ds['samples'])}")
        if snrs:
            print(f"   SNR     : {', '.join(snrs)}")
    else:
        print("   WARNING: No dataset found.")
    print(f"\n   Open  ->  http://127.0.0.1:5000\n{sep}\n")
    import threading, webbrowser
    threading.Timer(1.2, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(debug=False, host="127.0.0.1", port=5000, threaded=True)
