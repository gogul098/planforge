# ===================== FLOOR PLAN -> 3D (walls, doors, windows, stairs, floor) =====================
# Outputs in OUT_DIR: plan.json (vector plan, metres) · model.glb (3D preview) · model.ifc (BIM for Revit/ArchiCAD)
IMAGE = ""                    # floor-plan image URL or file path
HF_REPO = "floorplan-multitask-effb3-768"
MAX_SIDE = 1280               # resize longest side before inference (try 1024 / 1536 if walls look off)

# --- scale (pick ONE; first non-None wins, else door-width estimate)
METERS_PER_PIXEL = None       # if you know it exactly
BUILDING_WIDTH_M = None       # e.g. 9.0 = distance between outermost wall centre-lines, left to right
DOOR_WIDTH_M = 0.9            # fallback: median detected door width is assumed to be this

# --- heights (architect-editable defaults, metres)
WALL_HEIGHT, DOOR_HEIGHT = 3.0, 2.1
WINDOW_SILL, WINDOW_HEAD = 0.9, 2.1
SLAB, STAIR_RISE = 0.15, 0.175
OUT_DIR = "/kaggle/working/plan3d"

import os, sys, json, subprocess
if not os.environ.get("P3D_NO_INSTALL"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch==0.6.0",
                    "timm==1.0.31", "huggingface_hub", "trimesh", "shapely", "mapbox-earcut",
                    "ifcopenshell>=0.8.3"], check=False)
import cv2, numpy as np, matplotlib.pyplot as plt
from scipy import ndimage
from skimage.morphology import skeletonize

ROOMS = ["Background", "Outdoor", "Wall", "Kitchen", "LivingRoom", "Bedroom", "Bath",
         "Entry/Hallway", "Railing", "Storage", "Garage", "OtherRoom"]
WALL, WIN, DOOR, STAIRS = 2, 1, 2, 11   # room-head wall id; icon-head ids


# ----------------------------------------------------------------------------- 1. model inference
def _softmax(z, axis=0):
    z = z - z.max(axis, keepdims=True)
    np.exp(z, out=z)
    z /= z.sum(axis, keepdims=True)
    return z


def _net_probs(net, dev, img, flip=True):
    """RGB image -> (room probs 12xHxW, icon probs 12xHxW), with horizontal-flip averaging.
    `net` is a PyTorch model (Kaggle) or an ONNX Runtime session (server: no PyTorch needed)."""
    h, w = img.shape[:2]
    pad = cv2.copyMakeBorder(img, 0, (32 - h % 32) % 32, 0, (32 - w % 32) % 32,
                             cv2.BORDER_CONSTANT, value=(255, 255, 255))
    x = ((pad.astype(np.float32) - [123.675, 116.28, 103.53]) / [58.395, 57.12, 57.375]).astype(np.float32)
    x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
    if dev == "onnx":
        name = net.get_inputs()[0].name
        out = net.run(None, {name: x})[0][0]
        if flip:
            out += net.run(None, {name: np.ascontiguousarray(x[:, :, :, ::-1])})[0][0][:, :, ::-1]
            out /= 2
        out = out[:24, :h, :w]
        return _softmax(out[:12].copy()), _softmax(out[12:24].copy())
    import torch
    xt = torch.from_numpy(x).to(dev)
    with torch.no_grad(), torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
        out = (net(xt).float() + net(xt.flip(3)).float().flip(3)) / 2 if flip else net(xt).float()
    out = out[0, :, :h, :w]
    return (torch.softmax(out[:12], 0).cpu().numpy().astype(np.float32),
            torch.softmax(out[12:24], 0).cpu().numpy().astype(np.float32))


def _onnx_session(path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, os.cpu_count() or 1)
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_mem_pattern = False                      # inputs change size from plan to plan
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def use_onnx():
    """ONNX Runtime when PyTorch is not installed (the server image), or when asked (PLANFORGE_BACKEND=onnx)."""
    want = os.environ.get("PLANFORGE_BACKEND", "").lower()
    if want in ("onnx", "torch"):
        return want == "onnx"
    try:
        import torch  # noqa: F401
        return False
    except ImportError:
        return True


# ----------------------------------------------------------------------------- model store
# Where the weights come from:
#  * Kaggle: a local checkpoint, or Hugging Face directly.
#  * Server (AZURE_STORAGE_CONNECTION_STRING set): Azure Blob Storage is the copy the app loads from.
#    On every cold start the app asks Hugging Face which version the release tag points at; if Blob holds an
#    older one, the new files are copied HF -> Blob once, then loaded from Blob. If Hugging Face cannot be
#    reached, the version already in Blob is used. Within a running instance the files stay on local disk.
MODEL_CONTAINER = os.environ.get("PLANFORGE_MODEL_CONTAINER", "models")
MODEL_CACHE = os.environ.get("PLANFORGE_MODEL_CACHE", "/tmp/planforge-models")
_STORE_SEEN = {}                                   # repo -> (version, {file: local path}) for this process
import threading as _threading
_STORE_LOCK = _threading.Lock()                    # startup preload and a first request must not fetch at once


def _hf_token():
    token = os.environ.get("HF_TOKEN")
    if not token:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
    return token


def model_files(repo_name, files, revision=None, recheck=False):
    """Return (version, {file: local path}) for the given files of a Hugging Face model repo."""
    with _STORE_LOCK:
        return _model_files(repo_name, files, revision, recheck)


def _model_files(repo_name, files, revision=None, recheck=False):
    if repo_name in _STORE_SEEN and not recheck:
        return _STORE_SEEN[repo_name]
    from huggingface_hub import HfApi, hf_hub_download
    rev = revision or MODEL_REVISION
    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    sha, repo, token = None, None, None
    try:                                           # which version is released on Hugging Face?
        token = _hf_token()
        api = HfApi(token=token)
        repo = "{}/{}".format(api.whoami()["name"], repo_name)
        try:
            sha = api.model_info(repo, revision=rev).sha
        except Exception:
            sha = api.model_info(repo, revision="main").sha
    except Exception as e:
        print("Hugging Face not reachable ({}): using the stored model".format(type(e).__name__))
    if not conn:                                   # no Blob configured: plain Hugging Face (Kaggle, local)
        if sha is None:
            raise RuntimeError("model unavailable: Hugging Face unreachable and no Blob store configured")
        out = (sha, {f: hf_hub_download(repo, f, revision=sha, token=token) for f in files})
        _STORE_SEEN[repo_name] = out
        return out
    from azure.storage.blob import BlobServiceClient
    cont = BlobServiceClient.from_connection_string(conn).get_container_client(MODEL_CONTAINER)
    try:
        cont.create_container()
    except Exception:
        pass
    vblob = cont.get_blob_client("{}/VERSION".format(repo_name))
    try:
        stored = vblob.download_blob().readall().decode().strip()
    except Exception:
        stored = None
    if sha and sha != stored:                      # new release: copy it into Blob once
        print("new model release {} for {} (Blob had {}): copying to Blob".format(sha[:8], repo_name, (stored or "none")[:8]))
        for f in files:
            src = hf_hub_download(repo, f, revision=sha, token=token)
            with open(src, "rb") as fh:
                cont.get_blob_client("{}/{}/{}".format(repo_name, sha, f)).upload_blob(fh, overwrite=True)
        vblob.upload_blob(sha.encode(), overwrite=True)
        stored = sha
    if stored is None:
        raise RuntimeError("model unavailable: nothing in Blob and Hugging Face unreachable")
    # the version can already be in Blob without every file this engine needs (e.g. an earlier PyTorch build copied
    # best.pt for this release, the ONNX build now needs model.onnx): copy just the missing files, same revision
    for f in files:
        bc = cont.get_blob_client("{}/{}/{}".format(repo_name, stored, f))
        if bc.exists():
            continue
        if repo is None:
            raise RuntimeError("model file {} of release {} is not in Blob and Hugging Face is unreachable".format(
                f, stored[:8]))
        print("{} of release {} missing in Blob: copying it from Hugging Face".format(f, stored[:8]))
        src = hf_hub_download(repo, f, revision=stored, token=token)
        with open(src, "rb") as fh:
            bc.upload_blob(fh, overwrite=True)
    local = {}
    for f in files:                                # Blob -> local disk (same region: seconds)
        dst = os.path.join(MODEL_CACHE, repo_name, stored, f)
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            part = "{}.{}.part".format(dst, os.getpid())
            with open(part, "wb") as fh:
                cont.get_blob_client("{}/{}/{}".format(repo_name, stored, f)).download_blob().readinto(fh)
            os.replace(part, dst)
        local[f] = dst
    out = (stored, local)
    _STORE_SEEN[repo_name] = out
    return out


_MODEL = {"net": None, "dev": None, "rev": None, "checked": 0.0}
MODEL_REVISION = os.environ.get("PLANFORGE_MODEL_REV", "prod")   # HF tag/branch that is live; falls back to main
MODEL_CHECK_S = float(os.environ.get("PLANFORGE_MODEL_CHECK_S", "0"))  # 0 = check for a new release only at (cold) start


_LOAD_LOCK = _threading.Lock()


def load_model(force_check=False):
    with _LOAD_LOCK:
        return _load_model(force_check)


def _load_model(force_check=False):
    """The model is loaded once and kept in memory. Every MODEL_CHECK_S seconds (between plans) the
    Hugging Face release tag is checked; only if it moved is the new checkpoint downloaded and swapped in.
    If Hugging Face cannot be reached, the model already loaded keeps working."""
    import time
    now = time.time()
    if _MODEL["net"] is not None and not force_check and (MODEL_CHECK_S <= 0 or now - _MODEL["checked"] < MODEL_CHECK_S):
        return _MODEL["net"], _MODEL["dev"]
    _MODEL["checked"] = now
    if use_onnx():
        try:
            rev, paths = model_files(HF_REPO, ["checkpoints/model.onnx"], recheck=_MODEL["net"] is not None)
        except Exception as e:
            if _MODEL["net"] is not None:
                print("model update check failed ({}); keeping the loaded model".format(e))
                return _MODEL["net"], _MODEL["dev"]
            raise RuntimeError("ONNX model not found ({}). Run the export cell (kaggle/export_onnx.py) once to publish "
                               "checkpoints/model.onnx to Hugging Face.".format(e))
        if _MODEL["net"] is not None and rev == _MODEL["rev"]:
            return _MODEL["net"], _MODEL["dev"]
        sess = _onnx_session(paths["checkpoints/model.onnx"])
        meta = sess.get_modelmeta().custom_metadata_map
        _MODEL.update(net=sess, dev="onnx", rev=rev, info="{}, ONNX Runtime".format(meta.get("info", "")),
                      rotation_trained=meta.get("rotation_trained") == "1")
        print("model loaded (ONNX): revision", rev)
        return sess, "onnx"
    local = "/kaggle/working/checkpoints/best.pt"
    if os.path.exists(local):
        ckpt, rev = local, "local"
    else:
        try:
            rev, paths = model_files(HF_REPO, ["checkpoints/best.pt"], recheck=_MODEL["net"] is not None)
        except Exception as e:
            if _MODEL["net"] is not None:
                print("model update check failed ({}); keeping the loaded model".format(e))
                return _MODEL["net"], _MODEL["dev"]
            raise
        if _MODEL["net"] is not None and rev == _MODEL["rev"]:
            return _MODEL["net"], _MODEL["dev"]
        ckpt = paths["checkpoints/best.pt"]
    import torch, segmentation_models_pytorch as smp
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = smp.Unet(ck.get("encoder", "tu-efficientnet_b3"), encoder_weights=None, classes=27,
                   decoder_attention_type="scse")
    net.load_state_dict(ck["ema"]); net.to(dev).eval()
    info = "epoch {}, val score {:.3f}".format(ck.get("epoch", -1) + 1, ck.get("metrics", {}).get("score", float("nan")))
    _MODEL.update(net=net, dev=dev, rev=rev, info=info, rotation_trained=bool(ck.get("rotation_trained")))
    print("model loaded: revision", rev)
    return net, dev


def predict(image_src):
    import urllib.request
    _lap()
    net, dev = load_model()
    _lap("model load")
    if image_src.startswith("http"):
        req = urllib.request.Request(image_src, headers={"User-Agent": "Mozilla/5.0"})
        img = cv2.imdecode(np.frombuffer(urllib.request.urlopen(req, timeout=60).read(), np.uint8),
                           cv2.IMREAD_UNCHANGED)
    else:
        img = cv2.imread(image_src, cv2.IMREAD_UNCHANGED)
    assert img is not None, "could not read IMAGE"
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        a = img[:, :, 3:4] / 255.0
        img = (img[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    global LAST_ORIGINAL
    LAST_ORIGINAL = img                       # full-resolution copy for OCR (small text needs it)
    s = MAX_SIDE / max(img.shape[:2])
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    h, w = img.shape[:2]
    room_p, icon_p = _net_probs(net, dev, img, FLIP_TTA)
    views = 1
    if ROTATE_TTA and not _MODEL.get("rotation_trained"):
        # a model trained with rotated plans reads angled walls in one pass; older models learned from plans
        # drawn square to the page: show it extra copies rotated so each family
        # of angled walls lies flat, rotate its answers back and combine (rooms averaged, objects: most sure view).
        # Angles come from the walls the model found in the first pass, not from the ink: dimension ticks, hatching,
        # stair and furniture lines are diagonal too and would cost a full extra pass each.
        angs = _wall_angles(room_p.argmax(0) == WALL, 8.0)[:3]
        r_sum, r_w = room_p.copy(), np.ones((h, w), np.float32)
        for ang in angs:
            M, Mi, size = _rot_frame((h, w), ang)
            rimg = cv2.warpAffine(img, M, size, flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
            rp, ip = _net_probs(net, dev, rimg, FLIP_TTA)
            valid = cv2.warpAffine(np.ones(size[::-1], np.float32), Mi, (w, h), flags=cv2.INTER_NEAREST)
            back = lambda pr: np.stack([cv2.warpAffine(c, Mi, (w, h), flags=cv2.INTER_LINEAR) for c in pr])
            rp, ip = back(rp), back(ip)
            r_sum += rp * valid
            r_w += valid
            icon_p[1:] = np.maximum(icon_p[1:], ip[1:] * valid)
            views += 1
        room_p = r_sum / r_w
        if angs:
            print("model: also read at {} so angled walls / doors lie flat".format(
                ", ".join("{:.0f} deg".format(a_) for a_ in angs)))
    icon_p[0] = np.clip(1.0 - icon_p[1:].sum(0), 0, 1) if views > 1 else icon_p[0]
    room = room_p.argmax(0).astype(np.uint8)
    icon = icon_p.argmax(0).astype(np.uint8)
    global LAST_ICON_PROB
    LAST_ICON_PROB = (icon_p / np.maximum(icon_p.sum(0, keepdims=True), 1e-6)).astype(np.float16)
    print(f"model: {_MODEL.get('info', '')}, image {w}x{h}, {dev}, {views} view(s){' x2 (mirrored)' if FLIP_TTA else ''}")
    _lap("floor-plan model")
    return img, room, icon


# ----------------------------------------------------------------------------- 2. vectorise
def _components(mask, min_area):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [(lab == i, st[i]) for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_area]


def _cluster(values, tol):
    """Merge nearly equal coordinates so walls line up (grid snapping)."""
    order = np.argsort(values)
    out, group = np.array(values, float), [order[0]] if len(values) else []
    for a, b in zip(order[:-1], order[1:]):
        if values[b] - values[a] <= tol:
            group.append(b)
        else:
            out[group] = np.mean([values[i] for i in group]); group = [b]
    if group:
        out[group] = np.mean([values[i] for i in group])
    return out


def _straight_runs(c, L, G, T):
    """Split a blob (runs along axis 0) into straight, solid wall runs: (centre, a0, a1, thickness)."""
    across = c.sum(0)
    strong = across >= max(L * 0.5, 4)
    out, j = [], 0
    while j < len(strong):
        if not strong[j]:
            j += 1
            continue
        b0 = j
        while j + 1 < len(strong) and (strong[j + 1] or (j + 2 < len(strong) and strong[j + 2])):
            j += 1
        b1 = j
        j += 1
        bands = [(b0, b1)]
        if (b1 - b0 + 1) > 3 * T:
            # wide blob: keep only the columns that run much longer (a wall touching a solid block)
            a = across[b0:b1 + 1]
            keep = a >= 0.6 * a.max()
            sub, k = [], 0
            while k < len(keep):
                if keep[k]:
                    k0 = k
                    while k + 1 < len(keep) and keep[k + 1]:
                        k += 1
                    if (k - k0 + 1) <= 3 * T:
                        sub.append((b0 + k0, b0 + k))
                k += 1
            bands = sub                                 # nothing narrow enough -> blob, skip
        for b0, b1 in bands:
            out.extend(_band_runs(c, b0, b1, L, G))
    return out


def _band_runs(c, b0, b1, L, G):
    out = []
    band = c[:, b0:b1 + 1]
    rows = band.any(1)
    idx = np.where(rows)[0]
    if len(idx) == 0:
        return out
    # split where the band has a long gap (two collinear walls)
    cuts = np.where(np.diff(idx) > G)[0]
    for r0, r1 in zip(np.r_[idx[0], idx[cuts + 1]], np.r_[idx[cuts], idx[-1]]):
        if r1 - r0 + 1 < L:
            continue
        seg = band[r0:r1 + 1]
        if seg.any(1).mean() < 0.6:                # mostly empty -> bridged noise, not a wall
            continue
        widths = seg.sum(1)
        t = float(np.median(widths[widths > 0]))
        cols = seg.sum(0)
        centre = b0 + float((cols * np.arange(len(cols))).sum() / cols.sum())
        out.append((centre, float(r0) - 0.5, float(r1) + 0.5, t))   # pixel centres at integers, like `centre`
    return out


def _frame_walls(full, openings_any, icon, T):
    """Horizontal / vertical walls + their openings in one frame (the image, or the image rotated so a
    family of angled walls runs horizontally)."""
    # straight segments: horizontal / vertical runs longer than ~1.6 T. The run length must stay above the
    # thickness of the walls crossing this direction, otherwise a wall's cross-section would read as a wall.
    # Nothing is bridged: a break the model left in a wall stays a break (only 1-2 px antialiasing gaps close).
    L = max(int(round(1.6 * T)), 5) | 1          # odd sizes: an even kernel shifts every wall by 1 px
    segs = []
    G = 3
    for o, ker, gap in (("h", np.ones((1, L), np.uint8), np.ones((1, G), np.uint8)),
                        ("v", np.ones((L, 1), np.uint8), np.ones((G, 1), np.uint8))):
        # merge double-line walls (window drawn as two thin parallel lines) into one solid band
        P = max(3, int(round(0.8 * T))) | 1
        perp = np.ones((P, 1), np.uint8) if o == "h" else np.ones((1, P), np.uint8)
        m = cv2.morphologyEx(full.astype(np.uint8), cv2.MORPH_CLOSE, perp)
        # keep only runs in this direction, then close 1-2 px gaps (no larger bridging: the model decides)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, gap)
        for comp, st in _components(m, L):
            x, y, w, h = st[:4]
            c = comp[y:y + h, x:x + w]
            if o == "h":                         # work in "vertical" orientation: runs go down rows
                c, off_run, off_across = c.T, x, y
            else:
                off_run, off_across = y, x
            for centre, a0, a1, t in _straight_runs(c, L, G, T):
                segs.append({"o": o, "c": off_across + centre, "a0": float(off_run + a0),
                             "a1": float(off_run + a1), "t": t})      # thickness as the model drew it
    for o in "hv":
        idx = [i for i, s in enumerate(segs) if s["o"] == o]
        if idx:
            snapped = (_cluster([segs[i]["c"] for i in idx], 0.75 * T) if "align_walls" in ASSIST
                       else [segs[i]["c"] for i in idx])
            for i, c in zip(idx, snapped):
                segs[i]["c"] = float(c)
    # merge collinear pieces of the same wall
    segs.sort(key=lambda s: (s["o"], round(s["c"], 1), s["a0"]))
    merged_segs = []
    for sg in segs:
        p = merged_segs[-1] if merged_segs else None
        bridged = False
        if p and p["o"] == sg["o"] and abs(p["c"] - sg["c"]) < 0.5 and sg["a0"] - p["a1"] < 30 * T:
            # a gap that contains door/window pixels is part of the same wall
            c0, c1 = int(sg["c"] - sg["t"] / 2 - 2), int(sg["c"] + sg["t"] / 2 + 3)
            g0, g1 = int(p["a1"]), int(sg["a0"]) + 1
            band = openings_any[max(c0, 0):c1, g0:g1] if sg["o"] == "h" else openings_any[g0:g1, max(c0, 0):c1]
            bridged = band.size > 0 and band.any(axis=0 if sg["o"] == "h" else 1).mean() > 0.25
        join_gap = 1.5 * T if "join_gaps" in ASSIST else 2.0      # px: only pieces that touch, unless asked
        if p and p["o"] == sg["o"] and abs(p["c"] - sg["c"]) < 0.5 and (sg["a0"] <= p["a1"] + join_gap or bridged):
            lp, ls = p["a1"] - p["a0"], sg["a1"] - sg["a0"]
            p["t"] = (p["t"] * lp + sg["t"] * ls) / (lp + ls)
            p["a1"] = max(p["a1"], sg["a1"])
        else:
            merged_segs.append(dict(sg))
    segs = merged_segs
    # snap ends onto perpendicular walls (clean corners / T-junctions); two passes so both meet
    for _ in range(2):
        for sg in segs:
            for p in segs:
                reach = sg["t"] / 2 + (1.5 * T if "snap_corners" in ASSIST else 2.0)
                if p["o"] == sg["o"] or not (p["a0"] - reach <= sg["c"] <= p["a1"] + reach):
                    continue
                tol = p["t"] / 2 + (1.5 * T if "snap_corners" in ASSIST else 2.0)
                if abs(sg["a0"] - p["c"]) <= tol and abs(sg["a1"] - p["c"]) <= tol:
                    continue                         # both ends inside the same wall: a block (column), not a joint
                if abs(sg["a0"] - p["c"]) <= tol:
                    sg["a0"] = p["c"] - p["t"] / 2
                    sg["j0"] = (p["c"], p["t"])      # joined: perpendicular centre-line + thickness
                if abs(sg["a1"] - p["c"]) <= tol:
                    sg["a1"] = p["c"] + p["t"] / 2
                    sg["j1"] = (p["c"], p["t"])
    segs = [sg for sg in segs if sg["a1"] - sg["a0"] >= 1.5 * T]

    # openings -> intervals on their host wall
    openings = []
    for kind, cls in (("door", DOOR), ("window", WIN)):
        for comp, st in _components(icon == cls, max(12, 0.3 * T * T)):
            x, y, w, h = [float(v) for v in st[:4]]
            x, y = x - 0.5, y - 0.5                     # pixel edges, same convention as the wall centre-lines
            cx, cy = x + w / 2, y + h / 2
            best, best_ov = None, 0.0
            for i, s in enumerate(segs):
                # blob centre must sit on the wall band; score = overlap along the wall axis
                if s["o"] == "h":
                    near, ov = abs(cy - s["c"]) <= s["t"] / 2 + T, min(x + w, s["a1"]) - max(x, s["a0"])
                else:
                    near, ov = abs(cx - s["c"]) <= s["t"] / 2 + T, min(y + h, s["a1"]) - max(y, s["a0"])
                if near and ov > best_ov:
                    best, best_ov = i, ov
            if best is None:
                continue
            s = segs[best]
            u0, u1 = (x, x + w) if s["o"] == "h" else (y, y + h)
            u0, u1 = max(u0, s["a0"]), min(u1, s["a1"])
            if u1 - u0 >= 0.5 * T:
                openings.append({"kind": kind, "wall": best, "u0": u0, "u1": u1})
    # merge overlapping / nearly touching openings on the same wall (doors win over windows)
    openings.sort(key=lambda o: (o["wall"], o["u0"]))
    merged = []
    for o in openings:
        p = merged[-1] if merged else None
        if p and p["wall"] == o["wall"] and o["u0"] <= p["u1"] + T:
            if o["kind"] == p["kind"] or "door" in (o["kind"], p["kind"]):
                p["u1"] = max(p["u1"], o["u1"])
                p["kind"] = "door" if "door" in (o["kind"], p["kind"]) else p["kind"]
                continue
        merged.append(dict(o))

    return segs, merged


def _wall_angles(full, T):
    """Dominant wall directions (degrees, mod 90) other than the image axes, e.g. 30 / 60 for a hexagonal plan.
    Measured from the direction of the mask's edges after smoothing over about a wall thickness, so ragged mask
    edges (JPEG / phone photos) still give clean peaks; straight-edge (Hough) detection breaks up on them."""
    m = full.astype(np.float32)
    sig = max(1.0, T / 2)
    m = cv2.GaussianBlur(m, (0, 0), sig)
    gx = cv2.Sobel(m, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(m, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    sel = mag > 0.25 * mag.max() if mag.max() > 0 else mag > 0
    if not sel.any():
        return []
    # edge normal -> wall direction (+90); weight by edge strength
    ang_all = (np.degrees(np.arctan2(gy[sel], gx[sel])) + 90.0) % 90.0
    hist = np.bincount(np.round(ang_all).astype(int) % 90, weights=mag[sel], minlength=90)[:90]
    sm = np.array([hist[[(i + k) % 90 for k in (-2, -1, 0, 1, 2)]].sum() for i in range(90)])
    out = []
    total = sm.sum()
    for i in np.argsort(-sm):
        if sm[i] < 0.04 * total:
            break
        if min(i, 90 - i) <= 3 or any(min(abs(i - j), 90 - abs(i - j)) <= 3 for j in out):
            continue                        # the image axes, or next to an angle already taken
        # refine to a weighted mean inside the peak
        ks = [(i + k) % 90 for k in range(-3, 4)]
        w = hist[ks]
        ang = (i + np.dot(w, np.arange(-3, 4)) / max(w.sum(), 1e-9)) % 90
        out.append(float(ang))
    return out


def _rot_frame(shape, ang):
    """Affine that rotates the image by -ang (degrees) into an enlarged canvas, and its inverse."""
    H, W = shape
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)       # positive ang = counter-clockwise on screen
    c, s_ = abs(M[0, 0]), abs(M[0, 1])
    nW, nH = int(H * s_ + W * c) + 2, int(H * c + W * s_) + 2
    M[0, 2] += nW / 2 - W / 2
    M[1, 2] += nH / 2 - H / 2
    Mi = cv2.invertAffineTransform(M)
    return M, Mi, (nW, nH)


def _hv_point(o, c, u):
    return (u, c) if o == "h" else (c, u)


def _join_across(segs, T):
    """Ends of walls that meet at an angle (not both h / v in the same frame): extend / trim each end to the
    intersection of the two centre-lines and remember the other wall's thickness (corner solids close)."""
    def line(sg):
        p0, p1 = np.array(sg["p0"]), np.array(sg["p1"])
        L = max(np.linalg.norm(p1 - p0), 1e-9)
        return p0, (p1 - p0) / L, L
    for _ in range(2):
        for sg in segs:
            p0, d, L = line(sg)
            for end, key in ((0, "j0"), (1, "j1")):
                if sg.get(key + "_hv"):
                    continue                                   # already joined to a perpendicular wall in its frame
                pe = np.array(sg["p0"] if end == 0 else sg["p1"])
                best = None
                for q in segs:
                    if q is sg:
                        continue
                    q0, e, M = line(q)
                    cr = d[0] * e[1] - d[1] * e[0]
                    if abs(cr) < np.sin(np.radians(12)):      # parallel-ish: not a corner
                        continue
                    # intersection of the two centre-lines
                    tq = ((pe[0] - q0[0]) * d[1] - (pe[1] - q0[1]) * d[0]) / (e[0] * d[1] - e[1] * d[0])
                    X = q0 + e * tq
                    slack = 2 * T if "snap_corners" in ASSIST else 2.0
                    reach_q = q["t"] / 2 + slack
                    if not (-reach_q <= tq <= M + reach_q):
                        continue
                    dist = np.linalg.norm(X - pe)
                    if dist > q["t"] / (2 * abs(cr)) + slack:
                        continue
                    # do not cut a wall in half: the new end must stay on the same side of the old one
                    u = (X - p0) @ d
                    if (end == 0 and u > 0.6 * L) or (end == 1 and u < 0.4 * L):
                        continue
                    if best is None or dist < best[0]:
                        best = (dist, X, q["t"])
                if best is not None:
                    if end == 0:
                        sg["p0"] = [float(best[1][0]), float(best[1][1])]
                    else:
                        sg["p1"] = [float(best[1][0]), float(best[1][1])]
                    sg[key] = (None, best[2])
    return segs


def _edge_direction(full, p0, p1, t, T):
    """Direction (degrees, mod 180) and coherence (0..1) of the model mask's edges around a wall piece."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    pad = int(np.ceil(t / 2 + T)) + 3
    H, W = full.shape
    x0 = max(0, int(min(p0[0], p1[0])) - pad); x1 = min(W, int(max(p0[0], p1[0])) + pad + 1)
    y0 = max(0, int(min(p0[1], p1[1])) - pad); y1 = min(H, int(max(p0[1], p1[1])) + pad + 1)
    win = cv2.GaussianBlur(full[y0:y1, x0:x1].astype(np.float32), (0, 0), max(1.0, T / 4))
    gx, gy = cv2.Sobel(win, cv2.CV_32F, 1, 0), cv2.Sobel(win, cv2.CV_32F, 0, 1)
    d = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)
    n = np.array([-d[1], d[0]]) * (t / 2 + T / 2)
    m = np.zeros(win.shape, np.uint8)
    o = np.array([x0, y0])
    cv2.fillPoly(m, [np.round(np.array([p0 - n, p1 - n, p1 + n, p0 + n]) - o).astype(np.int32)], 1)
    gx, gy = gx[m > 0], gy[m > 0]
    jxx, jyy, jxy = float((gx * gx).sum()), float((gy * gy).sum()), float((gx * gy).sum())
    if jxx + jyy < 1e-6:
        return None, 0.0
    coh = np.hypot(jxx - jyy, 2 * jxy) / (jxx + jyy)
    normal = 0.5 * np.degrees(np.arctan2(2 * jxy, jxx - jyy))
    return (normal + 90) % 180, float(coh)


def _drop_cross_stubs(full, segs, T):
    """A short straight-pass (h / v) piece whose surrounding mask edges run at a clearly different angle is a
    stub across a stair-stepped angled or curved wall band, not a wall: drop it so the tracer follows the band
    as one piece. Real short h / v walls and square columns have h / v edges and stay."""
    for sg in segs:
        if sg.get("drop") or sg["frame"] != 0:
            continue
        p0, p1 = np.array(sg["p0"]), np.array(sg["p1"])
        L = np.linalg.norm(p1 - p0)
        if L >= 4 * T:
            continue
        ang, coh = _edge_direction(full, p0, p1, sg["t"], T)
        if ang is None or coh < 0.5:
            continue
        own = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])) % 180
        diff = abs(ang - own) % 180
        diff = min(diff, 180 - diff)
        if diff > 25:
            sg["drop"] = True
    return segs


def _join_trace_ends(segs, traced, T):
    """Close the corners where a traced chain meets another wall. The straight wall already claimed the corner
    pixels, so the chain stops about half a wall short of it, often with a tiny kink where the mask's centre-line
    bends into the junction. Drop that kink (shorter than ~T, at a free chain end) and run the chain's end on
    to the other wall's centre-line, only through the corner the model marked (at most ~T past its old end)."""
    seglen = lambda sg: np.hypot(sg["p1"][0] - sg["p0"][0], sg["p1"][1] - sg["p0"][1])
    chains = {}
    for pc in traced:
        if "_chain" in pc:
            chains.setdefault(pc["_chain"], []).append(pc)
    for ch in chains.values():                    # pieces are in path order; trim tiny kinks at free ends
        while len(ch) > 1 and seglen(ch[0]) < T and "j0" not in ch[0]:
            ch.pop(0)["drop"] = True
            ch[0].pop("j0", None)
        while len(ch) > 1 and seglen(ch[-1]) < T and "j1" not in ch[-1]:
            ch.pop()["drop"] = True
            ch[-1].pop("j1", None)
    traced = [pc for pc in traced if not pc.pop("drop", False)]
    walls = segs + traced
    for pc in traced:
        if "_chain" not in pc:
            continue                              # solid blocks stand on their own
        p0, p1 = np.array(pc["p0"]), np.array(pc["p1"])
        L = max(np.linalg.norm(p1 - p0), 1e-9)
        d = (p1 - p0) / L
        for end, key in ((0, "j0"), (1, "j1")):
            if key in pc:
                continue
            pe = p0 if end == 0 else p1
            out_d = -d if end == 0 else d         # pointing away from the piece
            best = None
            for q in walls:
                if q is pc or q.get("_chain", -1) == pc["_chain"]:
                    continue
                q0, q1 = np.array(q["p0"]), np.array(q["p1"])
                M = max(np.linalg.norm(q1 - q0), 1e-9)
                e = (q1 - q0) / M
                cr = d[0] * e[1] - d[1] * e[0]
                if abs(cr) < np.sin(np.radians(12)):
                    continue
                tq = ((pe[0] - q0[0]) * d[1] - (pe[1] - q0[1]) * d[0]) / (e[0] * d[1] - e[1] * d[0])
                if not (-T <= tq <= M + T):
                    continue
                X = q0 + e * tq
                s = (X - pe) @ out_d               # how far the end moves (outwards > 0)
                if s < -q["t"] / 2 or s > q["t"] / (2 * abs(cr)) + T:
                    continue
                if abs(s) > 0.6 * L and s < 0:     # never eat most of the piece
                    continue
                if best is None or abs(s) < best[0]:
                    best = (abs(s), X, q["t"], q, tq, M)
            if best is not None:
                _, X, qt, q, tq, M = best
                pc["p0" if end == 0 else "p1"] = [float(X[0]), float(X[1])]
                pc[key] = (None, qt)
                # the other wall ends right there too (a free end inside the corner block): end it at the
                # shared corner instead of letting it poke through the joint
                reach = pc["t"] / 2 + T
                for qend, qkey, near in ((0, "j0", tq <= reach), (1, "j1", tq >= M - reach)):
                    if near and qkey not in q and not q.get(qkey + "_hv") and 0 < tq < M:
                        q["p0" if qend == 0 else "p1"] = [float(X[0]), float(X[1])]
                        q[qkey] = (None, pc["t"])
    for pc in traced:
        pc.pop("_chain", None)
    return [pc for pc in traced if seglen(pc) >= 1]


def _drawn_windows(segs, merged, image, T, model_shape):
    """Windows the model did not mark, read from the drawing: a gap between two collinear walls with 2+ thin
    lines running along the wall inside its thickness. The two walls become one wall with a window in it.
    Gaps without such lines (doors, passages) are left alone."""
    if image is None:
        return segs, merged, 0
    g = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    k = g.shape[1] / model_shape[1]                     # drawing px per model px
    Hg, Wg = g.shape
    found = 0
    changed = True
    while changed:
        changed = False
        for i, a in enumerate(segs):
            pa0, pa1 = np.array(a["p0"]), np.array(a["p1"])
            La = np.linalg.norm(pa1 - pa0)
            if La < 1e-6:
                continue
            d = (pa1 - pa0) / La
            n = np.array([-d[1], d[0]])
            for j, b in enumerate(segs):
                if j == i:
                    continue
                pb0, pb1 = np.array(b["p0"]), np.array(b["p1"])
                Lb = np.linalg.norm(pb1 - pb0)
                if Lb < 1e-6:
                    continue
                db = (pb1 - pb0) / Lb
                if abs(d[0] * db[1] - d[1] * db[0]) > 0.035:
                    continue                                   # not parallel
                # b must start (one of its ends) just after a ends, on the same line
                ends_b = sorted([pb0, pb1], key=lambda q: (q - pa0) @ d)
                near, far = ends_b
                gap = (near - pa1) @ d
                if not (1.5 * T <= gap <= 14 * T) or abs((near - pa1) @ n) > 0.6 * T or abs((far - pa0) @ n) > 0.6 * T:
                    continue
                t = max(a["t"], b["t"])
                # perpendicular profiles across the wall thickness, along the gap (drawing resolution)
                hits, samples = 0, 0
                for f in np.linspace(0.15, 0.85, 12):
                    c = (pa1 + d * gap * f) * k
                    prof = []
                    for o in np.linspace(-(t / 2 + 1.5) * k, (t / 2 + 1.5) * k, max(9, int(3 * t * k))):
                        x, y = int(round(c[0] + n[0] * o)), int(round(c[1] + n[1] * o))
                        prof.append(g[y, x] < 140 if 0 <= x < Wg and 0 <= y < Hg else False)
                    runs = int(np.sum(np.diff(np.concatenate([[0], np.array(prof, int), [0]])) == 1))
                    samples += 1
                    hits += 2 <= runs <= 5
                if hits < 0.7 * samples:
                    continue
                # merge b into a, window in the gap
                new_end = far
                a["p1"] = [float(new_end[0]), float(new_end[1])]
                src = "j1" if far is pb1 else "j0"              # b's far end keeps its corner join
                for suffix in ("", "_hv"):
                    if src + suffix in b:
                        a["j1" + suffix] = b[src + suffix]
                    else:
                        a.pop("j1" + suffix, None)
                a["t"] = (a["t"] * La + b["t"] * Lb) / (La + Lb)
                for o in merged:
                    if o["wall"] == j:
                        o["wall"] = i
                merged.append({"kind": "window", "wall": i, "u0": 0.0, "u1": 0.0,
                               "q0": [float(pa1[0]), float(pa1[1])], "q1": [float(near[0]), float(near[1])],
                               "source": "drawing"})
                segs.pop(j)
                for o in merged:
                    if o["wall"] > j:
                        o["wall"] -= 1
                found += 1
                changed = True
                break
            if changed:
                break
    return segs, merged, found


def _skeleton_paths(sk):
    """Skeleton pixels -> list of ordered pixel paths, split at junctions and ends."""
    ys, xs = np.nonzero(sk)
    pix = set(zip(xs.tolist(), ys.tolist()))
    nb = lambda p: [(p[0] + dx, p[1] + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                    if (dx or dy) and (p[0] + dx, p[1] + dy) in pix]
    deg = {p: len(nb(p)) for p in pix}
    nodes = {p for p, d in deg.items() if d != 2}
    seen_edges, paths = set(), []

    def walk(a, b):
        path = [a, b]
        prev, cur = a, b
        while cur not in nodes:
            nxt = [q for q in nb(cur) if q != prev and (min(cur, q), max(cur, q)) not in seen_edges]
            if not nxt:
                break
            seen_edges.add((min(cur, nxt[0]), max(cur, nxt[0])))
            prev, cur = cur, nxt[0]
            path.append(cur)
        return path

    for a in nodes:
        for b in nb(a):
            e = (min(a, b), max(a, b))
            if e in seen_edges:
                continue
            seen_edges.add(e)
            paths.append(walk(a, b))
    left = [p for p in pix if p not in nodes and not any((min(p, q), max(p, q)) in seen_edges for q in nb(p))]
    while left:                                               # closed loops (no ends, no junctions)
        a = left[0]
        nodes.add(a)
        for b in nb(a)[:1]:
            seen_edges.add((min(a, b), max(a, b)))
            paths.append(walk(a, b))
        nodes.discard(a)
        done = {q for pth in paths for q in pth}
        left = [p for p in left if p not in done]
    return paths


K3_ = np.ones((3, 3), np.uint8)


def _prune_spurs(sk, T):
    """Remove skeleton side branches shorter than ~T (a free end to a junction): edge noise, not wall."""
    sk = sk.copy()
    for _ in range(3):
        paths = _skeleton_paths(sk)
        if len(paths) <= 1:
            break
        deg = {}
        for pth in paths:
            for q in (pth[0], pth[-1]):
                deg[q] = deg.get(q, 0) + 1
        cut = False
        for pth in paths:
            a, b = pth[0], pth[-1]
            free_a, free_b = deg[a] == 1, deg[b] == 1
            if len(pth) < T and (free_a != free_b):          # one free end, the other at a junction
                for q in pth:
                    if q != (b if free_a else a):
                        sk[q[1], q[0]] = False
                cut = True
        if not cut:
            break
    return sk


def _trace_leftovers(model_wall, segs, T):
    """Model wall pixels that no straight wall explains (curved walls, irregular pieces): follow the mask's
    centre-line with straight pieces (Revit walls are straight) no further than ~T/4 from it, thickness as drawn."""
    claimed = np.zeros(model_wall.shape, np.uint8)
    for sg in segs:
        p0, p1 = np.array(sg["p0"]), np.array(sg["p1"])
        d = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)
        n = np.array([-d[1], d[0]]) * (sg["t"] / 2 + 1)
        # joined ends reach on by half the other wall's thickness (the corner block the 3D / Revit wall fills)
        p0 = p0 - d * ((sg["j0"][1] / 2 if "j0" in sg else 0) + 1)
        p1 = p1 + d * ((sg["j1"][1] / 2 if "j1" in sg else 0) + 1)
        cv2.fillPoly(claimed, [np.round(np.array([p0 - n, p1 - n, p1 + n, p0 + n])).astype(np.int32)], 1)
    left = model_wall & ~(claimed > 0)
    out, chain_id = [], [0]
    for comp, st in _components(left, max(4, int(0.25 * T * T))):
        dist = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 3)
        # centre-line of the band with its ragged edges smoothed (JPEG / photo noise grows side branches that
        # cut the line into pieces too short to trace), then the remaining short side branches pruned
        sm = cv2.GaussianBlur(comp.astype(np.float32), (0, 0), max(1.0, T / 4)) > 0.5
        sk = _prune_spurs(skeletonize(sm & (cv2.dilate(comp.astype(np.uint8), K3_) > 0)), T)
        if comp.sum() < 2 * sk.sum():                        # a 1 px sliver along a wall's edge (rounding)
            continue
        ys_, xs_ = np.nonzero(comp)
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(np.stack([xs_, ys_], 1).astype(np.float32))
        rw, rh = rw + 1, rh + 1                              # pixel centres -> pixel edges
        if comp.sum() >= 0.8 * rw * rh and (sk.sum() < 1.5 * T or max(rw, rh) < 3 * min(rw, rh)):
            # a solid block (column, pier, short thick piece): one wall exactly over it, its real size and angle
            a_ = np.radians(ang if rw >= rh else ang + 90)
            d = np.array([np.cos(a_), np.sin(a_)]) * max(rw, rh) / 2
            out.append({"o": "a", "frame": "trace", "t": float(min(rw, rh)), "p0": [cx - d[0], cy - d[1]],
                        "p1": [cx + d[0], cy + d[1]], "j0_hv": False, "j1_hv": False})
            continue
        if sk.sum() < 1.5 * T:
            continue
        for path in _skeleton_paths(sk):
            chain_id[0] += 1
            if len(path) < 1.5 * T:
                continue
            pts = np.array(path, np.float32).reshape(-1, 1, 2)
            poly = cv2.approxPolyDP(pts, max(1.0, T / 4), False).reshape(-1, 2).astype(float)
            idx = [int(np.argmin(np.abs(np.array(path) - q).sum(1))) for q in poly]
            pieces = []
            for k in range(len(poly) - 1):
                seg_px = np.array(path[idx[k]:idx[k + 1] + 1])
                t = 2 * float(np.median(dist[seg_px[:, 1], seg_px[:, 0]])) if len(seg_px) else T
                pieces.append({"o": "a", "frame": "trace", "t": max(t, 1.0), "_chain": chain_id[0],
                               "p0": [float(poly[k][0]), float(poly[k][1])],
                               "p1": [float(poly[k + 1][0]), float(poly[k + 1][1])],
                               "j0_hv": False, "j1_hv": False})
            for k, pc in enumerate(pieces):                   # neighbours share their bend point: close the joint
                if k > 0:
                    pc["j0"] = (None, pieces[k - 1]["t"])
                if k < len(pieces) - 1:
                    pc["j1"] = (None, pieces[k + 1]["t"])
            out.extend(pc for pc in pieces if np.hypot(pc["p1"][0] - pc["p0"][0], pc["p1"][1] - pc["p0"][1]) >= 1)
    return out


def vectorise(room, icon, image=None):
    wall = room == WALL
    openings_any = np.isin(icon, [WIN, DOOR])
    # typical wall thickness T (px) from the medial axis. Measured on a lightly smoothed copy: ragged edges
    # (JPEG / phone plans, stair-stepped angled walls) give the raw skeleton many spurs near the edges, which
    # dragged T down to ~1/3 and broke every threshold that scales with it. Only T uses the smoothed copy.
    smooth = cv2.medianBlur(wall.astype(np.uint8) * 255, 5) > 127
    if smooth.sum() < 0.5 * wall.sum():                         # very thin walls: the filter would erase them
        smooth = wall
    dist = cv2.distanceTransform(smooth.astype(np.uint8), cv2.DIST_L2, 5)
    sk = skeletonize(smooth)
    T = float(np.clip(2 * np.median(dist[sk]) if sk.any() else 6, 3, 60))
    model_wall = wall.copy()
    # the model's wall mask is trusted: only 1-2 px specks are removed (no thickness filtering,
    # so thin partitions and walls detached from the main structure are kept)
    # (no morphological opening: it erases thin double-line window walls the model got right;
    #  specks are removed by size below)
    comps = _components(wall, max(4, int(0.25 * T * T)))     # only specks well under a wall's size
    if not comps:
        raise RuntimeError("No walls detected.")
    wall = np.any([m for m, s in comps], axis=0)
    # add whole door/window blobs that touch a wall (they fill the gaps in the wall lines)
    near = cv2.dilate(wall.astype(np.uint8), np.ones((int(2 * T),) * 2)) > 0
    lab, n = ndimage.label(openings_any)
    keep = np.unique(lab[near & (lab > 0)])
    full = wall | np.isin(lab, keep[keep > 0])
    full = cv2.morphologyEx(full.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0

    segs, merged = _frame_walls(full, openings_any, icon, T)
    for sg in segs:                                            # every wall also as two end points (image px)
        sg["frame"] = 0
        e0 = sg["j0"][0] if "j0" in sg else sg["a0"]
        e1 = sg["j1"][0] if "j1" in sg else sg["a1"]
        sg["p0"], sg["p1"] = list(_hv_point(sg["o"], sg["c"], e0)), list(_hv_point(sg["o"], sg["c"], e1))
        sg["j0_hv"], sg["j1_hv"] = "j0" in sg, "j1" in sg
    for o in merged:
        sg = segs[o["wall"]]
        o["q0"], o["q1"] = list(_hv_point(sg["o"], sg["c"], o["u0"])), list(_hv_point(sg["o"], sg["c"], o["u1"]))
    # angled walls (30 / 45 / 60 degree wings ...): rotate the image so they run straight, extract, rotate back
    claimed = np.zeros(full.shape, np.uint8)
    for sg in segs:
        cv2.line(claimed, tuple(int(round(v_)) for v_ in sg["p0"]), tuple(int(round(v_)) for v_ in sg["p1"]),
                 1, max(1, int(round(sg["t"] + T))))
    done_angles = []
    # angles from the WHOLE wall mask: the straight pass above already covers a 30 / 60 degree wall with
    # horizontal / vertical stubs (a 30 degree wall is 2x its thickness wide), so "what is still unexplained"
    # hides exactly the walls we are looking for. The duplicate check below keeps the clean angled wall.
    for ang in _wall_angles(full, T)[:6]:
        if any(min(abs(ang - b_), 90 - abs(ang - b_)) <= 2 for b_ in done_angles):
            continue
        done_angles.append(ang)
        M, Mi, size = _rot_frame(full.shape, ang)
        rest = full.astype(np.uint8)          # whole mask: frame 0 may have chopped this wall into stubs
        rf = cv2.warpAffine(rest, M, size, flags=cv2.INTER_NEAREST) > 0
        ro = cv2.warpAffine(openings_any.astype(np.uint8), M, size, flags=cv2.INTER_NEAREST) > 0
        ri = cv2.warpAffine(icon.astype(np.uint8), M, size, flags=cv2.INTER_NEAREST)
        fsegs, fops = _frame_walls(rf, ro, ri, T)
        back = lambda pt: [float(Mi[0, 0] * pt[0] + Mi[0, 1] * pt[1] + Mi[0, 2]),
                           float(Mi[1, 0] * pt[0] + Mi[1, 1] * pt[1] + Mi[1, 2])]
        base = len(segs)
        for o in fops:                                         # openings: frame coordinates -> image px
            fs = fsegs[o["wall"]]
            merged.append(dict(o, wall=base + o["wall"], q0=back(_hv_point(fs["o"], fs["c"], o["u0"])),
                               q1=back(_hv_point(fs["o"], fs["c"], o["u1"]))))
        for sg in fsegs:
            e0 = sg["j0"][0] if "j0" in sg else sg["a0"]
            e1 = sg["j1"][0] if "j1" in sg else sg["a1"]
            sg.update({"frame": ang, "p0": back(_hv_point(sg["o"], sg["c"], e0)),
                       "p1": back(_hv_point(sg["o"], sg["c"], e1)), "o": "a",
                       "j0_hv": "j0" in sg, "j1_hv": "j1" in sg})
            segs.append(sg)
            cv2.line(claimed, tuple(int(round(v_)) for v_ in sg["p0"]), tuple(int(round(v_)) for v_ in sg["p1"]),
                     1, max(1, int(round(sg["t"] + T))))
    # walls at their OWN angle: a photographed or hand-drawn plan has each wing at a slightly different angle
    # (26-33 and 59-64 degrees on one real plan), so one pass per main angle reads a 32 degree wall in the
    # 29 degree frame as a staircase. Every long straight wall line in the mask that no piece above follows is
    # read again in a small window turned to exactly its angle.
    n_local = 0
    sk_s = skeletonize(cv2.medianBlur(full.astype(np.uint8) * 255, 5) > 127).astype(np.uint8) * 255
    hl = cv2.HoughLinesP(sk_s, 1, np.pi / 720, threshold=max(20, int(3 * T)), minLineLength=max(int(5 * T), 30),
                         maxLineGap=max(int(1.2 * T), 4))
    groups = []
    for x0_, y0_, x1_, y1_ in (hl.reshape(-1, 4) if hl is not None else []):
        q0, q1 = np.array([x0_, y0_], float), np.array([x1_, y1_], float)
        ln = np.linalg.norm(q1 - q0)
        ang_ = float(np.degrees(np.arctan2(q1[1] - q0[1], q1[0] - q0[0])) % 180)
        tilt = ang_ % 90
        if min(tilt, 90 - tilt) < 0.75:                    # straight walls: the axis pass reads them
            continue
        d_ = (q1 - q0) / ln
        n_ = np.array([-d_[1], d_[0]])
        covered = 0.0
        for sg in segs:                                    # already followed by one piece at this angle?
            a_s, b_s = np.array(sg["p0"], float), np.array(sg["p1"], float)
            ls = np.linalg.norm(b_s - a_s)
            if ls < 1:
                continue
            ds = (b_s - a_s) / ls
            if abs(abs(ds @ d_) - 1) > 1.4e-5:             # more than ~0.3 degree apart (photos bend walls a
                continue                                   # little: a near miss is not "already read")
            if abs((a_s - q0) @ n_) > T / 2 or abs((b_s - q0) @ n_) > T / 2:
                continue
            u = sorted([(a_s - q0) @ d_, (b_s - q0) @ d_])
            covered = max(covered, (min(u[1], ln) - max(u[0], 0)) / ln)
        if covered >= 0.9:
            continue
        for g in groups:                                   # same wall line as a group already found?
            if abs(abs(g["d"] @ d_) - 1) < 3e-4 and abs((q0 - g["q0"]) @ g["n"]) < T and abs((q1 - g["q0"]) @ g["n"]) < T:
                g["pts"] += [q0, q1]
                g["w"] += ln
                break
        else:
            groups.append({"d": d_, "n": n_, "q0": q0, "pts": [q0, q1], "w": ln, "ang": ang_})
    groups.sort(key=lambda g: -g["w"])
    Hh0, Ww0 = full.shape
    for gi, g in enumerate(groups[:48]):
        P_ = np.array(g["pts"])
        pad = int(2 * T + 10)
        x0_, y0_ = max(int(P_[:, 0].min()) - pad, 0), max(int(P_[:, 1].min()) - pad, 0)
        x1_, y1_ = min(int(P_[:, 0].max()) + pad + 1, Ww0), min(int(P_[:, 1].max()) + pad + 1, Hh0)
        sub = full[y0_:y1_, x0_:x1_].astype(np.uint8)
        M, Mi, size = _rot_frame(sub.shape, g["ang"])
        rf = cv2.warpAffine(sub, M, size, flags=cv2.INTER_NEAREST) > 0
        ro = cv2.warpAffine(openings_any[y0_:y1_, x0_:x1_].astype(np.uint8), M, size, flags=cv2.INTER_NEAREST) > 0
        ri = cv2.warpAffine(icon[y0_:y1_, x0_:x1_].astype(np.uint8), M, size, flags=cv2.INTER_NEAREST)
        fsegs, fops = _frame_walls(rf, ro, ri, T)
        back = lambda pt, Mi=Mi, x0_=x0_, y0_=y0_: [float(Mi[0, 0] * pt[0] + Mi[0, 1] * pt[1] + Mi[0, 2] + x0_),
                                                   float(Mi[1, 0] * pt[0] + Mi[1, 1] * pt[1] + Mi[1, 2] + y0_)]
        key = "line{}".format(gi)
        keep = {}
        for k_, sg in enumerate(fsegs):
            e0 = sg["j0"][0] if "j0" in sg else sg["a0"]
            e1 = sg["j1"][0] if "j1" in sg else sg["a1"]
            p0_, p1_ = np.array(back(_hv_point(sg["o"], sg["c"], e0))), np.array(back(_hv_point(sg["o"], sg["c"], e1)))
            ls = np.linalg.norm(p1_ - p0_)
            if ls < 1.5 * T or abs(abs((p1_ - p0_) / ls @ g["d"]) - 1) > 3.5e-4:
                continue                                   # only pieces along this wall line
            mid = (p0_ + p1_) / 2
            if abs((mid - g["q0"]) @ g["n"]) > T:
                continue
            sg.update({"frame": key, "p0": list(p0_), "p1": list(p1_), "o": "a",
                       "j0_hv": "j0" in sg, "j1_hv": "j1" in sg})
            keep[k_] = len(segs)
            segs.append(sg)
            n_local += 1
        for o in fops:
            if o["wall"] in keep:
                fs = fsegs[o["wall"]]
                merged.append(dict(o, wall=keep[o["wall"]], q0=back(_hv_point(fs["o"], fs["c"], o["u0"])),
                                   q1=back(_hv_point(fs["o"], fs["c"], o["u1"]))))
        if keep:
            done_angles.append(g["ang"] % 90)
    if n_local:
        print("  walls read at their own angle: {}".format(n_local))
    # the same wall found in two frames (a straight angled wall + staircase stubs of it, a square column read as
    # a diamond in a 45 degree frame, ...): the piece whose band lies best on the model's wall pixels wins, then
    # the longer one; a piece mostly inside an already accepted wall of another frame is a duplicate
    if done_angles:
        seg_len = lambda sg: np.hypot(sg["p1"][0] - sg["p0"][0], sg["p1"][1] - sg["p0"][1])

        def band(sg, extra=0.0):
            m = np.zeros(full.shape, np.uint8)
            p0, p1 = np.array(sg["p0"]), np.array(sg["p1"])
            d = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)
            n = np.array([-d[1], d[0]]) * (sg["t"] + extra) / 2
            cv2.fillPoly(m, [np.round(np.array([p0 - n, p1 - n, p1 + n, p0 + n])).astype(np.int32)], 1)
            return m

        full_d = cv2.dilate(full.astype(np.uint8), np.ones((3, 3), np.uint8))
        for sg in segs:
            m = band(sg)
            sg["_fit"] = float((m & full_d).sum() / max(1, m.sum()))
        bands = {}
        # pieces that lie well on the mask (>= 90 %) first, longest first: one long wall beats a chain of short
        # pieces that each fit their few pixels perfectly (a wall read in a slightly-off angle frame)
        for sg in sorted(segs, key=lambda sg: (-(sg["_fit"] >= 0.9), -seg_len(sg), -sg["_fit"])):
            if sg["frame"] != 0 and (sg["_fit"] < 0.9 or sg["t"] <= 2):   # or a rotation sliver
                sg["drop"] = True        # a chord across a curve, not a straight wall: the tracer follows it
                continue
            m = band(sg)
            area = m.sum()
            other = np.zeros(full.shape, np.uint8)
            for fr, bm in bands.items():
                if fr != sg["frame"]:
                    other |= bm
            ov = (m & other).sum() / area if area else 1.0
            if ov > 0.8 or (ov > 0.5 and seg_len(sg) < 8 * T):
                sg["drop"] = True
                continue
            bm = bands.setdefault(sg["frame"], np.zeros(full.shape, np.uint8))
            bm |= band(sg, T)
        # straight-pass slivers along the stair-stepped edge of an angled wall (1-2 px thick, so they "fit"
        # perfectly and were accepted first): inside an accepted angled wall's band they are part of that wall
        ang_band = np.zeros(full.shape, np.uint8)
        for fr, bm in bands.items():
            if fr != 0:
                ang_band |= bm
        for sg in segs:
            if sg.get("drop") or sg["frame"] != 0 or sg["t"] > 0.5 * T:
                continue
            m = band(sg)
            if m.sum() and (m & ang_band).sum() / m.sum() > 0.6:
                sg["drop"] = True
        for sg in segs:
            sg.pop("_fit", None)
    # every wall must run ON wall pixels for most of its length: a thick wall seen from a slightly different
    # angle leaves a short diagonal "wall" that only crosses the real band (and sticks out of it)
    Hh, Ww = full.shape
    full_dil = cv2.dilate(full.astype(np.uint8), np.ones((3, 3), np.uint8))
    for sg in segs:
        if sg.get("drop"):
            continue
        p0, p1 = np.array(sg["p0"]), np.array(sg["p1"])
        n_ = max(2, int(np.hypot(*(p1 - p0))))
        ts = np.linspace(0, 1, n_)
        xs = np.clip(np.round(p0[0] + (p1[0] - p0[0]) * ts).astype(int), 0, Ww - 1)
        ys = np.clip(np.round(p0[1] + (p1[1] - p0[1]) * ts).astype(int), 0, Hh - 1)
        on = full_dil[ys, xs].mean()
        if on < 0.6:
            sg["drop"] = True
    _drop_cross_stubs(full, segs, T)
    keep_idx = [i for i, sg in enumerate(segs) if not sg.get("drop")]
    kmap = {i: k for k, i in enumerate(keep_idx)}
    merged = [dict(o, wall=kmap[o["wall"]]) for o in merged if o["wall"] in kmap]
    segs = _join_across([segs[i] for i in keep_idx], T)
    segs_ok = [i for i, sg in enumerate(segs) if not sg.get("drop") and
               np.hypot(sg["p1"][0] - sg["p0"][0], sg["p1"][1] - sg["p0"][1]) >= 1.5 * T]
    if not segs_ok:
        raise RuntimeError("No straight walls found. Try MAX_SIDE = 1024 or 1536, or a cleaner plan image.")
    remap = {i: k for k, i in enumerate(segs_ok)}
    segs = [segs[i] for i in segs_ok]
    merged = [dict(o, wall=remap[o["wall"]]) for o in merged if o["wall"] in remap]
    traced = _join_trace_ends(segs, _trace_leftovers(model_wall, segs, T), T)
    if traced:
        print("  walls traced along the model's mask (curves / irregular pieces): {}".format(len(traced)))
    segs = segs + traced
    n_win = 0
    if "drawn_windows" in ASSIST:
        segs, merged, n_win = _drawn_windows(segs, merged, image, T, full.shape)
    if n_win:
        print("  windows read from the drawing (not marked by the model): {}".format(n_win))

    stairs = []
    for comp, st in _components(icon == STAIRS, 4 * T * T):
        x, y, w, h = [int(v) for v in st[:4]]
        stairs.append({"x0": float(x), "y0": float(y), "x1": float(x + w), "y1": float(y + h),
                       "ox": x, "oy": y, "mask": comp[y:y + h, x:x + w].copy()})
    struct = np.zeros(full.shape, np.uint8)
    for sg in segs:
        p0, p1 = np.array(sg["p0"]), np.array(sg["p1"])
        d = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)
        n = np.array([-d[1], d[0]]) * sg["t"] / 2
        cv2.fillPoly(struct, [np.round(np.array([p0 - n, p1 - n, p1 + n, p0 + n])).astype(np.int32)], 1)
    # faithfulness, both ways, with 1 px tolerance (edge rounding only):
    #   coverage = share of the model's wall pixels that are inside a vector wall
    #   added    = share of the vector walls' pixels where the model marked no wall and no door / window
    k3 = np.ones((3, 3), np.uint8)
    missed = model_wall & ~(cv2.dilate(struct, k3) > 0)
    extra = (struct > 0) & ~(cv2.dilate((model_wall | openings_any).astype(np.uint8), k3) > 0)
    coverage = 1.0 - missed.sum() / max(1, model_wall.sum())
    added = extra.sum() / max(1, int((struct > 0).sum()))
    return {"T": T, "walls": segs, "openings": merged, "stairs": stairs, "full": struct > 0,
            "model_wall": model_wall, "missed": missed, "coverage": float(coverage),
            "added": float(added), "extra": extra,
            "openings_mask": full & ~wall}


# ----------------------------------------------------------------------------- 3. scale + rooms + json
def to_metres(v, room):
    segs, ops = v["walls"], v["openings"]
    doors = [o["u1"] - o["u0"] for o in ops if o["kind"] == "door"]
    vc = [s["c"] for s in segs if s["o"] == "v"]
    if METERS_PER_PIXEL:
        mpp, src = METERS_PER_PIXEL, "METERS_PER_PIXEL"
    elif BUILDING_WIDTH_M and len(vc) >= 2:
        mpp, src = BUILDING_WIDTH_M / (max(vc) - min(vc)), f"BUILDING_WIDTH_M={BUILDING_WIDTH_M}"
    elif doors:
        mpp, src = DOOR_WIDTH_M / float(np.median(doors)), f"median door = {DOOR_WIDTH_M} m ({len(doors)} doors)"
    else:
        mpp, src = 0.2 / v["T"], "wall thickness = 0.2 m (no doors found)"

    # footprint + rooms from the closed wall structure
    closed = cv2.morphologyEx(v["full"].astype(np.uint8), cv2.MORPH_CLOSE,
                              np.ones((int(2 * v["T"]),) * 2, np.uint8)) > 0
    filled = ndimage.binary_fill_holes(closed)
    comps = _components(filled, 1)
    fp_mask = max(comps, key=lambda c: c[1][cv2.CC_STAT_AREA])[0]
    ys_, xs_ = np.where(v["full"])
    ox, oy = float(xs_.min()), float(ys_.min())
    P = lambda x, y: [round((x - ox) * mpp, 3), round((y - oy) * mpp, 3)]

    def poly(mask, eps):
        cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = cv2.approxPolyDP(max(cs, key=cv2.contourArea), eps, True)[:, 0]
        return [P(x, y) for x, y in c]

    rooms = []
    interior = fp_mask & ~(cv2.dilate(v["full"].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    for comp, st in _components(interior, 1.0 / mpp ** 2):
        lab = room[comp]
        lab = lab[(lab > 2)]
        name = ROOMS[int(np.bincount(lab, minlength=12).argmax())] if lab.size else "Room"
        rooms.append({"name": name, "area_m2": round(float(comp.sum()) * mpp ** 2, 2),
                      "polygon": poly(comp, v["T"] / 2)})

    walls = []
    for i, s in enumerate(v["walls"]):
        if s["o"] == "h":
            a, b = P(s["a0"], s["c"]), P(s["a1"], s["c"])
        else:
            a, b = P(s["c"], s["a0"]), P(s["c"], s["a1"])
        walls.append({"id": i, "start": a, "end": b, "thickness": round(s["t"] * mpp, 3),
                      "height": WALL_HEIGHT})
    openings = []
    for o in ops:
        if (o["u1"] - o["u0"]) * mpp < (0.5 if o["kind"] == "door" else 0.3):
            continue                                     # specks, not real openings
        s = v["walls"][o["wall"]]
        sill, head = (0.0, DOOR_HEIGHT) if o["kind"] == "door" else (WINDOW_SILL, WINDOW_HEAD)
        openings.append({"type": o["kind"], "wall": o["wall"],
                         "offset": round((o["u0"] - s["a0"]) * mpp, 3),
                         "width": round((o["u1"] - o["u0"]) * mpp, 3), "sill": sill, "head": head})
    stairs = []
    for s in v["stairs"]:
        a, b = P(s["x0"], s["y0"]), P(s["x1"], s["y1"])
        n = int(np.clip(round(WALL_HEIGHT / STAIR_RISE), 3, 30))
        stairs.append({"min": a, "max": b, "steps": n, "rise": round(WALL_HEIGHT / n, 3),
                       "note": "climb direction assumed along the long side from the top/left end"})
    plan = {"units": "m", "scale_source": src, "meters_per_pixel": round(mpp, 6),
            "origin_px": [ox, oy], "footprint": poly(fp_mask, v["T"] / 2), "walls": walls,
            "openings": openings, "rooms": rooms, "stairs": stairs,
            "defaults": {"wall_height": WALL_HEIGHT, "door_height": DOOR_HEIGHT,
                         "window_sill": WINDOW_SILL, "window_head": WINDOW_HEAD, "slab": SLAB}}
    return plan


# ----------------------------------------------------------------------------- 4. geometry helpers
def wall_frame(w):
    """Unit direction, normal and length of a wall in plan metres."""
    a, b = np.array(w["start"], float), np.array(w["end"], float)
    d = b - a
    length = float(np.linalg.norm(d))
    d = d / max(length, 1e-9)
    return a, d, np.array([-d[1], d[0]]), length


def wall_pieces(plan):
    """Split every wall into solid boxes around its openings. Yields (category, corners2d(4), z0, z1)."""
    by_wall = {}
    for o in plan["openings"]:
        by_wall.setdefault(o["wall"], []).append(o)
    for w in plan["walls"]:
        a, d, n, length = wall_frame(w)
        t = w["thickness"]
        rect = lambda u0, u1, th: [a + d * u0 - n * th / 2, a + d * u1 - n * th / 2,
                                   a + d * u1 + n * th / 2, a + d * u0 + n * th / 2]
        cur = 0.0
        for o in sorted(by_wall.get(w["id"], []), key=lambda o: o["offset"]):
            u0, u1 = max(o["offset"], cur), min(o["offset"] + o["width"], length)
            if u1 <= u0:
                continue
            if u0 > cur:
                yield "wall", rect(cur, u0, t), 0.0, w["height"]
            if o["sill"] > 0:
                yield "wall", rect(u0, u1, t), 0.0, o["sill"]
            if o["head"] < w["height"]:
                yield "wall", rect(u0, u1, t), o["head"], w["height"]
            inset = min(0.03, (u1 - u0) / 4)
            if o["type"] == "door":
                yield "door", rect(u0 + inset, u1 - inset, 0.05), 0.0, o["head"]
            else:
                yield "glass", rect(u0 + inset, u1 - inset, 0.03), o["sill"], o["head"]
            cur = u1
        if cur < length:
            yield "wall", rect(cur, length, t), 0.0, w["height"]
    for s in plan["stairs"]:
        (x0, y0), (x1, y1) = s["min"], s["max"]
        along_x = (x1 - x0) >= (y1 - y0)
        for i in range(s["steps"]):
            f0, f1 = i / s["steps"], (i + 1) / s["steps"]
            if along_x:
                c = [[x0 + (x1 - x0) * f0, y0], [x0 + (x1 - x0) * f1, y0],
                     [x0 + (x1 - x0) * f1, y1], [x0 + (x1 - x0) * f0, y1]]
            else:
                c = [[x0, y0 + (y1 - y0) * f0], [x1, y0 + (y1 - y0) * f0],
                     [x1, y0 + (y1 - y0) * f1], [x0, y0 + (y1 - y0) * f1]]
            yield "stairs", [np.array(p, float) for p in c], 0.0, s["rise"] * (i + 1)


def prism(corners, z0, z1):
    """Box from a 2D quad (plan x, plan y) and heights -> (vertices in Y-up glTF axes, outward faces)."""
    c = [np.asarray(p, float) for p in corners]
    v = np.array([[p[0], z0, p[1]] for p in c] + [[p[0], z1, p[1]] for p in c])
    f = [[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6]]
    for i in range(4):
        j = (i + 1) % 4
        f += [[i, 4 + i, j], [j, 4 + i, 4 + j]]
    f = np.array(f)
    t = v[f]
    if np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() < 0:   # inward -> flip
        f = f[:, ::-1]
    return v, f


COLORS = {"wall": [235, 232, 225, 255], "door": [150, 95, 55, 255], "glass": [140, 190, 235, 140],
          "stairs": [190, 190, 190, 255], "floor": [205, 185, 150, 255]}


def build_glb(plan, path):
    import trimesh
    from shapely.geometry import Polygon
    parts = {}
    for cat, quad, z0, z1 in wall_pieces(plan):
        v, f = prism(quad, z0, z1)
        parts.setdefault(cat, []).append(trimesh.Trimesh(v, f, process=False))
    poly = Polygon(plan["footprint"]).buffer(0)
    if not poly.is_empty:
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
        fl = trimesh.creation.extrude_polygon(Polygon([(x, -y) for x, y in poly.exterior.coords]), SLAB)
        fl.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
        fl.apply_translation([0, -SLAB, 0])
        parts["floor"] = [fl]
    scene = trimesh.Scene()
    for cat, ms in parts.items():
        m = trimesh.util.concatenate(ms)
        m.fix_normals()
        m.visual = trimesh.visual.ColorVisuals(m, face_colors=np.tile(COLORS[cat], (len(m.faces), 1)))
        scene.add_geometry(m, geom_name=cat)
    scene.export(path)
    return scene


# ----------------------------------------------------------------------------- 5. IFC (BIM)
def build_ifc(plan, path):
    from ifcopenshell.api import root, unit, context, aggregate, spatial, geometry, feature, project
    f = project.create_file(version="IFC4")
    proj = root.create_entity(f, ifc_class="IfcProject", name="Floorplan to 3D")
    unit.assign_unit(f, units=[unit.add_si_unit(f, unit_type="LENGTHUNIT"),
                               unit.add_si_unit(f, unit_type="AREAUNIT")])
    model = context.add_context(f, context_type="Model")
    body = context.add_context(f, context_type="Model", context_identifier="Body",
                               target_view="MODEL_VIEW", parent=model)
    site = root.create_entity(f, ifc_class="IfcSite", name="Site")
    bldg = root.create_entity(f, ifc_class="IfcBuilding", name="Building")
    storey = root.create_entity(f, ifc_class="IfcBuildingStorey", name="Ground Floor")
    aggregate.assign_object(f, products=[site], relating_object=proj)
    aggregate.assign_object(f, products=[bldg], relating_object=site)
    aggregate.assign_object(f, products=[storey], relating_object=bldg)
    for e in (site, bldg, storey):
        geometry.edit_object_placement(f, product=e, matrix=np.eye(4))

    def place(e, origin2d, d2d, z=0.0):
        # IFC is Z-up; plan y (image down) -> IFC -Y so the model is not mirrored
        x_ax = np.array([d2d[0], -d2d[1], 0.0])
        z_ax = np.array([0.0, 0.0, 1.0])
        M = np.eye(4)
        M[:3, 0], M[:3, 1], M[:3, 2] = x_ax, np.cross(z_ax, x_ax), z_ax
        M[:3, 3] = [origin2d[0], -origin2d[1], z]
        geometry.edit_object_placement(f, product=e, matrix=M)

    walls = {}
    for w in plan["walls"]:
        a, d, n, length = wall_frame(w)
        e = root.create_entity(f, ifc_class="IfcWall", name=f"Wall {w['id']}")
        geometry.assign_representation(f, product=e, representation=geometry.add_wall_representation(
            f, context=body, length=length, height=w["height"], thickness=w["thickness"],
            offset=-w["thickness"] / 2))
        place(e, a, d)
        spatial.assign_container(f, products=[e], relating_structure=storey)
        walls[w["id"]] = (e, a, d, w)
    for k, o in enumerate(plan["openings"]):
        e_wall, a, d, w = walls[o["wall"]]
        h = o["head"] - o["sill"]
        org = a + d * o["offset"]
        op = root.create_entity(f, ifc_class="IfcOpeningElement", name=f"Opening {k}")
        geometry.assign_representation(f, product=op, representation=geometry.add_wall_representation(
            f, context=body, length=o["width"], height=h, thickness=w["thickness"] + 0.2,
            offset=-(w["thickness"] + 0.2) / 2))
        place(op, org, d, o["sill"])
        feature.add_feature(f, feature=op, element=e_wall)
        cls = "IfcDoor" if o["type"] == "door" else "IfcWindow"
        fill = root.create_entity(f, ifc_class=cls, name=f"{o['type'].title()} {k}")
        fill.OverallWidth, fill.OverallHeight = o["width"], h
        geometry.assign_representation(f, product=fill, representation=geometry.add_wall_representation(
            f, context=body, length=o["width"], height=h, thickness=0.05, offset=-0.025))
        place(fill, org, d, o["sill"])
        feature.add_filling(f, opening=op, element=fill)
        spatial.assign_container(f, products=[fill], relating_structure=storey)
    slab = root.create_entity(f, ifc_class="IfcSlab", name="Ground slab", predefined_type="FLOOR")
    geometry.assign_representation(f, product=slab, representation=geometry.add_slab_representation(
        f, context=body, depth=SLAB, polyline=[(x, -y) for x, y in plan["footprint"]]))
    geometry.edit_object_placement(f, product=slab, matrix=np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -SLAB], [0, 0, 0, 1]], float))
    spatial.assign_container(f, products=[slab], relating_structure=storey)
    for r in plan["rooms"]:
        sp = root.create_entity(f, ifc_class="IfcSpace", name=r["name"])
        sp.LongName = f"{r['name']} ({r['area_m2']} m²)"
        geometry.assign_representation(f, product=sp, representation=geometry.add_slab_representation(
            f, context=body, depth=WALL_HEIGHT, polyline=[(x, -y) for x, y in r["polygon"]]))
        geometry.edit_object_placement(f, product=sp, matrix=np.eye(4))
        aggregate.assign_object(f, products=[sp], relating_object=storey)
    for k, s in enumerate(plan["stairs"]):
        verts, faces, off = [], [], 0
        for cat, quad, z0, z1 in wall_pieces({"walls": [], "openings": [], "stairs": [s]}):
            v, fc = prism(quad, z0, z1)
            verts += [[p[0], -p[2], p[1]] for p in v]            # glTF (x, y-up, z) -> IFC (x, -z, y)
            faces += [[int(i) + off for i in tri] for tri in fc]
            off += len(v)
        st = root.create_entity(f, ifc_class="IfcStair", name=f"Stair {k}")
        geometry.assign_representation(f, product=st, representation=geometry.add_mesh_representation(
            f, context=body, vertices=[verts], faces=[faces]))
        geometry.edit_object_placement(f, product=st, matrix=np.eye(4))
        spatial.assign_container(f, products=[st], relating_structure=storey)
    f.write(path)
    return f


# ----------------------------------------------------------------------------- 6. previews
def previews(img, v, plan, scene, out_dir):
    fig, ax = plt.subplots(1, 2, figsize=(22, 10))
    ax[0].imshow(img); ax[0].set_title("vectorised plan (on input)", fontsize=15); ax[0].axis("off")
    for s in v["walls"]:
        x0, x1 = (s["a0"], s["a1"]) if s["o"] == "h" else (s["c"] - s["t"] / 2, s["c"] + s["t"] / 2)
        y0, y1 = (s["c"] - s["t"] / 2, s["c"] + s["t"] / 2) if s["o"] == "h" else (s["a0"], s["a1"])
        ax[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fc=(0.9, 0.1, 0.1, 0.45), ec="r", lw=0.6))
    for o in v["openings"]:
        s = v["walls"][o["wall"]]
        xs, ys = ([o["u0"], o["u1"]], [s["c"]] * 2) if s["o"] == "h" else ([s["c"]] * 2, [o["u0"], o["u1"]])
        ax[0].plot(xs, ys, color="orange" if o["kind"] == "door" else "deepskyblue", lw=5,
                   solid_capstyle="butt")
    for s in v["stairs"]:
        ax[0].add_patch(plt.Rectangle((s["x0"], s["y0"]), s["x1"] - s["x0"], s["y1"] - s["y0"],
                                      fc=(0.2, 0.8, 0.8, 0.3), ec="c"))
    # static 3D render (one collection -> per-face depth sorting)
    ax3 = fig.add_subplot(1, 2, 2, projection="3d"); ax[1].axis("off")
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    tris, cols = [], []
    for name, g in scene.geometry.items():
        t = g.vertices[g.faces][:, :, [0, 2, 1]] * [1, -1, 1]       # (X, Y-up, Z) -> (x, -y, up)
        tris.append(t); cols.append(np.tile(np.array(COLORS[name]) / 255.0, (len(t), 1)))
    tris, cols = np.vstack(tris), np.vstack(cols)
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    shade = 0.55 + 0.45 * np.clip(nrm @ np.array([0.35, -0.45, 0.82]), 0, 1)
    cols[:, :3] *= shade[:, None]
    ax3.add_collection3d(Poly3DCollection(tris, facecolors=cols, edgecolors=cols, linewidths=0.3))
    lo, hi = tris.reshape(-1, 3).min(0), tris.reshape(-1, 3).max(0)
    ax3.set_xlim(lo[0], hi[0]); ax3.set_ylim(lo[1], hi[1]); ax3.set_zlim(lo[2], hi[2])
    ax3.set_box_aspect(tuple(hi - lo)); ax3.set_axis_off()
    ax3.view_init(elev=40, azim=-60); ax3.set_title("3D model (static preview)", fontsize=15)
    plt.tight_layout()
    p = os.path.join(out_dir, "preview.png")
    plt.savefig(p, dpi=90)
    plt.show()
    return p


# ----------------------------------------------------------------------------- run
def main(img=None, room=None, icon=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    if room is None:
        assert IMAGE, "Set IMAGE at the top of the cell to a floor-plan URL or file path"
        img, room, icon = predict(IMAGE)
    v = vectorise(room, icon)
    plan = to_metres(v, room)
    json.dump(plan, open(os.path.join(OUT_DIR, "plan.json"), "w"), indent=1)
    scene = build_glb(plan, os.path.join(OUT_DIR, "model.glb"))
    try:
        build_ifc(plan, os.path.join(OUT_DIR, "model.ifc"))
        ifc_ok = True
    except Exception as e:
        print("IFC export failed (GLB/JSON still fine):", type(e).__name__, e)
        ifc_ok = False
    n_d = sum(o["type"] == "door" for o in plan["openings"])
    n_w = sum(o["type"] == "window" for o in plan["openings"])
    xs = [p[0] for p in plan["footprint"]]; ys = [p[1] for p in plan["footprint"]]
    print(f"scale: {plan['meters_per_pixel'] * 1000:.2f} mm/px  ({plan['scale_source']})")
    print(f"walls {len(plan['walls'])} | doors {n_d} | windows {n_w} | stairs {len(plan['stairs'])} | "
          f"rooms {len(plan['rooms'])} | footprint {max(xs) - min(xs):.2f} x {max(ys) - min(ys):.2f} m")
    previews(img, v, plan, scene, OUT_DIR)
    print("files:", ", ".join(sorted(os.listdir(OUT_DIR))), "->", OUT_DIR, "" if ifc_ok else "(no IFC)")
    try:   # interactive viewer inside the notebook (needs internet for three.js)
        from IPython.display import display
        display(scene.show(viewer="jupyter"))
    except Exception as e:
        print("interactive viewer unavailable:", e)
    return plan


if not os.environ.get("P3D_NO_MAIN"):
    plan = main()
