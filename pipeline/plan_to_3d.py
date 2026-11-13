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
def _net_probs(net, dev, img):
    """RGB image -> (room probs 12xHxW, icon probs 12xHxW), with horizontal-flip averaging."""
    import torch
    h, w = img.shape[:2]
    pad = cv2.copyMakeBorder(img, 0, (32 - h % 32) % 32, 0, (32 - w % 32) % 32,
                             cv2.BORDER_CONSTANT, value=(255, 255, 255))
    x = (pad.astype(np.float32) - [123.675, 116.28, 103.53]) / [58.395, 57.12, 57.375]
    x = torch.from_numpy(x.transpose(2, 0, 1)).float()[None].to(dev)
    with torch.no_grad(), torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
        out = (net(x).float() + net(x.flip(3)).float().flip(3)) / 2
    out = out[0, :, :h, :w]
    return (torch.softmax(out[:12], 0).cpu().numpy().astype(np.float32),
            torch.softmax(out[12:24], 0).cpu().numpy().astype(np.float32))


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
    import time, torch, segmentation_models_pytorch as smp
    now = time.time()
    if _MODEL["net"] is not None and not force_check and (MODEL_CHECK_S <= 0 or now - _MODEL["checked"] < MODEL_CHECK_S):
        return _MODEL["net"], _MODEL["dev"]
    _MODEL["checked"] = now
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
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = smp.Unet(ck.get("encoder", "tu-efficientnet_b3"), encoder_weights=None, classes=27,
                   decoder_attention_type="scse")
    net.load_state_dict(ck["ema"]); net.to(dev).eval()
    info = "epoch {}, val score {:.3f}".format(ck.get("epoch", -1) + 1, ck.get("metrics", {}).get("score", float("nan")))
    _MODEL.update(net=net, dev=dev, rev=rev, info=info)
    print("model loaded: revision", rev)
    return net, dev


def predict(image_src):
    import urllib.request
    net, dev = load_model()
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
    room_p, icon_p = _net_probs(net, dev, img)
    views = 1
    if ROTATE_TTA:
        # the model learned from plans drawn square to the page: show it extra copies rotated so each family
        # of angled walls lies flat, rotate its answers back and combine (rooms averaged, objects: most sure view)
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        walls_ink = cv2.morphologyEx((g < 128).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
        angs = _wall_angles(walls_ink, 8.0)[:3]
        r_sum, r_w = room_p.copy(), np.ones((h, w), np.float32)
        for ang in angs:
            M, Mi, size = _rot_frame((h, w), ang)
            rimg = cv2.warpAffine(img, M, size, flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
            rp, ip = _net_probs(net, dev, rimg)
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
    print(f"model: {_MODEL.get('info', '')}, image {w}x{h}, {dev}")
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
        out.append((centre, float(r0), float(r1 + 1), t))
    return out


def _frame_walls(full, openings_any, icon, T):
    """Horizontal / vertical walls + their openings in one frame (the image, or the image rotated so a
    family of angled walls runs horizontally)."""
    # straight segments: horizontal / vertical runs longer than ~2.5 T
    L = max(int(2.5 * T), 12)
    segs = []
    G = max(int(3 * T), 6)
    for o, ker, gap in (("h", np.ones((1, L), np.uint8), np.ones((1, G), np.uint8)),
                        ("v", np.ones((L, 1), np.uint8), np.ones((G, 1), np.uint8))):
        # merge double-line walls (window drawn as two thin parallel lines) into one solid band
        P = max(3, int(round(0.8 * T)))
        perp = np.ones((P, 1), np.uint8) if o == "h" else np.ones((1, P), np.uint8)
        m = cv2.morphologyEx(full.astype(np.uint8), cv2.MORPH_CLOSE, perp)
        # keep only genuinely long runs in this direction FIRST, then bridge small gaps between them
        # (bridging first would join side-by-side perpendicular walls into fake material)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, gap)
        for comp, st in _components(m, L * 2):
            x, y, w, h = st[:4]
            c = comp[y:y + h, x:x + w]
            if o == "h":                         # work in "vertical" orientation: runs go down rows
                c, off_run, off_across = c.T, x, y
            else:
                off_run, off_across = y, x
            for centre, a0, a1, t in _straight_runs(c, L, G, T):
                segs.append({"o": o, "c": off_across + centre, "a0": float(off_run + a0),
                             "a1": float(off_run + a1), "t": max(t, 0.6 * T)})
    for o in "hv":
        idx = [i for i, s in enumerate(segs) if s["o"] == o]
        if idx:
            snapped = _cluster([segs[i]["c"] for i in idx], 0.75 * T)
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
        if p and p["o"] == sg["o"] and abs(p["c"] - sg["c"]) < 0.5 and (sg["a0"] <= p["a1"] + 1.5 * T or bridged):
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
                reach = sg["t"] / 2 + 1.5 * T
                if p["o"] == sg["o"] or not (p["a0"] - reach <= sg["c"] <= p["a1"] + reach):
                    continue
                tol = p["t"] / 2 + 1.5 * T
                if abs(sg["a0"] - p["c"]) <= tol:
                    sg["a0"] = p["c"] - p["t"] / 2
                    sg["j0"] = (p["c"], p["t"])      # joined: perpendicular centre-line + thickness
                if abs(sg["a1"] - p["c"]) <= tol:
                    sg["a1"] = p["c"] + p["t"] / 2
                    sg["j1"] = (p["c"], p["t"])
    segs = [sg for sg in segs if sg["a1"] - sg["a0"] >= 2 * T]

    # openings -> intervals on their host wall
    openings = []
    for kind, cls in (("door", DOOR), ("window", WIN)):
        for comp, st in _components(icon == cls, max(12, 0.3 * T * T)):
            x, y, w, h = [float(v) for v in st[:4]]
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
    """Dominant wall directions (degrees, mod 90) other than the image axes, e.g. 30 / 60 for a hexagonal plan."""
    edges = cv2.Canny(full.astype(np.uint8) * 255, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=max(20, int(3 * T)),
                            minLineLength=max(int(4 * T), 20), maxLineGap=int(T))
    if lines is None:
        return []
    hist = np.zeros(90)
    for x0, y0, x1, y1 in lines.reshape(-1, 4):
        ang = np.degrees(np.arctan2(y1 - y0, x1 - x0)) % 90
        hist[int(round(ang)) % 90] += np.hypot(x1 - x0, y1 - y0)
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
                    reach_q = q["t"] / 2 + 2 * T
                    if not (-reach_q <= tq <= M + reach_q):
                        continue
                    dist = np.linalg.norm(X - pe)
                    if dist > q["t"] / (2 * abs(cr)) + 2 * T:
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


def vectorise(room, icon, image=None):
    wall = room == WALL
    openings_any = np.isin(icon, [WIN, DOOR])
    # typical wall thickness T (px) from the medial axis
    dist = cv2.distanceTransform(wall.astype(np.uint8), cv2.DIST_L2, 5)
    sk = skeletonize(wall)
    T = float(np.clip(2 * np.median(dist[sk]) if sk.any() else 6, 3, 60))
    model_wall = wall.copy()
    # the model's wall mask is trusted: only 1-2 px specks are removed (no thickness filtering,
    # so thin partitions and walls detached from the main structure are kept)
    # (no morphological opening: it erases thin double-line window walls the model got right;
    #  specks are removed by size below)
    comps = _components(wall, max(10, int(1.5 * T * T)))
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
    for _round_ in range(6):                                   # find angles in what is still unexplained
        angles = [a_ for a_ in _wall_angles(full & ~(claimed > 0), T)
                  if all(min(abs(a_ - b_), 90 - abs(a_ - b_)) > 2 for b_ in done_angles)]
        if not angles:
            break
        ang = angles[0]
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
    # the same wall found in two frames (a straight angled wall + staircase stubs of it, or two close angles):
    # longest first; a piece mostly inside an already accepted wall of another frame is a duplicate
    if done_angles:
        seg_len = lambda sg: np.hypot(sg["p1"][0] - sg["p0"][0], sg["p1"][1] - sg["p0"][1])
        bands = {}
        for sg in sorted(segs, key=lambda sg: -seg_len(sg)):
            m = np.zeros(full.shape, np.uint8)
            cv2.line(m, tuple(int(round(v_)) for v_ in sg["p0"]), tuple(int(round(v_)) for v_ in sg["p1"]),
                     1, max(1, int(round(sg["t"]))))
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
            cv2.line(bm, tuple(int(round(v_)) for v_ in sg["p0"]), tuple(int(round(v_)) for v_ in sg["p1"]),
                     1, max(1, int(round(sg["t"] + T))))
    # every wall must run ON wall pixels for most of its length: a thick wall seen from a slightly different
    # angle leaves a short diagonal "wall" that only crosses the real band (and sticks out of it)
    Hh, Ww = full.shape
    for sg in segs:
        if sg.get("drop"):
            continue
        p0, p1 = np.array(sg["p0"]), np.array(sg["p1"])
        n_ = max(2, int(np.hypot(*(p1 - p0))))
        ts = np.linspace(0, 1, n_)
        xs = np.clip(np.round(p0[0] + (p1[0] - p0[0]) * ts).astype(int), 0, Ww - 1)
        ys = np.clip(np.round(p0[1] + (p1[1] - p0[1]) * ts).astype(int), 0, Hh - 1)
        on = cv2.dilate(full.astype(np.uint8), np.ones((3, 3), np.uint8))[ys, xs].mean()
        if on < 0.6:
            sg["drop"] = True
    keep_idx = [i for i, sg in enumerate(segs) if not sg.get("drop")]
    kmap = {i: k for k, i in enumerate(keep_idx)}
    merged = [dict(o, wall=kmap[o["wall"]]) for o in merged if o["wall"] in kmap]
    segs = _join_across([segs[i] for i in keep_idx], T)
    segs_ok = [i for i, sg in enumerate(segs) if not sg.get("drop") and
               np.hypot(sg["p1"][0] - sg["p0"][0], sg["p1"][1] - sg["p0"][1]) >= 2 * T]
    if not segs_ok:
        raise RuntimeError("No straight walls found. Try MAX_SIDE = 1024 or 1536, or a cleaner plan image.")
    remap = {i: k for k, i in enumerate(segs_ok)}
    segs = [segs[i] for i in segs_ok]
    merged = [dict(o, wall=remap[o["wall"]]) for o in merged if o["wall"] in remap]
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
    # how much of the model's wall prediction made it into the vector walls
    covered = cv2.dilate(struct, np.ones((max(3, int(T)),) * 2, np.uint8)) > 0
    missed = model_wall & ~covered
    coverage = 1.0 - missed.sum() / max(1, model_wall.sum())
    return {"T": T, "walls": segs, "openings": merged, "stairs": stairs, "full": struct > 0,
            "model_wall": model_wall, "missed": missed, "coverage": float(coverage),
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
