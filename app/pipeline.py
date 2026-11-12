# generated from pipeline/ by assemble.py - do not edit
# ===================== FLOOR PLAN -> EXACT 3D FOR REVIT =====================
# Reads the plan with your model + OCR, solves wall positions from the WRITTEN dimensions,
# and writes plan.json (for build_in_revit.py), model.ifc (Revit: Open/Link IFC) and model.glb (preview).
IMAGE = ""                    # floor-plan image URL or file path (clean scan/export, dimensions visible)
HF_REPO = "floorplan-multitask-effb3-768"
MAX_SIDE = 1280               # inference size (try 1536 for big/dense sheets)
ROTATE_TTA = True             # also read the plan rotated so angled walls / doors lie flat (slower, x(1+angles))

# --- scale: written dimensions are used automatically. Only if the plan has NO dimensions:
METERS_PER_PIXEL = None
BUILDING_WIDTH_M = None       # distance between outermost wall centre-lines
DOOR_WIDTH_M = 0.9            # last-resort estimate

# --- standards (metres). Wall thickness snaps to the nearest option within 35%.
WALL_THICKNESSES = [0.115, 0.15, 0.2, 0.23, 0.3]
EXTERNAL_WALL_M = None        # e.g. 0.23 to force all external walls
INTERNAL_WALL_M = None        # e.g. 0.115 to force all internal walls
DOOR_WIDTHS = [0.75, 0.8, 0.9, 1.0, 1.2]
WALL_HEIGHT, DOOR_HEIGHT = 3.0, 2.1
WINDOW_SILL, WINDOW_HEAD = 0.9, 2.1
SLAB, STAIR_RISE = 0.15, 0.175
# --- stairs / entrance steps: written values on the plan always win; these fill the gaps
PLINTH_HEIGHT = None          # floor level above ground (m), e.g. 0.45; None = from the plan
STAIR_RISE_EXT = 0.15         # riser assumed for outside steps / stepped areas when nothing is written
STEPPED_AREAS = "down"        # stepped areas (L / U steps, seating): "down" = sunken (floor cut out), "up" = raised
OUT_DIR = "/kaggle/working/plan_revit"

# --- furniture: fixtures from the floor-plan model + your trained furniture detector (if available)
MAKE_FURNITURE = True
FURNITURE_REPO = "floorplan-furniture-obb"   # HF model repo from the furniture notebook ("" to skip)
FURNITURE_CONF = 0.4          # always accepted at/above this
FURNITURE_LOW_CONF = 0.15     # weaker hits accepted only when the room fits (bed in a bedroom, ...)

import os, sys, re, json, subprocess
if not os.environ.get("P3D_NO_INSTALL"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch==0.6.0",
                    "timm==1.0.31", "huggingface_hub", "trimesh", "shapely", "mapbox-earcut",
                    "ifcopenshell>=0.8.3", "rapidocr_onnxruntime"], check=False)
import cv2, numpy as np, matplotlib.pyplot as plt
from scipy import ndimage
from skimage.morphology import skeletonize

WALL, WIN, DOOR, STAIRS = 2, 1, 2, 11   # room-head wall id; icon-head ids
LAST_ORIGINAL = None
LAST_ICON_PROB = None


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


def predict(image_src):
    import urllib.request, torch, segmentation_models_pytorch as smp
    from huggingface_hub import HfApi, hf_hub_download
    ckpt = "/kaggle/working/checkpoints/best.pt"
    if not os.path.exists(ckpt):
        token = os.environ.get("HF_TOKEN")
        if not token:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
        repo = f"{HfApi(token=token).whoami()['name']}/{HF_REPO}"
        ckpt = hf_hub_download(repo, "checkpoints/best.pt", token=token)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = smp.Unet(ck.get("encoder", "tu-efficientnet_b3"), encoder_weights=None, classes=27,
                   decoder_attention_type="scse")
    net.load_state_dict(ck["ema"]); net.to(dev).eval()
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
    print(f"model: epoch {ck['epoch'] + 1}, val score {ck['metrics']['score']:.3f}, image {w}x{h}, {dev}")
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


# ----------------------------------------------------------------------------- 2b. dimension solving (OCR)
NUM = re.compile(r"\d+(?:[.,]\d+)?")


def ocr_tokens(img_rgb, ocr=None):
    """Run RapidOCR; split multi-number boxes ('5.00 11.00') into separate tokens with estimated centres."""
    if ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()
    res, _ = ocr(img_rgb[:, :, ::-1].copy())
    toks = []
    res = list(res or [])
    used = set()
    joined = []
    for i, (box, txt, conf) in enumerate(res):                  # 'W x' on one line, 'D' on the next
        if i in used:
            continue
        if re.search(r"\d\s*(?:m|mm|cm)?\s*[xX×]\s*$", txt.strip()):
            b = np.array(box, float)
            h = b[:, 1].max() - b[:, 1].min()
            cx = b[:, 0].mean()
            best = None
            for j, (box2, txt2, conf2) in enumerate(res):
                if j == i or j in used or not re.match(r"^\s*\d", txt2):
                    continue
                b2 = np.array(box2, float)
                dy = b2[:, 1].min() - b[:, 1].max()
                if -0.3 * h <= dy <= 1.5 * h and abs(b2[:, 0].mean() - cx) <= 2.5 * h:
                    if best is None or dy < best[0]:
                        best = (dy, j, b2, txt2, conf2)
            if best:
                _, j, b2, txt2, conf2 = best
                used.add(j)
                joined.append((np.vstack([b, b2]).tolist(), txt.strip() + " " + txt2.strip(), min(conf, conf2)))
                continue
        joined.append((box, txt, conf))
    for box, txt, conf in joined:
        b = np.array(box, float)
        x0, x1, y0, y1 = b[:, 0].min(), b[:, 0].max(), b[:, 1].min(), b[:, 1].max()
        vertical_text = (y1 - y0) > 1.6 * (x1 - x0) and len(txt) > 2
        n = max(len(txt), 1)
        nums = list(NUM.finditer(txt))
        letters = sum(ch.isalpha() for ch in txt)
        size = parse_size(txt)
        if size:
            toks.append({"kind": "size", "value": size, "x": (x0 + x1) / 2, "y": (y0 + y1) / 2,
                         "text": txt.strip(), "raw": txt, "conf": float(conf)})
        elif re.fullmatch(r"\s*\(\s*\d{1,3}\s*\)\s*", txt):       # room numbers '(10)' are not dimensions
            toks.append({"kind": "label", "text": txt, "raw": txt, "x": (x0 + x1) / 2, "y": (y0 + y1) / 2,
                         "h": float(y1 - y0), "conf": float(conf)})
        elif nums and letters <= 1 and not re.search(r"[xX×]", txt):
            for m in nums:
                f = (m.start() + m.end()) / 2 / n
                cx, cy = (x0 + f * (x1 - x0), (y0 + y1) / 2) if not vertical_text else \
                         ((x0 + x1) / 2, y1 - f * (y1 - y0))
                toks.append({"kind": "num", "value": float(m.group().replace(",", ".")), "x": cx, "y": cy,
                             "text": m.group(), "raw": txt, "conf": float(conf),
                             "h": float(min(y1 - y0, x1 - x0) if vertical_text else (y1 - y0))})
        elif letters >= 2:
            toks.append({"kind": "label", "text": txt, "raw": txt, "x": (x0 + x1) / 2, "y": (y0 + y1) / 2,
                         "h": float(min(y1 - y0, x1 - x0) if vertical_text else (y1 - y0)), "conf": float(conf)})
    return toks


QUOTES = "'\"°′″`’”"


def parse_length(side):
    """One length from OCR text: feet-inches (14'-0\", 8-6\", 12 -5\") or metric (3.60, 360, 3600)."""
    nums = re.findall(r"\d+(?:\.\d+)?", side)
    if not nums or len(nums) > 2:
        return None
    imperial = any(c in side for c in QUOTES) or (len(nums) == 2 and "." not in "".join(nums))
    if imperial:
        ft = float(nums[0])
        inch = float(nums[1]) if len(nums) == 2 else 0.0
        if inch >= 12 or ft > 150:
            return None
        return (ft + inch / 12.0) * 0.3048
    v = float(nums[0])
    return v if v < 40 else (v / 100 if v < 1000 else v / 1000)


UNIT_WORDS = re.compile(r"(?<=\d)\s*(?:mm|cm|mtrs?|m)(?=\s*[xX×]|\b)\.?", re.I)


def parse_size(text):
    """'W x D' room-size label -> (a_m, b_m) or None.  '3.66 m x 4.88 m', '12'0\" x 10'6\"', '3600 X 4200'."""
    unit = re.search(r"\d\s*(mm|cm)\b", text, re.I)
    text = UNIT_WORDS.sub("", text)
    parts = re.split(r"\s*[xX×]\s*", text.strip())
    if len(parts) != 2 or sum(ch.isalpha() for ch in text if ch not in "xX") > 0:
        return None
    a, b = parse_length(parts[0]), parse_length(parts[1])
    if unit and a and b and unit.group(1).lower() == "cm" and a > 40:
        a, b = a / 100.0, b / 100.0
    if a and b and 0.6 <= a <= 40 and 0.6 <= b <= 40:
        return (a, b)
    return None


def _unit_factor(values):
    """Plans write metres (3.00), centimetres (300) or millimetres (3000). Return factor to metres."""
    v = np.median(values)
    return 1.0 if v < 40 else (0.01 if v < 400 else 0.001)


def _room_constraints(struct, segs, T, toks, gx, gy):
    """Room-size labels -> clear sizes. Rays are cast from the label to the walls around it
    (works for rooms whose doors leave gaps). Returns measurements between wall centre-lines (px)."""
    sizes = [t for t in toks if t["kind"] == "size"]
    if not sizes or struct is None:
        return []
    H, W = struct.shape

    def thick(o, c):
        ts = [sg["t"] for sg in segs if sg["o"] == o and abs(sg["c"] - c) < 0.75 * T]
        return float(np.median(ts)) if ts else T

    def ray(x, y, dx, dy):
        while 0 <= x < W and 0 <= y < H:
            if struct[y, x]:
                return x, y
            x += dx; y += dy
        return None

    def span(x, y, horizontal):
        """Most common wall-to-wall distance over several parallel rays near (x, y)."""
        found = []
        for off in np.linspace(-4 * T, 4 * T, 9):
            if horizontal:
                yy = int(round(y + off))
                a, b = ray(int(x), yy, -1, 0), ray(int(x), yy, 1, 0)
                if a and b:
                    found.append((a[0], b[0]))
            else:
                xx = int(round(x + off))
                a, b = ray(xx, int(y), 0, -1), ray(xx, int(y), 0, 1)
                if a and b:
                    found.append((a[1], b[1]))
        if len(found) < 3:
            return None
        lens = np.array([q - p for p, q in found])
        med = np.median(lens)
        agree = [f for f, l in zip(found, lens) if abs(l - med) <= 2]
        if len(agree) < 3:
            return None                                  # rays disagree (door gap, odd shape)
        return int(np.median([p for p, _ in agree])), int(np.median([q for _, q in agree]))

    out = []
    for t in sizes:
        sx, sy = span(t["x"], t["y"], True), span(t["x"], t["y"], False)
        if not sx or not sy:
            continue
        w, h = sx[1] - sx[0] - 1, sy[1] - sy[0] - 1      # clear size in px (between wall faces)
        if w < 2 * T or h < 2 * T:
            continue
        a, b = t["value"]
        if abs(np.log((a / w) / (b / h))) > abs(np.log((b / w) / (a / h))):
            a, b = b, a                                  # label order is not always W x D
        if abs(np.log((a / w) / (b / h))) > 0.08:
            continue                                     # proportions disagree -> wrong region
        for ax, (p0, p1), clear, cpx, g, o in (("x", sx, a, w, gx, "v"), ("y", sy, b, h, gy, "h")):
            lo = [i for i, q in enumerate(g) if q <= p0 + 1 and p0 - q <= 1.5 * T]
            hi = [i for i, q in enumerate(g) if q >= p1 - 1 and q - p1 <= 1.5 * T]
            if lo and hi and hi[0] > lo[-1]:
                i, j = lo[-1], hi[0]
                out.append({"axis": ax, "i": i, "j": j, "clear": clear, "clear_px": cpx,
                            "half_t_px": (thick(o, g[i]) + thick(o, g[j])) / 2, "text": t["text"]})
    return out


def solve(grid_x, grid_y, bbox, toks, struct=None, segs=None, T=6.0):
    """Exact grid positions from written dimension chains and room-size labels.
    Conflicting measurements are rejected instead of distorting the model."""
    grids = {"x": list(grid_x), "y": list(grid_y)}
    cons = []
    # ---- (a) dimension chains written outside the building
    nums = [t for t in toks if t["kind"] == "num" and t["value"] > 0]
    chain_ratios = []
    if nums:
        f = _unit_factor([t["value"] for t in nums])
        bx0, by0, bx1, by1 = bbox
        pad = 0.02 * max(bx1 - bx0, by1 - by0)
        for t in nums:
            outside_lr = t["x"] < bx0 - pad or t["x"] > bx1 + pad
            outside_tb = t["y"] < by0 - pad or t["y"] > by1 + pad
            ax = "x" if (outside_tb and not outside_lr) else ("y" if (outside_lr and not outside_tb) else None)
            if ax is None:
                continue
            g, c = grids[ax], (t["x"] if ax == "x" else t["y"])
            pairs = [(i, j) for i in range(len(g)) for j in range(i + 1, len(g))
                     if abs((g[i] + g[j]) / 2 - c) <= 0.2 * (g[j] - g[i]) + 15]
            for i, j in pairs:
                chain_ratios.append(t["value"] * f / (g[j] - g[i]))
            cons.append({"kind": "dim", "axis": ax, "tok": t, "d": t["value"] * f, "pairs": pairs})
    # ---- (b) room-size labels
    rooms = _room_constraints(struct, segs or [], T, toks, grids["x"], grids["y"])
    room_ratios = [r["clear"] / r["clear_px"] for r in rooms]

    def consensus(r):
        r = np.array(r)
        if len(r) == 0:
            return None, 0
        sup = [(np.abs(np.log(r / q)) < 0.03).sum() for q in r]
        best = r[int(np.argmax(sup))]
        sel = r[np.abs(np.log(r / best)) < 0.03]
        return float(np.median(sel)), len(sel)

    s_chain, n_chain = consensus(chain_ratios)
    s_room, n_room = consensus(room_ratios)
    if n_chain >= 2:
        s, source = s_chain, "written dimension chains"
    elif n_room >= 2:
        s, source = s_room, "room-size labels"
    elif n_chain or n_room:
        s, source = (s_chain or s_room), "single measurement (low confidence)"
    else:
        return None

    out = {"scale_m_per_px": s, "source": source, "axes": {}, "report": []}
    for ax in ("x", "y"):
        g = grids[ax]
        n = len(g)
        eqs = []                                   # (i, j, d, report-entry)
        for c in cons:
            if c["axis"] != ax:
                continue
            best = None
            for i, j in c["pairs"]:
                L = g[j] - g[i]
                err = abs(s * L - c["d"]) / c["d"]
                cost = err + 0.3 * abs((g[i] + g[j]) / 2 - (c["tok"]["x"] if ax == "x" else c["tok"]["y"])) / L
                if err < 0.08 and (best is None or cost < best[0]):
                    best = (cost, i, j)
            if best:
                eqs.append([best[1], best[2], c["d"], {"kind": "dimension", "axis": ax, "text": c["tok"]["text"],
                                                       "written_m": round(c["d"], 4)}])
        for r in rooms:
            if r["axis"] != ax:
                continue
            d = r["clear"] + r["half_t_px"] * s
            if abs(s * (g[r["j"]] - g[r["i"]]) - d) / d < 0.08:
                eqs.append([r["i"], r["j"], d, {"kind": "room clear size", "axis": ax, "text": r["text"],
                                                "written_m": round(r["clear"], 4), "half_t_m": r["half_t_px"] * s}])
        active = list(range(len(eqs)))
        rejected = []
        for _ in range(len(eqs) + 1):
            rows, rhs, w = [], [], []
            for k in active:
                i, j, d, _ = eqs[k]
                r_ = np.zeros(n); r_[j], r_[i] = 1, -1
                rows.append(r_); rhs.append(d); w.append(1.0)
            for k in range(1, n):
                r_ = np.zeros(n); r_[k], r_[k - 1] = 1, -1
                rows.append(r_); rhs.append(s * (g[k] - g[k - 1])); w.append(0.02)
            r_ = np.zeros(n); r_[0] = 1; rows.append(r_); rhs.append(0.0); w.append(10.0)
            A, b, W = np.array(rows), np.array(rhs), np.array(w)
            sol = np.linalg.lstsq(A * W[:, None], b * W, rcond=None)[0] if n else np.array([])
            res = [(abs(sol[eqs[k][1]] - sol[eqs[k][0]] - eqs[k][2]), k) for k in active]
            if not res:                                                  # no written dimension on this axis
                break
            worst = max(res)
            if worst[0] > max(0.02, 0.01 * eqs[worst[1]][2]):
                active.remove(worst[1]); rejected.append(worst[1])   # conflicts with the others
            else:
                break
        # never allow a distorted grid: spacing must stay within 20% of the global scale
        if n > 1 and np.any(np.abs(np.log(np.maximum(np.diff(sol), 1e-9) / (s * np.maximum(np.diff(g), 1e-9)))) > 0.2):
            sol = np.array([(q - g[0]) * s for q in g])
            out["report"].append({"axis": ax, "text": "grid", "status": "distorted -> uniform scale used"})
        out["axes"][ax] = {"px": g, "m": sol.tolist(), "n_dims": len(active)}
        for k, (i, j, d, rep) in enumerate(eqs):
            got = sol[j] - sol[i]
            rep = dict(rep)
            if "half_t_m" in rep:
                got -= rep.pop("half_t_m")
                d = rep["written_m"]
            rep.update({"model_m": round(float(got), 4), "error_mm": round((got - d) * 1000, 1),
                        "status": "rejected (conflicts)" if k in rejected else
                        ("ok" if abs(got - d) < 0.005 else "conflict")})
            out["report"].append(rep)
    return out


def make_mapper(sol, axis):
    """px -> metres. Exact at grid lines, linear in between, measured scale outside."""
    px = np.array(sol["axes"][axis]["px"], float)
    m = np.array(sol["axes"][axis]["m"], float)
    s = sol["scale_m_per_px"]

    def f(v):
        v = np.asarray(v, float)
        if len(px) == 0:
            return v * s
        if len(px) == 1:
            return m[0] + (v - px[0]) * s
        out = np.interp(v, px, m)
        out = np.where(v < px[0], m[0] + (v - px[0]) * s, out)
        out = np.where(v > px[-1], m[-1] + (v - px[-1]) * s, out)
        return out
    return f


# ----------------------------------------------------------------------------- furniture detector helpers
# FloorPlanCAD classes we train on (doors / windows / stairs / walls come from the floor-plan model)
FURN_CLASSES = ["bed", "sofa", "chair", "table", "bedside_cupboard", "tv_cabinet", "half_height_cabinet",
                "high_cabinet", "wardrobe", "sink", "bath", "bath_tub", "toilet", "squat_toilet", "urinal",
                "gas_stove", "refrigerator", "washing_machine"]
ZOOM_FACTORS = (0.75, 1.0, 1.4)
# typical real sizes (long side, metres) used to measure the drawing scale of the dataset
TYPICAL_LONG_M = {"bed": 2.0, "toilet": 0.7, "bath_tub": 1.7, "sofa": 2.0, "refrigerator": 0.7}


def norm_label(s):
    return str(s).strip().lower().replace(" ", "_").replace("-", "_")


def to_rgb_white(img):
    """PNG with alpha -> RGB on white (FloorPlanCAD renders are 4-channel)."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        a = img[:, :, 3:4].astype(np.float32) / 255.0
        return (img[:, :, :3].astype(np.float32) * a + 255.0 * (1 - a)).astype(np.uint8)
    return img


def det_to_obb(bbox, mask, W, H):
    """FiftyOne detection (normalised [x,y,w,h] + cropped mask) -> 4 corner points in pixels.
    The rotated rectangle is fitted to the drawn strokes, so it follows the furniture's rotation."""
    bx, by, bw, bh = [float(v) for v in bbox]
    x0, y0, w, h = bx * W, by * H, bw * W, bh * H
    pts = None
    if mask is not None:
        m = np.asarray(mask)
        if m.ndim == 3:
            m = m[..., 0]
        ys, xs = np.nonzero(m)
        if len(xs) >= 5:
            sx, sy = w / m.shape[1], h / m.shape[0]
            pts = np.stack([x0 + (xs + 0.5) * sx, y0 + (ys + 0.5) * sy], 1).astype(np.float32)
    if pts is None:
        pts = np.array([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]], np.float32)
    rect = cv2.minAreaRect(pts)
    (cw, ch) = rect[1]
    if cw < 2 or ch < 2:                     # degenerate -> axis-aligned box
        rect = ((x0 + w / 2, y0 + h / 2), (max(w, 2), max(h, 2)), 0.0)
    return cv2.boxPoints(rect), rect


def obb_line(cls_id, corners, W, H):
    c = np.clip(corners, 0, [W - 1, H - 1]) / [W, H]
    return "{} ".format(cls_id) + " ".join("{:.6f}".format(v) for v in c.reshape(-1))


def write_sample(img_path, dets, out_img, out_lbl, stats):
    """dets: list of (label, bbox, mask). Writes RGB jpg + YOLO-OBB label file. Returns #objects."""
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return 0
    img = to_rgb_white(img)
    H, W = img.shape[:2]
    lines = []
    for label, bbox, mask in dets:
        lab = norm_label(label)
        if lab not in FURN_CLASSES:
            continue
        corners, rect = det_to_obb(bbox, mask, W, H)
        lines.append(obb_line(FURN_CLASSES.index(lab), corners, W, H))
        stats.setdefault(lab, []).append(max(rect[1]))
    cv2.imwrite(out_img, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    with open(out_lbl, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def dataset_px_per_m(stats):
    """Drawing scale of the dataset from typical object sizes (median of several classes)."""
    est = [np.median(stats[k]) / v for k, v in TYPICAL_LONG_M.items() if len(stats.get(k, [])) >= 20]
    return float(np.median(est)) if est else None


# ----------------------------------------------------------------------------- inference on a plan
def detect_furniture(model, img_rgb, plan_px_per_m, train_px_per_m, conf=0.35, tile=1024, overlap=0.25):
    """Rescale the plan so furniture has its training size, run tiled OBB detection, merge.
    Returns list of dicts in ORIGINAL image pixels: cls, conf, corners(4x2)."""
    names = model.names
    base = train_px_per_m / plan_px_per_m if (train_px_per_m and plan_px_per_m) else 1.0
    scales = [base * f for f in ZOOM_FACTORS]              # several zooms, merged below
    found = []
    for s in scales:
        s = float(np.clip(s, 0.2, 6.0))
        im = cv2.resize(img_rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        H, W = im.shape[:2]
        step = int(tile * (1 - overlap))
        ys = list(range(0, max(H - tile, 0) + 1, step)) or [0]
        xs = list(range(0, max(W - tile, 0) + 1, step)) or [0]
        if ys[-1] + tile < H:
            ys.append(max(H - tile, 0))
        if xs[-1] + tile < W:
            xs.append(max(W - tile, 0))
        for y0 in ys:
            for x0 in xs:
                crop = np.full((tile, tile, 3), 255, np.uint8)
                c = im[y0:y0 + tile, x0:x0 + tile]
                crop[:c.shape[0], :c.shape[1]] = c
                res = model.predict(crop[:, :, ::-1], imgsz=tile, conf=conf, verbose=False)[0]
                if res.obb is None or len(res.obb) == 0:
                    continue
                pts = res.obb.xyxyxyxy.cpu().numpy()
                for k in range(len(pts)):
                    p = pts[k].reshape(4, 2)
                    cx, cy = p[:, 0].mean(), p[:, 1].mean()
                    # ignore detections centred in the overlap margin of an inner tile edge
                    m = tile * overlap / 2
                    if (x0 > 0 and cx < m) or (y0 > 0 and cy < m) or \
                       (x0 + tile < W and cx > tile - m) or (y0 + tile < H and cy > tile - m):
                        continue
                    found.append({"cls": names[int(res.obb.cls[k])], "conf": float(res.obb.conf[k]),
                                  "corners": (p + [x0, y0]) / s})
    return rotated_nms(found)


def rotated_nms(dets, iou_thr=0.4):
    from shapely.geometry import Polygon
    dets = sorted(dets, key=lambda d: -d["conf"])
    keep = []
    for d in dets:
        p = Polygon(d["corners"]).buffer(0)
        if p.area <= 0:
            continue
        ok = True
        for k in keep:
            q = k["_poly"]
            inter = p.intersection(q).area
            if inter / (p.area + q.area - inter + 1e-9) > iou_thr or inter / min(p.area, q.area) > 0.8:
                ok = False
                break
        if ok:
            d["_poly"] = p
            keep.append(d)
    for k in keep:
        k.pop("_poly", None)
    return keep


# ----------------------------------------------------------------------------- 4c. furniture & fixtures
# height (m), category, which side goes against a wall ("short" = head/tank end, "long" = back, None = free)
FURN_SPEC = {
    "bed": (0.55, "furniture", "short"), "sofa": (0.80, "furniture", "long"), "chair": (0.85, "furniture", None),
    "table": (0.75, "furniture", None), "bedside_cupboard": (0.55, "furniture", "long"),
    "tv_cabinet": (0.50, "furniture", "long"), "half_height_cabinet": (0.90, "furniture", "long"),
    "high_cabinet": (2.10, "furniture", "long"), "wardrobe": (2.10, "furniture", "long"),
    "closet": (2.10, "furniture", "long"), "sauna_bench": (0.45, "furniture", "long"),
    "sink": (0.85, "plumbing", "long"), "bath": (0.55, "plumbing", "long"), "bath_tub": (0.55, "plumbing", "long"),
    "toilet": (0.75, "plumbing", "short"), "squat_toilet": (0.10, "plumbing", "short"), "urinal": (0.60, "plumbing", "short"),
    "gas_stove": (0.90, "appliance", "long"), "refrigerator": (1.80, "appliance", "long"),
    "washing_machine": (0.85, "appliance", "long"), "appliance": (0.90, "appliance", "long"),
    "fireplace": (1.00, "appliance", "long"), "chimney": (1.20, "appliance", None),
}
# floor-plan model icon classes -> furniture types (doors/windows/stairs are handled elsewhere)
ICON_FIXTURES = {3: "closet", 4: "appliance", 5: "toilet", 6: "sink", 7: "sauna_bench", 8: "fireplace",
                 9: "bath_tub", 10: "chimney"}
SAME_KIND = [{"toilet", "squat_toilet", "urinal"}, {"sink"}, {"bath", "bath_tub"},
             {"closet", "wardrobe", "high_cabinet", "half_height_cabinet"},
             {"appliance", "refrigerator", "washing_machine", "gas_stove"}, {"fireplace", "chimney"}]


def _kind(t):
    for k in SAME_KIND:
        if t in k:
            return frozenset(k)
    return frozenset([t])


def icon_fixtures(icon, T):
    """Fixtures the floor-plan model already segments -> rotated rectangles in px."""
    out = []
    for cls, name in ICON_FIXTURES.items():
        m = cv2.morphologyEx((icon == cls).astype(np.uint8), cv2.MORPH_CLOSE,
                             np.ones((max(3, int(0.6 * T)),) * 2, np.uint8))     # join fragments of one object
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] < max(12, 0.5 * T * T):
                continue
            ys, xs = np.nonzero(lab == i)
            rect = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
            out.append({"type": name, "conf": 1.0, "corners": cv2.boxPoints(rect), "source": "floor-plan model"})
    return out


WET_ONLY = {"bath", "bath_tub", "toilet", "squat_toilet", "urinal"}
# floor-plan model room classes: 1 Outdoor, 3 Kitchen, 4 LivingRoom, 5 Bedroom, 6 Bath, 7 Entry/Hallway,
# 9 Storage, 10 Garage, 11 OtherRoom
DRY_ROOMS = {1, 3, 4, 5, 7, 10}                 # a bath / toilet can never be here
FITS_ROOM = {                                   # where a weak detection is believable
    "bed": {5, 11}, "bedside_cupboard": {5, 11}, "wardrobe": {5, 9, 11}, "high_cabinet": {3, 5, 9, 11},
    "half_height_cabinet": {3, 4, 9, 11}, "sofa": {4, 11}, "tv_cabinet": {4, 5, 11}, "chair": {3, 4, 5, 11},
    "table": {3, 4, 11}, "gas_stove": {3}, "refrigerator": {3, 9, 11}, "washing_machine": {6, 9, 11},
    "sink": {3, 6, 9, 11}, "bath": {6, 11}, "bath_tub": {6, 11}, "toilet": {6, 11}, "squat_toilet": {6, 11},
    "urinal": {6, 11}}


def _room_at(room, it, r):
    H, W = room.shape
    cx, cy = np.asarray(it["corners"], float).mean(0)
    x, y = int(np.clip(round(cx), 0, W - 1)), int(np.clip(round(cy), 0, H - 1))
    win = room[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].ravel()
    win = win[(win != 0) & (win != 2)]                       # ignore background / wall pixels
    indoor = win[win != 1]
    if indoor.size:                                          # pixels across an exterior wall are outdoors:
        return int(np.bincount(indoor, minlength=12).argmax())   # prefer the room the object is in
    return 1 if win.size else None


def filter_by_room(items, room, T):
    """Cross-check furniture with the floor-plan model's rooms.
    - a bath / toilet in a clearly dry room (hall, living, bedroom, kitchen, garage, outdoors) is a
      look-alike symbol (skylight, shaft) -> dropped
    - a weak detector hit (below FURNITURE_CONF) is kept only when the room fits (bed in a bedroom ...)"""
    keep, dropped = [], 0
    r = max(2, int(2 * T))
    for it in items:
        rc = _room_at(room, it, r)
        if it["type"] in WET_ONLY and rc in DRY_ROOMS:
            dropped += 1
            continue
        if it["source"] == "furniture detector" and it["conf"] < FURNITURE_CONF:
            if rc is None or rc not in FITS_ROOM.get(it["type"], set()):
                dropped += 1
                continue
        keep.append(it)
    return keep, dropped


def detector_furniture(img, mpp):
    """Run the furniture detector from your HF repo (skipped if it is not trained yet)."""
    if not FURNITURE_REPO:
        return []
    try:
        from huggingface_hub import HfApi, hf_hub_download
        token = os.environ.get("HF_TOKEN")
        if not token:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
        repo = "{}/{}".format(HfApi(token=token).whoami()["name"], FURNITURE_REPO)
        w = hf_hub_download(repo, "checkpoints/best.pt", token=token)
        meta = json.load(open(hf_hub_download(repo, "furniture_meta.json", token=token)))
    except Exception as e:
        print("furniture detector not available yet ({}): using fixtures from the floor-plan model only".format(
            type(e).__name__))
        return []
    if not os.environ.get("P3D_NO_INSTALL"):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.3.253"], check=False)
    from ultralytics import YOLO
    dets = detect_furniture(YOLO(w), img, plan_px_per_m=1.0 / mpp, train_px_per_m=meta.get("train_px_per_m"),
                            conf=FURNITURE_LOW_CONF)          # weak hits are checked against the rooms later
    return [{"type": d["cls"], "conf": round(d["conf"], 3), "corners": d["corners"], "source": "furniture detector"}
            for d in dets if d["cls"] in FURN_SPEC]


def build_furniture(items_px, plan):
    """Rotated px rectangles -> metres, with the back placed against the nearest wall."""
    from shapely.geometry import Polygon, Point
    from shapely.ops import unary_union
    mx, my = make_mapper(plan["mapping"], "x"), make_mapper(plan["mapping"], "y")
    walls = []
    for w in plan["walls"]:
        a, d, n, L = wall_frame(w)
        h = w["thickness"] / 2
        walls.append(Polygon([a - n * h, a + d * L - n * h, a + d * L + n * h, a + n * h]))
    wall_union = unary_union(walls) if walls else None
    # detector first (exact rotation), then model fixtures that do not duplicate a detected object
    items_px = sorted(items_px, key=lambda d: (d["source"] != "furniture detector", -d["conf"]))
    out, polys = [], []
    for it in items_px:
        c = np.array([[float(mx(x)), float(my(y))] for x, y in it["corners"]])
        poly = Polygon(c).buffer(0)
        if poly.area < 0.04:
            continue
        if (it["type"] in WET_ONLY and it["source"] == "furniture detector" and wall_union is not None
                and wall_union.distance(poly) > 0.3):
            continue          # baths / toilets stand against a wall; free-floating X-boxes are skylights
        dup = False
        for q, o in zip(polys, out):
            inter = poly.intersection(q).area
            if inter / min(poly.area, q.area) > 0.5 and _kind(it["type"]) == _kind(o["type"]):
                dup = True
                break
        if dup:
            continue
        e01, e12 = np.linalg.norm(c[1] - c[0]), np.linalg.norm(c[2] - c[1])
        if e01 >= e12:
            long_dir, width, depth = (c[1] - c[0]) / e01, e01, e12
        else:
            long_dir, width, depth = (c[2] - c[1]) / e12, e12, e01
        centre = c.mean(0)
        # furniture sits square to the walls: snap the rotation to the wall grid when it is close
        ang = np.degrees(np.arctan2(long_dir[1], long_dir[0])) % 90
        tol = 25 if it["source"] == "floor-plan model" else 12      # model blobs give rough angles
        if min(ang, 90 - ang) <= tol:
            k = np.round(np.degrees(np.arctan2(long_dir[1], long_dir[0])) / 90.0) * 90.0
            long_dir = np.array([np.cos(np.radians(k)), np.sin(np.radians(k))])
            if it["source"] == "floor-plan model":
                # rough blob: measure the real extent along the snapped axes
                sd = np.array([-long_dir[1], long_dir[0]])
                width = float(np.ptp(c @ long_dir)); depth = float(np.ptp(c @ sd))
                if depth > width:
                    long_dir, width, depth = sd, depth, width
            short = np.array([-long_dir[1], long_dir[0]])
            c = np.array([centre + sa * long_dir * width / 2 + sb * short * depth / 2
                          for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        # real size settles look-alike symbols (seats in particular)
        if it["type"] == "chair" and width >= 1.1:
            it = dict(it, type="sofa")
        elif it["type"] == "sofa" and width < 0.9:
            it = dict(it, type="chair")
        height, cat, back = FURN_SPEC.get(it["type"], (0.8, "furniture", None))
        short_dir = np.array([-long_dir[1], long_dir[0]])
        front = short_dir
        if back and wall_union is not None:
            if back == "short":   # head / tank end against the wall -> faces along the long axis
                cands = [(centre + long_dir * width / 2, -long_dir), (centre - long_dir * width / 2, long_dir)]
            else:                 # back of the unit against the wall -> faces across
                cands = [(centre + short_dir * depth / 2, -short_dir), (centre - short_dir * depth / 2, short_dir)]
            front = min(cands, key=lambda pc: wall_union.distance(Point(pc[0])))[1]
        out.append({"id": len(out), "type": it["type"], "category": cat, "source": it["source"],
                    "confidence": it["conf"], "center": [round(float(v), 4) for v in centre],
                    "width": round(float(width), 3), "depth": round(float(depth), 3), "height": height,
                    "front": [round(float(v), 4) for v in front],
                    "rotation_deg": round(float(np.degrees(np.arctan2(front[1], front[0]))), 2),
                    "corners": [[round(float(x), 4), round(float(y), 4)] for x, y in c]})
        polys.append(poly)
    return out


FURN_COLORS = {"furniture": [176, 132, 92, 255], "plumbing": [240, 242, 246, 255], "appliance": [150, 155, 162, 255]}


def furniture_meshes(furniture):
    import trimesh
    parts = {}
    for f in furniture:
        v, fc = prism(f["corners"], 0.0, f["height"])
        parts.setdefault(f["category"], []).append(trimesh.Trimesh(v, fc, process=False))
    meshes = {}
    for cat, ms in parts.items():
        m = trimesh.util.concatenate(ms)
        m.visual = trimesh.visual.ColorVisuals(m, face_colors=np.tile(FURN_COLORS[cat], (len(m.faces), 1)))
        meshes[cat] = m
    return meshes


# ----------------------------------------------------------------------------- 3b. stairs, entrance steps, stepped areas
# Every stair gets: kind (entrance / interior / terrace), shape (straight / flared / irregular), the number of
# risers, riser height, tread depth, width, run, base and top level, and one solid block per tread.
# Each value says where it came from, best source first:
#   written on the plan (dimension, "12R @ 150", "+0.45 PLINTH")  >  counted from the drawn tread lines
#   >  your setting (PLINTH_HEIGHT)  >  standard assumption (flagged in "warnings")
_N_RISERS = re.compile(r"\b(\d{1,2})\s*(?:NOS?\.?\s*)?(?:R|RISERS?)\b", re.I)
_N_STEPS = re.compile(r"\b(\d{1,2})\s*(?:NOS?\.?\s*)?(?:STEPS?|TREADS?)\b", re.I)
_RISER_MM = re.compile(r"(?:\bR(?:ISER)?\s*[@=:]?\s*|@\s*)(\d{2,3})\b", re.I)
_LEVEL = re.compile(r"(?:^|[\s(])\+\s*(\d+(?:\.\d+)?)")
_LEVEL_WORDS = re.compile(r"PLINTH|\bP\.?L\b|F\.?F\.?L|FLOOR\s*LEVEL|LVL", re.I)


def _raw(t):
    return t.get("raw") or t.get("text") or ""


def _plinth_from_levels(toks):
    """'+0.45', '+450', 'PLINTH LVL +0.60' -> plinth height in metres (or None)."""
    cands = []
    for t in toks:
        raw = _raw(t)
        for m in _LEVEL.finditer(" " + raw):
            v = float(m.group(1))
            v = v if v <= 3 else v / 1000.0
            if 0.05 <= v <= 2.0:
                cands.append((bool(_LEVEL_WORDS.search(raw)), v))
    if not cands:
        return None
    worded = [v for w, v in cands if w]
    return float(np.median(worded or [v for _, v in cands]))


def _stair_notes(toks, cx, cy, r_px):
    """Notes written next to a stair: riser count, riser height, UP / DN labels."""
    near = [t for t in toks if np.hypot(t["x"] - cx, t["y"] - cy) <= r_px]
    text = " ".join(dict.fromkeys(_raw(t) for t in near))
    out = {}
    m = _N_RISERS.search(text) or _N_STEPS.search(text)
    if m and 2 <= int(m.group(1)) <= 40:
        out["risers"] = int(m.group(1))
    for m in _RISER_MM.finditer(text):
        if 100 <= int(m.group(1)) <= 220:
            out["riser"] = int(m.group(1)) / 1000.0
    out["up"] = [(t["x"], t["y"]) for t in near if _raw(t).strip().upper().rstrip(".") == "UP"]
    out["dn"] = [(t["x"], t["y"]) for t in near if _raw(t).strip().upper().rstrip(".") in ("DN", "DOWN")]
    return out


def _zoom_labels(original, s, x0, y0, x1, y1):
    """Short words like UP / DN are easy to miss on the whole sheet: re-read the area around a stair, zoomed."""
    X0, Y0, X1, Y1 = int(x0 * s), int(y0 * s), int(x1 * s), int(y1 * s)
    crop = original[Y0:Y1, X0:X1]
    if crop.size == 0:
        return {}
    z = 2.0 if max(crop.shape[:2]) < 1500 else 1.0
    big = cv2.resize(crop, None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC) if z != 1 else crop
    if big.ndim == 2:
        big = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    out = {"up": [], "dn": []}
    for t in ocr_tokens(big):
        w = _raw(t).strip().upper().rstrip(".")
        pt = ((t["x"] / z + X0) / s, (t["y"] / z + Y0) / s)
        if w == "UP":
            out["up"].append(pt)
        elif w in ("DN", "DOWN"):
            out["dn"].append(pt)
    return out


def _written_length(toks, cx, cy, r_px, measured, tol=0.12):
    """A written dimension near the stair that matches a measured length (m) -> (value, text) or None."""
    best = None
    for t in toks:
        if t["kind"] != "num" or np.hypot(t["x"] - cx, t["y"] - cy) > r_px:
            continue
        for v in (t["value"], t["value"] / 100.0, t["value"] / 1000.0):
            if v > 0 and abs(v - measured) / measured <= tol:
                err = abs(v - measured) / measured
                if best is None or err < best[2]:
                    best = (v, _raw(t), err)
    return None if best is None else best[:2]


def _window(shape, x0, y0, x1, y1, pad):
    H, W = shape
    return max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)


def _count_treads(comp, box, D_model, original, s, m_per_opx):
    """Count treads between the drawn lines inside the stair (full-resolution drawing).
    comp: bool crop of the stair (model px) at box=(ox, oy); D_model: distance to the HIGH side, same crop.
    Returns (n, label image low->high 1..n, (X0, Y0)) in original px, or None."""
    ox, oy = box
    H, W = comp.shape
    X0, Y0 = int(ox * s), int(oy * s)
    X1, Y1 = min(original.shape[1], int(np.ceil((ox + W) * s))), min(original.shape[0], int(np.ceil((oy + H) * s)))
    if X1 - X0 < 8 or Y1 - Y0 < 8:
        return None
    crop = original[Y0:Y1, X0:X1]
    size = (X1 - X0, Y1 - Y0)
    region = cv2.resize(comp.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    D = cv2.resize(D_model.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    ink = (gray < float(np.clip(thr, 100, 200))).astype(np.uint8)   # Otsu can collapse on near-white crops
    ink = cv2.dilate(ink, np.ones((2, 2), np.uint8)) > 0              # close tiny breaks in tread lines
    free = region & ~ink
    n, lab, st, _ = cv2.connectedComponentsWithStats(free.astype(np.uint8), connectivity=4)
    min_area = max(20, 0.02 / m_per_opx ** 2)                         # smaller than a 0.2 x 0.1 m tread = noise
    comps = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < min_area:
            continue
        d = D[lab == i]
        comps.append((float(np.median(d)), float(np.percentile(d, 20)), float(np.percentile(d, 80)), i))
    if len(comps) < 2:
        return None
    # a strip along the side (inside the model's stair area, outside the drawn stair) joins many treads:
    # far too long in the climbing direction to be one tread -> not a tread
    spans = np.array([c[2] - c[1] for c in comps])
    typical = float(np.median(spans))
    comps = [c for c, sp in zip(comps, spans) if sp <= 2.5 * max(typical, 1.0)]
    if len(comps) < 2:
        return None
    comps.sort()
    # pieces of one tread split by the walking line / arrow sit at the same distance from the top -> one tread
    groups = [[comps[0]]]
    for c in comps[1:]:
        g = groups[-1]
        lo, hi = min(x[1] for x in g), max(x[2] for x in g)
        ov = min(hi, c[2]) - max(lo, c[1])
        if ov > 0.5 * min(hi - lo, c[2] - c[1]) + 1e-6:
            g.append(c)
        else:
            groups.append([c])
    k = len(groups)
    # count by pitch over the whole run, so treads merged or lost in the middle do not shrink the count
    if k >= 3:
        cen = np.array([np.median([c[0] for c in g]) for g in groups])
        pitch = float(np.median(np.diff(cen)))
        dr = D[region]
        span = float(np.percentile(dr, 99) - np.percentile(dr, 1)) if dr.size else 0.0
        if pitch > 0:
            k_pitch = int(round(span / pitch))
            if k_pitch > k:
                k_full = k_pitch
            else:
                k_full = k
        else:
            k_full = k
    else:
        k_full = k
    covered = sum(st[c[3], cv2.CC_STAT_AREA] for g in groups for c in g)
    if not (2 <= k_full <= 40) or covered < 0.25 * region.sum():
        return None
    out = np.zeros(lab.shape, np.int32)
    for j, g in enumerate(groups):                                   # groups: high (near anchor) -> low
        for c in g:
            out[lab == c[3]] = k - j                                 # label 1 = lowest step
    # give the line pixels to the nearest tread so the treads tile the stair
    _, idx = ndimage.distance_transform_edt(out == 0, return_indices=True)
    filled = out[idx[0], idx[1]]
    filled[~(region | (out > 0))] = 0
    return k_full, filled, (X0, Y0)


def _label_polys(labels, X0, Y0, s, mx, my, m_per_opx):
    """Tread label image (original px) -> list of shapely polygons in metres, index 0 = lowest."""
    from shapely.geometry import Polygon
    polys = []
    eps = max(1.0, 0.03 / m_per_opx)
    for k in range(1, labels.max() + 1):
        cs, _ = cv2.findContours((labels == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            polys.append(None)
            continue
        c = cv2.approxPolyDP(max(cs, key=cv2.contourArea), eps, True)[:, 0]
        if len(c) < 3:
            polys.append(None)
            continue
        p = Polygon([(float(mx((x + X0) / s)), float(my((y + Y0) / s))) for x, y in c]).buffer(0)
        if p.geom_type == "MultiPolygon":
            p = max(p.geoms, key=lambda g: g.area)
        polys.append(p if p.area > 0.01 else None)
    return polys


def _slice(poly, centre, d, L, k):
    """Cut a stair outline into k equal treads along the run direction d (low -> high)."""
    from shapely.geometry import Polygon
    nrm = np.array([-d[1], d[0]])
    big = 100.0
    out = []
    for i in range(k):
        u0, u1 = -L / 2 + L * i / k, -L / 2 + L * (i + 1) / k
        band = Polygon([centre + d * u0 - nrm * big, centre + d * u1 - nrm * big,
                        centre + d * u1 + nrm * big, centre + d * u0 + nrm * big])
        p = poly.intersection(band)
        if p.geom_type != "Polygon":
            ps = [g for g in getattr(p, "geoms", []) if g.geom_type == "Polygon"]
            p = max(ps, key=lambda g: g.area) if ps else None
        out.append(p if p is not None and p.area > 1e-4 else None)
    return out


def _poly_list(p):
    return [[round(float(x), 4), round(float(y), 4)] for x, y in list(p.exterior.coords)[:-1]]


def _chain_flights(infos):
    """Interior flights that touch each other (a split / dog-leg stair: main flight -> landing -> flights)
    share one storey height: the longest flight goes from the floor to the landing, the others from the
    landing up. Each flight is turned so it climbs away from the one below."""
    inter = [i for i in infos if i["kind"] == "interior"]
    seen = set()
    for a in range(len(inter)):
        if a in seen:
            continue
        group, todo = [], [a]
        while todo:
            k = todo.pop()
            if k in seen:
                continue
            seen.add(k)
            group.append(k)
            todo += [j for j in range(len(inter)) if j not in seen and inter[k]["hull"].distance(inter[j]["hull"]) < 0.4]
        if len(group) < 2:
            continue
        fl = [inter[k] for k in group]
        for f_ in fl:                                     # unknown counts: from the run (about 280 mm treads)
            if not f_["n"]:
                f_["n"] = max(2, int(round(f_["run"] / 0.28)) + 1)
                f_["sources"]["risers"] = "estimated from the run length"
                f_["warnings"].append("risers not written or readable: estimated from the run")
        main = max(fl, key=lambda f_: f_["run"])
        uppers = [f_ for f_ in fl if f_ is not main]
        ucentre = np.mean([u["centre"] for u in uppers], 0)
        ends = lambda f_: (f_["centre"] - f_["d"] * f_["run"] / 2, f_["centre"] + f_["d"] * f_["run"] / 2)
        lo, hi = ends(main)
        if np.linalg.norm(lo - ucentre) < np.linalg.norm(hi - ucentre):  # main climbs towards the landing
            main["d"] = -main["d"]
        main["direction"] = "towards the landing and the upper flights"
        top_main = main["centre"] + main["d"] * main["run"] / 2
        for u in uppers:
            # an upper flight climbs AWAY from the landing: on a split stair the wings turn sideways,
            # left one to the left, right one to the right
            off = u["centre"] - top_main
            nrm = np.array([-u["d"][1], u["d"][0]])
            if abs(np.dot(nrm, off)) > abs(np.dot(u["d"], off)) and len(uppers) > 1:
                u["d"] = nrm                               # the flight runs across what was taken as its width
                u["run"], u["width"] = u["width"], u["run"]
                u["width_low"] = u["width_high"] = u["width"]
            if np.dot(u["d"], off) < 0:
                u["d"] = -u["d"]                          # upper flights start at the landing
            u["direction"] = "from the landing up, away from the main flight"
        n_up = max(u["n"] for u in uppers)                # parallel flights (split stair) climb together
        r = WALL_HEIGHT / (main["n"] + n_up)
        main.update(chain_base=0.0, chain_total=r * main["n"], chain_size=len(fl))
        for u in uppers:
            u.update(chain_base=r * main["n"], chain_total=WALL_HEIGHT - r * main["n"], chain_size=len(fl))
            u["n"] = n_up
        for f_ in fl:
            f_["warnings"] = [w_ for w_ in f_["warnings"] if not w_.startswith("direction not marked")]


def drawn_step_regions(original, shape, have_px, wall_px, T):
    """Steps the model did not mark, read from the drawing: 4+ parallel thin lines, evenly spaced about a tread
    apart, overlapping each other (stepped seating, sunken areas, unmarked flights).
    Returns stair entries like vectorise() makes, in model px."""
    if original is None:
        return []
    g = original if original.ndim == 2 else cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    k = g.shape[1] / shape[1]                                  # drawing px per model px
    ink = (g < 140).astype(np.uint8)
    thick = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))
    thin = (ink & (1 - cv2.dilate(thick, np.ones((3, 3), np.uint8)))) * 255
    segs = cv2.HoughLinesP(thin, 1, np.pi / 360, 25, minLineLength=int(max(20, 2.5 * T * k)), maxLineGap=3)
    if segs is None:
        return []
    segs = segs.reshape(-1, 4).astype(float)
    ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])) % 180
    out, used = [], np.zeros(len(segs), bool)
    model_px = have_px.copy()                                 # stairs the model marked
    found_px = np.zeros(shape, bool)                          # stepped areas found here
    for a0 in np.unique(np.round(ang / 3) * 3):
        idx = np.where((np.abs(((ang - a0) + 90) % 180 - 90) <= 3) & ~used)[0]
        if len(idx) < 4:
            continue
        d = np.array([np.cos(np.radians(a0)), np.sin(np.radians(a0))])
        n = np.array([-d[1], d[0]])
        # one entry per drawn line: merge Hough duplicates (same offset, overlapping)
        lines_ = []
        for i in sorted(idx, key=lambda i: (segs[i, :2] @ n)):
            off = float(((segs[i, :2] + segs[i, 2:]) / 2) @ n)
            u0, u1 = sorted([float(segs[i, :2] @ d), float(segs[i, 2:] @ d)])
            for L_ in lines_:
                if abs(L_[0] - off) <= 2.5 and min(L_[2], u1) - max(L_[1], u0) > -3:
                    L_[1], L_[2] = min(L_[1], u0), max(L_[2], u1)
                    break
            else:
                lines_.append([off, u0, u1])
        lines_.sort()
        lo_s, hi_s = 0.6 * T * k, 2.6 * T * k
        i = 0
        while i < len(lines_) - 3:
            run = [lines_[i]]
            for L_ in lines_[i + 1:]:
                gap = L_[0] - run[-1][0]
                ov = min(L_[2], run[-1][2]) - max(L_[1], run[-1][1])
                if gap < lo_s * 0.5:
                    continue                                    # a double line, not the next tread
                ok_ov = ov >= 0.5 * min(L_[2] - L_[1], run[-1][2] - run[-1][1])
                med_len = float(np.median([q[2] - q[1] for q in run]))
                ok_ov = ok_ov and 0.4 * med_len <= (L_[2] - L_[1]) <= 2.5 * med_len   # tread lines, not dimensions
                if len(run) >= 2:
                    med = float(np.median(np.diff([q[0] for q in run])))
                    one = abs(gap - med) <= 0.35 * med
                    two = abs(gap - 2 * med) <= 0.35 * med       # the line finder missed one tread line
                    if ok_ov and (one or two):
                        run.append(L_)
                        continue
                    if gap > 2.5 * med:
                        break
                    continue
                if lo_s <= gap <= hi_s and ok_ov:
                    run.append(L_)
                elif gap > hi_s:
                    break
            if len(run) >= 4:
                pts = []
                for off, u0, u1 in run:
                    pts += [d * u0 + n * off, d * u1 + n * off]
                hull = cv2.convexHull(np.array(pts, np.float32) / k)[:, 0]
                m = np.zeros(shape, np.uint8)
                cv2.fillPoly(m, [np.round(hull).astype(np.int32)], 1)
                m = m > 0
                area = m.sum()
                # skip what the model already marked, and duplicates of an arm already found (the other arm
                # of an L overlaps it only at the corner)
                if area and (m & model_px).sum() < 0.3 * area and (m & found_px).sum() < 0.8 * area \
                        and (m & wall_px).sum() < 0.25 * area:
                    ys, xs = np.nonzero(m)
                    x0, y0 = xs.min(), ys.min()
                    out.append({"x0": float(x0), "y0": float(y0), "x1": float(xs.max() + 1), "y1": float(ys.max() + 1),
                                "ox": int(x0), "oy": int(y0), "mask": m[y0:ys.max() + 1, x0:xs.max() + 1].copy(),
                                "drawn": True, "n_lines": len(run), "line_dir": (float(d[0]), float(d[1]))})
                    found_px = found_px | m
                i += len(run)
            else:
                i += 1
    return out


def build_stairs(v, toks, room, plan, original=None):
    """Detected stair regions -> exact stairs, entrance steps and stepped areas (plan.json 'stairs' + 'levels')."""
    from shapely.geometry import Point, Polygon
    mx, my = make_mapper(plan["mapping"], "x"), make_mapper(plan["mapping"], "y")
    MX, MY = (lambda q: float(mx(q))), (lambda q: float(my(q)))
    mpp = float(plan["mapping"]["scale_m_per_px"])
    s = (original.shape[1] / room.shape[1]) if original is not None else None
    m_per_opx = mpp / s if s else None
    footprint = Polygon(plan["floor"]["polygon"]).buffer(0) if plan.get("floor") else None
    T = v["T"]
    pad = int(max(3 * T, 1.5 / mpp))
    stair_px = np.zeros(room.shape, bool)
    for st in v["stairs"]:
        if "mask" in st:
            stair_px[st["oy"]:st["oy"] + st["mask"].shape[0], st["ox"]:st["ox"] + st["mask"].shape[1]] |= st["mask"]
    walls_px = room == WALL
    extra = drawn_step_regions(original, room.shape, stair_px, walls_px, v["T"])
    if extra:
        print("  steps read from the drawing (not marked by the model): {} areas".format(len(extra)))
        for st in extra:
            stair_px[st["oy"]:st["oy"] + st["mask"].shape[0], st["ox"]:st["ox"] + st["mask"].shape[1]] |= st["mask"]
    all_st = list(v["stairs"]) + extra
    indoor = ~np.isin(room, (0, 1)) & (room != WALL) & ~stair_px

    infos = []
    for st in all_st:
        if "mask" not in st:
            continue
        comp, ox, oy = st["mask"], st["ox"], st["oy"]
        h, w = comp.shape
        ys, xs = np.nonzero(comp)
        pts = np.stack([xs + ox, ys + oy], 1).astype(np.float32)
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)
        hull = cv2.convexHull(pts)[:, 0]
        area = float(comp.sum())
        rect_sol = area / max(rect[1][0] * rect[1][1], 1.0)
        hull_sol = area / max(cv2.contourArea(hull), 1.0)
        shape = "straight" if rect_sol >= 0.85 else ("flared" if hull_sol >= 0.9 else "irregular")
        # metres
        hull_m = Polygon([(MX(x), MY(y)) for x, y in hull]).buffer(0)
        box_m = np.array([[MX(x), MY(y)] for x, y in box])
        e01, e12 = np.linalg.norm(box_m[1] - box_m[0]), np.linalg.norm(box_m[2] - box_m[1])
        if e01 >= e12:
            run, width, d = e01, e12, (box_m[1] - box_m[0]) / max(e01, 1e-9)
        else:
            run, width, d = e12, e01, (box_m[2] - box_m[1]) / max(e12, 1e-9)
        centre = box_m.mean(0)
        if width < 0.6 or run < 0.5 or hull_m.area < 0.8:
            continue                                                # too small for a stair -> false positive
        # kind: outside the building = entrance steps; L / U shaped stepped area inside = terrace
        out_frac = float(np.isin(room[oy:oy + h, ox:ox + w][comp], (0, 1)).mean())
        # the building outline decides; the model's "outdoor" pixels only count right at the outline
        # (an open lobby the model does not label as a room is still inside the building)
        if footprint is not None:
            pc = Point(*centre)
            outside = (not footprint.contains(pc)) or (out_frac >= 0.6 and footprint.exterior.distance(pc) < 1.0)
        else:
            outside = out_frac >= 0.4
        kind = "entrance" if outside else ("terrace" if (shape == "irregular" or st.get("drawn")) else "interior")
        # distance (model px) to the HIGH side of the stair
        wx0, wy0, wx1, wy1 = _window(room.shape, ox, oy, ox + w, oy + h, pad)
        cx_px, cy_px = float(xs.mean() + ox), float(ys.mean() + oy)
        notes = _stair_notes(toks, cx_px, cy_px, max(pad, 0.6 * max(h, w)))
        warnings, direction = [], None
        ends_px = []                                                # the two run ends (model px)
        bx = np.array(box, float)
        if np.linalg.norm(bx[1] - bx[0]) >= np.linalg.norm(bx[2] - bx[1]):
            ends_px = [(bx[0] + bx[3]) / 2, (bx[1] + bx[2]) / 2]
        else:
            ends_px = [(bx[0] + bx[1]) / 2, (bx[2] + bx[3]) / 2]
        if kind == "entrance":
            anchor = indoor[wy0:wy1, wx0:wx1]
            direction = "rises towards the building"
        elif kind == "terrace":
            anchor = walls_px[wy0:wy1, wx0:wx1]
            direction = "rises towards the walls (assumed for a stepped area)"
        else:
            anchor = None
        if anchor is not None and anchor.any():
            Dw = ndimage.distance_transform_edt(~anchor)
        else:
            if not (notes["up"] or notes["dn"]) and original is not None:
                notes.update({k: v_ for k, v_ in _zoom_labels(original, s, wx0, wy0, wx1, wy1).items() if v_})
            # interior: top = end away from the 'UP' label (UP is written at the foot), or at the 'DN' label
            top = 1
            if notes["up"]:
                u = np.array(notes["up"][0])
                top = int(np.argmax([np.linalg.norm(e - u) for e in ends_px]))
                direction = "from the UP label"
            elif notes["dn"]:
                u = np.array(notes["dn"][0])
                top = int(np.argmin([np.linalg.norm(e - u) for e in ends_px]))
                direction = "from the DN label"
            else:
                direction = "assumed (no UP / DN label)"
                warnings.append("direction not marked on the plan: check which end goes up")
            a = np.zeros((wy1 - wy0, wx1 - wx0), np.uint8)
            far = ends_px[top] + (ends_px[top] - ends_px[1 - top]) / max(np.linalg.norm(ends_px[top] - ends_px[1 - top]), 1e-9) * 2
            ax_ = (ends_px[top] - ends_px[1 - top]) / max(np.linalg.norm(ends_px[top] - ends_px[1 - top]), 1e-9)
            pn = np.array([-ax_[1], ax_[0]]) * (h + w)              # a line across the top end, not a point
            cv2.line(a, tuple(int(q) for q in (far + pn - [wx0, wy0])), tuple(int(q) for q in (far - pn - [wx0, wy0])), 1, 2)
            Dw = ndimage.distance_transform_edt(a == 0)
        D_crop = Dw[oy - wy0:oy - wy0 + h, ox - wx0:ox - wx0 + w]
        Dat = lambda e: Dw[int(np.clip(e[1] - wy0, 0, Dw.shape[0] - 1)), int(np.clip(e[0] - wx0, 0, Dw.shape[1] - 1))]
        if st.get("line_dir"):
            # read from the drawing: the steps climb across their lines
            ld = np.array(st["line_dir"]); nd = np.array([-ld[1], ld[0]])
            cpx = np.array([cx_px, cy_px])
            ext = 0.5 * max(h, w)
            ends_px = [cpx - nd * ext, cpx + nd * ext]
        elif kind != "interior":
            # the run is the axis that climbs: the pair of opposite edges with the biggest height difference
            # (entrance steps are often wider than deep, so the long side is not always the run)
            pairs = [[(bx[0] + bx[1]) / 2, (bx[2] + bx[3]) / 2], [(bx[1] + bx[2]) / 2, (bx[3] + bx[0]) / 2]]
            ends_px = max(pairs, key=lambda pr: abs(Dat(pr[0]) - Dat(pr[1])))
        # low -> high direction along the run
        lo_i = int(np.argmax([Dat(e) for e in ends_px]))
        lo_m = np.array([MX(ends_px[lo_i][0]), MY(ends_px[lo_i][1])])
        hi_m = np.array([MX(ends_px[1 - lo_i][0]), MY(ends_px[1 - lo_i][1])])
        if np.linalg.norm(hi_m - lo_m) > 1e-6:
            d = (hi_m - lo_m) / np.linalg.norm(hi_m - lo_m)
        # run = along d across the outline, width = across it (measured at both ends: flared if they differ)
        from shapely.geometry import LineString
        nrm = np.array([-d[1], d[0]])
        proj = np.array(hull_m.exterior.coords) @ d
        run = float(proj.max() - proj.min())
        centre = centre + d * (0.5 * (proj.max() + proj.min()) - centre @ d)

        def cross_width(u):
            c = centre + d * u
            return float(LineString([c - nrm * 50, c + nrm * 50]).intersection(hull_m).length)
        width_low, width_high = cross_width(-0.45 * run), cross_width(0.45 * run)
        width = max(width_low, width_high)
        if shape == "straight" and abs(width_low - width_high) > 0.08 * width:
            shape = "flared"
        # treads counted from the drawing
        counted = None
        if original is not None:
            try:
                counted = _count_treads(comp, (ox, oy), D_crop, original, s, m_per_opx)
            except Exception as e:
                warnings.append("tread lines could not be read ({})".format(type(e).__name__))
        # written width / run
        sources = {}
        r_px = max(pad, 0.8 * max(h, w))
        if shape == "straight":
            width_low = width_high = width = 0.5 * (width_low + width_high)
        wr = _written_length(toks, cx_px, cy_px, r_px, width)
        if wr:
            k = wr[0] / width
            width_low, width_high = width_low * k, width_high * k
            width, sources["width"] = wr[0], "written {}".format(wr[1])
        else:
            sources["width"] = "measured"
        rr = _written_length(toks, cx_px, cy_px, r_px, run)
        if rr and abs(rr[0] - width) > 1e-6:
            run, sources["run"] = rr[0], "written {}".format(rr[1])
        else:
            sources["run"] = "measured"
        if counted is None and st.get("n_lines"):
            counted = (max(2, st["n_lines"] - 1), None, None)    # tiers between the lines read from the drawing
        infos.append(dict(kind=kind, shape=shape, hull=hull_m, centre=centre, d=d, run=run, width=width,
                          width_low=width_low, width_high=width_high,
                          notes=notes, counted=counted, sources=sources, warnings=warnings, direction=direction,
                          lo=lo_m, hi=hi_m))

    # ---- plinth (ground -> floor level)
    lvl = _plinth_from_levels(toks)
    entr = [i for i in infos if i["kind"] == "entrance"]
    for i in infos:                                                 # risers from notes / drawing
        n_note, n_cnt = i["notes"].get("risers"), (i["counted"][0] if i["counted"] else None)
        if i["kind"] == "interior" and n_cnt:
            n_cnt += 1                                              # drawn treads + the riser onto the upper floor
        i["n"], i["sources"]["risers"] = ((n_note, "written on the plan") if n_note else
                                          (n_cnt, "counted from the drawn treads") if n_cnt else (None, None))
    _chain_flights(infos)
    if PLINTH_HEIGHT:
        plinth, psrc = float(PLINTH_HEIGHT), "your setting"
    elif lvl:
        plinth, psrc = lvl, "level mark on the plan"
    elif entr:
        est = [(i["n"] or 3) * i["notes"].get("riser", STAIR_RISE_EXT) for i in entr]
        plinth = float(np.median(est))
        psrc = ("riser note x steps" if any(i["notes"].get("riser") for i in entr)
                else "assumed {:.0f} mm risers x counted steps".format(STAIR_RISE_EXT * 1000))
    else:
        plinth, psrc = 0.0, "none (no entrance steps, no level mark)"

    stairs = []
    for k, i in enumerate(infos):
        kind, w = i["kind"], i["warnings"]
        if kind == "entrance":
            base, total, rsrc = -plinth, plinth, "plinth: " + psrc
            if not i["n"]:
                i["n"] = max(1, int(round(total / STAIR_RISE_EXT)))
                i["sources"]["risers"] = "estimated from the plinth height"
                w.append("number of steps not written or readable: estimated")
            if psrc.startswith("assumed"):
                w.append("plinth height not written: set 'Plinth height' to make the steps exact")
        elif kind == "interior" and "chain_base" in i:
            base, total = i["chain_base"], i["chain_total"]
            rsrc = "storey height shared by {} connected flights".format(i["chain_size"])
        elif kind == "interior":
            base, total, rsrc = 0.0, WALL_HEIGHT, "storey height (wall height setting)"
            if not i["n"]:
                i["n"] = int(np.clip(round(total / STAIR_RISE), 3, 30))
                i["sources"]["risers"] = "estimated from the storey height"
                w.append("risers not written or readable: estimated at about {:.0f} mm".format(STAIR_RISE * 1000))
        else:
            r = i["notes"].get("riser", STAIR_RISE_EXT)
            if not i["n"]:
                i["n"] = 3
                i["sources"]["risers"] = "assumed"
                w.append("tiers could not be counted: 3 assumed")
            total = r * i["n"]
            base = -total if STEPPED_AREAS == "down" else 0.0
            if STEPPED_AREAS == "down" and plinth > 0.01:
                # steps going down lead to the open ground: they drop the full plinth height
                total, base = plinth, -plinth
                rsrc = "plinth: " + psrc + " split over the drawn steps"
            if not (STEPPED_AREAS == "down" and plinth > 0.01):
                rsrc = "riser note" if "riser" in i["notes"] else "assumed {:.0f} mm per tier".format(r * 1000)
            i["direction"] = ("goes DOWN from the raised floor to the open ground (floor and plinth cut out)" if STEPPED_AREAS == "down"
                              else "rises towards the walls (raised seating)")
            w.append("stepped area going {}: tier height {}; check".format(
                STEPPED_AREAS, "written" if "riser" in i["notes"] else "assumed"))
        n = int(i["n"])
        riser = total / n if n else 0.0
        if kind != "terrace" and not (0.10 <= riser <= 0.22):
            w.append("riser works out to {:.0f} mm: check the step count / levels".format(riser * 1000))
        # tread blocks: entrance -> n blocks (top one is the landing at floor level);
        # interior -> n-1 treads (the last riser lands on the upper floor); terrace -> n tiers
        blocks = n - 1 if kind == "interior" else n
        blocks = max(blocks, 1)
        polys = None
        if i["shape"] == "irregular" and i["counted"]:
            _, lab, (X0, Y0) = i["counted"]
            polys = _label_polys(lab, X0, Y0, s, mx, my, m_per_opx)
            polys = [p for p in polys if p is not None]
            if len(polys) != i["counted"][0]:
                polys = None
            else:
                blocks = len(polys)
        if polys is None:
            if i["shape"] == "straight":
                from shapely.geometry import Polygon as _P
                d, c, L, W = i["d"], i["centre"], i["run"], i["width"]
                nrm = np.array([-d[1], d[0]])
                outline = _P([c - d * L / 2 - nrm * W / 2, c + d * L / 2 - nrm * W / 2,
                              c + d * L / 2 + nrm * W / 2, c - d * L / 2 + nrm * W / 2])
            else:
                outline = i["hull"]
            polys = _slice(outline, i["centre"], i["d"], i["run"], blocks)
        treads = []
        for j, p in enumerate(polys):
            if p is None:
                continue
            if kind == "terrace" and STEPPED_AREAS == "down":
                top = -riser * (j + 1)                         # first tier (atrium side) one step down
            else:
                top = base + riser * (j + 1)
            treads.append({"polygon": _poly_list(p), "z0": round(base, 4), "z1": round(top, 4)})
        tread = i["run"] / blocks
        if kind != "terrace" and not (0.22 <= tread <= 0.45):
            w.append("tread works out to {:.0f} mm: check".format(tread * 1000))
        outline = i["hull"] if i["shape"] != "straight" else None
        if outline is None:
            d, c, L, W = i["d"], i["centre"], i["run"], i["width"]
            nrm = np.array([-d[1], d[0]])
            ol = [c - d * L / 2 - nrm * W / 2, c + d * L / 2 - nrm * W / 2,
                  c + d * L / 2 + nrm * W / 2, c - d * L / 2 + nrm * W / 2]
            outline_l = [[round(float(x), 4), round(float(y), 4)] for x, y in ol]
        else:
            outline_l = _poly_list(outline)
        c, d, L = i["centre"], i["d"], i["run"]
        stairs.append({
            "id": k, "kind": kind, "shape": i["shape"],
            "outline": outline_l,
            "start": [round(float(v_), 4) for v_ in c - d * L / 2],          # low end (walking line)
            "end": [round(float(v_), 4) for v_ in c + d * L / 2],            # high end
            "width": round(i["width"], 4), "width_low": round(i["width_low"], 4),
            "width_high": round(i["width_high"], 4), "run": round(L, 4),
            "risers": n, "riser": round(riser, 4), "tread": round(tread, 4),
            "base_z": round(base, 4), "top_z": round(0.0 if (kind == "terrace" and STEPPED_AREAS == "down") else base + total, 4),
            "sunken": bool(kind == "terrace" and STEPPED_AREAS == "down"),
            "direction": i["direction"],
            "sources": dict(i["sources"], rise=rsrc),
            "treads": treads, "warnings": w,
            "straight_run_fits": bool(i["shape"] == "straight" and kind != "terrace"),
        })
    levels = {"floor": 0.0, "ground": round(-plinth, 4), "plinth": round(plinth, 4), "plinth_source": psrc}
    return stairs, levels


# ----------------------------------------------------------------------------- 3. exact plan in metres
def _snap(v, options, tol):
    best = min(options, key=lambda o: abs(o - v))
    return best if abs(best - v) <= tol * best else v


def _round(v, step):
    return round(round(v / step) * step, 4)


NOTE_WORDS = ("VAULT", "SKY", "SHOWER", "SOAKER", "TUB", "FRIDGE", "FREEZ", "OVEN", "BENCH", "HOOKS",
              "WALL HT", "PK. HT", "CABINET", "BUFFET", "LINEN", "STRG", "WORKSPACE", "PLANTER", "ROOF",
              "OPENING", "ABOVE", "BELOW", "DROP", "UP", "DN", "SCALE", "FLOOR PLAN", "AREA:", "SF")


def _is_note(text):
    t = text.strip().upper()
    if len(re.sub(r"[^A-Z]", "", t)) < 2:
        return True
    return any(t.startswith(w) or (" " + w) in (" " + t) for w in NOTE_WORDS) and not any(
        k in t for k in ("ROOM", "BED", "BATH", "KITCHEN", "LIVING", "DINING", "TOILET", "STORE"))


def _room_name(text):
    t = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", text.strip())
    t = re.sub(r"(?i)^W\.*I?\.*C\.*$", "W.I.C.", t)                         # OCR: W..C. -> W.I.C.
    t = re.sub(r"(?i)(primary|master|guest|kids|main)(bed|living|bath)", r"\1 \2", t)
    if "." in t and len(t) <= 8:
        return t.upper()                                                     # abbreviations: W.I.C., C.R.
    return t.title() if len(t) >= 3 and t.isupper() else (t[:1].upper() + t[1:])


def build_plan(v, sol, toks, room):
    from shapely.geometry import Polygon, Point, LineString
    from shapely.ops import unary_union
    segs = v["walls"]
    if sol is None:          # no readable dimensions -> fallback scale
        doors = [o["u1"] - o["u0"] for o in v["openings"] if o["kind"] == "door"]
        s = DOOR_WIDTH_M / float(np.median(doors)) if doors else 0.2 / v["T"]
        if METERS_PER_PIXEL:
            s = METERS_PER_PIXEL
        gx = sorted({sg["c"] for sg in segs if sg["o"] == "v"})
        gy = sorted({sg["c"] for sg in segs if sg["o"] == "h"})
        sol = {"scale_m_per_px": s, "axes": {"x": {"px": gx, "m": [(g - gx[0]) * s for g in gx], "n_dims": 0},
                                             "y": {"px": gy, "m": [(g - gy[0]) * s for g in gy], "n_dims": 0}},
               "report": [], "fallback": True}
    mx, my = make_mapper(sol, "x"), make_mapper(sol, "y")
    s = sol["scale_m_per_px"]
    MX = lambda q: float(mx(q))
    MY = lambda q: float(my(q))

    # walls: centre-lines meeting at the centre-line of the wall they join (Revit joins them)
    walls = []
    for sg in segs:
        t = _snap(sg["t"] * s, WALL_THICKNESSES, 0.35) if WALL_THICKNESSES else _round(sg["t"] * s, 0.005)
        a, b = [MX(sg["p0"][0]), MY(sg["p0"][1])], [MX(sg["p1"][0]), MY(sg["p1"][1])]
        walls.append({"start": [round(c, 4) for c in a], "end": [round(c, 4) for c in b], "thickness": t,
                      "height": WALL_HEIGHT, "joined": ["j0" in sg, "j1" in sg]})
    # extensions so solids close the corner (half the joined wall's thickness)
    for w, sg in zip(walls, segs):
        e = []
        for key in ("j0", "j1"):
            if key in sg:
                e.append(round(_snap(sg[key][1] * s, WALL_THICKNESSES, 0.35) / 2 if WALL_THICKNESSES
                               else sg[key][1] * s / 2, 4))
            else:
                e.append(0.0)
        w["ext"] = e
    for i, w in enumerate(walls):
        w["id"] = i

    def wall_poly(w):
        a, b = np.array(w["start"]), np.array(w["end"])
        d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        n = np.array([-d[1], d[0]])
        a2, b2 = a - d * w["ext"][0], b + d * w["ext"][1]
        h = w["thickness"] / 2
        return Polygon([a2 - n * h, b2 - n * h, b2 + n * h, a2 + n * h])

    union = unary_union([wall_poly(w) for w in walls])
    polys = [union] if union.geom_type == "Polygon" else list(union.geoms)
    main = max(polys, key=lambda p: p.area)
    # outline: close door/window-sized gaps (<= 2 m) between wall groups, keep the biggest piece
    # close entrance / door gaps until one outline covers most of the building (open plans with big
    # missing walls need a larger radius than a house with every door detected)
    hull = union.convex_hull
    for r in (1.0, 1.5, 2.0, 3.0, 4.5, 6.0, None):
        if r is None:                     # walls too broken to close: the convex outline of all walls
            footprint = Polygon(hull.exterior)
            break
        closed = union.buffer(r, join_style=2).buffer(-r, join_style=2)
        cpolys = [closed] if closed.geom_type == "Polygon" else list(closed.geoms)
        footprint = Polygon(max(cpolys, key=lambda p: p.area).exterior)
        if footprint.area >= 0.6 * hull.area:
            break
    ring = footprint.exterior
    for w in walls:
        w["external"] = bool(LineString([w["start"], w["end"]]).distance(ring) <= w["thickness"] * 0.75)
    if EXTERNAL_WALL_M or INTERNAL_WALL_M:
        for w in walls:
            w["thickness"] = (EXTERNAL_WALL_M if w["external"] else INTERNAL_WALL_M) or w["thickness"]

    # openings: centre point on the wall centre-line, exact width from the solved grid
    openings = []
    for o in v["openings"]:
        sg, w = segs[o["wall"]], walls[o["wall"]]
        q0 = np.array([MX(o["q0"][0]), MY(o["q0"][1])])
        q1 = np.array([MX(o["q1"][0]), MY(o["q1"][1])])
        a_, d_, _, _ = wall_frame(w)
        along = float(((q0 + q1) / 2 - a_) @ d_)
        centre = list(a_ + d_ * along)                     # exactly on the wall centre-line
        u0, u1 = 0.0, float(np.linalg.norm(q1 - q0))
        width = abs(u1 - u0)
        if o["kind"] == "door" and width < 0.5:
            # the model found a door but marked only part of it (e.g. the hinge / arc): use a standard width
            # centred on the mark, as long as it fits on the wall
            L_w = float(np.linalg.norm(np.array(w["end"]) - np.array(w["start"])))
            std = DOOR_WIDTHS[len(DOOR_WIDTHS) // 2] if DOOR_WIDTHS else 0.9
            room_ = min(along, L_w - along) * 2 - 0.1
            if room_ < 0.6:
                continue
            width = min(std, room_)
        elif width < (0.5 if o["kind"] == "door" else 0.3):
            continue
        if o["kind"] == "door":
            width = _snap(width, DOOR_WIDTHS, 0.09)
            width = width if width in DOOR_WIDTHS else _round(width, 0.05)
            sill, head = 0.0, DOOR_HEIGHT
        else:
            width = _round(width, 0.05)
            sill, head = WINDOW_SILL, WINDOW_HEAD
        openings.append({"id": len(openings), "type": o["kind"], "wall": o["wall"],
                         "center": [round(c, 4) for c in centre], "offset": round(abs(along), 4),
                         "width": width, "height": round(head - sill, 4), "sill": sill})

    openings.sort(key=lambda o: (o["wall"], o["offset"]))
    merged = []
    for o in openings:
        p = merged[-1] if merged else None
        if p and p["wall"] == o["wall"] and p["type"] == o["type"] == "window" and \
                (o["offset"] - o["width"] / 2) - (p["offset"] + p["width"] / 2) < 0.25:
            lo, hi = p["offset"] - p["width"] / 2, o["offset"] + o["width"] / 2
            w_ = walls[p["wall"]]
            a_, d_, _, _ = wall_frame(w_)
            p["width"] = _round(hi - lo, 0.05)
            p["offset"] = round((lo + hi) / 2, 4)
            p["center"] = [round(c, 4) for c in (a_ + d_ * p["offset"])]
            continue
        merged.append(o)
    openings = merged
    for i, o in enumerate(openings):
        o["id"] = i

    # rooms = closed regions between walls; names from the text written inside them
    NOT_ROOMS = {"UP", "DN", "DOWN", "UP.", "DN."}
    labels = [t for t in toks if t["kind"] == "label"]
    room_labels = [t for t in labels if t["text"].strip().upper() not in NOT_ROOMS]
    rooms = []
    # close door-sized gaps so rooms are closed regions; sizes relative to the wall thickness, so a
    # wrong first scale (fixed afterwards by the room-size check) does not drop every room
    tm = v["T"] * s
    gap = 3.2 * tm                       # closes gaps up to ~1.5 m (single and double doors the model missed)
    shut = union.buffer(gap, join_style=2).buffer(-gap, join_style=2)
    spolys = [shut] if shut.geom_type == "Polygon" else list(shut.geoms)
    holes = [h for pg in spolys for h in pg.interiors]
    for hole in holes:
        poly = Polygon(hole)
        if poly.area < 40 * tm * tm:
            continue
        inside = [t for t in room_labels if poly.contains(Point(MX(t["x"]), MY(t["y"])))
                  and not _is_note(t["text"])]
        # the room title is the biggest text in the room (notes like VAULT / SKYLIGHT are smaller)
        inside.sort(key=lambda t: -t.get("h", 0))
        names = []
        for t in inside:                      # open-plan areas keep every title of the same size
            n = _room_name(t["text"])
            if t.get("h", 0) >= 0.85 * inside[0].get("h", 0) and n not in names and len(names) < 3:
                names.append(n)
        names = names or ["Room {}".format(len(rooms) + 1)]
        p = poly.representative_point()
        rooms.append({"id": len(rooms), "name": " / ".join(names), "area_m2": round(poly.area, 2),
                      "point": [round(p.x, 4), round(p.y, 4)],
                      "polygon": [[round(x, 4), round(y, 4)] for x, y in list(poly.exterior.coords)[:-1]]})

    stairs = []                                          # filled by build_stairs() in main()

    # structural grids (labelled 1..n across, A.. down)
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"

    def major(axis):
        """Grid only where a dimension lands or where walls run long (keeps Revit tidy)."""
        load = {}
        for w in walls:
            vert = abs(w["start"][0] - w["end"][0]) < 1e-6
            if (axis == "x") != vert:
                continue
            key = round(w["start"][0] if vert else w["start"][1], 3)
            load[key] = load.get(key, 0.0) + abs((w["end"][1] - w["start"][1]) if vert else (w["end"][0] - w["start"][0]))
        ext = (max(load) - min(load)) if load else 0.0
        keep = sorted(k for k, L in load.items() if L >= 0.25 * ext)
        merged = []
        for k in keep:
            if not merged or k - merged[-1] > 0.5:
                merged.append(k)
        return merged
    xs, ys = major("x"), major("y")
    bx0, by0, bx1, by1 = footprint.bounds
    grids = [{"name": str(i + 1), "start": [x, by0 - 1.5], "end": [x, by1 + 1.5]} for i, x in enumerate(xs)]
    grids += [{"name": letters[i % len(letters)], "start": [bx0 - 1.5, y], "end": [bx1 + 1.5, y]}
              for i, y in enumerate(ys)]

    rep = sol["report"]
    ok = [r for r in rep if r.get("status") == "ok"]
    mapping = {"scale_m_per_px": s, "axes": {a: {"px": [float(q) for q in sol["axes"][a]["px"]],
                                                  "m": [float(q) for q in sol["axes"][a]["m"]]} for a in "xy"}}
    return {"units": "m", "mapping": mapping, "axes": "x right, y down the sheet (Revit script flips y)",
            "level": {"name": "Ground Floor", "elevation": 0.0, "wall_height": WALL_HEIGHT},
            "precision": {"dimensions_read": len([r for r in rep if "written_m" in r]),
                          "dimensions_exact": len(ok), "report": rep,
                          "scale_source": sol.get("source", "written dimensions") if not sol.get("fallback")
                          else "estimated (no dimensions read) - set BUILDING_WIDTH_M or METERS_PER_PIXEL"},
            "walls": walls, "openings": openings, "rooms": rooms, "stairs": stairs, "grids": grids,
            "floor": {"polygon": [[round(x, 4), round(y, 4)] for x, y in list(footprint.exterior.coords)[:-1]],
                      "thickness": SLAB},
            "defaults": {"wall_height": WALL_HEIGHT, "door_height": DOOR_HEIGHT, "window_sill": WINDOW_SILL,
                         "window_head": WINDOW_HEAD, "slab": SLAB}}


# ----------------------------------------------------------------------------- 4. 3D preview mesh (GLB)
def wall_frame(w):
    a, b = np.array(w["start"], float), np.array(w["end"], float)
    d = b - a
    L = float(np.linalg.norm(d))
    d = d / max(L, 1e-9)
    return a, d, np.array([-d[1], d[0]]), L


def pieces(plan):
    by_wall = {}
    for o in plan["openings"]:
        by_wall.setdefault(o["wall"], []).append(o)
    for w in plan["walls"]:
        a, d, n, L = wall_frame(w)
        t = w["thickness"]
        rect = lambda u0, u1, th: [a + d * u0 - n * th / 2, a + d * u1 - n * th / 2,
                                   a + d * u1 + n * th / 2, a + d * u0 + n * th / 2]
        cur = -w["ext"][0]
        for o in sorted(by_wall.get(w["id"], []), key=lambda o: o["offset"]):
            u0, u1 = o["offset"] - o["width"] / 2, o["offset"] + o["width"] / 2
            u0, u1 = max(u0, cur), min(u1, L + w["ext"][1])
            if u1 <= u0:
                continue
            top = o["sill"] + o["height"]
            if u0 > cur:
                yield "wall", rect(cur, u0, t), 0.0, w["height"]
            if o["sill"] > 0:
                yield "wall", rect(u0, u1, t), 0.0, o["sill"]
            if top < w["height"]:
                yield "wall", rect(u0, u1, t), top, w["height"]
            if o["type"] == "door":
                yield "door", rect(u0 + 0.03, u1 - 0.03, 0.05), 0.0, top
            else:
                yield "glass", rect(u0 + 0.03, u1 - 0.03, 0.03), o["sill"], top
            cur = u1
        if cur < L + w["ext"][1]:
            yield "wall", rect(cur, L + w["ext"][1], t), 0.0, w["height"]
    for s in plan["stairs"]:
        for t in s.get("treads", []):
            yield "stairs", t["polygon"], t["z0"], t["z1"]


def prism(corners, z0, z1):
    """Vertical prism over a simple polygon (3+ corners, convex or not), y-up mesh coordinates."""
    c = np.asarray(corners, float)[:, :2]
    if len(c) > 3 and np.allclose(c[0], c[-1]):
        c = c[:-1]
    m = len(c)
    v = np.array([[p[0], z0, p[1]] for p in c] + [[p[0], z1, p[1]] for p in c])
    if m == 4:
        tri = [[0, 1, 2], [0, 2, 3]]
    else:
        import mapbox_earcut as earcut
        idx = earcut.triangulate_float64(c, np.array([m], dtype=np.uint32)).reshape(-1, 3)
        area = 0.5 * np.sum(c[:, 0] * np.roll(c[:, 1], -1) - np.roll(c[:, 0], -1) * c[:, 1])
        tri = []
        for a_, b_, c_ in idx:                       # same winding as the outline
            t2 = 0.5 * ((c[b_, 0] - c[a_, 0]) * (c[c_, 1] - c[a_, 1]) - (c[c_, 0] - c[a_, 0]) * (c[b_, 1] - c[a_, 1]))
            tri.append([a_, b_, c_] if (t2 > 0) == (area > 0) else [a_, c_, b_])
    f = [list(t) for t in tri] + [[m + t[0], m + t[2], m + t[1]] for t in tri]
    for i in range(m):
        j = (i + 1) % m
        f += [[i, m + i, j], [j, m + i, m + j]]
    f = np.array(f)
    t = v[f]
    if np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() < 0:
        f = f[:, ::-1]
    return v, f


COLORS = {"wall": [235, 232, 225, 255], "door": [150, 95, 55, 255], "glass": [140, 190, 235, 140],
          "stairs": [190, 190, 190, 255], "floor": [205, 185, 150, 255],
          "plinth": [168, 160, 150, 255], "ground": [150, 165, 140, 255],
          "furniture": [176, 132, 92, 255], "plumbing": [240, 242, 246, 255], "appliance": [150, 155, 162, 255]}


def fill_floor_gaps(plan, v, room):
    """Checks the converted plan against the model before anything is built.
    1. floor: every area the model calls room or wall is inside the building -> no empty patches
       where a room is (the outline from the walls alone misses corners whose outer wall was lost).
    2. a broken outer wall is carried on to the next wall, and rooms / notches it closes get floor."""
    from shapely.geometry import Polygon, LineString, Point
    from shapely.ops import unary_union
    T = v["T"]
    mx, my = make_mapper(plan["mapping"], "x"), make_mapper(plan["mapping"], "y")
    # ---- 1. floor
    bld = ((room >= 2) & (room <= 11)).astype(np.uint8)
    k = int(2 * T) | 1
    bld = cv2.morphologyEx(bld, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, lab, stt, _ = cv2.connectedComponentsWithStats(bld, 8)
    floor = Polygon(plan["floor"]["polygon"]).buffer(0)
    if n > 1:
        big = 1 + int(np.argmax(stt[1:, cv2.CC_STAT_AREA]))
        cs, _ = cv2.findContours((lab == big).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cs, key=cv2.contourArea)[:, 0, :].astype(float)
        mpoly = Polygon([(float(mx(x)), float(my(y))) for x, y in c]).buffer(0).simplify(0.05)
        gaps = mpoly.buffer(0).difference(floor).buffer(-0.3).buffer(0.3)     # thin slivers along walls are not gaps
        gaps = [g for g in getattr(gaps, "geoms", [gaps]) if g.area > 1.0 and g.buffer(0.05).intersects(floor)]
        new = unary_union([floor] + [g.buffer(0.05) for g in gaps])
        if new.geom_type != "Polygon":
            new = max(new.geoms, key=lambda g: g.area)
        new = Polygon(new.exterior).simplify(0.03)
        added = new.area - floor.area
        if added > 0.5:
            print("  floor check: floor filled where the model sees rooms but the wall outline left a gap (+{:.1f} m2)".format(added))
            plan["floor"]["polygon"] = [[round(float(x), 4), round(float(y), 4)] for x, y in list(new.exterior.coords)[:-1]]
            floor = new
    # a broken outer wall: a wall end that stops in the open, with another wall ahead on the same line
    # and the gap OUTSIDE the floor (so it is not a doorway) -> carry the wall on until it meets that wall
    lines0 = [LineString([w["start"], w["end"]]) for w in plan["walls"]]
    closed = 0
    for i, w in enumerate(plan["walls"]):
        a, b = np.array(w["start"], float), np.array(w["end"], float)
        L = np.linalg.norm(b - a)
        if L < 0.3:
            continue
        d = (b - a) / L
        for end, sgn in (("end", 1), ("start", -1)):
            p = b if sgn > 0 else a
            if any(lines0[j].distance(Point(p)) < 0.3 for j in range(len(lines0)) if j != i):
                continue                                          # already meets a wall
            ray = LineString([p, p + d * sgn * 4.0])
            hits = [(ray.intersection(lines0[j]), j) for j in range(len(lines0)) if j != i and ray.intersects(lines0[j])]
            hits = [(Point(p).distance(h), h) for h, j in hits if h.geom_type == "Point"]
            if not hits:
                continue
            dist_, h = min(hits, key=lambda t: t[0])
            gap = LineString([p, (h.x, h.y)])
            if dist_ > 0.3 and gap.difference(floor.buffer(-0.15)).length > 0.8 * gap.length:
                w[end] = [round(h.x, 4), round(h.y, 4)]
                lines0[i] = LineString([w["start"], w["end"]])
                a, b = np.array(w["start"], float), np.array(w["end"], float)
                closed += 1
    if closed:
        print("  floor check: {} broken outer wall(s) carried on to the next wall".format(closed))
    # rooms closed by walls (door gaps bridged) belong to the building too
    wl = unary_union([LineString([w["start"], w["end"]]).buffer(w["thickness"] / 2 + 0.5, cap_style=2, join_style=2) for w in plan["walls"]])
    cells = [Polygon(r).buffer(0.45, join_style=2) for g in getattr(wl, "geoms", [wl]) for r in g.interiors]
    cells = [c for c in cells if c.area > 1.0 and c.difference(floor).area > 0.5]
    if cells:
        new = unary_union([floor] + cells)
        if new.geom_type != "Polygon":
            new = max(new.geoms, key=lambda g: g.area)
        new = Polygon(new.exterior).simplify(0.03)
        print("  floor check: {} room(s) closed by walls were outside the floor -> floor added (+{:.1f} m2)".format(
            len(cells), new.area - floor.area))
        plan["floor"]["polygon"] = [[round(float(x), 4), round(float(y), 4)] for x, y in list(new.exterior.coords)[:-1]]
        floor = new


    # notches in the outline whose open side is closed by a wall are rooms too
    wl2 = unary_union([LineString([w["start"], w["end"]]).buffer(w["thickness"] / 2 + 0.5) for w in plan["walls"]])
    notch = floor.convex_hull.difference(floor.buffer(0.02))
    added = []
    for g in getattr(notch, "geoms", [notch]):
        if g.area < 1.0:
            continue
        mouth = g.exterior.difference(floor.buffer(0.1))
        if mouth.length > 0.3 and mouth.intersection(wl2).length > 0.7 * mouth.length:
            added.append(g.buffer(0.03, join_style=2))
    if added:
        new = unary_union([floor] + added)
        if new.geom_type != "Polygon":
            new = max(new.geoms, key=lambda g: g.area)
        new = Polygon(new.exterior).simplify(0.03)
        print("  floor check: {} notch(es) closed off by walls were left empty -> floor added (+{:.1f} m2)".format(
            len(added), new.area - floor.area))
        plan["floor"]["polygon"] = [[round(float(x), 4), round(float(y), 4)] for x, y in list(new.exterior.coords)[:-1]]
        floor = new


def open_ground(plan):
    """Steps that go down lead to an open space at ground level.
    For each group of down-steps: find the side that faces the nearest walls (that is the low side),
    sweep the steps that way until the walls stop them -> that space is left empty (no slab, no plinth),
    re-order the tiers so they fall from the floor towards it. Returns the slab openings (steps + space)."""
    from shapely.geometry import Polygon, LineString
    from shapely.ops import unary_union
    from shapely import affinity
    down = [st for st in plan["stairs"] if st.get("sunken")]
    if not down:
        return []
    walls = unary_union([Polygon(q).buffer(0) for cat, q, z0, z1 in pieces(plan) if cat == "wall"])
    floor = Polygon(plan["floor"]["polygon"]).buffer(0)
    free = floor.difference(walls.buffer(0.02))
    pl = float(plan["levels"].get("plinth", 0.0))
    holes = []
    for st in down:
        ol = Polygon(st["outline"]).buffer(0)
        a, b = np.array(st["start"]), np.array(st["end"])
        d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        c = np.array(ol.centroid.coords[0])
        best = None
        for sgn in (1, -1):                              # how far can the area sweep each way before a wall?
            dn = d * sgn
            reach = 6.0
            for t in np.arange(0.2, 6.01, 0.2):
                moved = affinity.translate(ol, *(dn * t))
                if moved.intersection(walls).area > 0.15 * ol.area or not floor.buffer(0.05).contains(moved.centroid):
                    reach = t
                    break
            if best is None or reach < best[0]:
                best = (reach, dn)
        reach, dn = best
        sweep = unary_union([affinity.translate(ol, *(dn * t)) for t in np.linspace(0, reach, 12)]).convex_hull
        cell = [g for g in getattr(free, "geoms", [free]) if g.intersects(ol)]
        space = sweep.intersection(unary_union(cell)) if cell else sweep
        st["open_ground"] = round(float(space.difference(ol).area), 2)
        st["direction"] = "goes DOWN from the raised floor ({:+.2f}) to the open ground ({:+.2f})".format(0.0, -pl)
        # tiers: highest next to the floor, lowest next to the open space
        trs = sorted(st["treads"], key=lambda tr: float(np.mean(np.array(tr["polygon"]) @ dn)))
        n = len(trs)
        for r, tr in enumerate(trs):
            tr["z0"] = round(-pl, 4) if pl else tr["z0"]
            tr["z1"] = round(-st["riser"] * (r + 1), 4) if r < n - 1 else round(-pl + st["riser"], 4) if pl else tr["z1"]
        st["treads"] = trs
        holes.append(unary_union([ol, space]).buffer(0))
    merged = unary_union(holes).buffer(0.01).buffer(-0.01)
    out = []
    for g in getattr(merged, "geoms", [merged]):
        g = g.simplify(0.02)
        if g.area > 0.3:
            out.append([[round(float(x), 4), round(float(y), 4)] for x, y in list(g.exterior.coords)[:-1]])
    print("  open ground left for {} down-going step areas: {:.1f} m2 (floor + plinth cut out)".format(
        len(down), sum(Polygon(o).area for o in out)))
    return out


def build_glb(plan, path, extra=None):
    import trimesh
    from shapely.geometry import Polygon
    parts = {}
    for cat, quad, z0, z1 in pieces(plan):
        vv, ff = prism(quad, z0, z1)
        parts.setdefault(cat, []).append(trimesh.Trimesh(vv, ff, process=False))
    fpoly = Polygon([(x, -y) for x, y in plan["floor"]["polygon"]]).buffer(0)
    for h in plan["floor"].get("holes", []):         # steps going down + the open ground they lead to
        fpoly = fpoly.difference(Polygon([(x, -y) for x, y in h]).buffer(0))
    fpieces = [fpoly] if fpoly.geom_type == "Polygon" else [q for q in getattr(fpoly, "geoms", []) if q.area > 0.05]
    fl = trimesh.util.concatenate([trimesh.creation.extrude_polygon(q, SLAB) for q in fpieces])
    fl.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    fl.apply_translation([0, -SLAB, 0])
    parts["floor"] = [fl]
    pl = float(plan.get("levels", {}).get("plinth", 0.0))
    if pl > 0.01:                       # building raised on a plinth, entrance steps come down to the ground
        fp = Polygon([(x, -y) for x, y in plan["floor"]["polygon"]]).buffer(0)
        for h in plan["floor"].get("holes", []):
            fp = fp.difference(Polygon([(x, -y) for x, y in h]).buffer(0))
        fp = max(getattr(fp, "geoms", [fp]), key=lambda g: g.area) if fp.geom_type != "Polygon" else fp
        if pl > SLAB + 0.01:
            pb = trimesh.creation.extrude_polygon(fp, pl - SLAB)
            pb.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
            pb.apply_translation([0, -pl, 0])
            parts["plinth"] = [pb]
        xs = [p[0] for p in plan["floor"]["polygon"]] + [q[0] for st in plan["stairs"] for q in st["outline"]]
        ys = [p[1] for p in plan["floor"]["polygon"]] + [q[1] for st in plan["stairs"] for q in st["outline"]]
        g = [[min(xs) - 3, min(ys) - 3], [max(xs) + 3, min(ys) - 3], [max(xs) + 3, max(ys) + 3], [min(xs) - 3, max(ys) + 3]]
        vv, ff = prism(g, -pl - 0.05, -pl)
        parts["ground"] = [trimesh.Trimesh(vv, ff, process=False)]
    scene = trimesh.Scene()
    for cat, ms in parts.items():
        m = trimesh.util.concatenate(ms)
        m.visual = trimesh.visual.ColorVisuals(m, face_colors=np.tile(COLORS[cat], (len(m.faces), 1)))
        scene.add_geometry(m, geom_name=cat)
    for cat, m in (extra or {}).items():
        scene.add_geometry(m.copy(), geom_name=cat)
    scene.export(path)
    return scene


# ----------------------------------------------------------------------------- 4b. faithful projection of the model's masks
def mask_polygons(mask, mx, my, eps_px=1.0):
    """Binary mask -> shapely polygons in metres (holes kept), via the OCR-solved px->m mapping."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    cs, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return None
    hier = hier[0]
    to_m = lambda c: [(float(mx(x)), float(my(y))) for x, y in cv2.approxPolyDP(c, eps_px, True)[:, 0]]
    polys = []
    for i, c in enumerate(cs):
        if hier[i][3] != -1 or len(c) < 3:
            continue
        holes, k = [], hier[i][2]
        while k != -1:
            if len(cs[k]) >= 3:
                h = to_m(cs[k])
                if len(h) >= 3:
                    holes.append(h)
            k = hier[k][0]
        shell = to_m(c)
        if len(shell) >= 3:
            p = Polygon(shell, holes).buffer(0)
            if not p.is_empty:
                polys.append(p)
    return unary_union(polys) if polys else None


def build_faithful(v, icon, mapping, path, extra=None):
    """Every wall/door/window pixel the model predicted, extruded to its height. Nothing is dropped."""
    import trimesh
    mx, my = make_mapper(mapping, "x"), make_mapper(mapping, "y")
    wall = v["model_wall"]
    T = v["T"]
    near = cv2.dilate(wall.astype(np.uint8), np.ones((int(2 * T),) * 2, np.uint8)) > 0
    lab, _ = ndimage.label(np.isin(icon, [WIN, DOOR]))
    keep = np.unique(lab[near & (lab > 0)])
    keep = keep[keep > 0]
    door = np.isin(lab, keep) & (icon == DOOR) & ~wall
    win = np.isin(lab, keep) & (icon == WIN) & ~wall
    stairs = (icon == STAIRS)
    layers = [("wall", wall, 0.0, WALL_HEIGHT),
              ("wall", door, DOOR_HEIGHT, WALL_HEIGHT), ("door", door, 0.0, DOOR_HEIGHT),
              ("wall", win, 0.0, WINDOW_SILL), ("glass", win, WINDOW_SILL, WINDOW_HEAD),
              ("wall", win, WINDOW_HEAD, WALL_HEIGHT), ("stairs", stairs, 0.0, 0.05)]
    parts = {}
    for cat, mask, z0, z1 in layers:
        if not mask.any() or z1 <= z0:
            continue
        geom = mask_polygons(mask, mx, my)
        if geom is None:
            continue
        for p in ([geom] if geom.geom_type == "Polygon" else list(geom.geoms)):
            if p.area < 1e-4:
                continue
            from shapely.geometry import Polygon
            flipped = Polygon([(x, -y) for x, y in p.exterior.coords],
                              [[(x, -y) for x, y in r.coords] for r in p.interiors])
            try:
                m = trimesh.creation.extrude_polygon(flipped, z1 - z0)
            except Exception:
                continue
            m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
            m.apply_translation([0, z0, 0])
            parts.setdefault(cat, []).append(m)
    scene = trimesh.Scene()
    for cat, ms in parts.items():
        m = trimesh.util.concatenate(ms)
        m.visual = trimesh.visual.ColorVisuals(m, face_colors=np.tile(COLORS[cat], (len(m.faces), 1)))
        scene.add_geometry(m, geom_name=cat)
    for cat, m in (extra or {}).items():
        scene.add_geometry(m.copy(), geom_name=cat)
    scene.export(path)
    return scene


# ----------------------------------------------------------------------------- 5. IFC with wall/door/window types
def build_ifc(plan, path):
    from ifcopenshell.api import root, unit, context, aggregate, spatial, geometry, feature, project, material
    from ifcopenshell.api import type as ifctype
    f = project.create_file(version="IFC4")
    proj = root.create_entity(f, ifc_class="IfcProject", name="Floorplan to Revit")
    unit.assign_unit(f, units=[unit.add_si_unit(f, unit_type="LENGTHUNIT"),
                               unit.add_si_unit(f, unit_type="AREAUNIT")])
    model = context.add_context(f, context_type="Model")
    body = context.add_context(f, context_type="Model", context_identifier="Body",
                               target_view="MODEL_VIEW", parent=model)
    axis_ctx = context.add_context(f, context_type="Plan", context_identifier="Axis",
                                   target_view="GRAPH_VIEW", parent=model)
    site = root.create_entity(f, ifc_class="IfcSite", name="Site")
    bldg = root.create_entity(f, ifc_class="IfcBuilding", name="Building")
    storey = root.create_entity(f, ifc_class="IfcBuildingStorey", name=plan["level"]["name"])
    storey.Elevation = 0.0
    aggregate.assign_object(f, products=[site], relating_object=proj)
    aggregate.assign_object(f, products=[bldg], relating_object=site)
    aggregate.assign_object(f, products=[storey], relating_object=bldg)
    for e in (site, bldg, storey):
        geometry.edit_object_placement(f, product=e, matrix=np.eye(4))
    masonry = material.add_material(f, name="Masonry", category="brick")

    def M(origin, d2, z=0.0):
        x_ax = np.array([d2[0], -d2[1], 0.0]); z_ax = np.array([0.0, 0.0, 1.0])
        m = np.eye(4)
        m[:3, 0], m[:3, 1], m[:3, 2] = x_ax, np.cross(z_ax, x_ax), z_ax
        m[:3, 3] = [origin[0], -origin[1], z]
        return m

    wall_types = {}

    def wall_type(t):
        key = int(round(t * 1000))
        if key not in wall_types:
            wt = root.create_entity(f, ifc_class="IfcWallType", name="Masonry {} mm".format(key),
                                    predefined_type="STANDARD")
            ls = material.add_material_set(f, name="Masonry {} mm".format(key), set_type="IfcMaterialLayerSet")
            layer = material.add_layer(f, layer_set=ls, material=masonry)
            material.edit_layer(f, layer=layer, attributes={"LayerThickness": t})
            material.assign_material(f, products=[wt], type="IfcMaterialLayerSet", material=ls)
            wall_types[key] = wt
        return wall_types[key]

    ifc_walls = {}
    for w in plan["walls"]:
        a, d, n, L = wall_frame(w)
        e = root.create_entity(f, ifc_class="IfcWall", name="Wall {}".format(w["id"]), predefined_type="STANDARD")
        ifctype.assign_type(f, related_objects=[e], relating_type=wall_type(w["thickness"]))
        start = a - d * w["ext"][0]
        length = L + w["ext"][0] + w["ext"][1]
        geometry.assign_representation(f, product=e, representation=geometry.add_wall_representation(
            f, context=body, length=length, height=w["height"], thickness=w["thickness"],
            offset=-w["thickness"] / 2))
        geometry.assign_representation(f, product=e, representation=geometry.add_axis_representation(
            f, context=axis_ctx, axis=[(0.0, 0.0), (length, 0.0)]))
        geometry.edit_object_placement(f, product=e, matrix=M(start, d))
        spatial.assign_container(f, products=[e], relating_structure=storey)
        ifc_walls[w["id"]] = (e, start, d, w)

    op_types = {}
    for o in plan["openings"]:
        e_wall, start, d, w = ifc_walls[o["wall"]]
        org = np.array(w["start"]) + d * (o["offset"] - o["width"] / 2)
        op = root.create_entity(f, ifc_class="IfcOpeningElement", name="Opening {}".format(o["id"]))
        geometry.assign_representation(f, product=op, representation=geometry.add_wall_representation(
            f, context=body, length=o["width"], height=o["height"], thickness=w["thickness"] + 0.2,
            offset=-(w["thickness"] + 0.2) / 2))
        geometry.edit_object_placement(f, product=op, matrix=M(org, d, o["sill"]))
        feature.add_feature(f, feature=op, element=e_wall)
        cls = "IfcDoor" if o["type"] == "door" else "IfcWindow"
        key = (cls, int(o["width"] * 1000), int(o["height"] * 1000))
        if key not in op_types:
            op_types[key] = root.create_entity(f, ifc_class=cls + "Type",
                                               name="{} {} x {}".format(cls[3:], key[1], key[2]))
        fill = root.create_entity(f, ifc_class=cls, name="{} {}".format(cls[3:], o["id"]))
        fill.OverallWidth, fill.OverallHeight = o["width"], o["height"]
        ifctype.assign_type(f, related_objects=[fill], relating_type=op_types[key])
        geometry.assign_representation(f, product=fill, representation=geometry.add_wall_representation(
            f, context=body, length=o["width"], height=o["height"], thickness=0.05, offset=-0.025))
        geometry.edit_object_placement(f, product=fill, matrix=M(org, d, o["sill"]))
        feature.add_filling(f, opening=op, element=fill)
        spatial.assign_container(f, products=[fill], relating_structure=storey)

    slab = root.create_entity(f, ifc_class="IfcSlab", name="Ground floor slab", predefined_type="FLOOR")
    geometry.assign_representation(f, product=slab, representation=geometry.add_slab_representation(
        f, context=body, depth=SLAB, polyline=[(x, -y) for x, y in plan["floor"]["polygon"]]))
    mm = np.eye(4); mm[2, 3] = -SLAB
    geometry.edit_object_placement(f, product=slab, matrix=mm)
    spatial.assign_container(f, products=[slab], relating_structure=storey)
    for r in plan["rooms"]:
        sp = root.create_entity(f, ifc_class="IfcSpace", name=r["name"])
        sp.LongName = "{} ({} m2)".format(r["name"], r["area_m2"])
        geometry.assign_representation(f, product=sp, representation=geometry.add_slab_representation(
            f, context=body, depth=WALL_HEIGHT, polyline=[(x, -y) for x, y in r["polygon"]]))
        geometry.edit_object_placement(f, product=sp, matrix=np.eye(4))
        aggregate.assign_object(f, products=[sp], relating_object=storey)
    ifc_cls = {"furniture": "IfcFurniture", "plumbing": "IfcSanitaryTerminal", "appliance": "IfcElectricAppliance"}
    for fu in plan.get("furniture", []):
        c = np.array(fu["corners"], float)
        e01, e12 = np.linalg.norm(c[1] - c[0]), np.linalg.norm(c[2] - c[1])
        d = (c[1] - c[0]) / e01 if e01 >= e12 else (c[2] - c[1]) / e12
        ctr = np.array(fu["center"])
        el = root.create_entity(f, ifc_class=ifc_cls.get(fu["category"], "IfcFurniture"),
                                name=fu["type"].replace("_", " ").title())
        geometry.assign_representation(f, product=el, representation=geometry.add_wall_representation(
            f, context=body, length=fu["width"], height=fu["height"], thickness=fu["depth"], offset=-fu["depth"] / 2))
        geometry.edit_object_placement(f, product=el, matrix=M(ctr - d * fu["width"] / 2, d))
        spatial.assign_container(f, products=[el], relating_structure=storey)
    for k, s in enumerate(plan["stairs"]):
        verts, faces, off = [], [], 0
        for _, quad, z0, z1 in pieces({"walls": [], "openings": [], "stairs": [s]}):
            vv, ff = prism(quad, z0, z1)
            verts += [[p[0], -p[2], p[1]] for p in vv]
            faces += [[int(i) + off for i in tri] for tri in ff]
            off += len(vv)
        st = root.create_entity(f, ifc_class="IfcStair", name="Stair {}".format(k))
        geometry.assign_representation(f, product=st, representation=geometry.add_mesh_representation(
            f, context=body, vertices=[verts], faces=[faces]))
        geometry.edit_object_placement(f, product=st, matrix=np.eye(4))
        spatial.assign_container(f, products=[st], relating_structure=storey)
    f.write(path)


# ----------------------------------------------------------------------------- 6. previews + report
def _render3d(ax3, scene, title):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    tris, cols = [], []
    for name, g in scene.geometry.items():
        if name == "ground" or name not in COLORS:
            continue                      # a flat sheet hides the walls in this simple renderer
        t = g.vertices[g.faces][:, :, [0, 2, 1]] * [1, -1, 1]
        tris.append(t); cols.append(np.tile(np.array(COLORS[name]) / 255.0, (len(t), 1)))
    tris, cols = np.vstack(tris), np.vstack(cols)
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    cols[:, :3] *= (0.55 + 0.45 * np.clip(nrm @ np.array([0.35, -0.45, 0.82]), 0, 1))[:, None]
    ax3.add_collection3d(Poly3DCollection(tris, facecolors=cols, edgecolors=cols, linewidths=0.3))
    lo, hi = tris.reshape(-1, 3).min(0), tris.reshape(-1, 3).max(0)
    ax3.set_xlim(lo[0], hi[0]); ax3.set_ylim(lo[1], hi[1]); ax3.set_zlim(lo[2], hi[2])
    ax3.set_box_aspect(tuple(hi - lo)); ax3.set_axis_off(); ax3.view_init(elev=40, azim=-60)
    ax3.set_title(title, fontsize=14)


def previews(img, v, plan, sol, scene, faithful, out_dir):
    fig = plt.figure(figsize=(30, 10))
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(img); ax.axis("off")
    ax.set_title("model walls kept {:.1f}%  (yellow = model wall pixels not in Revit walls)".format(
        100 * v["coverage"]), fontsize=13)
    mp = plan["mapping"]
    for axis in ("x", "y"):
        for px, m in zip(mp["axes"][axis]["px"], mp["axes"][axis]["m"]):
            if axis == "x":
                ax.axvline(px, color="m", lw=0.5, ls="--")
            else:
                ax.axhline(px, color="m", lw=0.5, ls="--")
    for sg in v["walls"]:
        p0, p1 = np.array(sg["p0"]), np.array(sg["p1"])
        d = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)
        n = np.array([-d[1], d[0]]) * sg["t"] / 2
        ax.add_patch(plt.Polygon([p0 - n, p1 - n, p1 + n, p0 + n], fc=(0.9, 0.1, 0.1, 0.45), ec="r", lw=0.6))
    miss = np.zeros(v["missed"].shape + (4,))
    miss[v["missed"]] = (1.0, 0.85, 0.0, 1.0)
    ax.imshow(miss)
    _render3d(fig.add_subplot(1, 3, 2, projection="3d"), faithful, "3D straight from the model (model_faithful.glb)")
    _render3d(fig.add_subplot(1, 3, 3, projection="3d"), scene, "Revit-ready walls (plan.json / model.ifc)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "preview.png"), dpi=80)
    plt.show()


def print_report(plan):
    pr = plan["precision"]
    print("\n=== PRECISION ({}) ===".format(pr["scale_source"]))
    for r in pr["report"]:
        if "written_m" in r:
            mark = "OK " if r["status"] == "ok" else ("-- " if r["status"].startswith("rejected") else "!! ")
            print("  {} {:<15} {}  {:<22} written {:>7.3f} m  model {:>7.3f} m  error {:+7.1f} mm{}".format(
                mark, r.get("kind", "dimension"), r["axis"], r["text"][:22], r["written_m"], r["model_m"],
                r["error_mm"], "  (rejected: conflicts)" if r["status"].startswith("rejected") else ""))
        else:
            print("  ?? {}: {} {}".format(r["axis"], r["text"], r.get("status", "")))
    print("  {}/{} written dimensions reproduced exactly".format(pr["dimensions_exact"], pr["dimensions_read"]))
    nd = sum(o["type"] == "door" for o in plan["openings"])
    nw = len(plan["openings"]) - nd
    xs = [c for w in plan["walls"] for c in (w["start"][0], w["end"][0])]
    ys = [c for w in plan["walls"] for c in (w["start"][1], w["end"][1])]
    print("\n=== MODEL ===\n  walls {} ({} external) | doors {} | windows {} | rooms {} | stairs {} | grids {}".format(
        len(plan["walls"]), sum(w["external"] for w in plan["walls"]), nd, nw, len(plan["rooms"]),
        len(plan["stairs"]), len(plan["grids"])))
    print("  building size {:.3f} x {:.3f} m (wall centre-lines)".format(max(xs) - min(xs), max(ys) - min(ys)))
    for r in plan["rooms"]:
        print("  room: {:<34} {:>6.2f} m2".format(r["name"], r["area_m2"]))
    print("  model wall pixels kept in Revit walls: {:.1f}%".format(100 * plan.get("coverage", 0)))
    if plan.get("furniture"):
        cnt = {}
        for fu in plan["furniture"]:
            cnt[fu["type"]] = cnt.get(fu["type"], 0) + 1
        print("  furniture/fixtures:", ", ".join("{} {}".format(v, k) for k, v in sorted(cnt.items())))
    lv = plan.get("levels", {})
    if plan["stairs"] or lv.get("plinth"):
        print("  plinth (floor above ground): {:.3f} m  [{}]".format(lv.get("plinth", 0), lv.get("plinth_source", "")))
    for st in plan["stairs"]:
        print("  {} stair {} ({}): {} risers x {:.0f} mm, tread {:.0f} mm, width {:.2f} m, run {:.2f} m, "
              "{:+.2f} -> {:+.2f} m".format(st["kind"], st["id"], st["shape"], st["risers"], st["riser"] * 1000,
                                           st["tread"] * 1000, st["width"], st["run"], st["base_z"], st["top_z"]))
        print("      from: " + "; ".join("{} {}".format(k, v) for k, v in st["sources"].items()))
        for wn in st["warnings"]:
            print("      CHECK: " + wn)
    th = sorted({w["thickness"] for w in plan["walls"]})
    print("  wall thicknesses (m):", th)


# ----------------------------------------------------------------------------- run
def ocr_full_res(img, original=None):
    """OCR on the full-resolution plan (room sizes / dimensions are tiny after the 1280 px resize),
    token positions mapped back to the inference image."""
    if original is None or original.shape[:2] == img.shape[:2]:
        return ocr_tokens(img)
    f = img.shape[1] / original.shape[1]
    big = original
    if max(big.shape[:2]) > 4000:                  # keep OCR fast on huge scans
        k = 4000 / max(big.shape[:2])
        big = cv2.resize(big, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        f = img.shape[1] / big.shape[1]
    toks = ocr_tokens(big)
    for t in toks:
        t["x"] *= f
        t["y"] *= f
    return toks


def dimension_line_scale(toks, original, img_w):
    """Each written dimension sits next to its dimension line: measure that line on the drawing, at any angle
    (21.34 along the top, 18.29 along a slanted side ...). The lengths vote for one scale.
    Returns (metres per inference px, number agreeing, details) or (None, 0, details)."""
    if original is None:
        return None, 0, []
    f = img_w / original.shape[1]                     # inference px per original px
    g = original if original.ndim == 2 else cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    ink = (g < 140).astype(np.uint8)
    thick = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))
    thin = (ink & (1 - cv2.dilate(thick, np.ones((3, 3), np.uint8)))) * 255   # dimension lines are thin
    segs = cv2.HoughLinesP(thin, 1, np.pi / 720, threshold=30, minLineLength=25, maxLineGap=4)
    if segs is None:
        return None, 0, []
    segs = segs.reshape(-1, 4).astype(float)
    nums = [t for t in toks if t["kind"] == "num"]
    if not nums:
        return None, 0, []
    unit = _unit_factor([t["value"] for t in nums])
    cands = []
    for t in nums:
        val = t["value"] * unit
        if not (0.3 <= val <= 200) or re.match(r"^\s*[.,]", t.get("raw", "")):
            continue                                       # '.29m' = a dimension OCR cut in half
        cx, cy = t["x"] / f, t["y"] / f
        h = max(t.get("h", 10.0) / f, 6.0)
        best = None
        for x0, y0, x1, y1 in segs:
            d = np.array([x1 - x0, y1 - y0]); L = np.linalg.norm(d)
            if L < 3 * h:
                continue
            d /= L
            nrm = np.array([-d[1], d[0]])
            rel = np.array([cx - x0, cy - y0])
            perp = abs(rel @ nrm)
            if perp > 2.5 * h:
                continue
            # the whole straight line through this piece: merge collinear pieces (text often breaks the line)
            ang = np.arctan2(d[1], d[0])
            same = []
            for q in segs:
                e = np.array([q[2] - q[0], q[3] - q[1]]); M = np.linalg.norm(e)
                if M < 1e-6:
                    continue
                if abs(np.sin(np.arctan2(e[1], e[0]) - ang)) > 0.02:
                    continue
                if abs(np.array([q[0] - x0, q[1] - y0]) @ nrm) > 2.5 or abs(np.array([q[2] - x0, q[3] - y0]) @ nrm) > 2.5:
                    continue
                same.append(q)
            iv = sorted(tuple(sorted((np.array([q[0] - x0, q[1] - y0]) @ d, np.array([q[2] - x0, q[3] - y0]) @ d)))
                        for q in same)
            # merge overlapping pieces; gaps up to the text size are the text breaking the line
            tc = rel @ d
            lo = hi = None
            runs = [list(iv[0])]
            for u0, u1 in iv[1:]:
                if u0 <= runs[-1][1] + 4 * h:
                    runs[-1][1] = max(runs[-1][1], u1)
                else:
                    runs.append([u0, u1])
            for r0, r1 in runs:
                if r0 - h <= tc <= r1 + h:
                    lo, hi = r0, r1
            if lo is None:
                continue
            # a chain (4.05 | 7.89 | 4.06) is one straight line: cut it at the ticks / extension lines crossing it
            # and keep the piece the number belongs to
            p0 = np.array([x0, y0])
            Hg, Wg = g.shape

            def ink_at(q):
                x_, y_ = int(round(q[0])), int(round(q[1]))
                return 0 <= x_ < Wg and 0 <= y_ < Hg and g[y_, x_] < 140

            def crossing(u):
                c_ = p0 + d * u
                return sum(ink_at(c_ + nrm * o) for o in (-7, -6, -5, 5, 6, 7)) >= 4
            cuts = [u for u in np.arange(lo + 3, hi - 3, 1.0) if crossing(u)]
            if cuts:
                groups_ = np.split(np.array(cuts), np.where(np.diff(cuts) > 3)[0] + 1)
                marks = [lo] + [float(gq.mean()) for gq in groups_] + [hi]
                for m0, m1 in zip(marks[:-1], marks[1:]):
                    if m0 <= tc <= m1:
                        lo, hi = m0, m1
                        break
            # filled arrowheads were removed with the thick strokes: walk on to the arrow tips
            for sgn in (-1, 1):
                end = hi if sgn > 0 else lo
                step = 0
                while step < 30 and any(ink_at(p0 + d * (end + sgn * (step + 1)) + nrm * o) for o in (-1, 0, 1)) \
                        and not crossing(end + sgn * (step + 1)):
                    step += 1
                if sgn > 0:
                    hi = end + step
                else:
                    lo = end - step
            length = hi - lo
            if length < 3 * h or not (lo + 0.15 * length <= tc <= hi - 0.15 * length):
                continue                                   # the number is written at the middle of its line
            if best is None or perp < best[0]:
                best = (perp, length)
        if best:
            cands.append((val / best[1] / f, t.get("raw", t["text"]), val, best[1]))
    if len(cands) < 3:
        return None, 0, cands
    sc = np.array([c[0] for c in cands])
    if len(cands) >= 3:
        med = np.median(sc)
        cands = [c for c in cands if 1 / 1.6 <= c[0] / med <= 1.6]   # drop misreads (wrong number for a line)
        sc = np.array([c[0] for c in cands])
    agree = [(np.abs(sc / x - 1) < 0.04).sum() for x in sc]
    i = int(np.argmax(agree))
    group = sc[np.abs(sc / sc[i] - 1) < 0.04]
    if len(group) < 3:
        return None, len(group), cands
    return float(np.median(group)), len(group), cands


def uniform_sol(v, s, source, report=()):
    """One scale for the whole sheet (no per-grid corrections)."""
    gx = sorted({sg["c"] for sg in v["walls"] if sg["o"] == "v"}) or [0.0, 1.0]
    gy = sorted({sg["c"] for sg in v["walls"] if sg["o"] == "h"}) or [0.0, 1.0]
    rep = [dict(r, status="rejected: conflicts with room sizes") if r.get("status") == "ok" else r for r in report]
    return {"scale_m_per_px": s, "source": source, "report": rep,
            "axes": {"x": {"px": gx, "m": [(g - gx[0]) * s for g in gx], "n_dims": 0},
                     "y": {"px": gy, "m": [(g - gy[0]) * s for g in gy], "n_dims": 0}}}


def room_mask_scale(room, icon, toks, s):
    """Written room sizes vs the floor-plan model's own room regions (clear area, any angle, no closed walls
    needed). Returns (factor, rooms used) or (None, n)."""
    block = np.isin(icon, (DOOR, WIN)) | np.isin(room, (0, 1, WALL))
    labs = {}
    for c in np.unique(room):
        if c in (0, 1, WALL):
            continue
        n, lab = cv2.connectedComponents(((room == c) & ~block).astype(np.uint8), connectivity=4)
        labs[c] = (lab, np.bincount(lab.ravel()))
    H, W = room.shape
    ks = []
    for t in toks:
        if t["kind"] != "size":
            continue
        x, y = int(np.clip(round(t["x"]), 0, W - 1)), int(np.clip(round(t["y"]), 0, H - 1))
        win = room[max(0, y - 6):y + 7, max(0, x - 6):x + 7].ravel()
        win = win[~np.isin(win, (0, 1, WALL))]
        if not win.size:
            continue
        c = int(np.bincount(win).argmax())
        lab, cnt = labs[c]
        ys, xs = np.nonzero(lab[max(0, y - 6):y + 7, max(0, x - 6):x + 7])
        if not len(ys):
            continue
        k_id = int(np.bincount(lab[max(0, y - 6):y + 7, max(0, x - 6):x + 7][ys, xs]).argmax())
        area_m2 = cnt[k_id] * s * s
        a, b = t["value"]
        if area_m2 > 0:
            ks.append(np.sqrt(a * b / area_m2))
    if len(ks) < 2:
        return None, len(ks)
    k = float(np.median(ks))
    agree = np.mean([abs(x_ / k - 1) < 0.12 for x_ in ks])
    return (k, int(sum(abs(x_ / k - 1) < 0.12 for x_ in ks))) if agree >= 0.5 else (None, len(ks))


def room_size_scale(plan, toks):
    """Written room sizes ('3.66 m x 4.88 m') vs the drawn room areas -> scale correction factor.
    Works for rooms at any angle. Returns (factor, rooms used) or (None, n) when they don't agree."""
    from shapely.geometry import Point, Polygon
    mx, my = make_mapper(plan["mapping"], "x"), make_mapper(plan["mapping"], "y")
    th = float(np.median([w["thickness"] for w in plan["walls"]])) if plan["walls"] else 0.2
    rooms = [(Polygon(r["polygon"]).buffer(0), r) for r in plan["rooms"]]
    ks = []
    for t in toks:
        if t["kind"] != "size":
            continue
        pt = Point(float(mx(t["x"])), float(my(t["y"])))
        inside = [(p.area, p) for p, r in rooms if p.area > 0 and p.contains(pt)]
        if not inside:
            continue
        area = min(inside, key=lambda q: q[0])[0]
        a, b = t["value"]
        ks.append(np.sqrt((a + th) * (b + th) / area))      # room polygons follow wall centre-lines
    if len(ks) < 2:
        return None, len(ks)
    k = float(np.median(ks))
    agree = np.mean([abs(x / k - 1) < 0.1 for x in ks])
    return (k, len(ks)) if agree >= 0.6 else (None, len(ks))


def main(img=None, room=None, icon=None, original=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    if room is None:
        assert IMAGE, "Set IMAGE at the top of the cell to a floor-plan URL or file path"
        img, room, icon = predict(IMAGE)
        original = LAST_ORIGINAL
    v = vectorise(room, icon, original if original is not None else img)
    toks = ocr_full_res(img, original)
    print("OCR: {} dimension numbers, {} room sizes, {} labels".format(
        sum(t["kind"] == "num" for t in toks), sum(t["kind"] == "size" for t in toks),
        sum(t["kind"] == "label" for t in toks)))
    gx = sorted({round(sg["c"], 2) for sg in v["walls"] if sg["o"] == "v"})
    gy = sorted({round(sg["c"], 2) for sg in v["walls"] if sg["o"] == "h"})
    ys_, xs_ = np.where(v["full"])
    sol = None if (METERS_PER_PIXEL or BUILDING_WIDTH_M) else solve(
        gx, gy, (xs_.min(), ys_.min(), xs_.max(), ys_.max()), toks, struct=v["full"], segs=v["walls"], T=v["T"])
    if sol is None and BUILDING_WIDTH_M:
        s = BUILDING_WIDTH_M / (gx[-1] - gx[0])
        sol = {"scale_m_per_px": s, "axes": {"x": {"px": gx, "m": [(g - gx[0]) * s for g in gx], "n_dims": 0},
                                             "y": {"px": gy, "m": [(g - gy[0]) * s for g in gy], "n_dims": 0}},
               "report": [], "fallback": True}
    plan = build_plan(v, sol, toks, room)
    # every dimension measured on its own dimension line (any angle): if they agree with each other but not
    # with the solved scale, the solver matched dimensions to the wrong walls -> use theirs
    sv, n_dl, dl = dimension_line_scale(toks, original, img.shape[1])
    dl_ok = sv is not None
    if dl_ok and abs(plan["mapping"]["scale_m_per_px"] / sv - 1) > 0.04 and not METERS_PER_PIXEL and not BUILDING_WIDTH_M:
        print("  scale check: {} dimension lines agree on a scale {:+.0f}% from the solved one -> corrected".format(
            n_dl, 100 * (sv / plan["mapping"]["scale_m_per_px"] - 1)))
        sol = uniform_sol(v, sv, "dimension lines measured on the drawing ({} agree)".format(n_dl),
                          (sol or {}).get("report", []))
        plan = build_plan(v, sol, toks, room)
    elif not dl_ok and dl:
        # the drawing is not to scale (its dimensions contradict each other). Keep the solved scale only if it
        # is believable; otherwise use the longest overall dimension, which fixes the building size
        sc = [c[0] for c in dl]
        spread = 100 * (max(sc) / min(sc) - 1)
        tm = float(np.median([sg["t"] for sg in v["walls"]])) * plan["mapping"]["scale_m_per_px"]
        n_solved = plan["precision"]["dimensions_exact"]
        print("  scale check: the plan's dimensions disagree by up to {:.0f}% - the drawing is not to scale".format(spread))
        if (tm < 0.08 or tm > 0.6 or n_solved < 3) and not METERS_PER_PIXEL and not BUILDING_WIDTH_M:
            longest = max(dl, key=lambda c: c[2])
            print("  scale check: using the longest dimension ({} m) for the building size".format(longest[2]))
            sol = uniform_sol(v, longest[0], "longest written dimension {} m (dimensions disagree by up to {:.0f}%)".format(
                longest[2], spread), (sol or {}).get("report", []))
            plan = build_plan(v, sol, toks, room)
        dl_ok = True                                       # do not let room sizes override this
    # written room sizes vs drawn room areas: if they disagree with the scale the dimensions gave (e.g. only a
    # couple of small dimensions were read and matched to the wrong walls), trust the rooms (two passes,
    # because wall-thickness snapping depends on the scale)
    for _ in range(0 if dl_ok else 2):
        k, n_rooms = room_mask_scale(room, icon, toks, plan["mapping"]["scale_m_per_px"])
        if k is None:
            k, n_rooms = room_size_scale(plan, toks)
        if k is None or abs(k - 1) <= 0.03 or METERS_PER_PIXEL or BUILDING_WIDTH_M:
            break
        s_new = plan["mapping"]["scale_m_per_px"] * k
        print("  scale check: {} room-size labels say the scale is off by {:+.0f}% -> corrected".format(
            n_rooms, 100 * (k - 1)))
        sol = uniform_sol(v, s_new, "room sizes written on the plan ({} rooms)".format(n_rooms),
                          (sol or {}).get("report", []))
        plan = build_plan(v, sol, toks, room)
    fill_floor_gaps(plan, v, room)
    plan["stairs"], plan["levels"] = build_stairs(v, toks, room, plan, original)
    plan["floor"]["holes"] = open_ground(plan)
    plan["coverage"] = round(v["coverage"], 4)
    if MAKE_FURNITURE:
        items = icon_fixtures(icon, v["T"]) + detector_furniture(img, plan["mapping"]["scale_m_per_px"])
        items, dropped = filter_by_room(items, room, v["T"])
        if dropped:
            print("  furniture: {} detections rejected by room check (wrong room for the object)".format(dropped))
        plan["furniture"] = build_furniture(items, plan)
        from shapely.geometry import Point as _Pt, Polygon as _Pg
        fp_ = _Pg(plan["floor"]["polygon"]).buffer(0.3)
        inside_ = [fu for fu in plan["furniture"] if fp_.contains(_Pt(*fu["center"]))]
        if len(inside_) < len(plan["furniture"]):
            print("  furniture: {} items outside the building dropped (title block / legend / notes)".format(
                len(plan["furniture"]) - len(inside_)))
        plan["furniture"] = inside_
        for k_, fu in enumerate(plan["furniture"]):
            fu["id"] = k_
    else:
        plan["furniture"] = []
    jp = os.path.join(OUT_DIR, "plan.json")
    json.dump(plan, open(jp, "w"), indent=1)
    fmesh = furniture_meshes(plan["furniture"]) if plan["furniture"] else {}
    scene = build_glb(plan, os.path.join(OUT_DIR, "model.glb"), extra=fmesh)
    faithful = build_faithful(v, icon, plan["mapping"], os.path.join(OUT_DIR, "model_faithful.glb"), extra=fmesh)
    try:
        build_ifc(plan, os.path.join(OUT_DIR, "model.ifc"))
    except Exception as e:
        print("IFC export failed (plan.json / GLB are fine):", type(e).__name__, e)
    print_report(plan)
    previews(img, v, plan, sol, scene, faithful, OUT_DIR)
    print("\nfiles in", OUT_DIR, ":", ", ".join(sorted(os.listdir(OUT_DIR))))
    print("Next: download plan.json -> run build_in_revit.py inside Revit (pyRevit / RevitPythonShell).")
    return plan


if not os.environ.get("P3D_NO_MAIN"):
    plan = main()
