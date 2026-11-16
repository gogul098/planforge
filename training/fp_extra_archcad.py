"""ArchCAD-400K (40K public samples, real CAD drawings of mostly public / commercial buildings) -> floor-plan model
cache entries, and -> furniture / door-type detector samples.

Each sample is an SVG of a 14 m x 14 m slice of a drawing. Every drawn element carries its class:
    <path ... semantic="single_door" instance="single_door_23"/>
(the numeric id table of the dataset card is accepted too: semantic="1" / semantic-id="1"). The SVG is rendered
here (black strokes on white) so image and labels line up exactly; then the same stroke -> area rules as
FloorPlanCAD (fp_extra_fpcad.convert_strokes) give the label maps:
    wall                       -> room head Wall
    concrete / steel column    -> filled outline = Wall (the converter projects columns as wall blocks)
    glass                      -> closed along its line = Window (ArchCAD has no separate window class)
    doors (4 kinds)            -> wall gap next to the symbol = Door only with teach_doors (else unknown)
    staircase, toilets, sinks, bathtub -> matching icon class
    axis & grid lines, beams, text, furniture, everything else -> known NOT wall (this is the lesson the
                                  model is missing: drawing notations are not walls)
Slices of structural drawings (foundation, pile, rebar) are skipped. No room types (room head = NOT_WALL).
Licence: cc-by-nc-4.0, gated on Hugging Face (accept the access agreement once with the account whose token
the notebook uses). Non-commercial use only.
"""
import os
import re
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from fp_common import IGNORE
from fp_extra_fpcad import Crop, convert_strokes

HF_REPO = "jackluoluo/ArchCAD"
SLICE_M = 14.0
# the dataset card's id table
ID_NAMES = {0: "axis_grid", 1: "single_door", 2: "double_door", 3: "parent_child_door", 4: "other_door",
            5: "elevator", 6: "staircase", 7: "sink", 8: "urinal", 9: "toilet", 10: "bathtub", 11: "squat_toilet",
            12: "other_fixtures", 13: "drain", 14: "table", 15: "chair", 16: "bed", 17: "sofa", 18: "hole",
            19: "glass", 20: "wall", 21: "concrete_column", 22: "steel_column", 23: "concrete_beam",
            24: "steel_beam", 25: "parking_space", 26: "foundation", 27: "pile", 28: "rebar", 29: "fire_hydrant",
            100: "others"}
# ArchCAD name -> name the FloorPlanCAD rules know
TO_FPCAD = {"staircase": "stairs", "stair": "stairs", "bathtub": "bath_tub", "parent_child_door": "double_door",
            "other_door": "door", "concrete_column": "column", "steel_column": "column"}
SOLID = ("column",)
GLASS = ("glass",)
UNSURE = {"elevator", "hole", "other_fixtures", "drain", "fire_hydrant"}
STRUCTURAL = {"foundation", "pile", "rebar"}
# detector (furniture + door types) labels from ArchCAD
DET_MAP = {"single_door": "single_door", "double_door": "double_door", "parent_child_door": "parent_child_door",
           "table": "table", "chair": "chair", "bed": "bed", "sofa": "sofa", "sink": "sink", "toilet": "toilet",
           "squat_toilet": "squat_toilet", "urinal": "urinal", "bathtub": "bath_tub"}


def norm(s):
    s = str(s).strip().lower()
    if re.fullmatch(r"\d+", s):
        return ID_NAMES.get(int(s), "others")
    s = re.sub(r"[\s\-&]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return {"axis_and_grid": "axis_grid", "axis": "axis_grid", "grid": "axis_grid",
            "parent_child": "parent_child_door", "mother_child_door": "parent_child_door"}.get(s, s)


# ----------------------------------------------------------------------------------------- SVG -> primitives
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def _floats(s):
    return [float(v) for v in re.findall(_NUM, s or "")]


def _mat(tr):
    """SVG transform attribute -> 3x3 matrix."""
    M = np.eye(3)
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", tr or ""):
        a = _floats(args)
        T = np.eye(3)
        if name == "matrix" and len(a) == 6:
            T = np.array([[a[0], a[2], a[4]], [a[1], a[3], a[5]], [0, 0, 1]])
        elif name == "translate":
            T[0, 2], T[1, 2] = a[0], (a[1] if len(a) > 1 else 0.0)
        elif name == "scale":
            T[0, 0], T[1, 1] = a[0], (a[1] if len(a) > 1 else a[0])
        elif name == "rotate":
            c, s_ = np.cos(np.radians(a[0])), np.sin(np.radians(a[0]))
            R = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
            if len(a) == 3:
                T1, T2 = np.eye(3), np.eye(3)
                T1[:2, 2], T2[:2, 2] = a[1:3], [-a[1], -a[2]]
                R = T1 @ R @ T2
            T = R
        M = M @ T
    return M


def _arc(p0, rx, ry, phi, large, sweep, p1, n=16):
    """SVG elliptical arc -> points (endpoint parameterisation, SVG spec F.6.5)."""
    if rx == 0 or ry == 0:
        return [p1]
    phi = np.radians(phi)
    c, s_ = np.cos(phi), np.sin(phi)
    dx, dy = (p0[0] - p1[0]) / 2, (p0[1] - p1[1]) / 2
    x1, y1 = c * dx + s_ * dy, -s_ * dx + c * dy
    rx, ry = abs(rx), abs(ry)
    lam = x1 ** 2 / rx ** 2 + y1 ** 2 / ry ** 2
    if lam > 1:
        rx, ry = rx * np.sqrt(lam), ry * np.sqrt(lam)
    num = rx ** 2 * ry ** 2 - rx ** 2 * y1 ** 2 - ry ** 2 * x1 ** 2
    co = np.sqrt(max(num, 0) / max(rx ** 2 * y1 ** 2 + ry ** 2 * x1 ** 2, 1e-12))
    if large == sweep:
        co = -co
    cx1, cy1 = co * rx * y1 / ry, -co * ry * x1 / rx
    cx, cy = c * cx1 - s_ * cy1 + (p0[0] + p1[0]) / 2, s_ * cx1 + c * cy1 + (p0[1] + p1[1]) / 2
    ang = lambda ux, uy, vx, vy: np.arctan2(ux * vy - uy * vx, ux * vx + uy * vy)
    t1 = ang(1, 0, (x1 - cx1) / rx, (y1 - cy1) / ry)
    dt = ang((x1 - cx1) / rx, (y1 - cy1) / ry, (-x1 - cx1) / rx, (-y1 - cy1) / ry)
    if not sweep and dt > 0:
        dt -= 2 * np.pi
    elif sweep and dt < 0:
        dt += 2 * np.pi
    ts = t1 + dt * np.linspace(0, 1, n + 1)[1:]
    return [(c * rx * np.cos(t) - s_ * ry * np.sin(t) + cx, s_ * rx * np.cos(t) + c * ry * np.sin(t) + cy) for t in ts]


def _path(d):
    """SVG path data -> list of polylines (subpaths), each [(x, y), ...]. Curves are sampled."""
    toks = re.findall(r"[MmLlHhVvCcSsQqTtAaZz]|" + _NUM, d or "")
    subs, cur, pos, start, cmd, i = [], [], (0.0, 0.0), (0.0, 0.0), None, 0
    last_ctrl = None

    def take(n):
        nonlocal i
        v = [float(t) for t in toks[i:i + n]]
        i += n
        return v
    while i < len(toks):
        if re.fullmatch(r"[A-Za-z]", toks[i]):
            cmd = toks[i]
            i += 1
            if cmd in "Zz":
                if cur:
                    cur.append(start)
                    subs.append(cur)
                cur, pos = [], start
                continue
        if cmd is None:
            break
        rel = cmd.islower()
        C = cmd.upper()
        ox, oy = pos if rel else (0.0, 0.0)
        try:
            if C == "M":
                x, y = take(2)
                pos = (x + ox, y + oy)
                if cur:
                    subs.append(cur)
                cur, start = [pos], pos
                cmd = "l" if rel else "L"
            elif C == "L":
                x, y = take(2)
                pos = (x + ox, y + oy)
                cur.append(pos)
            elif C == "H":
                (x,) = take(1)
                pos = (x + ox, pos[1])
                cur.append(pos)
            elif C == "V":
                (y,) = take(1)
                pos = (pos[0], y + oy)
                cur.append(pos)
            elif C in "CS":
                if C == "C":
                    x1, y1, x2, y2, x, y = take(6)
                    c1 = (x1 + ox, y1 + oy)
                else:
                    x2, y2, x, y = take(4)
                    c1 = (2 * pos[0] - last_ctrl[0], 2 * pos[1] - last_ctrl[1]) if last_ctrl else pos
                c2, p1 = (x2 + ox, y2 + oy), (x + ox, y + oy)
                p0 = pos
                for t in np.linspace(0, 1, 9)[1:]:
                    u = 1 - t
                    cur.append((u ** 3 * p0[0] + 3 * u * u * t * c1[0] + 3 * u * t * t * c2[0] + t ** 3 * p1[0],
                                u ** 3 * p0[1] + 3 * u * u * t * c1[1] + 3 * u * t * t * c2[1] + t ** 3 * p1[1]))
                pos, last_ctrl = p1, c2
                continue
            elif C in "QT":
                if C == "Q":
                    x1, y1, x, y = take(4)
                    c1 = (x1 + ox, y1 + oy)
                else:
                    x, y = take(2)
                    c1 = (2 * pos[0] - last_ctrl[0], 2 * pos[1] - last_ctrl[1]) if last_ctrl else pos
                p0, p1 = pos, (x + ox, y + oy)
                for t in np.linspace(0, 1, 9)[1:]:
                    u = 1 - t
                    cur.append((u * u * p0[0] + 2 * u * t * c1[0] + t * t * p1[0],
                                u * u * p0[1] + 2 * u * t * c1[1] + t * t * p1[1]))
                pos, last_ctrl = p1, c1
                continue
            elif C == "A":
                rx, ry, phi, large, sweep, x, y = take(7)
                p1 = (x + ox, y + oy)
                cur += _arc(pos, rx, ry, phi, int(large), int(sweep), p1)
                pos = p1
            else:
                i += 1
        except ValueError:
            break
        last_ctrl = None
    if cur:
        subs.append(cur)
    return [s_ for s_ in subs if len(s_) >= 1]


def _shape_polys(el, tag):
    """One SVG element -> (polylines, closed?, filled?) in its own user units."""
    g = lambda k, d=0.0: float(el.get(k, d) or d)
    if tag == "path":
        return _path(el.get("d")), False
    if tag == "line":
        return [[(g("x1"), g("y1")), (g("x2"), g("y2"))]], False
    if tag in ("polyline", "polygon"):
        v = _floats(el.get("points"))
        pts = list(zip(v[0::2], v[1::2]))
        return ([pts + pts[:1]] if tag == "polygon" and pts else [pts]), tag == "polygon"
    if tag == "rect":
        x, y, w, h = g("x"), g("y"), g("width"), g("height")
        return [[(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]], True
    if tag in ("circle", "ellipse"):
        cx, cy = g("cx"), g("cy")
        rx = g("r") if tag == "circle" else g("rx")
        ry = g("r") if tag == "circle" else g("ry")
        t = np.linspace(0, 2 * np.pi, 33)
        return [list(zip(cx + rx * np.cos(t), cy + ry * np.sin(t)))], True
    return [], False


def parse_svg(path_or_text):
    """-> (elements, (minx, miny, w, h)) with elements = [(label, instance, [polyline xy arrays], is_text)].
    Labels are normalised names (norm). Coordinates are in the root user space (transforms applied)."""
    txt = open(path_or_text, encoding="utf-8", errors="ignore").read() if os.path.exists(str(path_or_text)) \
        else str(path_or_text)
    root = ET.fromstring(txt)
    vb = _floats(root.get("viewBox"))
    out = []

    def attr(el, *names):
        for n in names:
            for k, v in el.attrib.items():
                if k.split("}")[-1] == n:
                    return v
        return None

    def walk(el, M, inherited):
        tag = el.tag.split("}")[-1]
        M = M @ _mat(el.get("transform"))
        sem = attr(el, "semantic", "semantic-id", "semanticId", "semantic_id") or inherited[0]
        ins = attr(el, "instance", "instance-id", "instanceId", "instance_id") or inherited[1]
        if tag in ("path", "line", "polyline", "polygon", "rect", "circle", "ellipse"):
            polys, _closed = _shape_polys(el, tag)
            arrs = []
            for p in polys:
                if not p:
                    continue
                a = np.asarray(p, float)
                a = (M @ np.c_[a, np.ones(len(a))].T).T[:, :2]
                arrs.append(a)
            if arrs:
                out.append((norm(sem) if sem is not None else "others", ins, arrs, False))
        elif tag == "text":
            x, y = (_floats(el.get("x")) or [0.0])[0], (_floats(el.get("y")) or [0.0])[0]
            fs = el.get("font-size") or (re.findall(r"font-size:\s*([^;]+)", el.get("style", "") or "") or [""])[0]
            size = (_floats(fs) or [10.0])[0]
            s = "".join(el.itertext()).strip()
            if s:
                p = (M @ np.array([x, y, 1.0]))[:2]
                h = size * np.sqrt(abs(np.linalg.det(M[:2, :2])) or 1.0)
                out.append((norm(sem) if sem is not None else "others", ins, [np.array([p, [p[0] + h * 0.6 * len(s), p[1]]])],
                            (s, h)))
        for ch in el:
            walk(ch, M, (sem, ins))

    walk(root, np.eye(3), (None, None))
    if len(vb) == 4:
        box = tuple(vb)
    else:
        allp = np.concatenate([a for e in out for a in e[2]]) if out else np.zeros((1, 2))
        mn, mx = allp.min(0), allp.max(0)
        box = (mn[0], mn[1], max(mx[0] - mn[0], 1e-6), max(mx[1] - mn[1], 1e-6))
    return out, box


def render(elements, box, size_px, line_px=1, rng=None):
    """Draw the slice: black strokes on white, size_px on the longer side. Returns
    (rgb, {label: [mask per instance]}, {label: [point array per instance]} (for detector boxes), ppm)."""
    minx, miny, w, h = box
    s = size_px / max(w, h)
    W, H = max(8, int(round(w * s))), max(8, int(round(h * s)))
    img = np.full((H, W), 255, np.uint8)
    # one full mask per class for elements without an instance (walls, glass, grid ...); countable objects
    # are drawn on a mask of their own and kept as a bounding-box crop (thousands of full masks = no RAM)
    inst_masks, inst_pts = {}, {}
    for lab, ins, arrs, text in elements:
        key = (lab, ins)
        m = inst_masks.get(key)
        if m is None:
            m = inst_masks[key] = np.zeros((H, W), np.uint8)
        elif isinstance(m, Crop):                          # more strokes of an instance cropped early
            m = inst_masks[key] = m.full(H, W).astype(np.uint8)
        pts_all = []
        for a in arrs:
            q = np.round((a - [minx, miny]) * s).astype(np.int32)
            pts_all.append(q)
            if text:
                (st, hh) = text
                hp = max(6, int(round(hh * s)))
                scale = hp / 22.0
                cv2.putText(img, st, tuple(int(v) for v in q[0]), cv2.FONT_HERSHEY_SIMPLEX, scale, 0,
                            max(1, int(round(scale * 1.5))), cv2.LINE_AA)
                cv2.putText(m, st, tuple(int(v) for v in q[0]), cv2.FONT_HERSHEY_SIMPLEX, scale, 1,
                            max(1, int(round(scale * 1.5))), cv2.LINE_8)
                continue
            if len(q) == 1:
                q = np.vstack([q, q])
            cv2.polylines(img, [q], False, 0, line_px, cv2.LINE_AA)
            cv2.polylines(m, [q], False, 1, max(1, line_px), cv2.LINE_8)
        inst_pts.setdefault(key, []).extend(pts_all)
        if ins is not None and len(inst_masks) > 64:      # keep RAM bounded: crop finished instances early
            _crop_done(inst_masks)
    _crop_done(inst_masks, final=True)
    strokes, points = {}, {}
    for (lab, _), m in inst_masks.items():
        strokes.setdefault(lab, []).append(m if isinstance(m, Crop) else m > 0)
    for (lab, _), pts in inst_pts.items():
        points.setdefault(lab, []).append(np.concatenate(pts) if pts else np.zeros((0, 2)))
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return rgb, strokes, points, size_px / SLICE_M


def _crop_done(inst_masks, final=False):
    """Replace full-size instance masks (keys with an instance id) by their bounding-box crops."""
    for key, m in list(inst_masks.items()):
        if key[1] is None or isinstance(m, Crop):
            continue
        ys, xs = np.nonzero(m)
        if not len(ys):
            inst_masks[key] = Crop(0, 0, np.zeros((1, 1), bool))
            continue
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        inst_masks[key] = Crop(y0, x0, m[y0:y1, x0:x1] > 0)


def convert_sample(svg, size_px=1400, teach_doors=False, line_px=1):
    """One ArchCAD SVG -> (rgb, room, icon) or None (structural slice / nothing useful in it)."""
    elements, box = parse_svg(svg)
    labs = {e[0] for e in elements}
    if labs & STRUCTURAL or not (labs - {"others", "axis_grid"}):
        return None
    rgb, strokes, _, ppm = render(elements, box, size_px, line_px)
    H, W = rgb.shape[:2]
    fp = {}
    for lab, ms in strokes.items():
        fp.setdefault(TO_FPCAD.get(lab, lab), []).extend(ms)
    room, icon = convert_strokes(fp, H, W, ppm, teach_doors, solid=SOLID, glass=GLASS)
    for lab in UNSURE:                                        # neither taught nor denied
        for m in strokes.get(lab, []):
            ys, xs = np.nonzero(m.full(H, W) if isinstance(m, Crop) else m)
            if len(xs) >= 3:
                hull = np.zeros((H, W), np.uint8)
                cv2.fillPoly(hull, [cv2.convexHull(np.stack([xs, ys], 1).astype(np.int32))], 1)
                icon[(hull > 0) & (icon == 0)] = IGNORE
    return rgb, room, icon


def detector_boxes(svg, size_px):
    """One ArchCAD SVG -> (rgb, [(detector label, 4x2 corners px)]) for the furniture / door-type detector."""
    elements, box = parse_svg(svg)
    if {e[0] for e in elements} & STRUCTURAL:
        return None
    rgb, _, points, _ = render(elements, box, size_px)
    out = []
    for lab, plist in points.items():
        if lab not in DET_MAP:
            continue
        for pts in plist:
            if len(pts) < 3:
                continue
            rect = cv2.minAreaRect(pts.astype(np.float32))
            if min(rect[1]) < 2:
                continue
            out.append((DET_MAP[lab], cv2.boxPoints(rect)))
    return rgb, out


def find_svgs(root):
    out = []
    for dp, _, fs in os.walk(root):
        out += [os.path.join(dp, f) for f in fs if f.lower().endswith(".svg")]
    return sorted(out)


def download(work_dir, token, log=print):
    """Gated HF dataset -> local folder with the SVGs (archives are unpacked)."""
    import tarfile
    import zipfile
    from huggingface_hub import snapshot_download
    d = os.path.join(work_dir, "ArchCAD")
    try:
        # the repo holds data/{svg,png,json,point,caption}.zip (2.6 GB); only the SVGs (311 MB) are needed
        snapshot_download(HF_REPO, repo_type="dataset", token=token, local_dir=d,
                          allow_patterns=["data/svg.zip", "*.svg", "README.md"])
    except Exception as e:
        raise RuntimeError("ArchCAD download failed ({}: {}). It is a gated dataset: open "
                           "https://huggingface.co/datasets/{} with the account of this HF token and accept the "
                           "access agreement (approval can take up to 3 business days).".format(type(e).__name__, e, HF_REPO))
    if not find_svgs(d):
        for dp, _, fs in os.walk(d):
            for f in fs:
                p = os.path.join(dp, f)
                if f.endswith(".zip"):
                    with zipfile.ZipFile(p) as z:
                        z.extractall(dp)
                elif f.endswith((".tar", ".tar.gz", ".tgz")):
                    with tarfile.open(p) as t:
                        t.extractall(dp)
    svgs = find_svgs(d)
    if not svgs:
        listing = [os.path.relpath(os.path.join(dp, f), d) for dp, _, fs in os.walk(d) for f in fs][:40]
        raise RuntimeError("ArchCAD: no SVG files found after download. Files seen: {}".format(listing))
    log("  ArchCAD: {} SVG slices".format(len(svgs)))
    return svgs


def probe(svgs, n=5, log=print):
    """Print what the first files contain (labels found), so a format change shows up at once."""
    import collections
    c = collections.Counter()
    for p in svgs[:n]:
        els, box = parse_svg(p)
        c.update(e[0] for e in els)
    log("  ArchCAD probe ({} files): {}".format(min(n, len(svgs)), dict(c.most_common())))
    if not (set(c) & {"wall", "single_door", "staircase", "glass"}):
        raise RuntimeError("ArchCAD: no known class names in the SVGs - the label attribute format changed: "
                           "{}".format(dict(c.most_common(15))))


def _convert_job(job):
    """Worker: one slice -> cache entry. Returns (k, written?)."""
    from fp_render import write_entry
    k, p, data_dir, max_side, teach_doors, line_px = job
    try:
        got = convert_sample(p, size_px=min(max_side, 1400), teach_doors=teach_doors, line_px=line_px)
    except Exception:
        got = None
    if got is None:
        return k, False
    rgb, room, icon = got
    write_entry(data_dir, "archcad{:06d}".format(k), rgb, room, icon, max_side)
    return k, True


def build_archcad(cache_root, work_dir, token, max_side=1536, limit=None, val_fraction=0.05, seed=0,
                  teach_doors=False, log=print):
    """Download, convert and write cache entries. Returns {"train": [...], "val": [...]}."""
    import random
    from fp_render import write_entry
    svgs = download(work_dir, token, log)
    probe(svgs, log=log)
    rng = random.Random(seed)
    rng.shuffle(svgs)
    if limit:
        svgs = svgs[:limit]
    data_dir = os.path.join(cache_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    out = {"train": [], "val": []}
    skipped = 0
    jobs = [(k, p, data_dir, max_side, teach_doors, 1 + int(rng.random() < 0.3)) for k, p in enumerate(svgs)]
    from multiprocessing import Pool
    with Pool(max(1, (os.cpu_count() or 2))) as pool:             # ~0.16 s per slice per core
        for n, (k, ok) in enumerate(pool.imap_unordered(_convert_job, jobs, chunksize=16)):
            if not ok:
                skipped += 1
            else:
                split = "val" if random.Random(seed * 1000003 + k).random() < val_fraction else "train"
                out[split].append({"id": "archcad{:06d}".format(k), "source": "archcad"})
            if (n + 1) % 2000 == 0:
                log("  archcad converted {} / {} (skipped {})".format(n + 1, len(svgs), skipped))
    log("  archcad: {} train, {} val, {} skipped (structural / empty / unreadable)".format(
        len(out["train"]), len(out["val"]), skipped))
    return out
