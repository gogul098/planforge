"""FloorPlanCAD (real CAD drawings, Voxel51 FiftyOne mirror) -> floor-plan model cache entries.

FloorPlanCAD labels drawn strokes (a wall = its two parallel lines, a door = arc + leaf). CubiCasa5K - and the
model - work with areas: a wall is the filled band, a door / window is the opening rectangle in the wall. So:
  wall   strokes closed across the wall thickness            -> room head Wall
  window hull of the window strokes (they sit in the wall)    -> icon head Window
  door   the wall gap next to the door symbol                 -> icon head Door only with teach_doors=True
         (rebuilt from strokes it is ~75% right, and CubiCasa5K / ResPlan / synthetic plans give exact door
         labels, so by default the whole door area is unknown here)
  stairs, toilet, sink, bath, appliances: hull of the symbol  -> matching icon class
  other furniture / unlabelled ink / paper                    -> icon head NoIcon
The room head only learns walls here: everything else is NOT_WALL (no room types in FloorPlanCAD), a 1 px band
around each wall and the openings are unknown (IGNORE). No junction labels.
Licence: FloorPlanCAD is CC BY-NC 4.0 (non-commercial).
"""
import os

import cv2
import numpy as np

from fp_common import ICON_CLASSES, ROOM_CLASSES, IGNORE, NOT_WALL

I = {n: i for i, n in enumerate(ICON_CLASSES)}
WALL = ROOM_CLASSES.index("Wall")
RAILING = ROOM_CLASSES.index("Railing")

DOOR = {"single_door", "double_door", "sliding_door", "folding_door", "revolving_door", "rolling_door", "door"}
WINDOW = {"window", "bay_window", "blind_window", "opening_symbol"}
ICON_MAP = {"stairs": "Stairs", "toilet": "Toilet", "squat_toilet": "Toilet", "urinal": "Toilet", "sink": "Sink",
            "bath": "Bathtub", "bath_tub": "Bathtub", "refrigerator": "ElectricalAppliance",
            "washing_machine": "ElectricalAppliance", "gas_stove": "ElectricalAppliance"}
UNSURE = {"curtain_wall", "wardrobe", "high_cabinet", "elevator", "escalator"}   # neither taught nor denied


def norm(s):
    return str(s).strip().lower().replace(" ", "_").replace("-", "_")


def _full_mask(bbox, mask, W, H):
    """FiftyOne detection (normalised bbox + mask cropped to it) -> full-size boolean stroke mask."""
    bx, by, bw, bh = [float(v) for v in bbox]
    x0, y0 = int(round(bx * W)), int(round(by * H))
    w, h = max(1, int(round(bw * W))), max(1, int(round(bh * H)))
    out = np.zeros((H, W), bool)
    if mask is None:
        return out
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    x1, y1 = min(W, x0 + w), min(H, y0 + h)
    if x1 <= max(x0, 0) or y1 <= max(y0, 0):
        return out
    out[max(y0, 0):y1, max(x0, 0):x1] = m[max(0, -y0):y1 - y0, max(0, -x0):x1 - x0]
    return out


def _hull(m):
    ys, xs = np.nonzero(m)
    out = np.zeros(m.shape, np.uint8)
    if len(xs) >= 3:
        cv2.fillPoly(out, [cv2.convexHull(np.stack([xs, ys], 1).astype(np.int32))], 1)
    return out > 0


def convert(img_rgb, dets, ppm, teach_doors=False):
    """dets: [(label, bbox_norm, mask)], ppm: drawing pixels per metre. Returns (room, icon) label maps."""
    H, W = img_rgb.shape[:2]
    strokes = {}
    for label, bbox, mask in dets:
        strokes.setdefault(norm(label), []).append(_full_mask(bbox, mask, W, H))
    k = max(3, int(round(0.35 * ppm))) | 1                     # bridges a wall's two lines (walls < 35 cm)
    wall_st = np.zeros((H, W), bool)
    for m in strokes.get("wall", []):
        wall_st |= m
    wall = cv2.morphologyEx(wall_st.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)) > 0
    room = np.full((H, W), NOT_WALL, np.uint8)
    room[wall] = WALL
    for m in strokes.get("railing", []):
        room[cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = RAILING
    edge = cv2.morphologyEx(wall.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    room[edge] = IGNORE
    icon = np.full((H, W), I["NoIcon"], np.uint8)
    for lab, ms in strokes.items():
        for m in ms:
            hull = _hull(m)
            if lab in WINDOW:
                icon[hull] = I["Window"]
                room[hull] = IGNORE
            elif lab in DOOR:
                icon[hull] = IGNORE                            # swing area: unknown
                # the opening: wall gap bridged next to the symbol (closing wide enough for the door width)
                ys, xs = np.nonzero(hull)
                if len(xs) == 0:
                    continue
                pad = int(0.6 * ppm)
                x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, W)
                y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, H)
                loc = wall[y0:y1, x0:x1].astype(np.uint8)
                kd = int(1.5 * max(xs.max() - xs.min(), ys.max() - ys.min())) + 3   # > the opening width
                near = cv2.dilate(hull[y0:y1, x0:x1].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                (_, _), (_, _), ang = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
                gap = np.zeros(loc.shape, bool)
                for a in (ang, ang + 90):                      # the wall runs along one side of the swing square
                    ker = np.zeros((kd, kd), np.uint8)
                    c, dx, dy = kd // 2, np.cos(np.radians(a)) * kd / 2, np.sin(np.radians(a)) * kd / 2
                    cv2.line(ker, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1, 1)
                    filled = ((cv2.morphologyEx(loc, cv2.MORPH_CLOSE, ker) > 0) & (loc == 0)).astype(np.uint8)
                    n_g, cc, stt, _ = cv2.connectedComponentsWithStats(filled)
                    for j in range(1, n_g):
                        comp = cc == j
                        if not (comp & near).any():
                            continue
                        ys2, xs2 = np.nonzero(comp)
                        (_, _), (rw, rh), _ = cv2.minAreaRect(np.stack([xs2, ys2], 1).astype(np.float32))
                        if min(rw, rh) <= 1.6 * k and max(rw, rh) >= 0.4 * ppm:   # one wall thick, door wide
                            gap |= comp
                if gap.sum() >= 4:
                    icon[y0:y1, x0:x1][gap] = I["Door"] if teach_doors else IGNORE
                    room[y0:y1, x0:x1][gap] = IGNORE
            elif lab in ICON_MAP:
                icon[hull] = I[ICON_MAP[lab]]
            elif lab in UNSURE:
                icon[hull & (icon == I["NoIcon"])] = IGNORE
    return room, icon


def build_fpcad(cache_root, max_side=1536, val_fraction=0.08, seed=0, teach_doors=False, log=print):
    """Download the FiftyOne mirror (same one the furniture notebook uses) and write cache entries."""
    import random
    import subprocess
    import sys
    from fp_render import write_entry
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "fiftyone"], check=True)
    import fiftyone.utils.huggingface as fouh
    ds = fouh.load_from_hub("Voxel51/FloorPlanCAD")
    schema = ds.get_field_schema()
    field = [k for k, v in schema.items() if "Detections" in str(getattr(v, "document_type", ""))][0]
    data_dir = os.path.join(cache_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    rng = random.Random(seed)
    out = {"train": [], "val": []}
    for k, s in enumerate(ds.iter_samples()):
        img = cv2.imread(s.filepath, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 3 and img.shape[2] == 4:                # transparent PNG -> on white
            a = img[:, :, 3:4].astype(np.float32) / 255.0
            img = (img[:, :, :3].astype(np.float32) * a + 255.0 * (1 - a)).astype(np.uint8)
        rgb = cv2.cvtColor(img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
        dets = []
        for d in (s[field].detections if s[field] else []):
            m = d.mask
            if m is None and getattr(d, "mask_path", None):
                m = cv2.imread(d.mask_path, cv2.IMREAD_GRAYSCALE)
            dets.append((d.label, d.bounding_box, m))
        ppm = max(rgb.shape[:2]) / 20.0                        # FloorPlanCAD crops are 20 m x 20 m blocks
        room, icon = convert(rgb, dets, ppm, teach_doors)
        sid = "fpcad{:05d}".format(k)
        write_entry(data_dir, sid, rgb, room, icon, max_side)
        out["val" if rng.random() < val_fraction else "train"].append({"id": sid, "source": "floorplancad"})
        if (k + 1) % 500 == 0:
            log("  floorplancad converted {}".format(k + 1))
    return out
