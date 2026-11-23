"""Batch: every Structured3D house -> (plan drawing PNG, 3D truth JSON, label masks NPZ).
Drawing at a fixed 50 px per metre with randomised drafting style; masks use PlanForge's room / icon classes."""
import os, sys, json, random, traceback
import numpy as np
import cv2
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3d_convert as conv2

ROOT = os.environ.get("S3D_ROOT", "/kaggle/input/structured3d/Structured3D")
OUT = os.environ.get("S3D_OUT", "/kaggle/working/s3d_pairs")
PPM = 50.0                                     # pixels per metre
PAD = 1.0                                      # m of paper around the building
ROOMS = ["Background", "Outdoor", "Wall", "Kitchen", "LivingRoom", "Bedroom", "Bath", "Entry/Hallway",
         "Railing", "Storage", "Garage", "OtherRoom"]
ICON_WIN, ICON_DOOR = 1, 2
TYPE_TO_ROOM = {"kitchen": "Kitchen", "living room": "LivingRoom", "dining room": "LivingRoom",
                "bedroom": "Bedroom", "study": "Bedroom", "studio": "LivingRoom", "office": "OtherRoom",
                "bathroom": "Bath", "laundry room": "Storage", "store room": "Storage", "corridor": "Entry/Hallway",
                "balcony": "Outdoor", "garden": "Outdoor", "undefined": "OtherRoom"}


def render(res, seed):
    rng = random.Random(seed)
    x0, y0, x1, y1 = res["outline"].bounds
    W = int((x1 - x0) / 1000 * PPM + 2 * PAD * PPM)
    H = int((y1 - y0) / 1000 * PPM + 2 * PAD * PPM)
    if W * H > 4000 * 4000:
        raise ValueError("too large")

    def px(pts):                                  # mm -> px (y down)
        p = np.asarray(pts, float)
        return np.stack([(p[:, 0] - x0) / 1000 * PPM + PAD * PPM, (y1 - p[:, 1]) / 1000 * PPM + PAD * PPM], 1)

    paper = rng.choice([255, 250, 246])
    img = np.full((H, W, 3), paper, np.uint8)
    room = np.zeros((H, W), np.uint8)
    icon = np.zeros((H, W), np.uint8)
    lw = rng.choice([1, 1, 2])
    ink = (rng.randint(0, 40),) * 3
    style = rng.choice(["solid", "solid", "grey", "hatch", "outline"])

    def polys(g):
        return [p for p in getattr(g, "geoms", [g]) if p.area > 100]

    # rooms (masks)
    for r in res["rooms"]:
        cls = ROOMS.index(TYPE_TO_ROOM.get(r["type"], "OtherRoom"))
        for p in polys(r["poly"]):
            cv2.fillPoly(room, [np.round(px(p.exterior.coords)).astype(np.int32)], cls)
    # walls
    wall_px = np.zeros((H, W), np.uint8)
    for p in polys(res["walls"]):
        ext = np.round(px(p.exterior.coords)).astype(np.int32)
        holes = [np.round(px(h.coords)).astype(np.int32) for h in p.interiors]
        cv2.fillPoly(wall_px, [ext] + holes, 1)
    room[wall_px > 0] = ROOMS.index("Wall")
    if style in ("solid", "grey"):
        img[wall_px > 0] = ink if style == "solid" else (120, 120, 120)
    elif style == "hatch":
        yy, xx = np.mgrid[0:H, 0:W]
        hatch = ((xx + yy) % 8 < 2)
        img[(wall_px > 0) & hatch] = ink
    edges = cv2.morphologyEx(wall_px, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    if style != "solid":
        img[edges > 0] = ink
    # openings
    for o in res["openings"]:
        a, b = np.asarray(o["a"]), np.asarray(o["b"])
        w = np.linalg.norm(b - a)
        if w < 1:
            continue
        u = (b - a) / w
        n = np.array([-u[1], u[0]])
        h = o["depth"] / 2 + 20
        quad = np.round(px([a - n * h, b - n * h, b + n * h, a + n * h])).astype(np.int32)
        cv2.fillPoly(img, [quad], (paper,) * 3)
        cls = ICON_DOOR if o["type"] == "door" else ICON_WIN
        cv2.fillPoly(icon, [quad], cls)
        room[cv2.fillPoly(np.zeros_like(room), [quad], 1) > 0] = ROOMS.index("Wall")
        if o["type"] in ("window", "internal_window"):
            for k in (-0.6, 0, 0.6):
                q = np.round(px([a + n * h * k, b + n * h * k])).astype(np.int32)
                cv2.line(img, tuple(q[0]), tuple(q[1]), ink, lw)
            for e in (a, b):
                q = np.round(px([e - n * h, e + n * h])).astype(np.int32)
                cv2.line(img, tuple(q[0]), tuple(q[1]), ink, lw)
        elif o["type"] == "glass_partition":
            for k in (-0.25, 0.25):
                q = np.round(px([a + n * h * k, b + n * h * k])).astype(np.int32)
                cv2.line(img, tuple(q[0]), tuple(q[1]), ink, 1)
        else:
            k = o["swing"]
            hinge = np.asarray(o["hinge"])
            other = b if np.allclose(hinge, a) else a
            leaf_end = hinge + n * k * w
            q = np.round(px([hinge, leaf_end])).astype(np.int32)
            cv2.line(img, tuple(q[0]), tuple(q[1]), ink, lw)
            t = np.linspace(0, np.pi / 2, 24)
            d0 = (other - hinge) / w
            arc = [hinge + w * (np.cos(tt) * d0 + np.sin(tt) * n * k) for tt in t]
            cv2.polylines(img, [np.round(px(arc)).astype(np.int32)], False, ink, 1)
    # room names (as many plans have)
    if rng.random() < 0.8:
        for r in res["rooms"]:
            p = max(polys(r["poly"]), key=lambda q: q.area, default=None)
            if p is None or p.area < 3e6:
                continue
            c = px([p.representative_point().coords[0]])[0]
            txt = r["type"].upper() if r["type"] != "undefined" else ""
            sc = 0.35 if p.area > 8e6 else 0.28
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, sc, 1)
            cv2.putText(img, txt, (int(c[0] - tw / 2), int(c[1] + th / 2)), cv2.FONT_HERSHEY_SIMPLEX, sc, ink, 1,
                        cv2.LINE_AA)
    return img, room, icon


def one(sc):
    try:
        res = conv2.convert(os.path.join(ROOT, sc, "annotation_3d.json"))
        if conv2.is_junk(res):
            return sc, "junk"
        img, room, icon = render(res, hash(sc) & 0xffff)
        cv2.imwrite(os.path.join(OUT, "img", sc + ".png"), img)
        np.savez_compressed(os.path.join(OUT, "mask", sc + ".npz"), room=room, icon=icon)
        truth = conv2.to_truth(res, sc)
        truth["drawing"] = {"px_per_m": PPM, "pad_m": PAD, "origin_mm": [res["outline"].bounds[0], res["outline"].bounds[3]],
                            "size_px": [img.shape[1], img.shape[0]]}
        json.dump(truth, open(os.path.join(OUT, "truth", sc + ".json"), "w"))
        return sc, "ok"
    except Exception as e:
        return sc, "error: " + repr(e)[:100]


if __name__ == "__main__":
    for d in ("img", "mask", "truth"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    scenes = sorted(os.listdir(ROOT))
    with Pool(os.cpu_count()) as p:
        res = p.map(one, scenes, chunksize=10)
    import collections
    c = collections.Counter(s.split(":")[0] for _, s in res)
    print(c)
    print(collections.Counter(s for _, s in res if s.startswith("error")).most_common(5))
    json.dump(dict(res), open(os.path.join(OUT, "status.json"), "w"))
