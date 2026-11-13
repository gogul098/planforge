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


_YOLO = {}


def detector_furniture(img, mpp, roi=None):
    """Run the furniture detector from your HF repo (skipped if it is not trained yet)."""
    if not FURNITURE_REPO:
        return []
    onnx = use_onnx()
    wfile = "checkpoints/best.onnx" if onnx else "checkpoints/best.pt"
    try:
        _, fp = model_files(FURNITURE_REPO, [wfile, "furniture_meta.json"])
        w = fp[wfile]
        meta = json.load(open(fp["furniture_meta.json"]))
    except Exception as e:
        print("furniture detector not available yet ({}): using fixtures from the floor-plan model only".format(
            type(e).__name__))
        return []
    if onnx:
        if _YOLO.get("w") != w:
            _YOLO.update(w=w, net=_onnx_session(w))
        dets = detect_furniture(_YOLO["net"], img, plan_px_per_m=1.0 / mpp, train_px_per_m=meta.get("train_px_per_m"),
                                conf=FURNITURE_LOW_CONF, zooms=FURNITURE_ZOOMS, roi=roi)
        return [{"type": d["cls"], "conf": round(d["conf"], 3), "corners": d["corners"], "source": "furniture detector"}
                for d in dets if d["cls"] in FURN_SPEC]
    if not os.environ.get("P3D_NO_INSTALL"):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.3.253"], check=False)
    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/yolo")
    from ultralytics import YOLO, settings as _ysettings
    try:                                           # never write into the app folder (read-only for the app user)
        _ysettings.update({"runs_dir": os.path.join(os.environ.get("YOLO_CONFIG_DIR", "/tmp/yolo"), "runs")})
    except Exception:
        pass
    if _YOLO.get("w") != w:                       # load once, reload only when a new release arrives
        _YOLO.update(w=w, net=YOLO(w))
    dets = detect_furniture(_YOLO["net"], img, plan_px_per_m=1.0 / mpp, train_px_per_m=meta.get("train_px_per_m"),
                            conf=FURNITURE_LOW_CONF, zooms=FURNITURE_ZOOMS, roi=roi)   # weak hits are checked against the rooms later
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
