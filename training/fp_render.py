"""Extra training data for the floor-plan model, written into the same cache as CubiCasa5K.

Sources
  resplan     ResPlan (17k South-Asian residential plans, CC BY 4.0): vector geometry, rendered here as drawings
  synthetic   angled buildings composed from ResPlan wings at 30 / 45 / 60 degrees around a hub (with curved
              fronts), rendered the same way - exact labels for the plans no dataset has enough of
  floorplancad  see fp_extra_fpcad.py (walls, doors, windows, stairs, sanitary from real CAD drawings)

Every plan is drawn in a randomly picked style (solid / hatched / outlined walls, line weights, door swings,
window marks, stair treads, room names and sizes the way Indian plans write them, dimension strings, paper tone,
scan blur) so the model learns the plan, not one drawing style.

Labels follow the cache format: <id>_lbl.png  B = room class, G = icon class (fp_common ids). What a source does
not know is IGNORE (255), so it teaches nothing there: untyped interior space, the inside of a door opening (room
head), the swing area of a door (icon head). These sources have no junction labels (no _pts.npy).
"""
import json
import math
import os
import pickle
import random

import cv2
import numpy as np

from fp_common import ROOM_CLASSES, ICON_CLASSES, IGNORE

R = {n: i for i, n in enumerate(ROOM_CLASSES)}
I = {n: i for i, n in enumerate(ICON_CLASSES)}

# ResPlan layer -> room class (None: icon only / not a room)
RESPLAN_ROOMS = {"living": "LivingRoom", "bedroom": "Bedroom", "kitchen": "Kitchen", "bathroom": "Bath",
                 "balcony": "Outdoor", "garden": "Outdoor", "pool": "Outdoor", "parking": "Outdoor",
                 "storage": "Storage"}
ROOM_NAMES = {
    "LivingRoom": ["LIVING", "HALL", "LIVING ROOM", "DRAWING", "FAMILY LOUNGE", "LOUNGE"],
    "Bedroom": ["BEDROOM", "BED ROOM", "M. BEDROOM", "BED RM", "GUEST ROOM", "KIDS ROOM"],
    "Kitchen": ["KITCHEN", "KIT.", "KITCHEN & DINING", "PANTRY"],
    "Bath": ["TOILET", "BATH", "W.C.", "T & B", "WASH", "ATT. TOILET"],
    "Outdoor": ["BALCONY", "SIT OUT", "VERANDAH", "GARDEN", "OPEN TERRACE", "PORTICO"],
    "Storage": ["STORE", "STORE ROOM", "UTILITY"],
    "Entry/Hallway": ["LOBBY", "FOYER", "PASSAGE", "CORRIDOR", "RECEPTION"],
}
SAFE_PICKLE = {("shapely.io", "from_wkb"), ("numpy._core.multiarray", "scalar"), ("numpy", "dtype"),
               ("numpy.core.multiarray", "scalar")}


class _SafeUnpickler(pickle.Unpickler):
    """ResPlan ships as a pickle: allow only the shapely / numpy constructors it needs (no code execution)."""

    def find_class(self, module, name):
        if (module, name) not in SAFE_PICKLE:
            raise pickle.UnpicklingError("blocked in dataset pickle: {}.{}".format(module, name))
        return super().find_class(module, name)


def load_resplan(pkl_path):
    with open(pkl_path, "rb") as f:
        return _SafeUnpickler(f).load()


# ----------------------------------------------------------------------------- geometry
def _polys(g):
    if g is None or getattr(g, "is_empty", True):
        return []
    if g.geom_type == "Polygon":
        return [g]
    if hasattr(g, "geoms"):
        return [p for q in g.geoms for p in _polys(q)]
    return []


def resplan_plan(p):
    """ResPlan record -> plan dict used by the renderer (units: ResPlan units, metres per unit in 'mpu')."""
    from shapely.ops import unary_union
    rooms = []
    for key, cls in RESPLAN_ROOMS.items():
        rooms += [(cls, q) for q in _polys(p.get(key))]
    stairs = _polys(p.get("stair"))
    doors = _polys(p.get("door")) + _polys(p.get("front_door"))
    walls = unary_union(_polys(p.get("wall")))
    inner = unary_union(_polys(p.get("inner")))
    gross = unary_union([inner, walls]).area
    mpu = math.sqrt(float(p.get("area", 0)) / gross) if gross > 0 and p.get("area") else 0.05
    return {"walls": walls, "rooms": rooms, "doors": doors, "windows": _polys(p.get("window")),
            "stairs": stairs, "inner": inner, "mpu": mpu, "id": "rp{}".format(p.get("id"))}


def _transform(plan, fn):
    out = dict(plan)
    out["walls"] = fn(plan["walls"])
    out["inner"] = fn(plan["inner"])
    out["rooms"] = [(c, fn(q)) for c, q in plan["rooms"]]
    for k in ("doors", "windows", "stairs"):
        out[k] = [fn(q) for q in plan[k]]
    return out


def synthetic_angled(parts, rng):
    """2-3 ResPlan plans as wings, each turned by 30 / 45 / 60 degrees around a shared hub (lobby), sometimes
    with a curved front - the kind of building (school, office, clubhouse) the axis-aligned datasets lack."""
    from shapely import affinity
    from shapely.geometry import Point, Polygon
    from shapely.ops import unary_union
    mpu = parts[0]["mpu"]
    wd = max(parts[0]["walls"].area / max(parts[0]["walls"].length / 2, 1e-6), 2.0)   # wall depth (units)
    n = len(parts)
    step = rng.choice([30, 45, 60])
    base = rng.uniform(0, 360)
    angles = [base + (k - (n - 1) / 2) * 2 * step for k in range(n)]
    wings = []
    radius = 0.0
    for pl in parts:
        b = unary_union([pl["walls"], pl["inner"]])
        c = b.centroid
        pl = _transform(pl, lambda g, c=c: affinity.translate(g, -c.x, -c.y))
        radius = max(radius, math.sqrt(b.area) * 0.75)
        wings.append(pl)
    placed = []
    for pl, a in zip(wings, angles):
        r = radius
        for _ in range(40):          # push the wing out along its direction until it clears the others
            dx, dy = r * math.cos(math.radians(a)), r * math.sin(math.radians(a))
            cand = _transform(pl, lambda g, a=a, dx=dx, dy=dy: affinity.translate(
                affinity.rotate(g, a - 90, origin=(0, 0)), dx, dy))
            foot = unary_union([cand["walls"], cand["inner"]]).buffer(wd)
            if all(not foot.intersects(unary_union([q["walls"], q["inner"]])) for q in placed) and \
                    not foot.contains(Point(0, 0)):
                break
            r *= 1.12
        placed.append(cand)
    feet = unary_union([unary_union([q["walls"], q["inner"]]) for q in placed])
    hull = feet.convex_hull
    if rng.random() < 0.4:           # curved front: a circular bulge on the hub
        hull = unary_union([hull, Point(0, 0).buffer(radius * rng.uniform(0.9, 1.3), 48)]).convex_hull
    hub = hull.difference(feet.buffer(wd * 0.5))
    hub = max(_polys(hub), key=lambda q: q.area) if _polys(hub) else Polygon()
    hub_wall = hub.exterior.buffer(wd / 2, cap_style=2, join_style=2).difference(feet.buffer(-wd * 0.1)) \
        if not hub.is_empty else Polygon()
    hub_room = hub.buffer(-wd / 2)
    plan = {"walls": unary_union([q["walls"] for q in placed] + [hub_wall]),
            "rooms": [r_ for q in placed for r_ in q["rooms"]] + ([("Entry/Hallway", q) for q in _polys(hub_room)]),
            "doors": [d for q in placed for d in q["doors"]], "windows": [w for q in placed for w in q["windows"]],
            "stairs": [s for q in placed for s in q["stairs"]],
            "inner": unary_union([q["inner"] for q in placed] + _polys(hub_room)), "mpu": mpu,
            "id": "syn_" + "_".join(p["id"] for p in parts) + "_{}".format(int(step))}
    # the hub's walls cut through nothing: openings where a wing meets the hub stay as the wing drew them
    return plan


# ----------------------------------------------------------------------------- drawing
def _style(rng):
    return {
        "wall": rng.choice(["solid", "solid", "hatch", "outline", "grey"]),
        "line": rng.choice([1, 1, 2, 2, 3]),
        "ink": int(rng.uniform(0, 70)),
        "paper": int(rng.uniform(236, 256)),
        "names": rng.random() < 0.85,
        "sizes": rng.random() < 0.6,
        "feet": rng.random() < 0.35,
        "dims": rng.random() < 0.5,
        "door_swing": rng.random() < 0.85,
        "window": rng.choice(["three", "two", "box"]),
        "blur": rng.random() < 0.3,
        "font": rng.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_PLAIN,
                            cv2.FONT_HERSHEY_COMPLEX]),
    }


def _pts(poly, T):
    return np.round(T(np.asarray(poly.exterior.coords))).astype(np.int32)


def _fill(img, polys, T, val):
    for q in polys:
        if q.is_empty or q.area <= 0:
            continue
        cv2.fillPoly(img, [_pts(q, T)] + [np.round(T(np.asarray(h.coords))).astype(np.int32)
                                         for h in q.interiors], val)


def _feet(m):
    ft = m / 0.3048
    f, i = int(ft), int(round((ft - int(ft)) * 12))
    if i == 12:
        f, i = f + 1, 0
    return "{}'{}\"".format(f, i)


def render(plan, ppm, rng):
    """Draw `plan` at `ppm` pixels per metre. Returns (rgb image, room label, icon label)."""
    from shapely.ops import unary_union
    st = _style(rng)
    foot = unary_union([plan["walls"], plan["inner"]])
    minx, miny, maxx, maxy = foot.bounds
    k = plan["mpu"] * ppm                                    # pixels per unit
    margin = int(rng.uniform(0.8, 2.0) * ppm)
    W = int((maxx - minx) * k) + 2 * margin
    H = int((maxy - miny) * k) + 2 * margin
    if W * H > 9e6 or W < 64 or H < 64:
        return None
    T = lambda xy: np.stack([(xy[:, 0] - minx) * k + margin, (xy[:, 1] - miny) * k + margin], 1)
    paper, ink = st["paper"], st["ink"]
    img = np.full((H, W), paper, np.uint8)
    room = np.full((H, W), R["Background"], np.uint8)
    icon = np.full((H, W), I["NoIcon"], np.uint8)
    lw = max(1, int(round(st["line"] * ppm / 80)))

    # interior: untyped space is unknown, typed rooms get their class
    _fill(room, _polys(plan["inner"]), T, IGNORE)
    for cls, q in plan["rooms"]:
        _fill(room, _polys(q), T, R[cls])
    for q in plan["stairs"]:
        _fill(room, _polys(q), T, IGNORE)
        _fill(icon, _polys(q), T, I["Stairs"])

    # walls
    walls = _polys(plan["walls"])
    openings = [q for q in plan["doors"] + plan["windows"]]
    _fill(room, walls, T, R["Wall"])
    wall_px = np.zeros((H, W), np.uint8)
    _fill(wall_px, walls, T, 1)
    op_px = np.zeros((H, W), np.uint8)
    _fill(op_px, [q for o in openings for q in _polys(o)], T, 1)
    draw_wall = (wall_px > 0) & (op_px == 0)
    if st["wall"] == "solid":
        img[draw_wall] = ink
    elif st["wall"] == "grey":
        img[draw_wall] = int(rng.uniform(90, 170))
    elif st["wall"] == "hatch":
        hatch = np.zeros((H, W), np.uint8)
        sp = max(4, int(ppm * 0.08))
        for c in range(-H, W, sp):
            cv2.line(hatch, (c, 0), (c + H, H), 1, max(1, lw // 2))
        img[draw_wall & (hatch > 0)] = ink
    edges = cv2.morphologyEx(draw_wall.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    img[edges] = ink
    if lw > 1:
        img[cv2.dilate(edges.astype(np.uint8), np.ones((lw, lw), np.uint8)) > 0] = ink

    # openings: the room head does not know whether an opening counts as wall -> IGNORE; icon head knows
    for q in [p for o in plan["windows"] for p in _polys(o)]:
        _fill(room, [q], T, IGNORE)
        _fill(icon, [q], T, I["Window"])
        rect = np.array(q.minimum_rotated_rectangle.exterior.coords)[:4]
        P = T(rect)
        e = [np.linalg.norm(P[(j + 1) % 4] - P[j]) for j in range(4)]
        j = int(np.argmax(e))
        a, b = P[j], P[(j + 1) % 4]
        n = P[(j + 2) % 4] - b
        cv2.polylines(img, [np.round(P).astype(np.int32)], True, ink, 1)
        lines = {"three": (0.25, 0.5, 0.75), "two": (0.35, 0.65), "box": ()}[st["window"]]
        for f in lines:
            cv2.line(img, tuple(np.round(a + n * f).astype(int)), tuple(np.round(b + n * f).astype(int)), ink, 1)
    for q in [p for o in plan["doors"] for p in _polys(o)]:
        _fill(room, [q], T, IGNORE)
        _fill(icon, [q], T, I["Door"])
        if not st["door_swing"]:
            continue
        rect = np.array(q.minimum_rotated_rectangle.exterior.coords)[:4]
        P = T(rect)
        e = [np.linalg.norm(P[(j + 1) % 4] - P[j]) for j in range(4)]
        j = int(np.argmax(e))
        a, b = P[j], P[(j + 1) % 4]
        width = np.linalg.norm(b - a)
        if width < 6:
            continue
        nrm = P[(j + 3) % 4] - a
        nrm = nrm / max(np.linalg.norm(nrm), 1e-6)
        side = 1 if rng.random() < 0.5 else -1                 # opens to one side or the other
        hinge, other = (a, b) if rng.random() < 0.5 else (b, a)
        leaf_dir = -side * nrm
        leaf = hinge + leaf_dir * width
        cv2.line(img, tuple(np.round(hinge).astype(int)), tuple(np.round(leaf).astype(int)), ink, max(1, lw))
        a0 = math.degrees(math.atan2(*(other - hinge)[::-1]))
        a1 = math.degrees(math.atan2(*(leaf - hinge)[::-1]))
        d = (a1 - a0 + 540) % 360 - 180
        cv2.ellipse(img, tuple(np.round(hinge).astype(int)), (int(width), int(width)), 0, a0, a0 + d, ink, 1)
        swing = np.zeros((H, W), np.uint8)
        cv2.ellipse(swing, tuple(np.round(hinge).astype(int)), (int(width), int(width)), 0, a0, a0 + d, 1, -1)
        icon[(swing > 0) & (icon == I["NoIcon"])] = IGNORE     # swing area: the icon head is not told

    # stairs: treads across the long side
    for q in [p for s in plan["stairs"] for p in _polys(s)]:
        rect = np.array(q.minimum_rotated_rectangle.exterior.coords)[:4]
        P = T(rect)
        e = [np.linalg.norm(P[(j + 1) % 4] - P[j]) for j in range(4)]
        j = int(np.argmax(e))
        a, b, c = P[j], P[(j + 1) % 4], P[(j + 2) % 4]
        nt = max(3, int(e[j] / max(ppm * 0.28, 3)))
        for t in range(nt + 1):
            u = a + (b - a) * t / nt
            cv2.line(img, tuple(np.round(u).astype(int)), tuple(np.round(u + (c - b)).astype(int)), ink, 1)
        cv2.polylines(img, [np.round(P).astype(np.int32)], True, ink, 1)

    # room names and sizes, dimension strings
    fs = max(0.3, ppm / 110) * rng.uniform(0.7, 1.2)
    for cls, q in plan["rooms"]:
        for p in _polys(q):
            if p.area * plan["mpu"] ** 2 < 2.0 or not st["names"]:
                continue
            c = T(np.array([[p.representative_point().x, p.representative_point().y]]))[0]
            name = rng.choice(ROOM_NAMES.get(cls, ["ROOM"]))
            (tw, th), _ = cv2.getTextSize(name, st["font"], fs, 1)
            org = (int(c[0] - tw / 2), int(c[1]))
            cv2.putText(img, name, org, st["font"], fs, ink, 1, cv2.LINE_AA)
            if st["sizes"]:
                bx = p.minimum_rotated_rectangle
                xy = np.array(bx.exterior.coords)[:3] * plan["mpu"]
                l1, l2 = np.linalg.norm(xy[1] - xy[0]), np.linalg.norm(xy[2] - xy[1])
                txt = "{} X {}".format(_feet(l1), _feet(l2)) if st["feet"] else "{:.2f} x {:.2f}".format(l1, l2)
                (tw2, _), _ = cv2.getTextSize(txt, st["font"], fs * 0.8, 1)
                cv2.putText(img, txt, (int(c[0] - tw2 / 2), int(c[1] + th * 1.8)), st["font"], fs * 0.8, ink, 1,
                            cv2.LINE_AA)
    if st["dims"]:
        x0, y0 = margin, int(margin * 0.45)
        x1 = W - margin
        cv2.line(img, (x0, y0), (x1, y0), ink, 1)
        for x in (x0, x1):
            cv2.line(img, (x - 6, y0 + 6), (x + 6, y0 - 6), ink, 1)
        L = (maxx - minx) * plan["mpu"]
        txt = _feet(L) if st["feet"] else "{:.2f} m".format(L)
        cv2.putText(img, txt, ((x0 + x1) // 2 - 30, y0 - 4), st["font"], fs, ink, 1, cv2.LINE_AA)

    if st["blur"]:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 0.9))
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return rgb, room, icon


# ----------------------------------------------------------------------------- cache writer
def write_entry(data_dir, sid, rgb, room, icon, max_side):
    h, w = room.shape
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        room = cv2.resize(room, (int(w * s), int(h * s)), interpolation=cv2.INTER_NEAREST)
        icon = cv2.resize(icon, (int(w * s), int(h * s)), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(data_dir, sid + "_img.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(data_dir, sid + "_lbl.png"), np.dstack([room, icon, np.zeros_like(room)]))


def build_rendered(cache_root, resplan_pkl, split_json=None, n_resplan=None, n_synthetic=3000, max_side=1536,
                   seed=0, log=print):
    """Render ResPlan plans and synthetic angled buildings into the cache. Returns index entries per split."""
    rng = random.Random(seed)
    data_dir = os.path.join(cache_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    recs = load_resplan(resplan_pkl)
    split = {}
    if split_json and os.path.exists(split_json):
        sj = json.load(open(split_json))
        for name, ids in sj.items():
            for i in ids:
                split[int(i)] = "val" if name in ("val", "valid", "validation") else ("test" if name == "test"
                                                                                       else "train")
    out = {"train": [], "val": [], "test": []}
    recs = recs[:n_resplan] if n_resplan else recs
    plans = []
    for p in recs:
        try:
            pl = resplan_plan(p)
        except Exception:
            continue
        plans.append((split.get(int(p.get("id", -1)), "train" if rng.random() < 0.9 else "val"), pl))
    for n_done, (sp, pl) in enumerate(plans):
        res = render(pl, rng.uniform(55, 130), rng)
        if res is None:
            continue
        write_entry(data_dir, pl["id"], *res, max_side)
        out[sp].append({"id": pl["id"], "source": "resplan"})
        if n_done % 1000 == 0:
            log("  resplan rendered {}/{}".format(n_done, len(plans)))
    train_plans = [pl for sp, pl in plans if sp == "train"]
    val_plans = [pl for sp, pl in plans if sp == "val"]
    made = 0
    for t in range(n_synthetic * 3):
        if made >= n_synthetic:
            break
        sp = "val" if t % 10 == 0 else "train"
        pool = val_plans if sp == "val" and len(val_plans) > 3 else train_plans
        try:
            pl = synthetic_angled(rng.sample(pool, rng.choice([2, 2, 3])), rng)
            res = render(pl, rng.uniform(45, 100), rng)
        except Exception:
            continue
        if res is None:
            continue
        sid = "syn{:05d}".format(made)
        write_entry(data_dir, sid, *res, max_side)
        out[sp].append({"id": sid, "source": "synthetic"})
        made += 1
        if made % 500 == 0:
            log("  synthetic angled plans {}/{}".format(made, n_synthetic))
    return out
