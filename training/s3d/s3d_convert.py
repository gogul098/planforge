"""Structured3D annotation_3d.json -> 3D truth for PlanForge training (rooms, walls with thickness and type,
openings with sill / head / swing, storey heights). Units: metres in the output.

Walls:
  * solid walls = inside the outer wall faces and outside every room;
  * two rooms that touch with a wall plane on the shared edge (a zero-thickness wall in the source) get a
    partition of the thickness of the nearest door / window in that line, else 120 mm, cut from both rooms;
  * a shared edge with no wall plane is an open boundary (open plan) and stays open;
  * a room edge on the outer face (no outer wall drawn) gets an outer wall of the house's outer thickness,
    added outwards.
"""
import json
import numpy as np
from shapely.geometry import LineString, Polygon, Point
from shapely.ops import unary_union, polygonize, linemerge

MM = 1000.0
DEFAULT_PART, DEFAULT_OUT = 120.0, 200.0
OPEN_TYPES = {"door", "window"}
SWING_PRIORITY = ["bathroom", "bedroom", "study", "kitchen", "store room", "laundry room", "office", "studio",
                  "balcony", "garden", "dining room", "corridor", "living room", "undefined"]


class Scene:
    def __init__(self, path):
        d = json.load(open(path))
        self.d = d
        self.J = np.array([j["coordinate"] for j in d["junctions"]], float)
        self.LJ = np.array(d["lineJunctionMatrix"], bool)
        self.PL = np.array(d["planeLineMatrix"], bool)
        self.planes = {p["ID"]: p for p in d["planes"]}

    def segments(self, pid, z=None):
        out = []
        for l in np.nonzero(self.PL[pid])[0]:
            js = np.nonzero(self.LJ[l])[0]
            if len(js) < 2:
                continue
            a, b = self.J[js[0]], self.J[js[1]]
            if z is not None and (abs(a[2] - z) > 1 or abs(b[2] - z) > 1):
                continue
            if np.linalg.norm(a[:2] - b[:2]) > 1:
                out.append(LineString([tuple(a[:2]), tuple(b[:2])]))
        return out

    def junctions(self, pid):
        return sorted({int(j) for l in np.nonzero(self.PL[pid])[0] for j in np.nonzero(self.LJ[l])[0]})


def _poly(segs):
    ps = list(polygonize(unary_union(segs))) if segs else []
    return unary_union(ps) if ps else None


def _on_plane(seg, plane, tol=6.0):
    n, off = np.array(plane["normal"][:2], float), float(plane["offset"])
    if abs(plane["normal"][2]) > 1e-6:
        return False
    return all(abs(float(np.dot(n, c)) + off) < tol for c in seg.coords)


def convert(path):
    S = Scene(path)
    rooms, openings, outwall = [], [], None
    for s in S.d["semantics"]:
        t = s["type"]
        if t in OPEN_TYPES:
            js = sorted({j for p in s["planeID"] for j in S.junctions(p)})
            P = S.J[js]
            xy = P[:, :2]
            c = xy.mean(0)
            _, _, vt = np.linalg.svd(xy - c)
            ax, nrm = vt[0], vt[1]
            pr, pn = (xy - c) @ ax, (xy - c) @ nrm
            mid = c + nrm * (pn.max() + pn.min()) / 2
            openings.append({"type": t, "a": mid + ax * pr.min(), "b": mid + ax * pr.max(),
                             "depth": float(pn.max() - pn.min()), "z0": float(P[:, 2].min()),
                             "z1": float(P[:, 2].max())})
        elif t == "outwall":
            outwall = s
        else:
            fl = [p for p in s["planeID"] if S.planes[p]["type"] == "floor"]
            ce = [p for p in s["planeID"] if S.planes[p]["type"] == "ceiling"]
            poly = _poly(S.segments(fl[0])) if fl else None
            if poly is None or poly.area < 1e4:                       # under 0.01 m2: noise
                continue
            rooms.append({"type": t, "poly": poly.buffer(0),
                          "walls": [S.planes[p] for p in s["planeID"] if S.planes[p]["type"] == "wall"],
                          "ceiling": abs(S.planes[ce[0]]["offset"]) if ce else 2800.0})
    if not rooms or outwall is None:
        raise ValueError("no rooms / no outer wall")
    footprint = _poly([g for p in outwall["planeID"] for g in S.segments(p, z=0.0)])
    if footprint is None:
        raise ValueError("no outer outline")
    allrooms = unary_union([r["poly"] for r in rooms])
    footprint = footprint.union(allrooms).buffer(0)

    def near_thickness(geom, default):
        best, bd = default, 400.0
        for o in openings:
            dd = geom.distance(LineString([tuple(o["a"]), tuple(o["b"])]))
            if dd < bd and 60 <= o["depth"] <= 400:
                best, bd = o["depth"], dd
        return best

    outer_t = [o["depth"] for o in openings if 60 <= o["depth"] <= 400 and
               footprint.boundary.distance(Point(*(o["a"] + o["b"]) / 2)) < 400]
    outer_t = float(np.median(outer_t)) if outer_t else DEFAULT_OUT
    added_part, added_out, open_edges = [], [], []
    # shared edges between rooms
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            A, B = rooms[i], rooms[j]
            sh = A["poly"].boundary.intersection(B["poly"].buffer(6))
            pieces = [g for g in getattr(sh, "geoms", [sh]) if g.geom_type in ("LineString", "MultiLineString")]
            for g in pieces:
                for seg in getattr(g, "geoms", [g]):
                    if seg.length < 50:
                        continue
                    walled = any(_on_plane(seg, p) for p in A["walls"] + B["walls"])
                    if walled:
                        t = near_thickness(seg, DEFAULT_PART)
                        added_part.append(seg.buffer(t / 2, cap_style=2, join_style=2))
                    else:
                        open_edges.append(seg)
    # room edges lying on the outer face: give them an outer wall, added outwards
    ext = footprint.boundary
    for r in rooms:
        on = r["poly"].boundary.intersection(ext.buffer(6))
        for seg in getattr(on, "geoms", [on]):
            if seg.geom_type == "LineString" and seg.length >= 50:
                added_out.append(seg.buffer(outer_t, cap_style=2, join_style=2).difference(allrooms))
    walls = footprint.difference(allrooms)
    if added_out:
        walls = walls.union(unary_union(added_out))
    if added_part:
        walls = walls.union(unary_union(added_part))
    walls = walls.buffer(0)
    outline = footprint.union(walls).buffer(1, join_style=2).buffer(-1, join_style=2)
    for r in rooms:
        r["poly"] = r["poly"].difference(walls).buffer(0)
    # openings: the rooms on each side, glass partitions, door swing
    for o in openings:
        a, b = np.asarray(o["a"]), np.asarray(o["b"])
        u = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        n = np.array([-u[1], u[0]])
        m = (a + b) / 2
        d = o["depth"] / 2 + 250
        side = {}
        for k in (1, -1):
            pt = Point(*(m + n * d * k))
            side[k] = next((r["type"] for r in rooms if r["poly"].contains(pt)), None)
        o["rooms"] = [side[1], side[-1]]
        if o["type"] == "window" and side[1] and side[-1] and o["z0"] < 50:
            o["type"] = "glass_partition"
        elif o["type"] == "window" and side[1] and side[-1]:
            o["type"] = "internal_window"
        if o["type"] == "door":
            cands = [k for k in (1, -1) if side[k]]
            if len(cands) == 2:
                cands.sort(key=lambda k: SWING_PRIORITY.index(side[k]) if side[k] in SWING_PRIORITY else 99)
            k = cands[0] if cands else 1                              # outside -> swings in (no room: k=1)
            if not side[k] and side[-k]:
                k = -k
            w = np.linalg.norm(b - a)
            best = None
            for hinge, other in ((a, b), (b, a)):
                arc = Point(*hinge).buffer(w).intersection(
                    Polygon([hinge, other, other + n * k * w, hinge + n * k * w]))
                clash = arc.intersection(walls).area
                if best is None or clash < best[0]:
                    best = (clash, hinge, other)
            o["swing"] = int(k)
            o["hinge"] = best[1]
    return {"rooms": rooms, "openings": openings, "walls": walls, "outline": outline,
            "open_edges": open_edges, "outer_t": outer_t,
            "ceiling": float(np.median([r["ceiling"] for r in rooms]))}


def is_junk(res):
    """Halls, warehouses and broken scenes that are not homes."""
    area = res["outline"].area / 1e6
    big = max(r["poly"].area for r in res["rooms"]) / 1e6
    named = sum(r["poly"].area for r in res["rooms"] if r["type"] != "undefined") / max(
        sum(r["poly"].area for r in res["rooms"]), 1)
    return area > 400 or big > 150 or (len(res["rooms"]) < 2) or (named < 0.3 and area > 80)


def to_truth(res, scene_id):
    """3D truth in metres: what the model must reproduce."""
    def P(g):
        return [[[round(x / MM, 4), round(y / MM, 4)] for x, y in p.exterior.coords]
                for p in getattr(g, "geoms", [g]) if p.area > 100]
    return {"scene": scene_id, "units": "m", "storey": {"floor_z": 0.0, "ceiling_z": round(res["ceiling"] / MM, 3)},
            "outer_wall_thickness": round(res["outer_t"] / MM, 3),
            "rooms": [{"type": r["type"], "polygon": P(r["poly"]), "area": round(r["poly"].area / 1e6, 3),
                       "ceiling_z": round(r["ceiling"] / MM, 3)} for r in res["rooms"]],
            "walls": P(res["walls"]),
            "openings": [{"type": o["type"], "a": [round(v / MM, 4) for v in o["a"]],
                          "b": [round(v / MM, 4) for v in o["b"]], "thickness": round(o["depth"] / MM, 3),
                          "sill": round(o["z0"] / MM, 3), "head": round(o["z1"] / MM, 3),
                          "rooms": o["rooms"], **({"swing": o["swing"], "hinge": [round(v / MM, 4) for v in o["hinge"]]}
                                                  if "swing" in o else {})} for o in res["openings"]],
            "open_edges": [[[round(x / MM, 4), round(y / MM, 4)] for x, y in e.coords] for e in res["open_edges"]]}
