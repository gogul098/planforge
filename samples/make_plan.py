"""School ground-floor plan drawn by code so that every written number IS the drawn geometry.
Outputs: school_ground_floor.dxf (metres), .png (300 dpi, 1:100), .pdf, and geometry.json."""
import json
import math

import ezdxf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Polygon as MPoly
from shapely.geometry import Polygon
from shapely.ops import unary_union

TE, TI = 0.23, 0.115          # external / internal wall thickness (m)
EXT = 0.115                    # wall ends run to the outer face of the wall they meet
WALL_H, RISER_IN = 3.0, None
PLINTH = 0.45

walls, openings, rooms, texts, lines, arcs, dims, furn = [], [], [], [], [], [], [], []
V = lambda *a: np.array(a, float)
rot = lambda d: V(-d[1], d[0])


def add_wall(a, b, t, ends=(True, True)):
    walls.append({"a": V(*a), "b": V(*b), "t": t, "ends": ends})
    return len(walls) - 1


def add_opening(w, centre_u, width, kind, swing=1, hinge=-1):
    """centre_u: distance from wall start a; swing: +1 = towards the wall's left normal; hinge: -1 = start side."""
    openings.append({"wall": w, "u": centre_u, "w": width, "kind": kind, "swing": swing, "hinge": hinge})


def text(p, s, size=2.5, angle=0.0, bold=False, ha="center", va="center"):
    texts.append({"p": V(*p), "s": s, "h": size, "angle": angle, "bold": bold, "ha": ha, "va": va})


def line(a, b, lw=0.18):
    lines.append((V(*a), V(*b), lw))


def dim(a, b, off_dir, off, label=None, size=2.5):
    """Aligned dimension between a and b, drawn at distance off along off_dir (unit)."""
    a, b, n = V(*a), V(*b), V(*off_dir)
    L = float(np.linalg.norm(b - a))
    dims.append({"a": a, "b": b, "n": n, "off": off, "label": label or "{:.2f}".format(L), "size": size})


# ----------------------------------------------------------------------------- blocks of rooms
def block(origin, u, v, rooms_w, depth, names, doors, outer_t=TE, inner_t=TI, windows=True, chain=True):
    """A straight row of rooms. origin = start point on the inner (atrium-side) wall centre-line; u = along the
    row, v = away from the atrium. rooms_w = clear widths along u; depth = clear depth along v.
    Returns end point (on the inner centre-line) and the length."""
    o, u, v = V(*origin), V(*u), V(*v)
    pos = [0.0]                                       # wall centre-lines along u
    x = TE / 2
    for k, wdt in enumerate(rooms_w):
        x += wdt
        if k < len(rooms_w) - 1:
            pos.append(x + TI / 2)
            x += TI
    x += TE / 2
    pos.append(x)
    L = x
    v_out = inner_t / 2 + depth + outer_t / 2       # outer wall centre-line
    P = lambda uu, vv: o + u * uu + v * vv
    iw = add_wall(P(0, 0), P(L, 0), inner_t)
    ow = add_wall(P(0, v_out), P(L, v_out), outer_t)
    xw = [add_wall(P(p, 0), P(p, v_out), TE if k in (0, len(pos) - 1) else TI) for k, p in enumerate(pos)]
    for k, wdt in enumerate(rooms_w):
        u0 = pos[k] + (TE if k == 0 else TI) / 2
        u1 = u0 + wdt
        uc = (u0 + u1) / 2
        c = P(uc, inner_t / 2 + depth / 2)
        ang = math.degrees(math.atan2(u[1], u[0]))
        ang = ang - 180 if ang > 90 else (ang + 180 if ang < -90 else ang)
        rooms.append({"name": names[k], "w": wdt, "d": depth, "centre": c,
                      "poly": [P(u0, inner_t / 2), P(u1, inner_t / 2), P(u1, inner_t / 2 + depth), P(u0, inner_t / 2 + depth)],
                      "angle": ang})
        dw = doors[k]
        if dw:
            add_opening(iw, u0 + 0.25 + dw / 2 if wdt > dw + 0.6 else uc, dw, "door",
                        swing=(1 if (u[0] * v[1] - u[1] * v[0]) > 0 else -1), hinge=-1)
        if windows and names[k].split()[0] not in ("STORE",):
            ww = min(1.5, wdt - 0.8) if names[k].split()[0] != "TOILET" else min(0.9, wdt - 0.8)
            if ww >= 0.6:
                add_opening(ow, uc, ww, "window")
    if chain:
        n_out = v
        base = v_out + outer_t / 2
        for k in range(len(pos) - 1):
            dim(P(pos[k], base), P(pos[k + 1], base), n_out, 1.0)
        dim(P(0, base), P(L, base), n_out, 1.9)
    return P(L, 0), L, v_out


# ----------------------------------------------------------------------------- layout
deg = math.radians
dR, dL = V(math.cos(deg(-60)), math.sin(deg(-60))), V(math.cos(deg(240)), math.sin(deg(240)))
A_R, A_L = V(4.0, 18.0), V(-4.0, 18.0)
# wings (outer side away from the atrium)
B_R, LR, voR = block(A_R, dR, rot(dR), [2.44, 3.66, 4.88], 5.49,
                     ["TOILET (BOYS)", "MEETING ROOM 2", "MULTIPURPOSE HALL"], [0.9, 0.9, 1.2])
B_L, LL, voL = block(A_L, dL, -rot(dL), [2.44, 3.66, 2.44, 2.44], 4.88,
                     ["TOILET (GIRLS)", "MEETING ROOM 1", "TOILET (STAFF)", "TOILET (ACCESSIBLE)"], [0.9, 0.9, 0.9, 1.0])
# lower blocks: same total length on both sides so the entrance wall is level
eR, eL = V(math.cos(deg(-120)), math.sin(deg(-120))), V(math.cos(deg(-60)), math.sin(deg(-60)))
left_w = [3.66, 1.83, 1.715]
total_left = TE + sum(left_w) + 2 * TI
lab = round(total_left + LL - LR - TE - TI - 1.83, 3)
C_R, MR, _ = block(B_R, eR, rot(eR), [lab, 1.83], 4.27, ["COMPUTER LAB", "STAFF ROOM"], [1.0, 0.9])
C_L, ML, _ = block(B_L, eL, -rot(eL), left_w, 4.27, ["OFFICE", "STORE", "PANTRY"], [0.9, 0.9, 0.9])
assert abs(C_R[1] - C_L[1]) < 1e-6, (C_R, C_L)
# atrium outline: top wall, entrance wall (external)
top = add_wall(A_L, A_R, TE)
bot = add_wall(C_L, C_R, TE)
mid_x = (C_L[0] + C_R[0]) / 2
add_opening(bot, mid_x - C_L[0], 1.8, "door2")                        # double entrance door
add_opening(top, 4.0 - 2.6, 1.5, "window"); add_opening(top, 4.0 + 2.6, 1.5, "window")
atrium = Polygon([A_L, A_R, B_R, C_R, C_L, B_L])

# ----------------------------------------------------------------------------- stair (to the first floor)
N_TREAD, TREAD, SW = 18, 0.30, 1.80
n_ris = N_TREAD + 1
riser = WALL_H / n_ris
y_top = A_R[1] - TE / 2
landing = 1.80
y1 = y_top - landing
y0 = y1 - N_TREAD * TREAD
x0, x1 = -SW / 2, SW / 2
for a, b in [((x0, y0), (x1, y0)), ((x1, y0), (x1, y_top)), ((x0, y0), (x0, y_top)), ((x0, y1), (x1, y1))]:
    line(a, b, 0.35)
for k in range(1, N_TREAD):
    line((x0, y0 + k * TREAD), (x1, y0 + k * TREAD))
arrow = (V(0, y0 + 0.15), V(0, y1 - 0.15))
text((0, y0 - 0.45), "UP", 3.0, bold=True)
text((x1 + 0.35, (y0 + y1) / 2), "{}R @ {:.0f}, T {:.0f}".format(n_ris, riser * 1000, TREAD * 1000), 2.2, angle=90)
dim((x0, y0), (x1, y0), (0, -1), 1.1)
dim((x1, y0), (x1, y1), (1, 0), 1.3)
text((0, (y1 + y_top) / 2), "LANDING", 2.0)

# ----------------------------------------------------------------------------- entrance steps
yb = C_L[1] - TE / 2
SW_E, N_E, T_E = 5.00, 3, 0.30
ex0, ex1 = mid_x - SW_E / 2, mid_x + SW_E / 2
ey0 = yb - N_E * T_E
for a, b in [((ex0, ey0), (ex1, ey0)), ((ex0, ey0), (ex0, yb)), ((ex1, ey0), (ex1, yb))]:
    line(a, b, 0.35)
for k in range(1, N_E):
    line((ex0, ey0 + k * T_E), (ex1, ey0 + k * T_E))
arrow_e = (V(ex1 - 0.5, ey0 + 0.1), V(ex1 - 0.5, yb - 0.1))
text((ex1 - 0.5, ey0 - 0.4), "UP", 2.5, bold=True)
text((ex0 - 0.2, (ey0 + yb) / 2), "{}R @ {:.0f}, T {:.0f}".format(N_E, PLINTH / N_E * 1000, T_E * 1000), 2.2, ha="right")
dim((ex0, ey0), (ex1, ey0), (0, -1), 0.9)
dim((ex1, ey0), (ex1, yb), (1, 0), 1.0)
text((mid_x + 2.2, yb + 1.0), "+0.45 FFL", 2.2)
text((mid_x - 3.6, ey0 - 1.2), "±0.00 GL", 2.2)

# ----------------------------------------------------------------------------- furniture (thin lines, real size)
def rect(c, w, d, ang=0.0):
    c = V(*c); a = math.radians(ang); u = V(math.cos(a), math.sin(a)); v = rot(u)
    return [c - u * w / 2 - v * d / 2, c + u * w / 2 - v * d / 2, c + u * w / 2 + v * d / 2, c - u * w / 2 + v * d / 2]


for r in rooms:
    nm = r["name"].split()[0]
    c, ang = r["centre"], r["angle"]
    a = math.radians(ang); u = V(math.cos(a), math.sin(a)); v = rot(u)
    if nm == "MEETING":
        furn.append(rect(c - v * 0.6, 1.8, 0.9, ang))
    elif nm == "OFFICE":
        furn.append(rect(c - v * 0.9, 1.4, 0.7, ang))
    elif nm == "COMPUTER":
        for k in (-1, 1):
            furn.append(rect(c + u * k * 1.3 - v * 0.9, 2.0, 0.6, ang))
    elif nm == "TOILET":
        furn.append(rect(c - v * (r["d"] / 2 - 0.45), 0.4, 0.7, ang))           # WC
furn.append(rect((mid_x, C_L[1] + 2.4), 2.4, 0.7))                               # reception desk
for sx in (-1, 1):
    furn.append(rect((mid_x + sx * 4.6, 12.0), 2.0, 0.9, -sx * 60))              # lounge sofas
    furn.append(rect((mid_x + sx * 3.7, 11.0), 0.9, 0.6, -sx * 60))              # coffee tables

# ----------------------------------------------------------------------------- labels
for r in rooms:
    text(r["centre"] + V(0, 0.28), r["name"], 2.5, angle=0, bold=True)
    text(r["centre"] - V(0, 0.28), "{:.2f} x {:.2f}".format(r["w"], r["d"]), 2.3)
ca = atrium.centroid
text((ca.x, ca.y - 1.8), "ATRIUM / RECEPTION", 3.0, bold=True)
text((ca.x, ca.y - 2.4), "{:.1f} m2".format(atrium.buffer(-TI / 2).area - SW * (y_top - y0)), 2.3)
text((-4.6, 13.2), "LOUNGE", 2.5, bold=True); text((4.6, 13.2), "LOUNGE", 2.5, bold=True)

# ----------------------------------------------------------------------------- geometry -> drawing primitives
def wall_pieces(k):
    w = walls[k]
    a, b, t = w["a"], w["b"], w["t"]
    L = float(np.linalg.norm(b - a)); d = (b - a) / L; n = rot(d)
    cuts = sorted([(o["u"] - o["w"] / 2, o["u"] + o["w"] / 2) for o in openings if o["wall"] == k])
    segs, cur = [], -EXT
    for c0, c1 in cuts:
        segs.append((cur, c0)); cur = c1
    segs.append((cur, L + EXT))
    return [[a + d * s0 - n * t / 2, a + d * s1 - n * t / 2, a + d * s1 + n * t / 2, a + d * s0 + n * t / 2]
            for s0, s1 in segs if s1 - s0 > 1e-6]


wall_polys = [p for k in range(len(walls)) for p in wall_pieces(k)]
solid = unary_union([Polygon(p) for p in wall_polys]).buffer(0)
open_lines, open_arcs = [], []
for o in openings:
    w = walls[o["wall"]]
    a, b, t = w["a"], w["b"], w["t"]
    L = float(np.linalg.norm(b - a)); d = (b - a) / L; n = rot(d)
    j0, j1 = a + d * (o["u"] - o["w"] / 2), a + d * (o["u"] + o["w"] / 2)
    if o["kind"] == "window":
        for off in (-t / 2, -t / 6, t / 6, t / 2):
            open_lines.append((j0 + n * off, j1 + n * off, 0.18))
        open_lines += [(j0 - n * t / 2, j0 + n * t / 2, 0.25), (j1 - n * t / 2, j1 + n * t / 2, 0.25)]
    elif o["kind"] == "door":
        hinge = j0 if o["hinge"] < 0 else j1
        other = j1 if o["hinge"] < 0 else j0
        sn = n * o["swing"]
        hinge_face = hinge + sn * t / 2
        tip = hinge_face + sn * o["w"]
        open_lines.append((hinge_face, tip, 0.35))
        a0 = math.degrees(math.atan2(*(tip - hinge_face)[::-1]))
        a1 = math.degrees(math.atan2(*((other + sn * t / 2) - hinge_face)[::-1]))
        lo, hi = (a0, a1) if (a1 - a0) % 360 <= 180 else (a1, a0)
        open_arcs.append((hinge_face, o["w"], lo, hi))
    else:                                                       # double door, swinging inwards (+n)
        half = o["w"] / 2
        for h, oth in ((j0, a + d * o["u"]), (j1, a + d * o["u"])):
            hf = h + n * t / 2
            tip = hf + n * half
            open_lines.append((hf, tip, 0.35))
            a0 = math.degrees(math.atan2(*(tip - hf)[::-1]))
            a1 = math.degrees(math.atan2(*((oth + n * t / 2) - hf)[::-1]))
            lo, hi = (a0, a1) if (a1 - a0) % 360 <= 180 else (a1, a0)
            open_arcs.append((hf, half, lo, hi))

# overall dimensions (horizontal + vertical) of the outer faces
outline = unary_union([Polygon(p) for p in wall_polys] + [atrium] + [Polygon(r["poly"]) for r in rooms]).buffer(0.01).buffer(-0.01)
bx0, by0, bx1, by1 = outline.bounds
dim((bx0, by1), (bx1, by1), (0, 1), 2.2)
dim((bx0, by0), (bx0, by1), (-1, 0), 2.0)

# ----------------------------------------------------------------------------- PNG / PDF at 1:100
margin = 6.0
X0, X1, Y0, Y1 = bx0 - margin, bx1 + margin, min(by0, ey0) - margin - 3.0, by1 + margin
mm = lambda m: m * 10.0                                     # 1:100 -> 1 m = 10 mm on paper
fig = plt.figure(figsize=(mm(X1 - X0) / 25.4, mm(Y1 - Y0) / 25.4))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1); ax.set_aspect("equal"); ax.axis("off")
pt = lambda mm_: mm_ * 72 / 25.4                           # paper mm -> points
geoms = [solid] if solid.geom_type == "Polygon" else list(solid.geoms)
for gpoly in geoms:
    ax.add_patch(MPoly(np.array(gpoly.exterior.coords), closed=True, fc="black", ec="black", lw=0.1))
    for hole in gpoly.interiors:
        ax.add_patch(MPoly(np.array(hole.coords), closed=True, fc="white", ec="black", lw=0.1))
for a, b, lw in lines + open_lines:
    ax.plot([a[0], b[0]], [a[1], b[1]], color="black", lw=pt(lw), solid_capstyle="butt")
for c, r, lo, hi in open_arcs:
    ax.add_patch(Arc(c, 2 * r, 2 * r, theta1=lo, theta2=hi, color="black", lw=pt(0.18)))
for poly in furn:
    ax.add_patch(MPoly(np.array(poly), closed=True, fill=False, ec="black", lw=pt(0.13)))
for (a, b) in (arrow, arrow_e):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", lw=pt(0.18), color="black"))
for t in texts:
    ax.text(*t["p"], t["s"], fontsize=pt(t["h"]), rotation=t["angle"], ha=t["ha"], va=t["va"],
            fontweight="bold" if t["bold"] else "normal", family="DejaVu Sans")
for dd in dims:
    a, b, n, off = dd["a"], dd["b"], dd["n"], dd["off"]
    pa, pb = a + n * off, b + n * off
    for p, q in ((a, pa), (b, pb)):                                    # extension lines
        ax.plot([p[0] + n[0] * 0.2, q[0] + n[0] * 0.3], [p[1] + n[1] * 0.2, q[1] + n[1] * 0.3], color="black", lw=pt(0.13))
    ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="black", lw=pt(0.18))
    dvec = (pb - pa) / np.linalg.norm(pb - pa)
    for p in (pa, pb):                                                 # 45 degree ticks
        tk = (dvec + n) / np.linalg.norm(dvec + n) * 0.18
        ax.plot([p[0] - tk[0], p[0] + tk[0]], [p[1] - tk[1], p[1] + tk[1]], color="black", lw=pt(0.35))
    ang = math.degrees(math.atan2(dvec[1], dvec[0]))
    ang = ang - 180 if ang > 90 else (ang + 180 if ang <= -90 else ang)
    mid = (pa + pb) / 2 + n * 0.28
    ax.text(*mid, dd["label"], fontsize=pt(dd["size"]), rotation=ang, ha="center", va="center", rotation_mode="anchor")
# title block, legend, north arrow (outside the building)
tx, ty = bx1 - 9.0, Y0 + 1.0
ax.add_patch(MPoly([[tx, ty], [tx + 8.5, ty], [tx + 8.5, ty + 3.4], [tx, ty + 3.4]], fill=False, ec="black", lw=pt(0.35)))
ax.text(tx + 4.25, ty + 2.5, "GROUND FLOOR PLAN", fontsize=pt(4.0), ha="center", fontweight="bold")
ax.text(tx + 4.25, ty + 1.5, "SCALE 1:100   ALL DIMENSIONS IN METRES", fontsize=pt(2.2), ha="center")
ax.text(tx + 4.25, ty + 0.7, "WALLS: EXTERNAL 230 mm, INTERNAL 115 mm   FFL +0.45", fontsize=pt(2.0), ha="center")
nx, ny = bx0 - 1.0, Y0 + 2.0
ax.annotate("", xy=(nx, ny + 2.0), xytext=(nx, ny), arrowprops=dict(arrowstyle="-|>", lw=pt(0.35), color="black"))
ax.text(nx, ny + 2.4, "N", fontsize=pt(3.5), ha="center", fontweight="bold")
for ext in ("png", "pdf"):
    fig.savefig("school_ground_floor." + ext, dpi=300 if ext == "png" else None, facecolor="white")
plt.close(fig)

# ----------------------------------------------------------------------------- DXF (metres)
doc = ezdxf.new("R2018", setup=True)
doc.header["$INSUNITS"] = 6
doc.header["$MEASUREMENT"] = 1
for name, color in (("WALL", 7), ("DOOR", 3), ("WINDOW", 5), ("STAIR", 6), ("FURNITURE", 8), ("TEXT", 7), ("DIM", 1)):
    doc.layers.add(name, color=color)
msp = doc.modelspace()
for gpoly in geoms:
    h = msp.add_hatch(color=7, dxfattribs={"layer": "WALL"})
    h.paths.add_polyline_path(list(gpoly.exterior.coords)[:-1], is_closed=True)
    for hole in gpoly.interiors:
        h.paths.add_polyline_path(list(hole.coords)[:-1], is_closed=True)
    msp.add_lwpolyline(list(gpoly.exterior.coords), dxfattribs={"layer": "WALL"})
for a, b, lw in open_lines:
    msp.add_line(tuple(a), tuple(b), dxfattribs={"layer": "DOOR" if lw >= 0.35 else "WINDOW"})
for c, r, lo, hi in open_arcs:
    msp.add_arc(tuple(c), r, lo, hi, dxfattribs={"layer": "DOOR"})
for a, b, lw in lines:
    msp.add_line(tuple(a), tuple(b), dxfattribs={"layer": "STAIR"})
for a, b in (arrow, arrow_e):
    msp.add_line(tuple(a), tuple(b), dxfattribs={"layer": "STAIR"})
for poly in furn:
    msp.add_lwpolyline([tuple(p) for p in poly], close=True, dxfattribs={"layer": "FURNITURE"})
for t in texts:
    mt = msp.add_text(t["s"], height=t["h"] / 10.0, rotation=t["angle"], dxfattribs={"layer": "TEXT"})
    mt.set_placement(tuple(t["p"]), align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER)
for dd in dims:
    a, b, n = dd["a"], dd["b"], dd["n"]
    msp.add_aligned_dim(p1=tuple(a), p2=tuple(b), distance=dd["off"] * (1 if ((b - a)[0] * n[1] - (b - a)[1] * n[0]) > 0 else -1),
                        dxfattribs={"layer": "DIM"}, override={"dimtxt": 0.25, "dimasz": 0.18, "dimdec": 2,
                                                                    "dimtsz": 0.18}).render()
doc.saveas("school_ground_floor.dxf")

# ----------------------------------------------------------------------------- record of the true geometry
json.dump({"walls": [{"a": w["a"].tolist(), "b": w["b"].tolist(), "t": w["t"]} for w in walls],
           "openings": [{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in o.items()} for o in openings],
           "rooms": [{"name": r["name"], "clear_w": r["w"], "clear_d": r["d"]} for r in rooms],
           "stair": {"width": SW, "run": N_TREAD * TREAD, "risers": n_ris, "riser_mm": round(riser * 1000, 1),
                     "tread_mm": TREAD * 1000},
           "entrance_steps": {"width": SW_E, "run": N_E * T_E, "risers": N_E, "plinth": PLINTH},
           "overall": {"width": round(bx1 - bx0, 3), "depth": round(by1 - by0, 3)}},
          open("geometry.json", "w"), indent=1)
print("overall {:.2f} x {:.2f} m | walls {} | doors {} | windows {} | rooms {}".format(
    bx1 - bx0, by1 - by0, len(walls), sum(o["kind"] != "window" for o in openings),
    sum(o["kind"] == "window" for o in openings), len(rooms)))
