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


def _split_unsupported(walls, openings, segs, room, s, T):
    """A wall exists only where the model marked wall pixels or an opening. Joining pieces of a wall across a gap
    (to carry a door / window the model marked in it) must not close the rest of the gap: every stretch with no
    model wall pixels and no opening is cut out again (an open porch front stays open)."""
    wallpx = cv2.dilate((room == WALL).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    Hh, Ww = wallpx.shape
    min_gap = max(0.3, 2 * T * s)
    out_w, remap = [], {}
    for i, (w, sg) in enumerate(zip(walls, segs)):
        p0, p1 = np.array(sg["p0"], float), np.array(sg["p1"], float)
        n = max(2, int(np.hypot(*(p1 - p0))))
        t = np.linspace(0, 1, n)
        xs = np.clip(np.round(p0[0] + (p1[0] - p0[0]) * t).astype(int), 0, Ww - 1)
        ys = np.clip(np.round(p0[1] + (p1[1] - p0[1]) * t).astype(int), 0, Hh - 1)
        cov = wallpx[ys, xs].copy()
        a, b = np.array(w["start"], float), np.array(w["end"], float)
        L = float(np.linalg.norm(b - a)) or 1e-9
        for o in openings:
            if o["wall"] == i:
                cov |= np.abs(t * L - o["offset"]) <= o["width"] / 2 + 0.05
        # cut stretches longer than min_gap with no support
        cuts, k = [], 0
        while k < n:
            if not cov[k]:
                j = k
                while j < n and not cov[j]:
                    j += 1
                if (j - k) / n * L >= min_gap:
                    cuts.append((t[k], t[j - 1]))
                k = j
            else:
                k += 1
        if not cuts:
            remap[i] = [(len(out_w), 0.0, 1.0)]
            out_w.append(w)
            continue
        print("  wall {}: {:.2f} m with no model wall pixels and no opening left open (not walled over)".format(
            i, sum(c1 - c0 for c0, c1 in cuts) * L))
        pieces, f = [], 0.0
        for c0, c1 in cuts:
            pieces.append((f, c0)); f = c1
        pieces.append((f, 1.0))
        remap[i] = []
        for f0, f1 in pieces:
            if (f1 - f0) * L < 0.2:
                continue
            nw = dict(w)
            nw["start"] = [round(float(c), 4) for c in a + (b - a) * f0]
            nw["end"] = [round(float(c), 4) for c in a + (b - a) * f1]
            j0, j1 = list(w.get("joined", [False, False]))
            nw["joined"] = [j0 if f0 == 0.0 else False, j1 if f1 == 1.0 else False]
            if "ext" in w:
                nw["ext"] = [w["ext"][0] if f0 == 0.0 else 0.0, w["ext"][1] if f1 == 1.0 else 0.0]
            remap[i].append((len(out_w), f0, f1))
            out_w.append(nw)
    out_o = []
    for o in openings:
        L = float(np.linalg.norm(np.array(walls[o["wall"]]["end"]) - np.array(walls[o["wall"]]["start"]))) or 1e-9
        f = o["offset"] / L
        for k_, f0, f1 in remap.get(o["wall"], []):
            if f0 - 1e-6 <= f <= f1 + 1e-6:
                out_o.append(dict(o, wall=k_, offset=round(o["offset"] - f0 * L, 4)))
                break
    for k_, o in enumerate(out_o):
        o["id"] = k_
    return out_w, out_o


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
        t = (_snap(sg["t"] * s, WALL_THICKNESSES, 0.35) if (WALL_THICKNESSES and "standard_sizes" in ASSIST)
             else _round(sg["t"] * s, 0.005))                 # measured, not snapped to a standard
        a, b = [MX(sg["p0"][0]), MY(sg["p0"][1])], [MX(sg["p1"][0]), MY(sg["p1"][1])]
        walls.append({"start": [round(c, 4) for c in a], "end": [round(c, 4) for c in b], "thickness": t,
                      "height": WALL_HEIGHT, "joined": ["j0" in sg, "j1" in sg]})
    # extensions so solids close the corner (half the joined wall's thickness)
    for w, sg in zip(walls, segs):
        e = []
        for key in ("j0", "j1"):
            if key in sg:
                e.append(round(_snap(sg[key][1] * s, WALL_THICKNESSES, 0.35) / 2 if (WALL_THICKNESSES and "standard_sizes" in ASSIST)
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
        if o["kind"] == "door" and width < 0.5 and "widen_doors" in ASSIST:
            # the model found a door but marked only part of it (e.g. the hinge / arc): use a standard width
            # centred on the mark, as long as it fits on the wall
            L_w = float(np.linalg.norm(np.array(w["end"]) - np.array(w["start"])))
            std = DOOR_WIDTHS[len(DOOR_WIDTHS) // 2] if DOOR_WIDTHS else 0.9
            room_ = min(along, L_w - along) * 2 - 0.1
            if room_ < 0.6:
                continue
            width = min(std, room_)
        elif width < 0.3:
            continue                                        # too small to cut an opening
        if o["kind"] == "door":
            if "standard_sizes" in ASSIST:
                width = _snap(width, DOOR_WIDTHS, 0.09)
                width = width if width in DOOR_WIDTHS else _round(width, 0.05)
            else:
                width = _round(width, 0.01)                     # as marked
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
        if "merge_windows" in ASSIST and p and p["wall"] == o["wall"] and p["type"] == o["type"] == "window" and \
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
    walls, openings = _split_unsupported(walls, openings, segs, room, s, v["T"])
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
    floor0 = Polygon(plan["floor"]["polygon"]).buffer(0)
    _fill_floor_gaps(plan, v, room, T, mx, my)
    # stairs outside the building stay outside: the added floor never covers them
    st_out = [Polygon(st["outline"]).buffer(0.1, join_style=2) for st in plan.get("stairs", [])
              if st["kind"] in ("entrance",) or (st["kind"] == "terrace" and not st.get("sunken"))]
    if st_out:
        new = Polygon(plan["floor"]["polygon"]).buffer(0)
        added = new.difference(floor0).difference(unary_union(st_out))
        new = unary_union([floor0, added])
        if new.geom_type != "Polygon":
            new = max(new.geoms, key=lambda g: g.area)
        new = Polygon(new.exterior).simplify(0.03)
        plan["floor"]["polygon"] = [[round(float(x), 4), round(float(y), 4)] for x, y in list(new.exterior.coords)[:-1]]


def _fill_floor_gaps(plan, v, room, T, mx, my):
    from shapely.geometry import Polygon, LineString, Point
    from shapely.ops import unary_union
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
            # a door / window gap: the same wall carries on further along the line -> never close it
            def parallel_ahead(j):
                wj = plan["walls"][j]
                dj = np.subtract(wj["end"], wj["start"]); dj = dj / max(np.linalg.norm(dj), 1e-9)
                return abs(np.dot(dj, d)) > 0.98 and lines0[j].distance(LineString([p, p + d * sgn * 6.0])) < 0.4
            if any(parallel_ahead(j) for j in range(len(lines0)) if j != i):
                continue
            mid = gap.interpolate(0.5, normalized=True)
            if dist_ > 0.3 and floor.distance(mid) > 0.3 and gap.difference(floor.buffer(-0.15)).length > 0.8 * gap.length:
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

    # ---- floor: structural slab + finish flooring on top, both open where steps go down to the ground
    from ifcopenshell.api import style as ifcstyle
    from shapely.geometry import Polygon as _Pg, LineString as _Ls
    from shapely.ops import split as _split, unary_union as _uu
    fpoly = _Pg(plan["floor"]["polygon"]).buffer(0)
    for h in plan["floor"].get("holes", []):
        fpoly = fpoly.difference(_Pg(h).buffer(0))

    def no_voids(g):                                 # IFC slab outlines here are single loops: cut through each void
        out, todo = [], list(getattr(g, "geoms", [g]))
        while todo:
            q = todo.pop()
            if q.geom_type != "Polygon" or q.area < 0.05:
                continue
            if not q.interiors:
                out.append(q)
                continue
            cx = _Pg(q.interiors[0]).centroid.x
            y0, y1 = q.bounds[1] - 1, q.bounds[3] + 1
            todo += list(_split(q, _Ls([(cx, y0), (cx, y1)])).geoms)
        return out

    concrete = material.add_material(f, name="Concrete slab", category="concrete")
    tiles = material.add_material(f, name="Vitrified floor tiles 600 x 600", category="tile")

    def colour(mat, rgb, name):
        st = ifcstyle.add_style(f, name=name)
        ifcstyle.add_surface_style(f, style=st, ifc_class="IfcSurfaceStyleShading",
                                   attributes={"SurfaceColour": {"Name": None, "Red": rgb[0], "Green": rgb[1], "Blue": rgb[2]}})
        ifcstyle.assign_material_style(f, material=mat, style=st, context=body)
    colour(concrete, (0.62, 0.62, 0.6), "Concrete")
    colour(tiles, (0.86, 0.78, 0.64), "Floor tiles")
    FINISH = 0.02
    pieces_ = no_voids(fpoly)
    for k, q in enumerate(pieces_):
        poly = [(x, -y) for x, y in list(q.exterior.coords)[:-1]]
        slab = root.create_entity(f, ifc_class="IfcSlab", name="Ground floor slab" + (" {}".format(k + 1) if len(pieces_) > 1 else ""),
                                  predefined_type="FLOOR")
        geometry.assign_representation(f, product=slab, representation=geometry.add_slab_representation(
            f, context=body, depth=SLAB - FINISH, polyline=poly))
        mm = np.eye(4); mm[2, 3] = -SLAB
        geometry.edit_object_placement(f, product=slab, matrix=mm)
        material.assign_material(f, products=[slab], material=concrete)
        spatial.assign_container(f, products=[slab], relating_structure=storey)
        fin = root.create_entity(f, ifc_class="IfcCovering", name="Flooring - vitrified tiles" + (" {}".format(k + 1) if len(pieces_) > 1 else ""),
                                 predefined_type="FLOORING")
        geometry.assign_representation(f, product=fin, representation=geometry.add_slab_representation(
            f, context=body, depth=FINISH, polyline=poly))
        mm = np.eye(4); mm[2, 3] = -FINISH
        geometry.edit_object_placement(f, product=fin, matrix=mm)
        material.assign_material(f, products=[fin], material=tiles)
        spatial.assign_container(f, products=[fin], relating_structure=storey)
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
        if "written_m" in r and r["status"] == "superseded":
            print("  ..  {:<15} {}  {:<22} written {:>7.3f} m  (read; one scale is used for the whole sheet)".format(
                r.get("kind", "dimension"), r["axis"], r["text"][:22], r["written_m"]))
        elif "written_m" in r:
            mark = "OK " if r["status"] == "ok" else ("-- " if r["status"].startswith("rejected") else "!! ")
            print("  {} {:<15} {}  {:<22} written {:>7.3f} m  model {:>7.3f} m  error {:+7.1f} mm{}".format(
                mark, r.get("kind", "dimension"), r["axis"], r["text"][:22], r["written_m"], r["model_m"],
                r["error_mm"], "  (rejected: conflicts)" if r["status"].startswith("rejected") else ""))
        else:
            print("  ?? {}: {} {}".format(r["axis"], r["text"], r.get("status", "")))
    if any(r.get("status") == "superseded" for r in pr["report"]):
        print("  scale: {} written dimensions read; one scale for the whole sheet from the {}".format(
            pr["dimensions_read"], pr["scale_source"]))
    else:
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
    rep = [dict(r, status="superseded") if "written_m" in r else r for r in report]
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
    plan["stairs"], plan["levels"] = build_stairs(v, toks, room, plan, original)
    if "fill_floor" in ASSIST:
        fill_floor_gaps(plan, v, room)
    plan["floor"]["holes"] = open_ground(plan)
    plan["coverage"] = round(v["coverage"], 4)
    if MAKE_FURNITURE:
        items = icon_fixtures(icon, v["T"]) + detector_furniture(img, plan["mapping"]["scale_m_per_px"])
        dropped = 0
        if "furniture_filter" in ASSIST:
            items, dropped = filter_by_room(items, room, v["T"])
        if dropped:
            print("  furniture: {} detections rejected by room check (wrong room for the object)".format(dropped))
        plan["furniture"] = build_furniture(items, plan)
        from shapely.geometry import Point as _Pt, Polygon as _Pg
        fp_ = _Pg(plan["floor"]["polygon"]).buffer(0.3)
        inside_ = [fu for fu in plan["furniture"] if "furniture_filter" not in ASSIST or fp_.contains(_Pt(*fu["center"]))]
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
