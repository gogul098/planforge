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
    for r in (1.0, 1.5, 2.0, 3.0):
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
        if width < (0.5 if o["kind"] == "door" else 0.3):
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


def build_glb(plan, path, extra=None):
    import trimesh
    from shapely.geometry import Polygon
    parts = {}
    for cat, quad, z0, z1 in pieces(plan):
        vv, ff = prism(quad, z0, z1)
        parts.setdefault(cat, []).append(trimesh.Trimesh(vv, ff, process=False))
    fl = trimesh.creation.extrude_polygon(Polygon([(x, -y) for x, y in plan["floor"]["polygon"]]), SLAB)
    fl.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    fl.apply_translation([0, -SLAB, 0])
    parts["floor"] = [fl]
    pl = float(plan.get("levels", {}).get("plinth", 0.0))
    if pl > 0.01:                       # building raised on a plinth, entrance steps come down to the ground
        fp = Polygon([(x, -y) for x, y in plan["floor"]["polygon"]]).buffer(0)
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
    v = vectorise(room, icon)
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
    plan["coverage"] = round(v["coverage"], 4)
    if MAKE_FURNITURE:
        items = icon_fixtures(icon, v["T"]) + detector_furniture(img, plan["mapping"]["scale_m_per_px"])
        items, dropped = filter_by_room(items, room, v["T"])
        if dropped:
            print("  furniture: {} detections rejected by room check (wrong room for the object)".format(dropped))
        plan["furniture"] = build_furniture(items, plan)
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
