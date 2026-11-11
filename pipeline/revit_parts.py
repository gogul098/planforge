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
        if sg["o"] == "h":
            y = MY(sg["c"])
            x0 = MX(sg["j0"][0]) if "j0" in sg else MX(sg["a0"])
            x1 = MX(sg["j1"][0]) if "j1" in sg else MX(sg["a1"])
            a, b = [x0, y], [x1, y]
        else:
            x = MX(sg["c"])
            y0 = MY(sg["j0"][0]) if "j0" in sg else MY(sg["a0"])
            y1 = MY(sg["j1"][0]) if "j1" in sg else MY(sg["a1"])
            a, b = [x, y0], [x, y1]
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
    closed = union.buffer(1.0, join_style=2).buffer(-1.0, join_style=2)
    cpolys = [closed] if closed.geom_type == "Polygon" else list(closed.geoms)
    footprint = Polygon(max(cpolys, key=lambda p: p.area).exterior)
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
        if sg["o"] == "h":
            u0, u1 = MX(o["u0"]), MX(o["u1"])
            centre = [(u0 + u1) / 2, w["start"][1]]
            along = centre[0] - w["start"][0]
        else:
            u0, u1 = MY(o["u0"]), MY(o["u1"])
            centre = [w["start"][0], (u0 + u1) / 2]
            along = centre[1] - w["start"][1]
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
    for hole in main.interiors:
        poly = Polygon(hole)
        if poly.area < 1.0:
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

    # stairs: run along the long side, starting at the end nearest the "UP" label
    stairs = []
    up = [t for t in labels if t["text"].strip().upper() in ("UP", "UP.")]
    for st in v["stairs"]:
        x0, y0, x1, y1 = MX(st["x0"]), MY(st["y0"]), MX(st["x1"]), MY(st["y1"])
        short, long_ = sorted([x1 - x0, y1 - y0])
        if short < 0.7 or long_ < 1.5 or short * long_ < 1.5:
            continue                                     # too small to be a real stair -> false positive
        along_x = (x1 - x0) >= (y1 - y0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if along_x:
            p0, p1, width = [x0, cy], [x1, cy], y1 - y0
        else:
            p0, p1, width = [cx, y0], [cx, y1], x1 - x0
        if up:
            ux, uy = MX(up[0]["x"]), MY(up[0]["y"])
            if np.hypot(ux - p1[0], uy - p1[1]) < np.hypot(ux - p0[0], uy - p0[1]):
                p0, p1 = p1, p0
        n = int(np.clip(round(WALL_HEIGHT / STAIR_RISE), 3, 30))
        run = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        stairs.append({"start": [round(c, 4) for c in p0], "end": [round(c, 4) for c in p1],
                       "width": round(width, 4), "risers": n, "riser": round(WALL_HEIGHT / n, 4),
                       "tread": round(run / max(n - 1, 1), 4),
                       "straight_run_fits": bool(run / max(n - 1, 1) >= 0.22),
                       "outline": [[round(x0, 4), round(y0, 4)], [round(x1, 4), round(y0, 4)],
                                   [round(x1, 4), round(y1, 4)], [round(x0, 4), round(y1, 4)]],
                       "direction_from_UP_label": bool(up)})

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
        a, b = np.array(s["start"]), np.array(s["end"])
        L = float(np.linalg.norm(b - a))
        d = (b - a) / max(L, 1e-9)
        n = np.array([-d[1], d[0]])
        h = s["width"] / 2
        for i in range(s["risers"]):
            u0, u1 = L * i / s["risers"], L * (i + 1) / s["risers"]
            q = [a + d * u0 - n * h, a + d * u1 - n * h, a + d * u1 + n * h, a + d * u0 + n * h]
            yield "stairs", q, 0.0, s["riser"] * (i + 1)


def prism(corners, z0, z1):
    c = [np.asarray(p, float) for p in corners]
    v = np.array([[p[0], z0, p[1]] for p in c] + [[p[0], z1, p[1]] for p in c])
    f = [[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6]]
    for i in range(4):
        j = (i + 1) % 4
        f += [[i, 4 + i, j], [j, 4 + i, 4 + j]]
    f = np.array(f)
    t = v[f]
    if np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() < 0:
        f = f[:, ::-1]
    return v, f


COLORS = {"wall": [235, 232, 225, 255], "door": [150, 95, 55, 255], "glass": [140, 190, 235, 140],
          "stairs": [190, 190, 190, 255], "floor": [205, 185, 150, 255],
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
        x0, x1 = (sg["a0"], sg["a1"]) if sg["o"] == "h" else (sg["c"] - sg["t"] / 2, sg["c"] + sg["t"] / 2)
        y0, y1 = (sg["c"] - sg["t"] / 2, sg["c"] + sg["t"] / 2) if sg["o"] == "h" else (sg["a0"], sg["a1"])
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fc=(0.9, 0.1, 0.1, 0.45), ec="r", lw=0.6))
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
