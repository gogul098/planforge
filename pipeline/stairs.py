import re
# ----------------------------------------------------------------------------- 3b. stairs, entrance steps, stepped areas
# Every stair gets: kind (entrance / interior / terrace), shape (straight / flared / irregular), the number of
# risers, riser height, tread depth, width, run, base and top level, and one solid block per tread.
# Each value says where it came from, best source first:
#   written on the plan (dimension, "12R @ 150", "+0.45 PLINTH")  >  counted from the drawn tread lines
#   >  your setting (PLINTH_HEIGHT)  >  standard assumption (flagged in "warnings")
_N_RISERS = re.compile(r"\b(\d{1,2})\s*(?:NOS?\.?\s*)?(?:R|RISERS?)\b", re.I)
_N_STEPS = re.compile(r"\b(\d{1,2})\s*(?:NOS?\.?\s*)?(?:STEPS?|TREADS?)\b", re.I)
_RISER_MM = re.compile(r"(?:\bR(?:ISER)?\s*[@=:]?\s*|@\s*)(\d{2,3})\b", re.I)
_LEVEL = re.compile(r"(?:^|[\s(])\+\s*(\d+(?:\.\d+)?)")
_LEVEL_WORDS = re.compile(r"PLINTH|\bP\.?L\b|F\.?F\.?L|FLOOR\s*LEVEL|LVL", re.I)


def _raw(t):
    return t.get("raw") or t.get("text") or ""


def _plinth_from_levels(toks):
    """'+0.45', '+450', 'PLINTH LVL +0.60' -> plinth height in metres (or None)."""
    cands = []
    for t in toks:
        raw = _raw(t)
        for m in _LEVEL.finditer(" " + raw):
            v = float(m.group(1))
            v = v if v <= 3 else v / 1000.0
            if 0.05 <= v <= 2.0:
                cands.append((bool(_LEVEL_WORDS.search(raw)), v))
    if not cands:
        return None
    worded = [v for w, v in cands if w]
    return float(np.median(worded or [v for _, v in cands]))


def _stair_notes(toks, cx, cy, r_px):
    """Notes written next to a stair: riser count, riser height, UP / DN labels."""
    near = [t for t in toks if np.hypot(t["x"] - cx, t["y"] - cy) <= r_px]
    text = " ".join(dict.fromkeys(_raw(t) for t in near))
    out = {}
    m = _N_RISERS.search(text) or _N_STEPS.search(text)
    if m and 2 <= int(m.group(1)) <= 40:
        out["risers"] = int(m.group(1))
    for m in _RISER_MM.finditer(text):
        if 100 <= int(m.group(1)) <= 220:
            out["riser"] = int(m.group(1)) / 1000.0
    out["up"] = [(t["x"], t["y"]) for t in near if _raw(t).strip().upper().rstrip(".") == "UP"]
    out["dn"] = [(t["x"], t["y"]) for t in near if _raw(t).strip().upper().rstrip(".") in ("DN", "DOWN")]
    return out


def _zoom_labels(original, s, x0, y0, x1, y1):
    """Short words like UP / DN are easy to miss on the whole sheet: re-read the area around a stair, zoomed."""
    X0, Y0, X1, Y1 = int(x0 * s), int(y0 * s), int(x1 * s), int(y1 * s)
    crop = original[Y0:Y1, X0:X1]
    if crop.size == 0:
        return {}
    z = 2.0 if max(crop.shape[:2]) < 1500 else 1.0
    big = cv2.resize(crop, None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC) if z != 1 else crop
    if big.ndim == 2:
        big = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    out = {"up": [], "dn": []}
    for t in ocr_tokens(big):
        w = _raw(t).strip().upper().rstrip(".")
        pt = ((t["x"] / z + X0) / s, (t["y"] / z + Y0) / s)
        if w == "UP":
            out["up"].append(pt)
        elif w in ("DN", "DOWN"):
            out["dn"].append(pt)
    return out


def _written_length(toks, cx, cy, r_px, measured, tol=0.12):
    """A written dimension near the stair that matches a measured length (m) -> (value, text) or None."""
    best = None
    for t in toks:
        if t["kind"] != "num" or np.hypot(t["x"] - cx, t["y"] - cy) > r_px:
            continue
        for v in (t["value"], t["value"] / 100.0, t["value"] / 1000.0):
            if v > 0 and abs(v - measured) / measured <= tol:
                err = abs(v - measured) / measured
                if best is None or err < best[2]:
                    best = (v, _raw(t), err)
    return None if best is None else best[:2]


def _window(shape, x0, y0, x1, y1, pad):
    H, W = shape
    return max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)


def _count_treads(comp, box, D_model, original, s, m_per_opx):
    """Count treads between the drawn lines inside the stair (full-resolution drawing).
    comp: bool crop of the stair (model px) at box=(ox, oy); D_model: distance to the HIGH side, same crop.
    Returns (n, label image low->high 1..n, (X0, Y0)) in original px, or None."""
    ox, oy = box
    H, W = comp.shape
    X0, Y0 = int(ox * s), int(oy * s)
    X1, Y1 = min(original.shape[1], int(np.ceil((ox + W) * s))), min(original.shape[0], int(np.ceil((oy + H) * s)))
    if X1 - X0 < 8 or Y1 - Y0 < 8:
        return None
    crop = original[Y0:Y1, X0:X1]
    size = (X1 - X0, Y1 - Y0)
    region = cv2.resize(comp.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    D = cv2.resize(D_model.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    ink = (gray < float(np.clip(thr, 100, 200))).astype(np.uint8)   # Otsu can collapse on near-white crops
    ink = cv2.dilate(ink, np.ones((2, 2), np.uint8)) > 0              # close tiny breaks in tread lines
    free = region & ~ink
    n, lab, st, _ = cv2.connectedComponentsWithStats(free.astype(np.uint8), connectivity=4)
    min_area = max(20, 0.02 / m_per_opx ** 2)                         # smaller than a 0.2 x 0.1 m tread = noise
    comps = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < min_area:
            continue
        d = D[lab == i]
        comps.append((float(np.median(d)), float(np.percentile(d, 20)), float(np.percentile(d, 80)), i))
    if len(comps) < 2:
        return None
    # a strip along the side (inside the model's stair area, outside the drawn stair) joins many treads:
    # far too long in the climbing direction to be one tread -> not a tread
    spans = np.array([c[2] - c[1] for c in comps])
    typical = float(np.median(spans))
    comps = [c for c, sp in zip(comps, spans) if sp <= 2.5 * max(typical, 1.0)]
    if len(comps) < 2:
        return None
    comps.sort()
    # pieces of one tread split by the walking line / arrow sit at the same distance from the top -> one tread
    groups = [[comps[0]]]
    for c in comps[1:]:
        g = groups[-1]
        lo, hi = min(x[1] for x in g), max(x[2] for x in g)
        ov = min(hi, c[2]) - max(lo, c[1])
        if ov > 0.5 * min(hi - lo, c[2] - c[1]) + 1e-6:
            g.append(c)
        else:
            groups.append([c])
    k = len(groups)
    # count by pitch over the whole run, so treads merged or lost in the middle do not shrink the count
    if k >= 3:
        cen = np.array([np.median([c[0] for c in g]) for g in groups])
        pitch = float(np.median(np.diff(cen)))
        dr = D[region]
        span = float(np.percentile(dr, 99) - np.percentile(dr, 1)) if dr.size else 0.0
        if pitch > 0:
            k_pitch = int(round(span / pitch))
            if k_pitch > k:
                k_full = k_pitch
            else:
                k_full = k
        else:
            k_full = k
    else:
        k_full = k
    covered = sum(st[c[3], cv2.CC_STAT_AREA] for g in groups for c in g)
    if not (2 <= k_full <= 40) or covered < 0.25 * region.sum():
        return None
    out = np.zeros(lab.shape, np.int32)
    for j, g in enumerate(groups):                                   # groups: high (near anchor) -> low
        for c in g:
            out[lab == c[3]] = k - j                                 # label 1 = lowest step
    # give the line pixels to the nearest tread so the treads tile the stair
    _, idx = ndimage.distance_transform_edt(out == 0, return_indices=True)
    filled = out[idx[0], idx[1]]
    filled[~(region | (out > 0))] = 0
    return k_full, filled, (X0, Y0)


def _label_polys(labels, X0, Y0, s, mx, my, m_per_opx):
    """Tread label image (original px) -> list of shapely polygons in metres, index 0 = lowest."""
    from shapely.geometry import Polygon
    polys = []
    eps = max(1.0, 0.03 / m_per_opx)
    for k in range(1, labels.max() + 1):
        cs, _ = cv2.findContours((labels == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            polys.append(None)
            continue
        c = cv2.approxPolyDP(max(cs, key=cv2.contourArea), eps, True)[:, 0]
        if len(c) < 3:
            polys.append(None)
            continue
        p = Polygon([(float(mx((x + X0) / s)), float(my((y + Y0) / s))) for x, y in c]).buffer(0)
        if p.geom_type == "MultiPolygon":
            p = max(p.geoms, key=lambda g: g.area)
        polys.append(p if p.area > 0.01 else None)
    return polys


def _slice(poly, centre, d, L, k):
    """Cut a stair outline into k equal treads along the run direction d (low -> high)."""
    from shapely.geometry import Polygon
    nrm = np.array([-d[1], d[0]])
    big = 100.0
    out = []
    for i in range(k):
        u0, u1 = -L / 2 + L * i / k, -L / 2 + L * (i + 1) / k
        band = Polygon([centre + d * u0 - nrm * big, centre + d * u1 - nrm * big,
                        centre + d * u1 + nrm * big, centre + d * u0 + nrm * big])
        p = poly.intersection(band)
        if p.geom_type != "Polygon":
            ps = [g for g in getattr(p, "geoms", []) if g.geom_type == "Polygon"]
            p = max(ps, key=lambda g: g.area) if ps else None
        out.append(p if p is not None and p.area > 1e-4 else None)
    return out


def _poly_list(p):
    return [[round(float(x), 4), round(float(y), 4)] for x, y in list(p.exterior.coords)[:-1]]


def _chain_flights(infos):
    """Interior flights that touch each other (a split / dog-leg stair: main flight -> landing -> flights)
    share one storey height: the longest flight goes from the floor to the landing, the others from the
    landing up. Each flight is turned so it climbs away from the one below."""
    inter = [i for i in infos if i["kind"] == "interior"]
    seen = set()
    for a in range(len(inter)):
        if a in seen:
            continue
        group, todo = [], [a]
        while todo:
            k = todo.pop()
            if k in seen:
                continue
            seen.add(k)
            group.append(k)
            todo += [j for j in range(len(inter)) if j not in seen and inter[k]["hull"].distance(inter[j]["hull"]) < 0.4]
        if len(group) < 2:
            continue
        fl = [inter[k] for k in group]
        for f_ in fl:                                     # unknown counts: from the run (about 280 mm treads)
            if not f_["n"]:
                f_["n"] = max(2, int(round(f_["run"] / 0.28)) + 1)
                f_["sources"]["risers"] = "estimated from the run length"
                f_["warnings"].append("risers not written or readable: estimated from the run")
        main = max(fl, key=lambda f_: f_["run"])
        uppers = [f_ for f_ in fl if f_ is not main]
        ucentre = np.mean([u["centre"] for u in uppers], 0)
        ends = lambda f_: (f_["centre"] - f_["d"] * f_["run"] / 2, f_["centre"] + f_["d"] * f_["run"] / 2)
        lo, hi = ends(main)
        if np.linalg.norm(lo - ucentre) < np.linalg.norm(hi - ucentre):  # main climbs towards the landing
            main["d"] = -main["d"]
        main["direction"] = "towards the landing and the upper flights"
        top_main = main["centre"] + main["d"] * main["run"] / 2
        for u in uppers:
            # an upper flight climbs AWAY from the landing: on a split stair the wings turn sideways,
            # left one to the left, right one to the right
            off = u["centre"] - top_main
            nrm = np.array([-u["d"][1], u["d"][0]])
            if abs(np.dot(nrm, off)) > abs(np.dot(u["d"], off)) and len(uppers) > 1:
                u["d"] = nrm                               # the flight runs across what was taken as its width
                u["run"], u["width"] = u["width"], u["run"]
                u["width_low"] = u["width_high"] = u["width"]
            if np.dot(u["d"], off) < 0:
                u["d"] = -u["d"]                          # upper flights start at the landing
            u["direction"] = "from the landing up, away from the main flight"
        n_up = max(u["n"] for u in uppers)                # parallel flights (split stair) climb together
        r = WALL_HEIGHT / (main["n"] + n_up)
        main.update(chain_base=0.0, chain_total=r * main["n"], chain_size=len(fl))
        for u in uppers:
            u.update(chain_base=r * main["n"], chain_total=WALL_HEIGHT - r * main["n"], chain_size=len(fl))
            u["n"] = n_up
        for f_ in fl:
            f_["warnings"] = [w_ for w_ in f_["warnings"] if not w_.startswith("direction not marked")]


def _closed_box(run, d, n, ink, thick, Tk):
    """Furniture symbols (sofas, beds, tables) are nested rectangles: their parallel lines are tied together
    at both ends by short cross lines. Step lines are not - they stop at an open edge, a wall or a stringer on
    at most one side. Look along both end edges of the group (wall ink excluded): drawn on both -> a box."""
    H, W = ink.shape
    free = ink & (1 - cv2.dilate(thick, np.ones((5, 5), np.uint8)))
    offs = [q[0] for q in run]
    o0, o1 = min(offs), max(offs)
    if o1 - o0 < 1:
        return False
    cover = []
    for u in (float(np.median([q[1] for q in run])), float(np.median([q[2] for q in run]))):
        hits, tot = 0, 0
        for o in np.linspace(o0, o1, max(8, int(o1 - o0))):
            best = 0
            for du in (-2, -1, 0, 1, 2):                       # the cross line may sit a pixel or two off
                x, y = (d * (u + du) + n * o).round().astype(int)
                if 0 <= x < W and 0 <= y < H and free[y, x]:
                    best = 1
                    break
            hits += best
            tot += 1
        cover.append(hits / max(tot, 1))
    width = float(np.median([q[2] - q[1] for q in run]))
    spacing = (o1 - o0) / max(len(run) - 1, 1)
    # a flight is wide for its tread spacing (a 1 m stair with 280 mm treads is already ~3.5x);
    # a furniture symbol is a small box: both ends drawn AND narrow
    return min(cover) >= 0.3 and width < 3.0 * spacing


def drawn_step_regions(original, shape, have_px, wall_px, T):
    """Steps the model did not mark, read from the drawing: 4+ parallel thin lines, evenly spaced about a tread
    apart, overlapping each other (stepped seating, sunken areas, unmarked flights).
    Returns stair entries like vectorise() makes, in model px."""
    if original is None:
        return []
    g = original if original.ndim == 2 else cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    k = g.shape[1] / shape[1]                                  # drawing px per model px
    ink = (g < 140).astype(np.uint8)
    thick = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))
    thin = (ink & (1 - cv2.dilate(thick, np.ones((3, 3), np.uint8)))) * 255
    segs = cv2.HoughLinesP(thin, 1, np.pi / 360, 25, minLineLength=int(max(20, 2.5 * T * k)), maxLineGap=3)
    if segs is None:
        return []
    segs = segs.reshape(-1, 4).astype(float)
    ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])) % 180
    out, used = [], np.zeros(len(segs), bool)
    model_px = have_px.copy()                                 # stairs the model marked
    found_px = np.zeros(shape, bool)                          # stepped areas found here
    for a0 in np.unique(np.round(ang / 3) * 3):
        idx = np.where((np.abs(((ang - a0) + 90) % 180 - 90) <= 3) & ~used)[0]
        if len(idx) < 4:
            continue
        d = np.array([np.cos(np.radians(a0)), np.sin(np.radians(a0))])
        n = np.array([-d[1], d[0]])
        # one entry per drawn line: merge Hough duplicates (same offset, overlapping)
        lines_ = []
        for i in sorted(idx, key=lambda i: (segs[i, :2] @ n)):
            off = float(((segs[i, :2] + segs[i, 2:]) / 2) @ n)
            u0, u1 = sorted([float(segs[i, :2] @ d), float(segs[i, 2:] @ d)])
            for L_ in lines_:
                if abs(L_[0] - off) <= 2.5 and min(L_[2], u1) - max(L_[1], u0) > -3:
                    L_[1], L_[2] = min(L_[1], u0), max(L_[2], u1)
                    break
            else:
                lines_.append([off, u0, u1])
        lines_.sort()
        lo_s, hi_s = 0.6 * T * k, 2.6 * T * k
        i = 0
        while i < len(lines_) - 3:
            run = [lines_[i]]
            for L_ in lines_[i + 1:]:
                gap = L_[0] - run[-1][0]
                ov = min(L_[2], run[-1][2]) - max(L_[1], run[-1][1])
                if gap < lo_s * 0.5:
                    continue                                    # a double line, not the next tread
                ok_ov = ov >= 0.5 * min(L_[2] - L_[1], run[-1][2] - run[-1][1])
                med_len = float(np.median([q[2] - q[1] for q in run]))
                ok_ov = ok_ov and 0.4 * med_len <= (L_[2] - L_[1]) <= 2.5 * med_len   # tread lines, not dimensions
                if len(run) >= 2:
                    med = float(np.median(np.diff([q[0] for q in run])))
                    one = abs(gap - med) <= 0.35 * med
                    two = abs(gap - 2 * med) <= 0.35 * med       # the line finder missed one tread line
                    if ok_ov and (one or two):
                        run.append(L_)
                        continue
                    if gap > 2.5 * med:
                        break
                    continue
                if lo_s <= gap <= hi_s and ok_ov:
                    run.append(L_)
                elif gap > hi_s:
                    break
            if len(run) >= 4 and _closed_box(run, d, n, ink, thick, k * T):
                print("  drawn lines rejected as steps: they close into boxes (furniture), not open treads")
                i += len(run)
                continue
            if len(run) >= 4:
                pts = []
                for off, u0, u1 in run:
                    pts += [d * u0 + n * off, d * u1 + n * off]
                hull = cv2.convexHull(np.array(pts, np.float32) / k)[:, 0]
                m = np.zeros(shape, np.uint8)
                cv2.fillPoly(m, [np.round(hull).astype(np.int32)], 1)
                m = m > 0
                area = m.sum()
                # skip what the model already marked, and duplicates of an arm already found (the other arm
                # of an L overlaps it only at the corner)
                if area and (m & model_px).sum() < 0.3 * area and (m & found_px).sum() < 0.8 * area \
                        and (m & wall_px).sum() < 0.25 * area:
                    ys, xs = np.nonzero(m)
                    x0, y0 = xs.min(), ys.min()
                    out.append({"x0": float(x0), "y0": float(y0), "x1": float(xs.max() + 1), "y1": float(ys.max() + 1),
                                "ox": int(x0), "oy": int(y0), "mask": m[y0:ys.max() + 1, x0:xs.max() + 1].copy(),
                                "drawn": True, "n_lines": len(run), "line_dir": (float(d[0]), float(d[1]))})
                    found_px = found_px | m
                i += len(run)
            else:
                i += 1
    return out


def _believable_drawn_steps(st, toks, mpp, pad):
    """Steps found only in the drawing must look like a place people walk: at least 1.2 m wide,
    unless the drawing says so itself (an UP / DN label or a level mark next to them).
    Furniture symbols (a sofa is ~1 m) fail this."""
    ys, xs = np.nonzero(st["mask"])
    d = np.array(st.get("line_dir", (1.0, 0.0)))
    pts = np.column_stack([xs + st["ox"], ys + st["oy"]]).astype(float)
    width_m = float(np.ptp(pts @ d)) * mpp
    cx, cy = pts.mean(0)
    evidence = False
    for t in toks or []:
        if abs(t["x"] - cx) > pad + 0.5 * np.ptp(pts[:, 0]) or abs(t["y"] - cy) > pad + 0.5 * np.ptp(pts[:, 1]):
            continue
        w = _raw(t).strip().upper().rstrip(".")
        if w in ("UP", "DN", "DOWN") or re.match(r"^[+\-±]\s*\d+[.,]\d+", w):
            evidence = True
            break
    if width_m < 1.2 and not evidence:
        print("  drawn lines rejected as steps: only {:.2f} m wide and nothing on the plan says they are steps "
              "(furniture?)".format(width_m))
        return False
    return True


def build_stairs(v, toks, room, plan, original=None):
    """Detected stair regions -> exact stairs, entrance steps and stepped areas (plan.json 'stairs' + 'levels')."""
    from shapely.geometry import Point, Polygon
    mx, my = make_mapper(plan["mapping"], "x"), make_mapper(plan["mapping"], "y")
    MX, MY = (lambda q: float(mx(q))), (lambda q: float(my(q)))
    mpp = float(plan["mapping"]["scale_m_per_px"])
    s = (original.shape[1] / room.shape[1]) if original is not None else None
    m_per_opx = mpp / s if s else None
    footprint = Polygon(plan["floor"]["polygon"]).buffer(0) if plan.get("floor") else None
    T = v["T"]
    pad = int(max(3 * T, 1.5 / mpp))
    stair_px = np.zeros(room.shape, bool)
    for st in v["stairs"]:
        if "mask" in st:
            stair_px[st["oy"]:st["oy"] + st["mask"].shape[0], st["ox"]:st["ox"] + st["mask"].shape[1]] |= st["mask"]
    walls_px = room == WALL
    extra = []
    if "drawn_steps" in ASSIST:
        extra = drawn_step_regions(original, room.shape, stair_px, walls_px, v["T"])
        extra = [st for st in extra if _believable_drawn_steps(st, toks, mpp, pad)]
    if extra:
        print("  steps read from the drawing (not marked by the model): {} areas".format(len(extra)))
        for st in extra:
            stair_px[st["oy"]:st["oy"] + st["mask"].shape[0], st["ox"]:st["ox"] + st["mask"].shape[1]] |= st["mask"]
    all_st = list(v["stairs"]) + extra
    indoor = ~np.isin(room, (0, 1)) & (room != WALL) & ~stair_px
    # open ground INSIDE the building, as the model marks it (background / outdoor within the outline): a sandpit,
    # sunken court, amphitheatre floor. Steps beside it go down into it - the model decides, nothing is guessed.
    open_inside, oi_lab = None, None
    if footprint is not None and not footprint.is_empty:
        from shapely import contains_xy
        op = np.isin(room, (0, 1)) & ~walls_px & ~stair_px
        ys_o, xs_o = np.nonzero(op)
        if len(xs_o):
            inside = contains_xy(footprint.buffer(-0.05), mx(xs_o.astype(float)), my(ys_o.astype(float)))
            open_inside = np.zeros(room.shape, np.uint8)
            open_inside[ys_o[inside], xs_o[inside]] = 1
            n_oi, oi_lab, oi_st, _ = cv2.connectedComponentsWithStats(open_inside)
            min_px = 1.0 / (mpp * mpp)                                # under 1 m2: a speck, not a pit
            for l_ in range(1, n_oi):
                if oi_st[l_, cv2.CC_STAT_AREA] < min_px:
                    open_inside[oi_lab == l_] = 0
                    oi_lab[oi_lab == l_] = 0
            open_inside = open_inside > 0

    infos = []
    for st in all_st:
        if "mask" not in st:
            continue
        comp, ox, oy = st["mask"], st["ox"], st["oy"]
        h, w = comp.shape
        ys, xs = np.nonzero(comp)
        pts = np.stack([xs + ox, ys + oy], 1).astype(np.float32)
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)
        hull = cv2.convexHull(pts)[:, 0]
        area = float(comp.sum())
        rect_sol = area / max(rect[1][0] * rect[1][1], 1.0)
        hull_sol = area / max(cv2.contourArea(hull), 1.0)
        shape = "straight" if rect_sol >= 0.85 else ("flared" if hull_sol >= 0.9 else "irregular")
        # metres
        hull_m = Polygon([(MX(x), MY(y)) for x, y in hull]).buffer(0)
        box_m = np.array([[MX(x), MY(y)] for x, y in box])
        e01, e12 = np.linalg.norm(box_m[1] - box_m[0]), np.linalg.norm(box_m[2] - box_m[1])
        if e01 >= e12:
            run, width, d = e01, e12, (box_m[1] - box_m[0]) / max(e01, 1e-9)
        else:
            run, width, d = e12, e01, (box_m[2] - box_m[1]) / max(e12, 1e-9)
        centre = box_m.mean(0)
        if width < 0.6 or run < 0.5 or hull_m.area < 0.8:
            continue                                                # too small for a stair -> false positive
        # kind: outside the building = entrance steps; L / U shaped stepped area inside = terrace
        out_frac = float(np.isin(room[oy:oy + h, ox:ox + w][comp], (0, 1)).mean())
        # the building outline decides; the model's "outdoor" pixels only count right at the outline
        # (an open lobby the model does not label as a room is still inside the building)
        if footprint is not None:
            pc = Point(*centre)
            outside = (not footprint.contains(pc)) or (out_frac >= 0.6 and footprint.exterior.distance(pc) < 1.0)
        else:
            outside = out_frac >= 0.4
        # a stair the model marks is a stair: outside the building outline it leads up to the floor,
        # inside it leads to the next floor. Stepped areas (sunken / raised) only with the "stepped_areas" helper.
        stepped = "stepped_areas" in ASSIST and (shape == "irregular" or st.get("drawn"))
        kind = "entrance" if outside else ("terrace" if stepped else "interior")
        pit_lab = []
        if not outside and open_inside is not None:
            r_ = max(int(round(0.6 / mpp)), int(T), 3)
            Hr, Wr = room.shape
            y0r, y1r, x0r, x1r = max(oy - r_, 0), min(oy + h + r_, Hr), max(ox - r_, 0), min(ox + w + r_, Wr)
            cw = np.zeros((y1r - y0r, x1r - x0r), np.uint8)
            cw[oy - y0r:oy - y0r + h, ox - x0r:ox - x0r + w] = comp
            ring = (cv2.dilate(cw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_ + 1,) * 2)) > 0) & (cw == 0)
            oi = open_inside[y0r:y1r, x0r:x1r]
            if ring.sum() and (oi & ring).sum() >= 0.15 * ring.sum():
                kind = "terrace"
                pit_lab = [int(l_) for l_ in np.unique(oi_lab[y0r:y1r, x0r:x1r][oi & ring]) if l_]
        # distance (model px) to the HIGH side of the stair
        wx0, wy0, wx1, wy1 = _window(room.shape, ox, oy, ox + w, oy + h, pad)
        cx_px, cy_px = float(xs.mean() + ox), float(ys.mean() + oy)
        notes = _stair_notes(toks, cx_px, cy_px, max(pad, 0.6 * max(h, w)))
        warnings, direction = [], None
        ends_px = []                                                # the two run ends (model px)
        bx = np.array(box, float)
        if np.linalg.norm(bx[1] - bx[0]) >= np.linalg.norm(bx[2] - bx[1]):
            ends_px = [(bx[0] + bx[3]) / 2, (bx[1] + bx[2]) / 2]
        else:
            ends_px = [(bx[0] + bx[1]) / 2, (bx[2] + bx[3]) / 2]
        if kind == "entrance":
            anchor = indoor[wy0:wy1, wx0:wx1]
            direction = "rises towards the building"
        elif kind == "terrace" and pit_lab:
            anchor = indoor[wy0:wy1, wx0:wx1]                    # high side = the floor, low side = the open ground
            direction = "goes DOWN to the open area beside it (the model marks it as open ground)"
        elif kind == "terrace":
            anchor = walls_px[wy0:wy1, wx0:wx1]
            direction = "rises towards the walls (assumed for a stepped area)"
        else:
            anchor = None
        if anchor is not None and anchor.any():
            Dw = ndimage.distance_transform_edt(~anchor)
        else:
            if not (notes["up"] or notes["dn"]) and original is not None:
                notes.update({k: v_ for k, v_ in _zoom_labels(original, s, wx0, wy0, wx1, wy1).items() if v_})
            # interior: top = end away from the 'UP' label (UP is written at the foot), or at the 'DN' label
            top = 1
            if notes["up"]:
                u = np.array(notes["up"][0])
                top = int(np.argmax([np.linalg.norm(e - u) for e in ends_px]))
                direction = "from the UP label"
            elif notes["dn"]:
                u = np.array(notes["dn"][0])
                top = int(np.argmin([np.linalg.norm(e - u) for e in ends_px]))
                direction = "from the DN label"
            else:
                direction = "assumed (no UP / DN label)"
                warnings.append("direction not marked on the plan: check which end goes up")
            a = np.zeros((wy1 - wy0, wx1 - wx0), np.uint8)
            far = ends_px[top] + (ends_px[top] - ends_px[1 - top]) / max(np.linalg.norm(ends_px[top] - ends_px[1 - top]), 1e-9) * 2
            ax_ = (ends_px[top] - ends_px[1 - top]) / max(np.linalg.norm(ends_px[top] - ends_px[1 - top]), 1e-9)
            pn = np.array([-ax_[1], ax_[0]]) * (h + w)              # a line across the top end, not a point
            cv2.line(a, tuple(int(q) for q in (far + pn - [wx0, wy0])), tuple(int(q) for q in (far - pn - [wx0, wy0])), 1, 2)
            Dw = ndimage.distance_transform_edt(a == 0)
        D_crop = Dw[oy - wy0:oy - wy0 + h, ox - wx0:ox - wx0 + w]
        Dat = lambda e: Dw[int(np.clip(e[1] - wy0, 0, Dw.shape[0] - 1)), int(np.clip(e[0] - wx0, 0, Dw.shape[1] - 1))]
        if st.get("line_dir"):
            # read from the drawing: the steps climb across their lines
            ld = np.array(st["line_dir"]); nd = np.array([-ld[1], ld[0]])
            cpx = np.array([cx_px, cy_px])
            ext = 0.5 * max(h, w)
            ends_px = [cpx - nd * ext, cpx + nd * ext]
        elif kind != "interior":
            # the run is the axis that climbs: the pair of opposite edges with the biggest height difference
            # (entrance steps are often wider than deep, so the long side is not always the run)
            pairs = [[(bx[0] + bx[1]) / 2, (bx[2] + bx[3]) / 2], [(bx[1] + bx[2]) / 2, (bx[3] + bx[0]) / 2]]
            ends_px = max(pairs, key=lambda pr: abs(Dat(pr[0]) - Dat(pr[1])))
        # low -> high direction along the run
        lo_i = int(np.argmax([Dat(e) for e in ends_px]))
        lo_m = np.array([MX(ends_px[lo_i][0]), MY(ends_px[lo_i][1])])
        hi_m = np.array([MX(ends_px[1 - lo_i][0]), MY(ends_px[1 - lo_i][1])])
        if np.linalg.norm(hi_m - lo_m) > 1e-6:
            d = (hi_m - lo_m) / np.linalg.norm(hi_m - lo_m)
        # run = along d across the outline, width = across it (measured at both ends: flared if they differ)
        from shapely.geometry import LineString
        nrm = np.array([-d[1], d[0]])
        proj = np.array(hull_m.exterior.coords) @ d
        run = float(proj.max() - proj.min())
        centre = centre + d * (0.5 * (proj.max() + proj.min()) - centre @ d)

        def cross_width(u):
            c = centre + d * u
            return float(LineString([c - nrm * 50, c + nrm * 50]).intersection(hull_m).length)
        width_low, width_high = cross_width(-0.45 * run), cross_width(0.45 * run)
        width = max(width_low, width_high)
        if shape == "straight" and abs(width_low - width_high) > 0.08 * width:
            shape = "flared"
        # treads counted from the drawing
        counted = None
        if original is not None:
            try:
                counted = _count_treads(comp, (ox, oy), D_crop, original, s, m_per_opx)
            except Exception as e:
                warnings.append("tread lines could not be read ({})".format(type(e).__name__))
        # written width / run
        sources = {}
        r_px = max(pad, 0.8 * max(h, w))
        if shape == "straight":
            width_low = width_high = width = 0.5 * (width_low + width_high)
        wr = _written_length(toks, cx_px, cy_px, r_px, width)
        if wr:
            k = wr[0] / width
            width_low, width_high = width_low * k, width_high * k
            width, sources["width"] = wr[0], "written {}".format(wr[1])
        else:
            sources["width"] = "measured"
        rr = _written_length(toks, cx_px, cy_px, r_px, run)
        if rr and abs(rr[0] - width) > 1e-6:
            run, sources["run"] = rr[0], "written {}".format(rr[1])
        else:
            sources["run"] = "measured"
        if counted is None and st.get("n_lines"):
            counted = (max(2, st["n_lines"] - 1), None, None)    # tiers between the lines read from the drawing
        pits = []
        for l_ in pit_lab:                                          # the open ground it leads to, in metres
            cs_, _ = cv2.findContours((oi_lab == l_).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cs_:
                cpts = max(cs_, key=cv2.contourArea)[:, 0].astype(float)
                pg = Polygon(np.stack([mx(cpts[:, 0]), my(cpts[:, 1])], 1)).buffer(0)
                if pg.area >= 1.0:
                    pits.append(pg)
        down = bool(pits) or (kind == "terrace" and STEPPED_AREAS == "down")
        infos.append(dict(kind=kind, shape=shape, hull=hull_m, centre=centre, d=d, run=run, width=width,
                          down=down, pits=pits,
                          width_low=width_low, width_high=width_high,
                          notes=notes, counted=counted, sources=sources, warnings=warnings, direction=direction,
                          lo=lo_m, hi=hi_m))

    # ---- plinth (ground -> floor level)
    lvl = _plinth_from_levels(toks)
    entr = [i for i in infos if i["kind"] == "entrance"]
    for i in infos:                                                 # risers from notes / drawing
        n_note, n_cnt = i["notes"].get("risers"), (i["counted"][0] if i["counted"] else None)
        if i["kind"] == "interior" and n_cnt:
            n_cnt += 1                                              # drawn treads + the riser onto the upper floor
        i["n"], i["sources"]["risers"] = ((n_note, "written on the plan") if n_note else
                                          (n_cnt, "counted from the drawn treads") if n_cnt else (None, None))
    _chain_flights(infos)
    if PLINTH_HEIGHT:
        plinth, psrc = float(PLINTH_HEIGHT), "your setting"
    elif lvl:
        plinth, psrc = lvl, "level mark on the plan"
    elif entr:
        est = [(i["n"] or 3) * i["notes"].get("riser", STAIR_RISE_EXT) for i in entr]
        plinth = float(np.median(est))
        psrc = ("riser note x steps" if any(i["notes"].get("riser") for i in entr)
                else "assumed {:.0f} mm risers x counted steps".format(STAIR_RISE_EXT * 1000))
    else:
        plinth, psrc = 0.0, "none (no entrance steps, no level mark)"

    stairs = []
    for k, i in enumerate(infos):
        kind, w = i["kind"], i["warnings"]
        if kind == "entrance":
            base, total, rsrc = -plinth, plinth, "plinth: " + psrc
            if not i["n"]:
                i["n"] = max(1, int(round(total / STAIR_RISE_EXT)))
                i["sources"]["risers"] = "estimated from the plinth height"
                w.append("number of steps not written or readable: estimated")
            if psrc.startswith("assumed"):
                w.append("plinth height not written: set 'Plinth height' to make the steps exact")
        elif kind == "interior" and "chain_base" in i:
            base, total = i["chain_base"], i["chain_total"]
            rsrc = "storey height shared by {} connected flights".format(i["chain_size"])
        elif kind == "interior":
            base, total, rsrc = 0.0, WALL_HEIGHT, "storey height (wall height setting)"
            if "fix_treads" in ASSIST and i["n"] and not (0.12 <= total / i["n"] <= 0.21) \
                    and i["sources"].get("risers") != "written on the plan":
                # the converter's own tread count cannot be right for this storey height: do not build it
                w.append("only {} treads could be read on the drawing: risers set from the storey height".format(i["n"] - 1))
                i["n"] = None
            if not i["n"]:
                i["n"] = int(np.clip(round(total / STAIR_RISE), 3, 30))
                i["sources"]["risers"] = "estimated from the storey height"
                w.append("risers not written or readable: estimated at about {:.0f} mm".format(STAIR_RISE * 1000))
        else:
            r = i["notes"].get("riser", STAIR_RISE_EXT)
            if not i["n"]:
                i["n"] = 3
                i["sources"]["risers"] = "assumed"
                w.append("tiers could not be counted: 3 assumed")
            total = r * i["n"]
            base = -total if i["down"] else 0.0
            if i["down"] and plinth > 0.01:
                # steps going down lead to the open ground: they drop the full plinth height
                total, base = plinth, -plinth
                rsrc = "plinth: " + psrc + " split over the drawn steps"
            if not (i["down"] and plinth > 0.01):
                rsrc = "riser note" if "riser" in i["notes"] else "assumed {:.0f} mm per tier".format(r * 1000)
            i["direction"] = ("goes DOWN from the raised floor to the open ground (floor and plinth cut out)" if i["down"]
                              else "rises towards the walls (raised seating)")
            w.append("stepped area going {}: tier height {}; check".format(
                "down" if i["down"] else "up", "written" if "riser" in i["notes"] else "assumed"))
        n = int(i["n"])
        riser = total / n if n else 0.0
        if kind != "terrace" and not (0.10 <= riser <= 0.22):
            w.append("riser works out to {:.0f} mm: check the step count / levels".format(riser * 1000))
        # tread blocks: entrance -> n blocks (top one is the landing at floor level);
        # interior -> n-1 treads (the last riser lands on the upper floor); terrace -> n tiers
        blocks = n - 1 if kind == "interior" else n
        blocks = max(blocks, 1)
        polys = None
        if i["shape"] == "irregular" and i["counted"]:
            _, lab, (X0, Y0) = i["counted"]
            polys = _label_polys(lab, X0, Y0, s, mx, my, m_per_opx)
            polys = [p for p in polys if p is not None]
            if len(polys) != i["counted"][0]:
                polys = None
            else:
                blocks = len(polys)
        if polys is None:
            if i["shape"] == "straight":
                from shapely.geometry import Polygon as _P
                d, c, L, W = i["d"], i["centre"], i["run"], i["width"]
                nrm = np.array([-d[1], d[0]])
                outline = _P([c - d * L / 2 - nrm * W / 2, c + d * L / 2 - nrm * W / 2,
                              c + d * L / 2 + nrm * W / 2, c - d * L / 2 + nrm * W / 2])
            else:
                outline = i["hull"]
            polys = _slice(outline, i["centre"], i["d"], i["run"], blocks)
        treads = []
        for j, p in enumerate(polys):
            if p is None:
                continue
            if kind == "terrace" and i["down"]:
                top = -riser * (j + 1)                         # first tier (atrium side) one step down
            else:
                top = base + riser * (j + 1)
            treads.append({"polygon": _poly_list(p), "z0": round(base, 4), "z1": round(top, 4)})
        tread = i["run"] / blocks
        if kind != "terrace" and not (0.22 <= tread <= 0.45):
            w.append("tread works out to {:.0f} mm: check".format(tread * 1000))
        outline = i["hull"] if i["shape"] != "straight" else None
        if outline is None:
            d, c, L, W = i["d"], i["centre"], i["run"], i["width"]
            nrm = np.array([-d[1], d[0]])
            ol = [c - d * L / 2 - nrm * W / 2, c + d * L / 2 - nrm * W / 2,
                  c + d * L / 2 + nrm * W / 2, c - d * L / 2 + nrm * W / 2]
            outline_l = [[round(float(x), 4), round(float(y), 4)] for x, y in ol]
        else:
            outline_l = _poly_list(outline)
        c, d, L = i["centre"], i["d"], i["run"]
        stairs.append({
            "id": k, "kind": kind, "shape": i["shape"],
            "outline": outline_l,
            "start": [round(float(v_), 4) for v_ in c - d * L / 2],          # low end (walking line)
            "end": [round(float(v_), 4) for v_ in c + d * L / 2],            # high end
            "width": round(i["width"], 4), "width_low": round(i["width_low"], 4),
            "width_high": round(i["width_high"], 4), "run": round(L, 4),
            "risers": n, "riser": round(riser, 4), "tread": round(tread, 4),
            "base_z": round(base, 4), "top_z": round(0.0 if (kind == "terrace" and i["down"]) else base + total, 4),
            "sunken": bool(kind == "terrace" and i["down"]),
            "pit": [_poly_list(pg) for pg in i.get("pits", [])],      # the open ground the model marked beside it
            "direction": i["direction"],
            "sources": dict(i["sources"], rise=rsrc),
            "treads": treads, "warnings": w,
            "straight_run_fits": bool(i["shape"] == "straight" and kind != "terrace"),
        })
    levels = {"floor": 0.0, "ground": round(-plinth, 4), "plinth": round(plinth, 4), "plinth_source": psrc}
    return stairs, levels
