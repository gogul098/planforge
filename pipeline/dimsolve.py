"""Dimension solving: read written dimensions, match them to wall grid lines, solve exact positions."""
import re

import numpy as np

NUM = re.compile(r"\d+(?:[.,]\d+)?")


_OCR = []


def ocr_tokens(img_rgb, ocr=None):
    """Run RapidOCR; split multi-number boxes ('5.00 11.00') into separate tokens with estimated centres."""
    if ocr is None:                       # one engine per process: creating it loads three ONNX models
        if not _OCR:
            from rapidocr_onnxruntime import RapidOCR
            _OCR.append(RapidOCR())
        ocr = _OCR[0]
    res, _ = ocr(img_rgb[:, :, ::-1].copy())
    toks = []
    res = list(res or [])
    used = set()
    joined = []
    for i, (box, txt, conf) in enumerate(res):                  # 'W x' on one line, 'D' on the next
        if i in used:
            continue
        if re.search(r"\d\s*(?:m|mm|cm)?\s*[xX×]\s*$", txt.strip()):
            b = np.array(box, float)
            h = b[:, 1].max() - b[:, 1].min()
            cx = b[:, 0].mean()
            best = None
            for j, (box2, txt2, conf2) in enumerate(res):
                if j == i or j in used or not re.match(r"^\s*\d", txt2):
                    continue
                b2 = np.array(box2, float)
                dy = b2[:, 1].min() - b[:, 1].max()
                if -0.3 * h <= dy <= 1.5 * h and abs(b2[:, 0].mean() - cx) <= 2.5 * h:
                    if best is None or dy < best[0]:
                        best = (dy, j, b2, txt2, conf2)
            if best:
                _, j, b2, txt2, conf2 = best
                used.add(j)
                joined.append((np.vstack([b, b2]).tolist(), txt.strip() + " " + txt2.strip(), min(conf, conf2)))
                continue
        joined.append((box, txt, conf))
    for box, txt, conf in joined:
        b = np.array(box, float)
        x0, x1, y0, y1 = b[:, 0].min(), b[:, 0].max(), b[:, 1].min(), b[:, 1].max()
        vertical_text = (y1 - y0) > 1.6 * (x1 - x0) and len(txt) > 2
        n = max(len(txt), 1)
        nums = list(NUM.finditer(txt))
        letters = sum(ch.isalpha() for ch in txt)
        size = parse_size(txt)
        if size:
            toks.append({"kind": "size", "value": size, "x": (x0 + x1) / 2, "y": (y0 + y1) / 2,
                         "text": txt.strip(), "raw": txt, "conf": float(conf)})
        elif re.fullmatch(r"\s*\(\s*\d{1,3}\s*\)\s*", txt):       # room numbers '(10)' are not dimensions
            toks.append({"kind": "label", "text": txt, "raw": txt, "x": (x0 + x1) / 2, "y": (y0 + y1) / 2,
                         "h": float(y1 - y0), "conf": float(conf)})
        elif nums and letters <= 1 and not re.search(r"[xX×]", txt):
            for m in nums:
                f = (m.start() + m.end()) / 2 / n
                cx, cy = (x0 + f * (x1 - x0), (y0 + y1) / 2) if not vertical_text else \
                         ((x0 + x1) / 2, y1 - f * (y1 - y0))
                toks.append({"kind": "num", "value": float(m.group().replace(",", ".")), "x": cx, "y": cy,
                             "text": m.group(), "raw": txt, "conf": float(conf),
                             "h": float(min(y1 - y0, x1 - x0) if vertical_text else (y1 - y0))})
        elif letters >= 2:
            toks.append({"kind": "label", "text": txt, "raw": txt, "x": (x0 + x1) / 2, "y": (y0 + y1) / 2,
                         "h": float(min(y1 - y0, x1 - x0) if vertical_text else (y1 - y0)), "conf": float(conf)})
    return toks


QUOTES = "'\"°′″`’”"


def parse_length(side):
    """One length from OCR text: feet-inches (14'-0\", 8-6\", 12 -5\") or metric (3.60, 360, 3600)."""
    nums = re.findall(r"\d+(?:\.\d+)?", side)
    if not nums or len(nums) > 2:
        return None
    imperial = any(c in side for c in QUOTES) or (len(nums) == 2 and "." not in "".join(nums))
    if imperial:
        ft = float(nums[0])
        inch = float(nums[1]) if len(nums) == 2 else 0.0
        if inch >= 12 or ft > 150:
            return None
        return (ft + inch / 12.0) * 0.3048
    v = float(nums[0])
    return v if v < 40 else (v / 100 if v < 1000 else v / 1000)


UNIT_WORDS = re.compile(r"(?<=\d)\s*(?:mm|cm|mtrs?|m)(?=\s*[xX×]|\b)\.?", re.I)


def parse_size(text):
    """'W x D' room-size label -> (a_m, b_m) or None.  '3.66 m x 4.88 m', '12'0\" x 10'6\"', '3600 X 4200'."""
    unit = re.search(r"\d\s*(mm|cm)\b", text, re.I)
    text = UNIT_WORDS.sub("", text)
    parts = re.split(r"\s*[xX×]\s*", text.strip())
    if len(parts) != 2 or sum(ch.isalpha() for ch in text if ch not in "xX") > 0:
        return None
    a, b = parse_length(parts[0]), parse_length(parts[1])
    if unit and a and b and unit.group(1).lower() == "cm" and a > 40:
        a, b = a / 100.0, b / 100.0
    if a and b and 0.6 <= a <= 40 and 0.6 <= b <= 40:
        return (a, b)
    return None


def _unit_factor(values):
    """Plans write metres (3.00), centimetres (300) or millimetres (3000). Return factor to metres."""
    v = np.median(values)
    return 1.0 if v < 40 else (0.01 if v < 400 else 0.001)


def _room_constraints(struct, segs, T, toks, gx, gy):
    """Room-size labels -> clear sizes. Rays are cast from the label to the walls around it
    (works for rooms whose doors leave gaps). Returns measurements between wall centre-lines (px)."""
    sizes = [t for t in toks if t["kind"] == "size"]
    if not sizes or struct is None:
        return []
    H, W = struct.shape

    def thick(o, c):
        ts = [sg["t"] for sg in segs if sg["o"] == o and abs(sg["c"] - c) < 0.75 * T]
        return float(np.median(ts)) if ts else T

    def ray(x, y, dx, dy):
        while 0 <= x < W and 0 <= y < H:
            if struct[y, x]:
                return x, y
            x += dx; y += dy
        return None

    def span(x, y, horizontal):
        """Most common wall-to-wall distance over several parallel rays near (x, y)."""
        found = []
        for off in np.linspace(-4 * T, 4 * T, 9):
            if horizontal:
                yy = int(round(y + off))
                a, b = ray(int(x), yy, -1, 0), ray(int(x), yy, 1, 0)
                if a and b:
                    found.append((a[0], b[0]))
            else:
                xx = int(round(x + off))
                a, b = ray(xx, int(y), 0, -1), ray(xx, int(y), 0, 1)
                if a and b:
                    found.append((a[1], b[1]))
        if len(found) < 3:
            return None
        lens = np.array([q - p for p, q in found])
        med = np.median(lens)
        agree = [f for f, l in zip(found, lens) if abs(l - med) <= 2]
        if len(agree) < 3:
            return None                                  # rays disagree (door gap, odd shape)
        return int(np.median([p for p, _ in agree])), int(np.median([q for _, q in agree]))

    out = []
    for t in sizes:
        sx, sy = span(t["x"], t["y"], True), span(t["x"], t["y"], False)
        if not sx or not sy:
            continue
        w, h = sx[1] - sx[0] - 1, sy[1] - sy[0] - 1      # clear size in px (between wall faces)
        if w < 2 * T or h < 2 * T:
            continue
        a, b = t["value"]
        if abs(np.log((a / w) / (b / h))) > abs(np.log((b / w) / (a / h))):
            a, b = b, a                                  # label order is not always W x D
        if abs(np.log((a / w) / (b / h))) > 0.08:
            continue                                     # proportions disagree -> wrong region
        for ax, (p0, p1), clear, cpx, g, o in (("x", sx, a, w, gx, "v"), ("y", sy, b, h, gy, "h")):
            lo = [i for i, q in enumerate(g) if q <= p0 + 1 and p0 - q <= 1.5 * T]
            hi = [i for i, q in enumerate(g) if q >= p1 - 1 and q - p1 <= 1.5 * T]
            if lo and hi and hi[0] > lo[-1]:
                i, j = lo[-1], hi[0]
                out.append({"axis": ax, "i": i, "j": j, "clear": clear, "clear_px": cpx,
                            "half_t_px": (thick(o, g[i]) + thick(o, g[j])) / 2, "text": t["text"]})
    return out


def solve(grid_x, grid_y, bbox, toks, struct=None, segs=None, T=6.0):
    """Exact grid positions from written dimension chains and room-size labels.
    Conflicting measurements are rejected instead of distorting the model."""
    grids = {"x": list(grid_x), "y": list(grid_y)}
    # a plan with no vertical (or no horizontal) wall at all - every wall at an angle - has no grid lines on
    # that axis: one anchor at the building's edge, the measured scale everywhere (no per-grid corrections)
    for ax, lo in (("x", bbox[0]), ("y", bbox[1])):
        if not grids[ax]:
            grids[ax] = [float(lo)]
    cons = []
    # ---- (a) dimension chains written outside the building
    nums = [t for t in toks if t["kind"] == "num" and t["value"] > 0]
    chain_ratios = []
    if nums:
        f = _unit_factor([t["value"] for t in nums])
        bx0, by0, bx1, by1 = bbox
        pad = 0.02 * max(bx1 - bx0, by1 - by0)
        for t in nums:
            outside_lr = t["x"] < bx0 - pad or t["x"] > bx1 + pad
            outside_tb = t["y"] < by0 - pad or t["y"] > by1 + pad
            ax = "x" if (outside_tb and not outside_lr) else ("y" if (outside_lr and not outside_tb) else None)
            if ax is None:
                continue
            g, c = grids[ax], (t["x"] if ax == "x" else t["y"])
            pairs = [(i, j) for i in range(len(g)) for j in range(i + 1, len(g))
                     if abs((g[i] + g[j]) / 2 - c) <= 0.2 * (g[j] - g[i]) + 15]
            for i, j in pairs:
                chain_ratios.append(t["value"] * f / (g[j] - g[i]))
            cons.append({"kind": "dim", "axis": ax, "tok": t, "d": t["value"] * f, "pairs": pairs})
    # ---- (b) room-size labels
    rooms = _room_constraints(struct, segs or [], T, toks, grids["x"], grids["y"])
    room_ratios = [r["clear"] / r["clear_px"] for r in rooms]

    def consensus(r):
        r = np.array(r)
        if len(r) == 0:
            return None, 0
        sup = [(np.abs(np.log(r / q)) < 0.03).sum() for q in r]
        best = r[int(np.argmax(sup))]
        sel = r[np.abs(np.log(r / best)) < 0.03]
        return float(np.median(sel)), len(sel)

    s_chain, n_chain = consensus(chain_ratios)
    s_room, n_room = consensus(room_ratios)
    if n_chain >= 2:
        s, source = s_chain, "written dimension chains"
    elif n_room >= 2:
        s, source = s_room, "room-size labels"
    elif n_chain or n_room:
        s, source = (s_chain or s_room), "single measurement (low confidence)"
    else:
        return None

    out = {"scale_m_per_px": s, "source": source, "axes": {}, "report": []}
    for ax in ("x", "y"):
        g = grids[ax]
        n = len(g)
        eqs = []                                   # (i, j, d, report-entry)
        for c in cons:
            if c["axis"] != ax:
                continue
            best = None
            for i, j in c["pairs"]:
                L = g[j] - g[i]
                err = abs(s * L - c["d"]) / c["d"]
                cost = err + 0.3 * abs((g[i] + g[j]) / 2 - (c["tok"]["x"] if ax == "x" else c["tok"]["y"])) / L
                if err < 0.08 and (best is None or cost < best[0]):
                    best = (cost, i, j)
            if best:
                eqs.append([best[1], best[2], c["d"], {"kind": "dimension", "axis": ax, "text": c["tok"]["text"],
                                                       "written_m": round(c["d"], 4)}])
        for r in rooms:
            if r["axis"] != ax:
                continue
            d = r["clear"] + r["half_t_px"] * s
            if abs(s * (g[r["j"]] - g[r["i"]]) - d) / d < 0.08:
                eqs.append([r["i"], r["j"], d, {"kind": "room clear size", "axis": ax, "text": r["text"],
                                                "written_m": round(r["clear"], 4), "half_t_m": r["half_t_px"] * s}])
        active = list(range(len(eqs)))
        rejected = []
        for _ in range(len(eqs) + 1):
            rows, rhs, w = [], [], []
            for k in active:
                i, j, d, _ = eqs[k]
                r_ = np.zeros(n); r_[j], r_[i] = 1, -1
                rows.append(r_); rhs.append(d); w.append(1.0)
            for k in range(1, n):
                r_ = np.zeros(n); r_[k], r_[k - 1] = 1, -1
                rows.append(r_); rhs.append(s * (g[k] - g[k - 1])); w.append(0.02)
            r_ = np.zeros(n); r_[0] = 1; rows.append(r_); rhs.append(0.0); w.append(10.0)
            A, b, W = np.array(rows), np.array(rhs), np.array(w)
            sol = np.linalg.lstsq(A * W[:, None], b * W, rcond=None)[0] if n else np.array([])
            res = [(abs(sol[eqs[k][1]] - sol[eqs[k][0]] - eqs[k][2]), k) for k in active]
            if not res:                                                  # no written dimension on this axis
                break
            worst = max(res)
            if worst[0] > max(0.02, 0.01 * eqs[worst[1]][2]):
                active.remove(worst[1]); rejected.append(worst[1])   # conflicts with the others
            else:
                break
        # never allow a distorted grid: spacing must stay within 20% of the global scale
        if n > 1 and np.any(np.abs(np.log(np.maximum(np.diff(sol), 1e-9) / (s * np.maximum(np.diff(g), 1e-9)))) > 0.2):
            sol = np.array([(q - g[0]) * s for q in g])
            out["report"].append({"axis": ax, "text": "grid", "status": "distorted -> uniform scale used"})
        out["axes"][ax] = {"px": g, "m": sol.tolist(), "n_dims": len(active)}
        for k, (i, j, d, rep) in enumerate(eqs):
            got = sol[j] - sol[i]
            rep = dict(rep)
            if "half_t_m" in rep:
                got -= rep.pop("half_t_m")
                d = rep["written_m"]
            rep.update({"model_m": round(float(got), 4), "error_mm": round((got - d) * 1000, 1),
                        "status": "rejected (conflicts)" if k in rejected else
                        ("ok" if abs(got - d) < 0.005 else "conflict")})
            out["report"].append(rep)
    return out


def make_mapper(sol, axis):
    """px -> metres. Exact at grid lines, linear in between, measured scale outside."""
    px = np.array(sol["axes"][axis]["px"], float)
    m = np.array(sol["axes"][axis]["m"], float)
    s = sol["scale_m_per_px"]

    def f(v):
        v = np.asarray(v, float)
        if len(px) == 0:
            return v * s
        if len(px) == 1:
            return m[0] + (v - px[0]) * s
        out = np.interp(v, px, m)
        out = np.where(v < px[0], m[0] + (v - px[0]) * s, out)
        out = np.where(v > px[-1], m[-1] + (v - px[-1]) * s, out)
        return out
    return f
