"""Multi-view drawing sheets from real plans, for the sheet-view detector (head 0).

Architects often put the ground floor, first floor and terrace plans of one building side by side on one sheet,
with a title under each view, a border, a title block, notes and schedules. This composes such sheets from real
plan drawings (ResPlan renders, Structured3D renders, CubiCasa5K scans: anything drawn), so the position of
every view is known exactly:

    plan_view       the drawing of one floor (not its title)
    title_block     the drawing's title block
    table           door / window schedules and area tables

Titles are taken from the names real sheets use (GROUND FLOOR PLAN, FIRST FLOOR PLAN, TERRACE PLAN ...). The
plans on one composed sheet are different buildings: these sheets teach FINDING the views, not stacking floors.
"""
import math
import os
import random

import cv2
import numpy as np

CLASSES = ["plan_view", "title_block", "table"]
TITLES = ["GROUND FLOOR PLAN", "FIRST FLOOR PLAN", "SECOND FLOOR PLAN", "TERRACE PLAN", "STILT FLOOR PLAN",
          "BASEMENT PLAN", "TYPICAL FLOOR PLAN", "GF PLAN", "FF PLAN", "SF PLAN", "ROOF PLAN", "MEZZANINE PLAN"]
WORDS = ["PROPOSED", "RESIDENCE", "BUILDING", "FOR", "MR", "SITE", "NO", "PLOT", "SURVEY", "DRAWN", "CHECKED",
         "APPROVED", "ARCHITECT", "SCALE", "DATE", "SHEET", "REV", "NOTES", "ALL", "DIMENSIONS", "ARE", "IN", "MM",
         "WALLS", "230", "THK", "BRICK", "LINTEL", "LEVEL", "PLINTH", "+0.45", "FFL", "1:100", "NTS", "CLIENT"]
FONTS = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX,
         cv2.FONT_HERSHEY_PLAIN]


def content_box(rgb, room=None, pad=4):
    """Bounding box of the drawing: labelled building pixels when given, else ink."""
    if room is not None:
        m = (room != 0) & (room != 255) & (room != 254)
    else:
        m = rgb.min(axis=2) < 200
    ys, xs = np.nonzero(m)
    if len(xs) < 20:
        return None
    x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
    x1, y1 = min(rgb.shape[1], xs.max() + pad + 1), min(rgb.shape[0], ys.max() + pad + 1)
    return x0, y0, x1, y1


def _text(img, s, x, y, scale, rng, ink):
    f = rng.choice(FONTS)
    th = max(1, int(round(scale * rng.uniform(1.2, 2.2))))
    cv2.putText(img, s, (int(x), int(y)), f, scale, ink, th, cv2.LINE_AA)
    (w, h), _ = cv2.getTextSize(s, f, scale, th)
    return w, h


def _sentence(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(n))


def _table(img, x, y, w, h, rng, ink):
    rows, cols = rng.randint(3, 9), rng.randint(2, 5)
    cv2.rectangle(img, (x, y), (x + w, y + h), ink, 2)
    for r in range(1, rows):
        yy = y + r * h // rows
        cv2.line(img, (x, yy), (x + w, yy), ink, 1)
    for c in range(1, cols):
        xx = x + c * w // cols
        cv2.line(img, (xx, y), (xx, y + h), ink, 1)
    sc = max(0.3, min(0.9, h / rows / 40))
    for r in range(rows):
        for c in range(cols):
            if rng.random() < 0.8:
                cell = rng.choice(["D1", "D2", "W1", "W2", "V1", "1000", "900", "2100", "1200", "MAIN DOOR",
                                   "WINDOW", "VENT", str(rng.randint(1, 30))])
                cv2.putText(img, cell, (x + c * w // cols + 4, y + r * h // rows + int(h / rows * 0.7)),
                            cv2.FONT_HERSHEY_SIMPLEX, sc, ink, 1, cv2.LINE_AA)


def _north(img, cx, cy, r, ink):
    cv2.circle(img, (cx, cy), r, ink, 2)
    pts = np.array([[cx, cy - r], [cx - r // 3, cy + r // 2], [cx, cy + r // 5], [cx + r // 3, cy + r // 2]], np.int32)
    cv2.fillPoly(img, [pts], ink)
    cv2.putText(img, "N", (cx - r // 4, cy - r - 6), cv2.FONT_HERSHEY_SIMPLEX, r / 30, ink, 2)


def compose(plans, rng, size=None):
    """plans: list of (rgb uint8, room labels or None). Returns (sheet rgb, [(class, x0, y0, x1, y1)])."""
    W = size or rng.choice([2400, 2800, 3300, 4000])
    H = int(W / rng.choice([1.414, 1.414, 1.5, 1.3]))
    paper = rng.choice([255, 252, 248])
    ink = (rng.randint(0, 40),) * 3
    img = np.full((H, W, 3), paper, np.uint8)
    boxes = []
    m = int(W * rng.uniform(0.015, 0.03))
    if rng.random() < 0.85:                                     # border frame
        cv2.rectangle(img, (m, m), (W - m, H - m), ink, rng.choice([2, 3, 4]))
        if rng.random() < 0.5:
            cv2.rectangle(img, (m + 10, m + 10), (W - m - 10, H - m - 10), ink, 1)
    # title block: bottom strip or right column
    right = rng.random() < 0.55
    if right:
        tbw = int(W * rng.uniform(0.14, 0.2))
        tb = (W - m - tbw, m, W - m, H - m)
        area = (m + 20, m + 20, W - m - tbw - 20, H - m - 20)
    else:
        tbh = int(H * rng.uniform(0.1, 0.15))
        tb = (m, H - m - tbh, W - m, H - m)
        area = (m + 20, m + 20, W - m - 20, H - m - tbh - 20)
    if rng.random() < 0.9:
        x0, y0, x1, y1 = tb
        cv2.rectangle(img, (x0, y0), (x1, y1), ink, 3)
        n = rng.randint(3, 7)
        for k in range(1, n):
            if right:
                yy = y0 + int((y1 - y0) * (0.5 + 0.5 * k / n))
                cv2.line(img, (x0, yy), (x1, yy), ink, 1)
            else:
                xx = x0 + int((x1 - x0) * (0.4 + 0.6 * k / n))
                cv2.line(img, (xx, y0), (xx, y1), ink, 1)
        for k in range(rng.randint(4, 10)):
            sc = rng.uniform(0.6, 1.2) * W / 3300
            tx = x0 + 12 + (rng.randint(0, max(1, (x1 - x0) // 3)) if not right else 0)
            ty = y0 + 30 + k * int(40 * W / 3300)
            if ty < y1 - 8:
                _text(img, _sentence(rng, rng.randint(1, 4)), tx, ty, sc, rng, ink)
        boxes.append(("title_block", x0, y0, x1, y1))
    # notes / tables in a side band of the drawing area, sometimes
    ax0, ay0, ax1, ay1 = area
    if rng.random() < 0.5:
        bw = int((ax1 - ax0) * rng.uniform(0.16, 0.24))
        bx = ax1 - bw
        y = ay0 + 10
        if rng.random() < 0.7:
            th = int((ay1 - ay0) * rng.uniform(0.15, 0.3))
            _table(img, bx, y, bw, th, rng, ink)
            boxes.append(("table", bx, y, bx + bw, y + th))
            y += th + 30
        for k in range(rng.randint(3, 10)):
            sc = 0.7 * W / 3300
            if y + 30 > ay1:
                break
            _text(img, "{}. {}".format(k + 1, _sentence(rng, rng.randint(3, 6))), bx, y + 20, sc, rng, ink)
            y += int(34 * W / 3300)
        ax1 = bx - 30
    # views in a grid
    n = len(plans)
    cols = n if n <= 3 and (ax1 - ax0) > 1.3 * (ay1 - ay0) else math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    cw, ch = (ax1 - ax0) // cols, (ay1 - ay0) // rows
    titles = rng.sample(TITLES[:4], min(n, 4)) if n <= 4 and rng.random() < 0.7 else \
        [rng.choice(TITLES) for _ in range(n)]
    for k, (rgb, room) in enumerate(plans):
        r, c = divmod(k, cols)
        cb = content_box(rgb, room)
        if cb is None:
            continue
        x0, y0, x1, y1 = cb
        crop = rgb[y0:y1, x0:x1]
        tsp = int(ch * 0.12)                                       # room for the title under the view
        s = min((cw * 0.92) / crop.shape[1], (ch * 0.92 - tsp) / crop.shape[0]) * rng.uniform(0.7, 1.0)
        if s <= 0:
            continue
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        hh, ww = crop.shape[:2]
        px = ax0 + c * cw + (cw - ww) // 2 + rng.randint(-cw // 20, cw // 20)
        py = ay0 + r * ch + (ch - tsp - hh) // 2 + rng.randint(-ch // 30, ch // 30)
        px, py = max(ax0, min(px, ax0 + (c + 1) * cw - ww)), max(ay0, py)
        if py + hh > H or px + ww > W:
            continue
        region = img[py:py + hh, px:px + ww]
        np.minimum(region, crop, out=region)                       # ink over paper
        boxes.append(("plan_view", px, py, px + ww, py + hh))
        if rng.random() < 0.9:                                      # its title
            sc = rng.uniform(0.9, 1.6) * W / 3300
            tw, th_ = cv2.getTextSize(titles[k], cv2.FONT_HERSHEY_DUPLEX, sc, 2)[0]
            tx, ty = px + ww // 2 - tw // 2, py + hh + int(tsp * 0.6)
            if ty < H - m:
                _text(img, titles[k], tx, ty, sc, rng, ink)
                if rng.random() < 0.6:
                    cv2.line(img, (tx, ty + 8), (tx + tw, ty + 8), ink, 2)
                if rng.random() < 0.5:
                    _text(img, "SCALE 1:{}".format(rng.choice([50, 100, 150, 200])), tx, ty + int(40 * W / 3300),
                          sc * 0.6, rng, ink)
    if rng.random() < 0.6:
        _north(img, ax0 + 60, ay0 + 70, int(40 * W / 3300), ink)
    return img, boxes


def yolo_lines(boxes, W, H):
    out = []
    for cls, x0, y0, x1, y1 in boxes:
        cx, cy, w, h = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H
        if w > 0.005 and h > 0.005:
            out.append("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(CLASSES.index(cls), cx, cy, w, h))
    return out
