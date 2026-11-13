"""Furniture detector: FloorPlanCAD -> YOLO-OBB conversion, and tiled inference on plans."""
import json
import math
import os

import cv2
import numpy as np

# FloorPlanCAD classes we train on (doors / windows / stairs / walls come from the floor-plan model)
FURN_CLASSES = ["bed", "sofa", "chair", "table", "bedside_cupboard", "tv_cabinet", "half_height_cabinet",
                "high_cabinet", "wardrobe", "sink", "bath", "bath_tub", "toilet", "squat_toilet", "urinal",
                "gas_stove", "refrigerator", "washing_machine"]
ZOOM_FACTORS = (0.75, 1.0, 1.4)
# typical real sizes (long side, metres) used to measure the drawing scale of the dataset
TYPICAL_LONG_M = {"bed": 2.0, "toilet": 0.7, "bath_tub": 1.7, "sofa": 2.0, "refrigerator": 0.7}


def norm_label(s):
    return str(s).strip().lower().replace(" ", "_").replace("-", "_")


def to_rgb_white(img):
    """PNG with alpha -> RGB on white (FloorPlanCAD renders are 4-channel)."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        a = img[:, :, 3:4].astype(np.float32) / 255.0
        return (img[:, :, :3].astype(np.float32) * a + 255.0 * (1 - a)).astype(np.uint8)
    return img


def det_to_obb(bbox, mask, W, H):
    """FiftyOne detection (normalised [x,y,w,h] + cropped mask) -> 4 corner points in pixels.
    The rotated rectangle is fitted to the drawn strokes, so it follows the furniture's rotation."""
    bx, by, bw, bh = [float(v) for v in bbox]
    x0, y0, w, h = bx * W, by * H, bw * W, bh * H
    pts = None
    if mask is not None:
        m = np.asarray(mask)
        if m.ndim == 3:
            m = m[..., 0]
        ys, xs = np.nonzero(m)
        if len(xs) >= 5:
            sx, sy = w / m.shape[1], h / m.shape[0]
            pts = np.stack([x0 + (xs + 0.5) * sx, y0 + (ys + 0.5) * sy], 1).astype(np.float32)
    if pts is None:
        pts = np.array([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]], np.float32)
    rect = cv2.minAreaRect(pts)
    (cw, ch) = rect[1]
    if cw < 2 or ch < 2:                     # degenerate -> axis-aligned box
        rect = ((x0 + w / 2, y0 + h / 2), (max(w, 2), max(h, 2)), 0.0)
    return cv2.boxPoints(rect), rect


def obb_line(cls_id, corners, W, H):
    c = np.clip(corners, 0, [W - 1, H - 1]) / [W, H]
    return "{} ".format(cls_id) + " ".join("{:.6f}".format(v) for v in c.reshape(-1))


def write_sample(img_path, dets, out_img, out_lbl, stats):
    """dets: list of (label, bbox, mask). Writes RGB jpg + YOLO-OBB label file. Returns #objects."""
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return 0
    img = to_rgb_white(img)
    H, W = img.shape[:2]
    lines = []
    for label, bbox, mask in dets:
        lab = norm_label(label)
        if lab not in FURN_CLASSES:
            continue
        corners, rect = det_to_obb(bbox, mask, W, H)
        lines.append(obb_line(FURN_CLASSES.index(lab), corners, W, H))
        stats.setdefault(lab, []).append(max(rect[1]))
    cv2.imwrite(out_img, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    with open(out_lbl, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def dataset_px_per_m(stats):
    """Drawing scale of the dataset from typical object sizes (median of several classes)."""
    est = [np.median(stats[k]) / v for k, v in TYPICAL_LONG_M.items() if len(stats.get(k, [])) >= 20]
    return float(np.median(est)) if est else None


# ----------------------------------------------------------------------------- inference on a plan
def _onnx_names(sess):
    import ast
    return ast.literal_eval(sess.get_modelmeta().custom_metadata_map.get("names", "{}"))


def _obb_onnx(sess, crop_rgb, conf):
    """Ultralytics YOLO-OBB exported to ONNX: output (1, 4 + classes + 1, N) = cx, cy, w, h, class scores, angle (rad).
    Returns [(corners 4x2 in crop px, class id, score)]; overlaps are merged later by rotated_nms."""
    x = np.ascontiguousarray((crop_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]          # (4 + nc + 1, N)
    nc = out.shape[0] - 5
    scores = out[4:4 + nc]
    cls = scores.argmax(0)
    sc = scores[cls, np.arange(scores.shape[1])]
    keep = np.nonzero(sc >= conf)[0]
    if len(keep) > 1:                                  # same NMS Ultralytics applies (rotated, per class)
        rects = [((float(out[0, k]), float(out[1, k])), (float(out[2, k]), float(out[3, k])),
                  float(np.degrees(out[4 + nc, k]))) for k in keep]
        sel = []
        for c_ in np.unique(cls[keep]):
            idx = [j for j, k in enumerate(keep) if cls[k] == c_]
            r = cv2.dnn.NMSBoxesRotated([rects[j] for j in idx], [float(sc[keep[j]]) for j in idx], conf, 0.7)
            sel += [idx[int(j)] for j in np.array(r).reshape(-1)]
        keep = keep[sorted(sel)]
    res = []
    for k in keep:
        cx, cy, w, h = out[:4, k]
        a = float(out[4 + nc, k])
        c, s_ = np.cos(a), np.sin(a)
        dx, dy = np.array([c, s_]) * w / 2, np.array([-s_, c]) * h / 2
        ctr = np.array([cx, cy])
        res.append((np.array([ctr + dx + dy, ctr + dx - dy, ctr - dx - dy, ctr - dx + dy]), int(cls[k]), float(sc[k])))
    return res


def detect_furniture(model, img_rgb, plan_px_per_m, train_px_per_m, conf=0.35, tile=1024, overlap=0.25,
                     zooms=None, roi=None):
    """Rescale the plan so furniture has its training size, run tiled OBB detection, merge.
    zooms: factors of the training scale (default ZOOM_FACTORS); roi: (x0, y0, x1, y1) in image px to search
    (the building), None = whole sheet. Tiles with almost no ink are skipped.
    Returns list of dicts in ORIGINAL image pixels: cls, conf, corners(4x2)."""
    names = model.names if hasattr(model, "names") else _onnx_names(model)
    ox, oy = 0, 0
    if roi is not None:
        Hi, Wi = img_rgb.shape[:2]
        x0_, y0_, x1_, y1_ = [int(round(v)) for v in roi]
        x0_, y0_, x1_, y1_ = max(x0_, 0), max(y0_, 0), min(x1_, Wi), min(y1_, Hi)
        if x1_ - x0_ > 16 and y1_ - y0_ > 16:
            img_rgb, ox, oy = img_rgb[y0_:y1_, x0_:x1_], x0_, y0_
    base = train_px_per_m / plan_px_per_m if (train_px_per_m and plan_px_per_m) else 1.0
    scales = [base * f for f in (zooms or ZOOM_FACTORS)]   # several zooms, merged below
    found = []
    for s in scales:
        s = float(np.clip(s, 0.2, 6.0))
        im = cv2.resize(img_rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        H, W = im.shape[:2]
        step = int(tile * (1 - overlap))
        ys = list(range(0, max(H - tile, 0) + 1, step)) or [0]
        xs = list(range(0, max(W - tile, 0) + 1, step)) or [0]
        if ys[-1] + tile < H:
            ys.append(max(H - tile, 0))
        if xs[-1] + tile < W:
            xs.append(max(W - tile, 0))
        for y0 in ys:
            for x0 in xs:
                crop = np.full((tile, tile, 3), 255, np.uint8)
                c = im[y0:y0 + tile, x0:x0 + tile]
                if (c.min(axis=2) < 160).mean() < 0.002:       # blank paper: nothing to detect, skip the pass
                    continue
                crop[:c.shape[0], :c.shape[1]] = c
                if hasattr(model, "run"):                     # ONNX Runtime session
                    boxes = _obb_onnx(model, crop, conf)
                else:
                    res = model.predict(crop[:, :, ::-1], imgsz=tile, conf=conf, verbose=False)[0]
                    if res.obb is None or len(res.obb) == 0:
                        continue
                    pts_ = res.obb.xyxyxyxy.cpu().numpy()
                    boxes = [(pts_[k].reshape(4, 2), int(res.obb.cls[k]), float(res.obb.conf[k])) for k in range(len(pts_))]
                for p, ci, cf in boxes:
                    cx, cy = p[:, 0].mean(), p[:, 1].mean()
                    # ignore detections centred in the overlap margin of an inner tile edge
                    m = tile * overlap / 2
                    if (x0 > 0 and cx < m) or (y0 > 0 and cy < m) or \
                       (x0 + tile < W and cx > tile - m) or (y0 + tile < H and cy > tile - m):
                        continue
                    found.append({"cls": names[int(ci)], "conf": float(cf), "corners": (p + [x0, y0]) / s + [ox, oy]})
    return rotated_nms(found)


def rotated_nms(dets, iou_thr=0.4):
    from shapely.geometry import Polygon
    dets = sorted(dets, key=lambda d: -d["conf"])
    keep = []
    for d in dets:
        p = Polygon(d["corners"]).buffer(0)
        if p.area <= 0:
            continue
        ok = True
        for k in keep:
            q = k["_poly"]
            inter = p.intersection(q).area
            if inter / (p.area + q.area - inter + 1e-9) > iou_thr or inter / min(p.area, q.area) > 0.8:
                ok = False
                break
        if ok:
            d["_poly"] = p
            keep.append(d)
    for k in keep:
        k.pop("_poly", None)
    return keep
