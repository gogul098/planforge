"""Structured3D (3,500 professionally designed apartments, 3D structure annotations) -> floor-plan model cache.

Each house's annotation_3d.json is turned into a drawn plan + label maps by training/s3d (s3d_convert: rooms,
walls with their real thickness, doors / windows; s3d_pairs.render: the drawing at 50 px/m in a random drafting
style). It teaches:
    room head   all room types, walls, background outside the flat
    icon head   Door, Window
    wall type   (derived in the data loader: walls the outside touches are exterior)
    level       every room floor is level 0 here: "Normal" (the counter-examples for raised / open-to-below)
No junction labels. Halls, warehouses and broken scenes are skipped (s3d_convert.is_junk).

Licence: Structured3D is for NON-COMMERCIAL research and education only and may NOT be redistributed in any
form. So this cache is rebuilt in every run from the copy attached to the notebook (a private Kaggle dataset)
and is never uploaded anywhere.
"""
import glob
import json
import os
import random
import sys
import zipfile
from multiprocessing import Pool

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "s3d"))

LISTING = "s3d_index.json"


def find_root(search=("/kaggle/input", "/kaggle/working", "/tmp")):
    """The folder holding scene_xxxxx/annotation_3d.json, extracting the zip when only the zip is attached."""
    for base in search:
        for p in glob.glob(os.path.join(base, "**", "scene_00000", "annotation_3d.json"), recursive=True):
            return os.path.dirname(os.path.dirname(p))
    for base in search:
        for z in glob.glob(os.path.join(base, "**", "*tructured3D*annotation*3d*.zip"), recursive=True):
            out = os.path.join("/kaggle/working" if os.path.isdir("/kaggle/working") else "/tmp", "s3d_raw")
            if not os.path.exists(os.path.join(out, "Structured3D")):
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(out)
            return os.path.join(out, "Structured3D")
    return None


def _one(job):
    root, scene, data_dir, max_side = job
    import s3d_convert
    import s3d_pairs
    try:
        res = s3d_convert.convert(os.path.join(root, scene, "annotation_3d.json"))
        if s3d_convert.is_junk(res):
            return None
        img, room, icon = s3d_pairs.render(res, hash(scene) & 0xffff)
        level = np.full(room.shape, 255, np.uint8)
        floor = (room != 0) & (room != 2)                     # every room (not background, not wall)
        level[floor] = 0
        h, w = room.shape
        s = min(1.0, max_side / max(h, w))
        if s < 1.0:
            size = (int(w * s), int(h * s))
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            room, icon, level = [cv2.resize(m, size, interpolation=cv2.INTER_NEAREST) for m in (room, icon, level)]
        sid = "s3d_" + scene
        cv2.imwrite(os.path.join(data_dir, sid + "_img.png"), img)
        cv2.imwrite(os.path.join(data_dir, sid + "_lbl.png"), np.dstack([room, icon, level]))
        return sid
    except Exception:
        return None


def prepare_s3d(cache_root, max_side=1536, limit=None, val_fraction=0.05, seed=0, workers=None, log=print):
    """Add Structured3D to the cache; returns the merged index, or None when no Structured3D copy is attached."""
    from fp_prep import _merge_index
    listing = os.path.join(cache_root, LISTING)
    if os.path.exists(listing):
        return _merge_index(cache_root, json.load(open(listing)))
    root = find_root()
    if root is None:
        log("Structured3D not attached (add your private Kaggle dataset with Structured3D_annotation_3d.zip); "
            "training continues without it.")
        return None
    scenes = sorted(d for d in os.listdir(root) if d.startswith("scene_"))
    if limit:
        scenes = scenes[:limit]
    data_dir = os.path.join(cache_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    log(f"Converting Structured3D: {len(scenes)} houses (drawn at 50 px/m; kept private, never uploaded) ...")
    jobs = [(root, s, data_dir, max_side) for s in scenes]
    ids = []
    with Pool(workers or os.cpu_count() or 2) as p:
        for k, sid in enumerate(p.imap_unordered(_one, jobs, chunksize=8), 1):
            if sid:
                ids.append(sid)
            if k % 500 == 0 or k == len(jobs):
                log(f"  structured3d {k}/{len(jobs)} ({len(ids)} kept)")
    ids.sort()
    rng = random.Random(seed)
    out = {"train": [], "val": []}
    for sid in ids:
        out["val" if rng.random() < val_fraction else "train"].append(
            {"id": sid, "source": "structured3d", "level": True})
    json.dump(out, open(listing, "w"))
    log(f"  structured3d: train {len(out['train'])}, val {len(out['val'])}")
    return _merge_index(cache_root, out)
