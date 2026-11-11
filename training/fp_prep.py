"""Data acquisition + one-time preprocessing of CubiCasa5K into a fast training cache.

Order of preference:
  1. cache tar already in your private HF dataset repo  -> download + untar (fast resume)
  2. CubiCasa5K attached as a Kaggle dataset            -> parse from /kaggle/input
  3. official Zenodo zip (5.5 GB, md5-checked)           -> download, parse
Then the cache is uploaded to the HF dataset repo so later runs skip steps 2/3.
"""
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import zipfile
from multiprocessing import Pool

import cv2
import numpy as np

ZENODO_URL = "https://zenodo.org/records/2613548/files/cubicasa5k.zip?download=1"
ZENODO_MD5 = "0ce0b203d1e3c125b51087b219bd23b9"
CACHE_VERSION = "v1"

_PARSER_DIR = None


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ----------------------------------------------------------------------------- raw data
def find_raw_root(search_roots=("/kaggle/input",)):
    for base in search_roots:
        if not os.path.isdir(base):
            continue
        for txt in glob.glob(os.path.join(base, "**", "train.txt"), recursive=True):
            root = os.path.dirname(txt)
            if (os.path.exists(os.path.join(root, "val.txt"))
                    and glob.glob(os.path.join(root, "high_quality*"))):
                return root
    return None


def md5sum(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _download(url, dst, label):
    """Resumable HTTP download that prints a progress line every ~2% (log-friendly)."""
    import urllib.request
    have = os.path.getsize(dst) if os.path.exists(dst) else 0
    req = urllib.request.Request(url, headers={"User-Agent": "floorplan-trainer",
                                               **({"Range": f"bytes={have}-"} if have else {})})
    with urllib.request.urlopen(req, timeout=120) as r:
        if have and r.status != 206:  # server ignored resume -> start over
            have = 0
        total = have + int(r.headers.get("Content-Length") or 0)
        t0, done, next_pct = time.time(), have, 0
        with open(dst, "ab" if have else "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                pct = 100 * done / total if total else 0
                if pct >= next_pct or done == total:
                    sp = (done - have) / max(time.time() - t0, 1e-6)
                    eta = (total - done) / sp if sp > 0 else 0
                    bar = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
                    log(f"{label} [{bar}] {pct:5.1f}%  {done / 1e9:.2f}/{total / 1e9:.2f} GB  "
                        f"{sp / 1e6:.1f} MB/s  ETA {int(eta // 60)}m{int(eta % 60):02d}s")
                    next_pct = int(pct) + 2
    if total and os.path.getsize(dst) < total:
        raise IOError("download incomplete")


def download_zenodo(work_dir, retries=6):
    os.makedirs(work_dir, exist_ok=True)
    zpath = os.path.join(work_dir, "cubicasa5k.zip")
    for attempt in range(1, retries + 1):
        log(f"Downloading CubiCasa5K from Zenodo, 5.5 GB (attempt {attempt}/{retries}) ...")
        try:
            _download(ZENODO_URL, zpath, "download")
            log("Download finished, verifying md5 (~1 min) ...")
            if md5sum(zpath) == ZENODO_MD5:
                log("md5 OK")
                return zpath
            log("md5 mismatch -> deleting and retrying")
            os.remove(zpath)
        except Exception as e:
            log(f"Download interrupted ({type(e).__name__}: {e}); resuming ...")
        time.sleep(min(60, 10 * attempt))
    raise RuntimeError("Could not download CubiCasa5K from Zenodo. Attach the dataset on Kaggle "
                       "(Add Data -> search 'cubicasa5k') and rerun.")


def extract_needed(zpath, out_dir):
    """Extract only split files, F1_scaled.png and model.svg (skips the big originals)."""
    log("Extracting needed files ...")
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist()
                 if n.endswith((".txt", "F1_scaled.png", "model.svg"))]
        for k, n in enumerate(names, 1):
            z.extract(n, out_dir)
            if k % 1000 == 0 or k == len(names):
                log(f"  extracted {k}/{len(names)} files ({100 * k / len(names):.0f}%)")
    root = find_raw_root([out_dir])
    if root is None:
        raise RuntimeError("Extracted archive has unexpected layout (train.txt not found).")
    log(f"Extracted {len(names)} files, root = {root}")
    return root


# ----------------------------------------------------------------------------- parser
def setup_parser(parser_dir):
    """Vendor the official CubiCasa5K SVG parser (house.py + svg_utils.py) without its
    lmdb/torch package imports."""
    if os.path.exists(os.path.join(parser_dir, "house.py")):
        return parser_dir
    tmp = parser_dir + "_repo"
    if not os.path.exists(tmp):
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/CubiCasa/CubiCasa5k.git", tmp], check=True)
    os.makedirs(parser_dir, exist_ok=True)
    for fn in ("house.py", "svg_utils.py"):
        src = open(os.path.join(tmp, "floortrans", "loaders", fn)).read()
        src = src.replace("from floortrans.loaders.svg_utils import", "from svg_utils import")
        src = src.replace("is 'H'", "== 'H'").replace("is 'V'", "== 'V'")
        open(os.path.join(parser_dir, fn), "w").write(src)
    return parser_dir


def _init_worker(parser_dir):
    global _PARSER_DIR
    _PARSER_DIR = parser_dir
    if parser_dir not in sys.path:
        sys.path.insert(0, parser_dir)
    cv2.setNumThreads(1)
    import warnings, logging
    warnings.filterwarnings("ignore")
    logging.getLogger().setLevel(logging.ERROR)


def _parse_stairs(svg_path, shape, icons):
    """Stairs are commented out in the official parser; we paint Flight/Winding polygons
    as icon class 11 where no other icon is present. Returns number of stair polygons."""
    from xml.dom import minidom
    from svg_utils import get_polygon
    h, w = shape
    n = 0
    dom = minidom.parse(svg_path)
    for g in dom.getElementsByTagName("g"):
        if "Stairs" not in g.getAttribute("class").split(" "):
            continue
        parts = [c for c in g.childNodes if c.nodeType == c.ELEMENT_NODE
                 and c.getAttribute("class") in ("Flight", "Winding")] or [g]
        for c in parts:
            try:
                rr, cc = get_polygon(c)
            except Exception:
                continue
            keep = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            rr, cc = rr[keep], cc[keep]
            if len(rr) == 0:
                continue
            sel = icons[rr, cc] == 0
            icons[rr[sel], cc[sel]] = 11
            n += 1
    return n


def _heat_group(ch):
    return 0 if ch <= 12 else (1 if ch <= 16 else 2)


def process_one(args):
    raw_root, folder, out_dir, max_side = args
    sid = folder.strip("/").replace("/", "__")
    try:
        import house as cc_house
        d = os.path.join(raw_root, folder.strip("/"))
        img = cv2.imread(os.path.join(d, "F1_scaled.png"), cv2.IMREAD_UNCHANGED)
        if img is None:
            return {"id": sid, "error": "missing image"}
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # composite transparency on white
            a = img[:, :, 3:4].astype(np.float32) / 255.0
            img = (img[:, :, :3].astype(np.float32) * a + 255.0 * (1 - a)).astype(np.uint8)
        h, w = img.shape[:2]
        house = cc_house.House(os.path.join(d, "model.svg"), h, w)
        room = house.walls.astype(np.uint8)
        icon = house.icons.astype(np.uint8)
        n_stairs = _parse_stairs(os.path.join(d, "model.svg"), (h, w), icon)
        pts = []
        for ch, plist in house.get_heatmap_dict().items():
            g = _heat_group(int(ch))
            pts += [(float(x), float(y), g) for x, y in plist]
        pts = np.array(pts, dtype=np.float32).reshape(-1, 3)

        s = min(1.0, max_side / max(h, w))
        if s < 1.0:
            nw, nh = int(round(w * s)), int(round(h * s))
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            room = cv2.resize(room, (nw, nh), interpolation=cv2.INTER_NEAREST)
            icon = cv2.resize(icon, (nw, nh), interpolation=cv2.INTER_NEAREST)
            pts[:, :2] *= s
            h, w = nh, nw
        cv2.imwrite(os.path.join(out_dir, sid + "_img.png"), img)  # BGR on disk
        lbl = np.dstack([room, icon, np.zeros_like(room)])
        cv2.imwrite(os.path.join(out_dir, sid + "_lbl.png"), lbl)
        np.save(os.path.join(out_dir, sid + "_pts.npy"), pts)
        return {"id": sid, "h": h, "w": w, "n_pts": int(len(pts)), "stairs": n_stairs}
    except Exception as e:
        return {"id": sid, "error": f"{type(e).__name__}: {e}",
                "tb": traceback.format_exc(limit=3)}


def build_cache(raw_root, cache_root, parser_dir, max_side, workers=None):
    out_dir = os.path.join(cache_root, "data")
    os.makedirs(out_dir, exist_ok=True)
    workers = workers or os.cpu_count() or 2
    index, failures, stairs_total = {}, [], 0
    for split in ("train", "val", "test"):
        folders = [l.strip() for l in open(os.path.join(raw_root, f"{split}.txt")) if l.strip()]
        log(f"Parsing {split}: {len(folders)} floorplans with {workers} workers ...")
        t0, items = time.time(), []
        jobs = [(raw_root, f, out_dir, max_side) for f in folders]
        with Pool(workers, initializer=_init_worker, initargs=(parser_dir,)) as pool:
            for k, r in enumerate(pool.imap_unordered(process_one, jobs, chunksize=4), 1):
                if "error" in r:
                    failures.append(r)
                else:
                    items.append(r)
                    stairs_total += r["stairs"]
                if k % 100 == 0 or k == len(jobs):
                    el = time.time() - t0
                    log(f"  parsing {split}: {k}/{len(jobs)} ({100 * k / len(jobs):.0f}%)  "
                        f"{el / k:.2f}s/plan  ETA {el / k * (len(jobs) - k) / 60:.1f} min")
        items.sort(key=lambda r: r["id"])
        index[split] = items
    n_total = sum(len(v) for v in index.values()) + len(failures)
    log(f"Done. ok={n_total - len(failures)} failed={len(failures)} stair polygons={stairs_total}")
    for f in failures[:5]:
        log("  failure:", f["id"], f["error"])
    if len(failures) > 0.05 * n_total:
        raise RuntimeError(f"Too many parse failures ({len(failures)}/{n_total}); see log above.")
    if stairs_total == 0:
        log("WARNING: no stairs parsed; the Stairs class will stay empty.")
    index["meta"] = {"max_side": max_side, "version": CACHE_VERSION,
                     "failures": [f["id"] for f in failures], "stairs_polygons": stairs_total}
    with open(os.path.join(cache_root, "index.json"), "w") as f:
        json.dump(index, f)
    return index


# ----------------------------------------------------------------------------- HF cache
def cache_name(max_side):
    return f"cubicasa5k_cache_side{max_side}_{CACHE_VERSION}.tar"


def try_download_cache(api, repo, max_side, cache_root, token):
    from huggingface_hub import hf_hub_download
    name = cache_name(max_side)
    try:
        files = api.list_repo_files(repo, repo_type="dataset", token=token)
    except Exception as e:
        log(f"Could not list dataset repo ({e}); will build cache.")
        return False
    if name not in files:
        return False
    log(f"Found cached data on HF ({name}); downloading ...")
    for attempt in range(3):
        try:
            p = hf_hub_download(repo, name, repo_type="dataset", token=token,
                                local_dir=os.path.dirname(cache_root.rstrip("/")) + "/hf_dl")
            os.makedirs(cache_root, exist_ok=True)
            with tarfile.open(p) as t:
                t.extractall(cache_root)
            os.remove(p)
            ok = os.path.exists(os.path.join(cache_root, "index.json"))
            log("Cache restored." if ok else "Cache tar was incomplete.")
            return ok
        except Exception as e:
            log(f"Cache download failed ({e}); retry {attempt + 1}/3")
            time.sleep(15)
    return False


def upload_cache(api, repo, max_side, cache_root, token):
    name = cache_name(max_side)
    tar_path = os.path.join(os.path.dirname(cache_root.rstrip("/")), name)
    log("Packing cache for HF upload ...")
    with tarfile.open(tar_path, "w") as t:
        t.add(os.path.join(cache_root, "index.json"), arcname="index.json")
        t.add(os.path.join(cache_root, "data"), arcname="data")
    log(f"Uploading {os.path.getsize(tar_path) / 1e9:.2f} GB to {repo} (dataset) ...")
    for attempt in range(3):
        try:
            api.upload_file(path_or_fileobj=tar_path, path_in_repo=name, repo_id=repo,
                            repo_type="dataset", token=token,
                            commit_message=f"Preprocessed CubiCasa5K cache (side {max_side})")
            log("Cache uploaded.")
            break
        except Exception as e:
            log(f"Cache upload failed ({e}); retry {attempt + 1}/3")
            time.sleep(30)
    else:
        log("WARNING: cache upload failed; training continues (next run will rebuild).")
    os.remove(tar_path)


def prepare_data(cache_root, work_dir, parser_dir, max_side, api, dataset_repo, token):
    if os.path.exists(os.path.join(cache_root, "index.json")):
        log("Local cache already present.")
        return json.load(open(os.path.join(cache_root, "index.json")))
    if api is not None and try_download_cache(api, dataset_repo, max_side, cache_root, token):
        return json.load(open(os.path.join(cache_root, "index.json")))
    raw = find_raw_root()
    if raw:
        log(f"Using Kaggle-attached CubiCasa5K at {raw}")
    else:
        zpath = download_zenodo(work_dir)
        raw = extract_needed(zpath, os.path.join(work_dir, "raw"))
        os.remove(zpath)
    setup_parser(parser_dir)
    index = build_cache(raw, cache_root, parser_dir, max_side)
    if raw.startswith(work_dir):
        shutil.rmtree(os.path.join(work_dir, "raw"), ignore_errors=True)
    if api is not None:
        upload_cache(api, dataset_repo, max_side, cache_root, token)
    return index
