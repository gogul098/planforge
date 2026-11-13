"""PlanForge: floor plan (DXF / PDF / PNG / JPG) -> exact 3D + Revit. FastAPI backend."""
import json
import os
import shutil
import tempfile
import threading
import time
import traceback
import uuid
import zipfile
import contextlib
import io

os.environ["P3D_NO_MAIN"] = "1"
os.environ["P3D_NO_INSTALL"] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

import pipeline as P

HERE = os.path.dirname(os.path.abspath(__file__))
JOBS = os.environ.get("PLANFORGE_JOBS") or os.path.join(tempfile.gettempdir(), "planforge_jobs")
os.makedirs(JOBS, exist_ok=True)
LOCK = threading.Lock()          # the pipeline uses module-level settings -> one plan at a time
STATUS = {}                      # job -> {"state", "progress", "msg", ...}
UNIT_M = {0: None, 1: 0.0254, 2: 0.3048, 4: 0.001, 5: 0.01, 6: 1.0, 14: 0.1}   # DXF $INSUNITS -> metres
OUT_FILES = ("plan.json", "model.ifc", "model.glb", "model_faithful.glb", "preview.png", "build_in_revit.py")
MAX_UPLOAD = 50 * 2 ** 20                                   # bytes
JOB_TTL_S = float(os.environ.get("PLANFORGE_JOB_TTL_H", "24")) * 3600   # finished jobs are deleted after this

app = FastAPI(title="PlanForge")


@app.on_event("startup")
def _warm():
    """Load the model once when the container starts, so no request waits for the download."""
    def go():
        try:
            P.load_model(force_check=True)
        except Exception as e:
            print("model preload failed (will retry on first plan):", e)
    threading.Thread(target=go, daemon=True).start()
    threading.Thread(target=_cleanup_loop, daemon=True).start()


def cleanup_jobs(now=None):
    """Delete job folders older than JOB_TTL_S so the instance's disk does not fill up. Running jobs are kept."""
    now = now or time.time()
    removed = 0
    for job in os.listdir(JOBS):
        d = os.path.join(JOBS, job)
        if not os.path.isdir(d) or STATUS.get(job, {}).get("state") in ("queued", "running"):
            continue
        try:
            age = now - os.path.getmtime(d)
            sp = os.path.join(d, "status.json")
            if os.path.exists(sp):
                age = min(age, now - os.path.getmtime(sp))
        except OSError:
            continue
        if age > JOB_TTL_S:
            shutil.rmtree(d, ignore_errors=True)
            STATUS.pop(job, None)
            removed += 1
    if removed:
        print("cleanup: removed {} old job(s)".format(removed))
    return removed


def _cleanup_loop():
    while True:
        try:
            cleanup_jobs()
        except Exception as e:
            print("cleanup failed:", e)
        time.sleep(3600)

from fastapi.middleware.cors import CORSMiddleware  # noqa: E402  the website may be hosted elsewhere
# The website lives on Vercel; this engine only answers /api/*. Set ALLOWED_ORIGINS (comma separated) to lock it to
# your site(s), e.g. "https://planforge.vercel.app,https://planforge.tech". Default "*" = any site.
_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"])


# ----------------------------------------------------------------------------- input formats
def dxf_to_png(path, out, unit_override=None, max_px=7000):
    """Rasterise a DXF exactly: every curve flattened in model units, drawn black on white.
    Returns (png, metres per pixel of the PNG)."""
    import cv2
    import ezdxf
    from ezdxf import bbox, disassemble, path as zpath

    doc = ezdxf.readfile(path)
    msp = doc.modelspace()
    upm = unit_override or UNIT_M.get(doc.header.get("$INSUNITS", 0))
    if upm is None:                           # unitless: guess from size (houses are 5-60 m)
        ext = bbox.extents(msp, fast=True)
        big = max(ext.size.x, ext.size.y) if ext.has_data else 0
        upm = 0.001 if big > 2000 else (0.01 if big > 200 else 1.0)
    polys = []
    for e in disassemble.recursive_decompose(msp):          # explodes blocks / inserts
        if e.dxftype() in ("TEXT", "MTEXT", "ATTRIB", "DIMENSION", "HATCH", "POINT", "VIEWPORT"):
            continue
        try:
            p = zpath.make_path(e)
        except Exception:
            continue
        for sub in p.sub_paths():
            pts = [(v.x, v.y) for v in sub.flattening(distance=0.002 / upm)]
            if len(pts) >= 2:
                polys.append(np.array(pts, float))
    if not polys:
        raise ValueError("DXF has no drawable lines")
    allp = np.vstack(polys)
    lo, hi = allp.min(0), allp.max(0)
    w, h = max(hi[0] - lo[0], 1e-9), max(hi[1] - lo[1], 1e-9)
    ppu = min(80 * upm, max_px / max(w, h))   # ~80 px per metre
    pad = 40
    W, H = int(np.ceil(w * ppu)) + 2 * pad, int(np.ceil(h * ppu)) + 2 * pad
    img = np.full((H, W), 255, np.uint8)
    for p in polys:
        q = np.stack([(p[:, 0] - lo[0]) * ppu + pad, H - ((p[:, 1] - lo[1]) * ppu + pad)], 1)
        cv2.polylines(img, [np.round(q * 4).astype(np.int32)], False, 0, 2, cv2.LINE_AA, shift=2)
    cv2.imwrite(out, img)
    return out, upm / ppu


def load_input(path, ext, dxf_units):
    """-> (png path, metres per pixel of that png or None)"""
    if ext == ".dxf":
        return dxf_to_png(path, path + ".png", UNIT_M.get(int(dxf_units)) if dxf_units else None)
    if ext == ".dwg":
        raise HTTPException(415, "DWG is a closed format: in AutoCAD use SAVEAS -> DXF, then upload the DXF.")
    if ext == ".pdf":
        import pymupdf
        doc = pymupdf.open(path)
        doc[0].get_pixmap(dpi=300).save(path + ".png")
        return path + ".png", None
    if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"):
        return path, None
    raise HTTPException(415, "Unsupported file type " + ext)


# ----------------------------------------------------------------------------- conversion job
def _set(job, **kw):
    STATUS.setdefault(job, {}).update(kw)
    try:                                   # persisted so a restarted server can still serve the job
        json.dump(STATUS[job], open(os.path.join(JOBS, job, "status.json"), "w"))
    except Exception:
        pass


def _clean(text):
    return "\n".join(l for l in text.splitlines() if l.strip() and "Warning" not in l and "warn(" not in l)


def _package(work, out):
    shutil.copy(os.path.join(HERE, "build_in_revit.py"), os.path.join(out, "build_in_revit.py"))
    zpath = os.path.join(work, "floorplan_3d.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in OUT_FILES:
            if os.path.exists(os.path.join(out, f)):
                z.write(os.path.join(out, f), f)


def _shell_glb(plan, out):
    """Building without furniture: the viewer draws furniture itself so it can be edited."""
    P.build_glb(plan, os.path.join(out, "shell.glb"), extra=None)


def run_job(job, src, ext, opt):
    work = os.path.join(JOBS, job)
    out = os.path.join(work, "out")
    log = io.StringIO()
    t0 = time.time()
    with LOCK:
        try:
            _set(job, state="running", progress=0.05, msg="Reading the plan")
            png, mpp_src = load_input(src, ext, opt.get("dxf_units"))
            P.OUT_DIR = out
            P.METERS_PER_PIXEL = None
            P.BUILDING_WIDTH_M = float(opt["building_width"]) if opt.get("building_width") else None
            P.WALL_HEIGHT = float(opt.get("wall_height", 3.0))
            P.DOOR_HEIGHT = float(opt.get("door_height", 2.1))
            P.WINDOW_SILL = float(opt.get("sill", 0.9))
            P.WINDOW_HEAD = float(opt.get("head", 2.1))
            P.EXTERNAL_WALL_M = float(opt["ext_wall"]) / 1000 if opt.get("ext_wall") else None
            P.INTERNAL_WALL_M = float(opt["int_wall"]) / 1000 if opt.get("int_wall") else None
            P.MAKE_FURNITURE = opt.get("furniture", "copy") == "copy"
            P.PLINTH_HEIGHT = float(opt["plinth"]) if opt.get("plinth") not in (None, "") else None
            # speed on a small CPU (each can be turned back on with an environment variable):
            P.FLIP_TTA = os.environ.get("PLANFORGE_FLIP_TTA", "0") == "1"          # mirrored second pass: 2x model time
            P.PREVIEW_3D = os.environ.get("PLANFORGE_PREVIEW_3D", "0") == "1"      # 3D renders in preview.png
            P.FURNITURE_ZOOMS = tuple(float(z) for z in os.environ.get("PLANFORGE_FURNITURE_ZOOMS", "1.0").split(","))
            P.TIMES.clear()
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                _set(job, progress=0.15, msg="Detecting walls, doors, windows, rooms")
                if os.environ.get("P3D_TEST_MASKS"):                 # offline test hook
                    d = os.environ["P3D_TEST_MASKS"]
                    img, room, icon = (np.load(os.path.join(d, n + ".npy")) for n in ("img", "room", "icon"))
                    op = os.path.join(d, "orig.npy")
                    P.LAST_ORIGINAL = np.load(op) if os.path.exists(op) else img
                else:
                    img, room, icon = P.predict(png)
                if mpp_src:                                          # DXF: exact scale from the drawing units
                    P.METERS_PER_PIXEL = mpp_src * P.LAST_ORIGINAL.shape[1] / img.shape[1]
                _set(job, progress=0.45, msg="Reading dimensions, solving geometry, furniture")
                P.main(img, room, icon, original=P.LAST_ORIGINAL)
            plan = json.load(open(os.path.join(out, "plan.json")))
            _shell_glb(plan, out)
            _package(work, out)
            _set(job, state="done", progress=1.0, msg="Done in {:.0f} s".format(time.time() - t0),
                 report=_clean(log.getvalue()), input_kind=ext.strip("."), exact_scale=bool(mpp_src))
        except Exception as e:
            traceback.print_exc()
            _set(job, state="error", msg=(e.detail if isinstance(e, HTTPException) else
                                          "{}: {}".format(type(e).__name__, str(e)[:300])),
                 report=_clean(log.getvalue())[-4000:])


@app.post("/api/convert")
async def convert(file: UploadFile = File(...), options: str = Form("{}")):
    ext = os.path.splitext(file.filename or "")[1].lower()
    job = uuid.uuid4().hex[:12]
    work = os.path.join(JOBS, job)
    os.makedirs(work)
    src = os.path.join(work, "input" + ext)
    size = 0
    with open(src, "wb") as f:                     # stop as soon as the limit is passed, not after the whole upload
        while True:
            chunk = await file.read(2 ** 20)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD:
                f.close()
                shutil.rmtree(work, ignore_errors=True)
                raise HTTPException(413, "File is larger than 50 MB")
            f.write(chunk)
    opt = json.loads(options or "{}")
    _set(job, state="queued", progress=0.0, msg="Waiting in queue", name=file.filename)
    threading.Thread(target=run_job, args=(job, src, ext, opt), daemon=True).start()
    return {"job": job}


@app.get("/api/status/{job}")
def status(job: str, wait: float = 0):
    """wait > 0: long poll - the answer comes as soon as the status changes, or after `wait` seconds
    (max 25 s, well under Azure's 230 s request limit). One request every ~25 s instead of every second."""
    job = os.path.basename(job)
    sp = os.path.join(JOBS, job, "status.json")
    if job not in STATUS and os.path.exists(sp):
        STATUS[job] = json.load(open(sp))
        if STATUS[job].get("state") in ("queued", "running"):
            STATUS[job].update(state="error", msg="Server restarted during conversion: please convert again")
    if job not in STATUS:
        raise HTTPException(404, "unknown job")
    before = json.dumps(STATUS[job], sort_keys=True, default=str)
    end = time.time() + min(max(wait, 0), 25)
    while time.time() < end and STATUS[job].get("state") not in ("done", "error"):
        if json.dumps(STATUS[job], sort_keys=True, default=str) != before:
            break
        time.sleep(0.5)
    return STATUS[job]


import logging  # noqa: E402


class _QuietStatus(logging.Filter):            # keep the log readable: no line per status check
    def filter(self, rec):
        return "/api/status/" not in rec.getMessage()


logging.getLogger("uvicorn.access").addFilter(_QuietStatus())


@app.get("/api/files/{job}/{name}")
def files(job: str, name: str):
    base = os.path.join(JOBS, os.path.basename(job))
    p = os.path.join(base, "floorplan_3d.zip") if name == "floorplan_3d.zip" else \
        os.path.join(base, "out", os.path.basename(name))
    if not os.path.exists(p):
        raise HTTPException(404, "not found")
    return FileResponse(p, filename=name, headers={"Cache-Control": "no-store"})


# ----------------------------------------------------------------------------- furniture edits
def furniture_from_edit(items):
    """[{type, center:[x,y], width, depth, angle_deg (long axis)}] -> plan.json furniture records."""
    out = []
    for i, it in enumerate(items):
        t = it["type"]
        height, cat, _ = P.FURN_SPEC.get(t, (0.8, "furniture", None))
        a = np.radians(float(it.get("angle_deg", 0.0)))
        L = np.array([np.cos(a), np.sin(a)])
        S = np.array([-L[1], L[0]])
        c = np.array(it["center"], float)
        w, d = float(it["width"]), float(it["depth"])
        corners = [c + sa * L * w / 2 + sb * S * d / 2 for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        front = np.array(it.get("front") or S, float)
        out.append({"id": i, "type": t, "category": cat, "source": it.get("source", "user"),
                    "confidence": it.get("confidence", 1.0), "center": [round(float(v), 4) for v in c],
                    "width": round(w, 3), "depth": round(d, 3), "height": float(it.get("height") or height),
                    "front": [round(float(v), 4) for v in front],
                    "rotation_deg": round(float(np.degrees(np.arctan2(front[1], front[0]))), 2),
                    "corners": [[round(float(x), 4), round(float(y), 4)] for x, y in corners]})
    return out


@app.post("/api/export/{job}")
def export(job: str, body: dict):
    work = os.path.join(JOBS, os.path.basename(job))
    out = os.path.join(work, "out")
    jp = os.path.join(out, "plan.json")
    if not os.path.exists(jp):
        raise HTTPException(404, "unknown job")
    plan = json.load(open(jp))
    plan["furniture"] = furniture_from_edit(body.get("furniture", []))
    json.dump(plan, open(jp, "w"), indent=1)
    with LOCK:
        fmesh = P.furniture_meshes(plan["furniture"]) if plan["furniture"] else {}
        P.build_glb(plan, os.path.join(out, "model.glb"), extra=fmesh)
        try:
            P.build_ifc(plan, os.path.join(out, "model.ifc"))
        except Exception as e:
            print("IFC export failed:", e)
        _package(work, out)
    return {"ok": True, "furniture": len(plan["furniture"])}


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/version")
def version():
    """Which build is running: the git commit it was built from and when."""
    return {"commit": os.environ.get("PLANFORGE_VERSION", "dev"), "built_at": os.environ.get("PLANFORGE_BUILT_AT", "unknown"),
            "model_revision": str(getattr(P, "_MODEL", {}).get("rev") or "not loaded yet")}


@app.get("/api/catalog")
def catalog():
    return {k: {"height": v[0], "category": v[1]} for k, v in P.FURN_SPEC.items()}


@app.get("/")
def root():
    """API only: the website is hosted separately (Vercel)."""
    site = os.environ.get("FRONTEND_URL")
    if site:
        from fastapi.responses import RedirectResponse
        return RedirectResponse(site)
    return {"service": "PlanForge engine", "api": ["/api/health", "/api/version", "/api/convert", "/api/status/{job}",
                                                    "/api/files/{job}/{name}"]}
