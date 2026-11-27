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
import hashlib
import hmac
import secrets
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
PDF_MAX_SIDE = int(os.environ.get("PLANFORGE_PDF_MAX_SIDE", "5000"))  # px of the rendered PDF page (long side)
MAX_QUEUE = int(os.environ.get("PLANFORGE_MAX_QUEUE", "6"))  # plans waiting or running; more -> "busy, try again"
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
from fastapi import Request  # noqa: E402
# The website lives on Vercel; this engine only answers /api/*. Set ALLOWED_ORIGINS (comma separated) to lock it to
# your site(s), e.g. "https://planforge.vercel.app,https://planforge.tech". Default "*" = any site.
_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_ORIGINS, allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"])


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


MAX_PIXELS = int(float(os.environ.get("PLANFORGE_MAX_MP", "80")) * 1e6)   # a large A0 scan; a bigger image would exhaust memory


def check_image(path):
    """Read only the header: refuse files that are not images and 'image bombs' (tiny file, huge pixel count)
    before anything decodes them into memory."""
    from PIL import Image
    try:
        with Image.open(path) as im:
            w, h = im.size
    except Image.DecompressionBombError:
        raise HTTPException(413, "The image is too large. Please send it at a lower resolution.")
    except Exception:
        raise HTTPException(415, "The file is not a readable image")
    if w * h > MAX_PIXELS or max(w, h) > 30000:
        raise HTTPException(413, "The image is too large ({} x {} px). Please send it at a lower resolution.".format(w, h))


def _num(opt, key, default, lo, hi):
    """A number from the user's options, kept inside a sensible range (no NaN, no absurd sizes)."""
    v = opt.get(key)
    if v in (None, ""):
        return default
    try:
        v = float(v)
    except (TypeError, ValueError):
        raise HTTPException(400, "Option {} must be a number".format(key))
    if not (lo <= v <= hi):
        raise HTTPException(400, "Option {} must be between {} and {}".format(key, lo, hi))
    return v


def clean_options(raw):
    try:
        opt = json.loads(raw or "{}")
    except ValueError:
        raise HTTPException(400, "Options are not valid JSON")
    if not isinstance(opt, dict):
        raise HTTPException(400, "Options must be an object")
    out = {
        "building_width": _num(opt, "building_width", None, 1, 1000),
        "wall_height": _num(opt, "wall_height", 3.0, 2, 12),
        "door_height": _num(opt, "door_height", 2.1, 1.5, 6),
        "head": _num(opt, "head", 2.1, 0.5, 10),
        "sill": _num(opt, "sill", 0.9, 0, 5),
        "ext_wall": _num(opt, "ext_wall", None, 50, 2000),
        "int_wall": _num(opt, "int_wall", None, 30, 1000),
        "plinth": _num(opt, "plinth", None, 0, 5),
        "furniture": opt.get("furniture") if opt.get("furniture") in ("copy", "manual", "none") else "copy",
        "dxf_units": None,
    }
    if opt.get("dxf_units") not in (None, ""):
        if str(opt["dxf_units"]) not in {str(k) for k in UNIT_M}:
            raise HTTPException(400, "Unknown DXF unit")
        out["dxf_units"] = int(opt["dxf_units"])
    return out


def load_input(path, ext, dxf_units):
    """-> (png path, metres per pixel of that png or None)"""
    if ext == ".dxf":
        return dxf_to_png(path, path + ".png", UNIT_M.get(int(dxf_units)) if dxf_units else None)
    if ext == ".dwg":
        raise HTTPException(415, "DWG is a closed format: in AutoCAD use SAVEAS -> DXF, then upload the DXF.")
    if ext == ".pdf":
        import pymupdf
        doc = pymupdf.open(path)
        r = doc[0].rect                                   # points (1/72 inch)
        dpi = max(72, min(300, int(PDF_MAX_SIDE * 72 / max(r.width, r.height, 1))))   # A1 at 300 dpi = 70 MP
        doc[0].get_pixmap(dpi=dpi).save(path + ".png")
        return path + ".png", None
    if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"):
        check_image(path)
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
                    src_w = getattr(P, "LAST_SOURCE_W", None) or P.LAST_ORIGINAL.shape[1]
                    P.METERS_PER_PIXEL = mpp_src * src_w / img.shape[1]
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


# ----------------------------------------------------------------------------- plan ownership
# Each upload gets a secret key, returned only to the browser that sent it. The server keeps just its hash.
# Every later request for that plan must carry the key (X-Plan-Key header), so knowing a job id is not enough.
def _hash(key):
    return hashlib.sha256(key.encode()).hexdigest()


def new_key(job):
    key = secrets.token_urlsafe(32)
    d = os.path.join(JOBS, os.path.basename(job))
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "owner.key"), "w") as f:
        f.write(_hash(key))
    return key


def owner(job, request):
    """Raise 404 (never 403: do not confirm that the plan exists) unless the request carries this plan's key."""
    job = os.path.basename(job)
    key = request.headers.get("x-plan-key") or ""
    kp = os.path.join(JOBS, job, "owner.key")
    try:
        stored = open(kp).read().strip()
    except OSError:
        raise HTTPException(404, "unknown job")
    if not key or not hmac.compare_digest(_hash(key), stored):
        raise HTTPException(404, "unknown job")
    return job


def _queue():
    """Jobs waiting or running, oldest first."""
    q = [(v.get("queued_at", 0), j) for j, v in STATUS.items() if v.get("state") in ("queued", "running")]
    return [j for _, j in sorted(q)]


@app.post("/api/convert")
async def convert(file: UploadFile = File(...), options: str = Form("{}")):
    if len(_queue()) >= MAX_QUEUE:      # one plan at a time: a long queue only ends in timeouts / out of memory
        raise HTTPException(503, "The engine is busy ({} plans in the queue). Please try again in a minute."
                            .format(len(_queue())))
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in (".dxf", ".dwg", ".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"):
        raise HTTPException(415, "Unsupported file type. Send DXF, PDF, PNG, JPG, WEBP, BMP or TIFF.")
    opt = clean_options(options)
    job = uuid.uuid4().hex
    work = os.path.join(JOBS, job)
    os.makedirs(work)
    key = new_key(job)
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
    ahead = len(_queue())
    _set(job, state="queued", progress=0.0, queued_at=time.time(), name=os.path.basename(file.filename or "plan")[:120],
         msg="Waiting in queue ({} plan{} ahead)".format(ahead, "" if ahead == 1 else "s") if ahead else "Starting")
    threading.Thread(target=run_job, args=(job, src, ext, opt), daemon=True).start()
    return {"job": job, "key": key}


@app.get("/api/status/{job}")
def status(job: str, request: Request, wait: float = 0):
    """wait > 0: long poll - the answer comes as soon as the status changes, or after `wait` seconds
    (max 25 s, well under Azure's 230 s request limit). One request every ~25 s instead of every second."""
    job = owner(job, request)
    sp = os.path.join(JOBS, job, "status.json")
    if job not in STATUS and os.path.exists(sp):
        STATUS[job] = json.load(open(sp))
        if STATUS[job].get("state") in ("queued", "running"):
            STATUS[job].update(state="error", msg="Server restarted during conversion: please convert again")
    if job not in STATUS:
        raise HTTPException(404, "unknown job")
    if STATUS[job].get("state") == "queued":                # tell the user where they are in the queue
        q = _queue()
        ahead = q.index(job) if job in q else 0
        if ahead:
            STATUS[job]["msg"] = "Waiting in queue ({} plan{} ahead)".format(ahead, "" if ahead == 1 else "s")
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
def files(job: str, name: str, request: Request):
    base = os.path.join(JOBS, owner(job, request))
    p = os.path.join(base, "floorplan_3d.zip") if name == "floorplan_3d.zip" else \
        os.path.join(base, "out", os.path.basename(name))
    if not os.path.exists(p):
        raise HTTPException(404, "not found")
    return FileResponse(p, filename=name, headers={"Cache-Control": "no-store"})


@app.delete("/api/jobs/{job}")
def delete_job(job: str, request: Request):
    """The user is done with this plan (downloaded it, or moved on): remove the upload and every output now
    instead of waiting for the 24 h cleanup. A plan still in the queue or converting is not touched."""
    job = owner(job, request)
    work = os.path.join(JOBS, job)
    if STATUS.get(job, {}).get("state") in ("queued", "running"):
        raise HTTPException(409, "This plan is still converting")
    existed = os.path.isdir(work)
    shutil.rmtree(work, ignore_errors=True)
    STATUS.pop(job, None)
    return {"deleted": existed}


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
def export(job: str, body: dict, request: Request):
    work = os.path.join(JOBS, owner(job, request))
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


# ----------------------------------------------------------------------------- stair edits
def apply_stair_edit(plan, sid, mode, depth=None):
    """The architect's call for one stair area: 'down' = steps go down from the floor to open ground (the
    ground beside them is cut out of the floor and plinth), 'up' = raised seating on the floor, 'flat' = no
    steps (plain floor), 'stair' = back to a stair to the floor above. depth = metres between floor and the
    low level (None = keep / plinth)."""
    sts = plan.get("stairs", [])
    st = next((x for x in sts if x.get("id") == sid), None)
    if st is None:
        raise HTTPException(404, "unknown stair")
    counted = "written" in str(st.get("sources", {}).get("risers", "")) or \
        "counted" in str(st.get("sources", {}).get("risers", ""))

    def reslice(tiers):
        """Step count was only estimated (for a full stair): cut the area into realistic steps instead."""
        from shapely.geometry import Polygon
        a, b = np.array(st["start"], float), np.array(st["end"], float)
        L = float(np.linalg.norm(b - a))
        if L < 1e-6 or tiers < 1:
            return
        d = (b - a) / L
        polys = [p for p in P._slice(Polygon(st["outline"]).buffer(0), (a + b) / 2, d, L, tiers) if p is not None]
        if polys:
            st["treads"] = [{"polygon": P._poly_list(p), "z0": 0.0, "z1": 0.0} for p in polys]
            st["tread"] = round(L / len(polys), 4)
            st["sources"] = dict(st.get("sources", {}), risers="re-cut for your edit at about 150 mm per step")
    st.setdefault("warnings", [])
    st["edited"] = mode
    if mode == "flat":
        plan["stairs"] = [x for x in sts if x is not st]
        plan.setdefault("stairs_removed", []).append(st)
    elif mode == "down":
        st["kind"], st["sunken"] = "terrace", True
        if depth:
            st["depth_user"] = float(depth)
        if not counted:
            dp = float(depth) if depth else float(plan.get("levels", {}).get("plinth") or 0.45)
            reslice(max(1, int(round(dp / 0.15)) - 1))           # n tiers between floor and pit = n + 1 risers
        st["sources"] = dict(st.get("sources", {}), direction="set by you: goes down")
    elif mode in ("up", "stair"):
        st["sunken"] = False
        st.pop("depth_user", None)
        if not counted:
            dp = float(depth) if depth else (0.45 if mode == "up" else P.WALL_HEIGHT)
            reslice(max(1, int(round(dp / (0.15 if mode == "up" else 0.17))) - (0 if mode == "up" else 1)))
        n = max(1, len(st.get("treads", [])))
        if mode == "up":
            st["kind"] = "terrace"
            total = float(depth) if depth else float(st.get("riser", 0.15)) * n
            rise = total / n
            for r, tr in enumerate(st.get("treads", [])):
                tr["z0"], tr["z1"] = 0.0, round(rise * (r + 1), 4)
            st["direction"] = "rises from the floor (raised seating, set by you)"
        else:
            st["kind"] = "interior"
            total = float(depth) if depth else P.WALL_HEIGHT
            rise = total / (n + 1)
            for r, tr in enumerate(st.get("treads", [])):
                tr["z0"], tr["z1"] = 0.0, round(rise * (r + 1), 4)
            st["direction"] = "stair to the floor above (set by you)"
        st["riser"], st["base_z"], st["top_z"] = round(rise, 4), 0.0, round(total, 4)
        st["sources"] = dict(st.get("sources", {}), direction="set by you: " + mode)
    else:
        raise HTTPException(400, "mode must be down, up, flat or stair")
    plan["floor"]["holes"] = P.open_ground(plan)
    return plan


def _rebuild(work, out, plan):
    jp = os.path.join(out, "plan.json")
    json.dump(plan, open(jp, "w"), indent=1)
    fmesh = P.furniture_meshes(plan["furniture"]) if plan.get("furniture") else {}
    P.build_glb(plan, os.path.join(out, "model.glb"), extra=fmesh)
    _shell_glb(plan, out)
    try:
        P.build_ifc(plan, os.path.join(out, "model.ifc"))
    except Exception as e:
        print("IFC export failed:", e)
    _package(work, out)


@app.post("/api/stair/{job}")
def stair_edit(job: str, body: dict, request: Request):
    work = os.path.join(JOBS, owner(job, request))
    out = os.path.join(work, "out")
    jp = os.path.join(out, "plan.json")
    if not os.path.exists(jp):
        raise HTTPException(404, "unknown job")
    plan = json.load(open(jp))
    depth = body.get("depth")
    with LOCK:
        plan = apply_stair_edit(plan, int(body.get("id", -1)), str(body.get("mode", "")),
                                float(depth) if depth not in (None, "") else None)
        _rebuild(work, out, plan)
    return {"ok": True, "stairs": len(plan.get("stairs", [])), "holes": len(plan["floor"].get("holes", []))}


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
