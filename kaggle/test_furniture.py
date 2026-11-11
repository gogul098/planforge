# ===================== TEST THE FURNITURE DETECTOR ON A FLOOR PLAN =====================
IMAGE = ""                              # floor-plan image URL or file path
FURN_REPO = "floorplan-furniture-obb"   # HF repo written by the training notebook
PLAN_PX_PER_M = None                    # pixels per metre of your image if known; None = auto-search
CONF = 0.35                             # lower = finds more (and more false hits)
LOCAL_WEIGHTS = None                    # optional local .pt instead of downloading

import os, sys, json, subprocess, urllib.request
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.3.253", "huggingface_hub", "shapely"],
               check=False)
import cv2, numpy as np, matplotlib.pyplot as plt
from shapely.geometry import Polygon
from ultralytics import YOLO

# ---- weights + training scale
if LOCAL_WEIGHTS:
    weights, meta = LOCAL_WEIGHTS, {"train_px_per_m": None}
else:
    from huggingface_hub import HfApi, hf_hub_download
    token = os.environ.get("HF_TOKEN")
    if not token:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
    repo = "{}/{}".format(HfApi(token=token).whoami()["name"], FURN_REPO)
    weights = hf_hub_download(repo, "checkpoints/best.pt", token=token, force_download=True)   # newest best
    meta = json.load(open(hf_hub_download(repo, "furniture_meta.json", token=token)))
model = YOLO(weights)
print("classes:", list(model.names.values()))
print("training drawing scale: {} px/m".format(meta.get("train_px_per_m")))

# ---- image
assert IMAGE, "Set IMAGE to a floor-plan URL or file path"
if IMAGE.startswith("http"):
    req = urllib.request.Request(IMAGE, headers={"User-Agent": "Mozilla/5.0"})
    img = cv2.imdecode(np.frombuffer(urllib.request.urlopen(req, timeout=60).read(), np.uint8), cv2.IMREAD_UNCHANGED)
else:
    img = cv2.imread(IMAGE, cv2.IMREAD_UNCHANGED)
assert img is not None, "could not read IMAGE"
if img.ndim == 2:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
elif img.shape[2] == 4:
    a = img[:, :, 3:4] / 255.0
    img = (img[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def detect(scale, conf, tile=1024, overlap=0.25):
    """Rescale so furniture has its training size, detect tile by tile, merge overlaps."""
    im = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    H, W = im.shape[:2]
    step = int(tile * (1 - overlap))
    ys = sorted(set(list(range(0, max(H - tile, 0) + 1, step)) + [max(H - tile, 0)]))
    xs = sorted(set(list(range(0, max(W - tile, 0) + 1, step)) + [max(W - tile, 0)]))
    found = []
    for y0 in ys:
        for x0 in xs:
            crop = np.full((tile, tile, 3), 255, np.uint8)
            c = im[y0:y0 + tile, x0:x0 + tile]
            crop[:c.shape[0], :c.shape[1]] = c
            r = model.predict(crop[:, :, ::-1], imgsz=tile, conf=conf, verbose=False)[0]
            if r.obb is None:
                continue
            for k, p in enumerate(r.obb.xyxyxyxy.cpu().numpy()):
                p = p.reshape(4, 2)
                cx, cy = p.mean(0)
                m = tile * overlap / 2
                if (x0 > 0 and cx < m) or (y0 > 0 and cy < m) or (x0 + tile < W and cx > tile - m) or \
                   (y0 + tile < H and cy > tile - m):
                    continue
                found.append({"cls": model.names[int(r.obb.cls[k])], "conf": float(r.obb.conf[k]),
                              "corners": (p + [x0, y0]) / scale})
    found.sort(key=lambda d: -d["conf"])
    keep = []
    for d in found:
        P = Polygon(d["corners"]).buffer(0)
        if P.area > 0 and all(P.intersection(k["_p"]).area / min(P.area, k["_p"].area) < 0.6 for k in keep):
            d["_p"] = P
            keep.append(d)
    return keep


# ---- scale: known, or search the zoom where the detector is most confident
tpm = meta.get("train_px_per_m")
if PLAN_PX_PER_M and tpm:
    scales = [tpm / PLAN_PX_PER_M]
else:
    scales = [0.5, 0.7, 1.0, 1.4, 2.0, 2.8]
    print("no plan scale given -> trying zoom levels", scales)
best = None
for s in scales:
    d = detect(s, CONF)
    score = sum(x["conf"] for x in d)
    print("  zoom {:.2f}: {} objects (confidence sum {:.1f})".format(s, len(d), score))
    if best is None or score > best[0]:
        best = (score, s, d)
_, scale, dets = best
print("\nusing zoom {:.2f}: {} objects".format(scale, len(dets)))
counts = {}
for d in dets:
    counts[d["cls"]] = counts.get(d["cls"], 0) + 1
for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
    print("  {:<20} {}".format(k, v))

# ---- picture
cmap = plt.get_cmap("tab20")
names = sorted(model.names.values())
fig, ax = plt.subplots(figsize=(16, 16 * rgb.shape[0] / rgb.shape[1]))
ax.imshow(rgb); ax.axis("off")
for d in dets:
    col = cmap(names.index(d["cls"]) % 20)
    c = np.vstack([d["corners"], d["corners"][:1]])
    ax.plot(c[:, 0], c[:, 1], color=col, lw=2)
    ax.fill(c[:, 0], c[:, 1], color=col, alpha=0.25)
    ax.text(c[:, 0].mean(), c[:, 1].mean(), "{} {:.2f}".format(d["cls"], d["conf"]), fontsize=7,
            ha="center", va="center", color="k", bbox=dict(fc="white", alpha=0.7, lw=0, pad=1))
ax.set_title("furniture detector: {} objects (zoom {:.2f}, conf >= {})".format(len(dets), scale, CONF), fontsize=14)
plt.tight_layout(); plt.savefig("furniture_test.png", dpi=110); plt.show()
