# ===================== DIAGNOSE: what the furniture detector sees =====================
IMAGE = ""                              # same plan you ran through plan_to_revit.py
FURN_REPO = "floorplan-furniture-obb"
LOOK_FOR = "bed"                        # class to diagnose
PLAN_PX_PER_M = None                    # optional: from plan_to_revit's plan.json -> 1 / mapping.scale_m_per_px

import os, sys, json, subprocess, urllib.request
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.3.253", "huggingface_hub"], check=False)
import cv2, numpy as np, matplotlib.pyplot as plt
from ultralytics import YOLO
from huggingface_hub import HfApi, hf_hub_download

token = os.environ.get("HF_TOKEN")
if not token:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("HF_TOKEN")
repo = "{}/{}".format(HfApi(token=token).whoami()["name"], FURN_REPO)
model = YOLO(hf_hub_download(repo, "checkpoints/best.pt", token=token, force_download=True))
meta = json.load(open(hf_hub_download(repo, "furniture_meta.json", token=token, force_download=True)))

print("=== 0. WHAT TRAINING IMAGES LOOK LIKE (FloorPlanCAD) ===")
pngs = [f for f in HfApi().list_repo_files("Voxel51/FloorPlanCAD", repo_type="dataset") if f.lower().endswith(".png")]
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
for ax, f in zip(axs, pngs[:3]):
    t = cv2.imread(hf_hub_download("Voxel51/FloorPlanCAD", f, repo_type="dataset"), cv2.IMREAD_UNCHANGED)
    if t.ndim == 3 and t.shape[2] == 4:
        a = t[:, :, 3:4] / 255.0
        t = (t[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
    hsv = cv2.cvtColor(t, cv2.COLOR_BGR2HSV)
    ink = hsv[:, :, 2] < 240
    ink |= hsv[:, :, 1] > 40
    coloured = (hsv[:, :, 1][ink] > 60).mean() if ink.any() else 0
    print("  {}: {}x{}, {:.0%} of line pixels are coloured".format(f.split("/")[-1], t.shape[1], t.shape[0], coloured))
    ax.imshow(t[:, :, ::-1]); ax.set_title("{:.0%} coloured lines".format(coloured)); ax.axis("off")
plt.tight_layout(); plt.show()

print("\n=== 1. TRAINING DATA ===")
print("training scale (px per metre):", meta.get("train_px_per_m"))
print("examples per class:", {k: v for k, v in sorted(meta.get("counts", {}).items(), key=lambda kv: -kv[1])})
print("'{}' examples: {}".format(LOOK_FOR, meta.get("counts", {}).get(LOOK_FOR, 0)))

if IMAGE.startswith("http"):
    req = urllib.request.Request(IMAGE, headers={"User-Agent": "Mozilla/5.0"})
    img = cv2.imdecode(np.frombuffer(urllib.request.urlopen(req, timeout=60).read(), np.uint8), cv2.IMREAD_UNCHANGED)
else:
    img = cv2.imread(IMAGE, cv2.IMREAD_UNCHANGED)
assert img is not None, "set IMAGE"
if img.ndim == 2:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
elif img.shape[2] == 4:
    a = img[:, :, 3:4] / 255.0
    img = (img[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
# same size the pipeline uses (MAX_SIDE = 1280)
s0 = 1280 / max(img.shape[:2])
img = cv2.resize(img, None, fx=s0, fy=s0, interpolation=cv2.INTER_AREA if s0 < 1 else cv2.INTER_CUBIC)
print("\n=== 2. PLAN ===\nimage after pipeline resize:", img.shape[1], "x", img.shape[0])
if PLAN_PX_PER_M and meta.get("train_px_per_m"):
    print("zoom the pipeline uses: {:.2f}".format(meta["train_px_per_m"] / PLAN_PX_PER_M))


def run(zoom, conf=0.05, tile=1024):
    im = cv2.resize(img, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_AREA if zoom < 1 else cv2.INTER_CUBIC)
    H, W = im.shape[:2]
    out = []
    for y0 in range(0, max(H - tile, 0) + tile, int(tile * 0.75)):
        for x0 in range(0, max(W - tile, 0) + tile, int(tile * 0.75)):
            crop = np.full((tile, tile, 3), 255, np.uint8)
            c = im[y0:y0 + tile, x0:x0 + tile]
            if c.size == 0:
                continue
            crop[:c.shape[0], :c.shape[1]] = c
            r = model.predict(crop, imgsz=tile, conf=conf, verbose=False)[0]
            if r.obb is None:
                continue
            for k, p in enumerate(r.obb.xyxyxyxy.cpu().numpy()):
                out.append((model.names[int(r.obb.cls[k])], float(r.obb.conf[k]), (p.reshape(4, 2) + [x0, y0]) / zoom))
    return out


print("\n=== 3. '{}' at each zoom (conf >= 0.05) ===".format(LOOK_FOR))
best = None
for z in [0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.5]:
    d = run(z)
    hits = sorted([x for x in d if x[0] == LOOK_FOR], key=lambda x: -x[1])
    allc = {}
    for x in d:
        if x[1] >= 0.3:
            allc[x[0]] = allc.get(x[0], 0) + 1
    print("zoom {:>4}: {:>3} {} (best conf {:.2f}) | everything >=0.3: {}".format(
        z, len([h for h in hits if h[1] >= 0.25]), LOOK_FOR, hits[0][1] if hits else 0, allc))
    if hits and (best is None or hits[0][1] > best[1][0][1]):
        best = (z, hits)

if best:
    z, hits = best
    fig, ax = plt.subplots(figsize=(14, 14 * img.shape[0] / img.shape[1]))
    ax.imshow(img[:, :, ::-1]); ax.axis("off")
    for _, cf, p in hits[:15]:
        q = np.vstack([p, p[:1]])
        ax.plot(q[:, 0], q[:, 1], "r-" if cf >= 0.25 else "y--", lw=2)
        ax.text(*p.mean(0), "{:.2f}".format(cf), color="r", fontsize=9, ha="center")
    ax.set_title("'{}' candidates at zoom {} (red >= 0.25, yellow weaker)".format(LOOK_FOR, z))
    plt.tight_layout(); plt.show()
else:
    print("\nNo '{}' at any zoom, even at confidence 0.05 -> the model has not learned this drawing style yet.".format(LOOK_FOR))
