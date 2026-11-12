# ===================== DEBUG: a marked image for every stage =====================
# Run this cell AFTER the plan_to_revit cell (same notebook, same IMAGE).
# Saves stage_01 ... stage_10 PNGs + one contact sheet in OUT_DIR/debug and shows the sheet.
import os, json
import cv2, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon as MPoly

DBG = os.path.join(OUT_DIR, "debug")
os.makedirs(DBG, exist_ok=True)
ROOM_NAMES = ["Background", "Outdoor", "Wall", "Kitchen", "Living", "Bedroom", "Bath", "Entry", "Railing",
              "Storage", "Garage", "Other room"]
ICON_NAMES = ["-", "Window", "Door", "Closet", "Appliance", "Toilet", "Sink", "Sauna bench", "Fireplace",
              "Bathtub", "Chimney", "Stairs"]
CMAP = plt.get_cmap("tab20")

# ---- run the pipeline once, keeping every intermediate result
if "DBG_INPUT" in globals():                  # (offline testing hook)
    img, room, icon, original = DBG_INPUT
else:
    img, room, icon = predict(IMAGE)
    original = LAST_ORIGINAL
v = vectorise(room, icon, original)
toks = ocr_full_res(img, original)
plan = main(img, room, icon, original=original)
f = img.shape[1] / original.shape[1]          # inference px per original px
sv, n_dl, dl = dimension_line_scale(toks, original, img.shape[1])
H, W = room.shape
base = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
faded = (0.35 * base + 0.65 * 255).astype(np.uint8)
stages = []


def save(fig, name, title):
    p = os.path.join(DBG, name)
    fig.suptitle(title, fontsize=13)
    fig.savefig(p, dpi=90, bbox_inches="tight")
    plt.close(fig)
    stages.append((p, title))


def canvas(t=None):
    fig, ax = plt.subplots(figsize=(11, 11 * H / W))
    ax.imshow(faded if t is None else t)
    ax.set_axis_off()
    return fig, ax


def overlay(classes, values, names, skip=(0,)):
    out = faded.astype(float).copy()
    handles = []
    for k in values:
        if k in skip:
            continue
        m = classes == k
        if not m.any():
            continue
        c = np.array(CMAP(k % 20)[:3]) * 255
        out[m] = 0.35 * out[m] + 0.65 * c
        handles.append(Patch(color=CMAP(k % 20), label="{} ({:.1f}%)".format(names[k], 100 * m.mean())))
    return out.astype(np.uint8), handles


# 01 input
fig, ax = canvas(base); save(fig, "stage_01_input.png", "1. Input given to the model ({}x{})".format(W, H))

# 02 model: rooms
o, hd = overlay(room, range(12), ROOM_NAMES)
fig, ax = canvas(o); ax.legend(handles=hd, loc="upper right", fontsize=8)
save(fig, "stage_02_model_rooms.png", "2. MODEL output - room head (every pixel's class)")

# 03 model: icons
o, hd = overlay(icon, range(12), ICON_NAMES)
fig, ax = canvas(o); ax.legend(handles=hd, loc="upper right", fontsize=8)
save(fig, "stage_03_model_icons.png", "3. MODEL output - icon head (doors, windows, stairs, fixtures)")

# 04 converter input: wall pixels
o = faded.copy()
o[room == WALL] = (40, 40, 40)
o[v["openings_mask"]] = (0, 170, 255)
fig, ax = canvas(o)
ax.legend(handles=[Patch(color=(0.16, 0.16, 0.16), label="model wall pixels"),
                   Patch(color=(0, 0.67, 1), label="door / window pixels joined to walls")], loc="upper right")
save(fig, "stage_04_wall_pixels.png", "4. What the 3D converter receives: model wall pixels (T = {:.1f} px)".format(v["T"]))

# 05 vector walls vs model pixels
fig, ax = canvas()
miss = np.zeros((H, W, 4)); miss[v["missed"]] = (1, 0.8, 0, 1)
ax.imshow(miss)
frames = sorted({round(sg.get("frame", 0), 1) for sg in v["walls"]})
for i, sg in enumerate(v["walls"]):
    p0, p1 = np.array(sg["p0"]), np.array(sg["p1"])
    d = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9); n = np.array([-d[1], d[0]]) * sg["t"] / 2
    col = CMAP(frames.index(round(sg.get("frame", 0), 1)) * 2 % 20)
    ax.add_patch(MPoly([p0 - n, p1 - n, p1 + n, p0 + n], fc=col, ec="k", lw=0.4, alpha=0.75))
    c = (p0 + p1) / 2
    ax.text(c[0], c[1], str(i), fontsize=6, ha="center", va="center")
ax.legend(handles=[Patch(color=CMAP(k * 2 % 20), label="walls at {} deg".format(fr)) for k, fr in enumerate(frames)] +
          [Patch(color=(1, 0.8, 0), label="model wall pixels NOT converted")], loc="upper right", fontsize=8)
save(fig, "stage_05_vector_walls.png", "5. Straight walls built from the model pixels: {} walls, {:.1f}% of pixels kept".format(
    len(v["walls"]), 100 * v["coverage"]))

# 06 openings
fig, ax = canvas()
for sg in v["walls"]:
    ax.plot([sg["p0"][0], sg["p1"][0]], [sg["p0"][1], sg["p1"][1]], color="0.55", lw=2)
for o_ in v["openings"]:
    col = "lime" if o_["kind"] == "door" else "deepskyblue"
    ax.plot([o_["q0"][0], o_["q1"][0]], [o_["q0"][1], o_["q1"][1]], color=col, lw=5)
ax.legend(handles=[Patch(color="lime", label="door ({})".format(sum(o_["kind"] == "door" for o_ in v["openings"]))),
                   Patch(color="deepskyblue", label="window ({})".format(sum(o_["kind"] != "door" for o_ in v["openings"])))],
          loc="upper right")
save(fig, "stage_06_openings.png", "6. Doors / windows (from the model) placed on their walls")

# 07 OCR + scale
fig, ax = canvas()
for t in toks:
    col = {"num": "red", "size": "green", "label": "purple"}[t["kind"]]
    ax.text(t["x"], t["y"], t.get("raw", t.get("text", "")), color=col, fontsize=7, ha="center", va="center",
            bbox=dict(fc="white", ec=col, lw=0.6, pad=1))
med = np.median([c[0] for c in dl]) if dl else None
for c in dl:
    ok = sv is not None and abs(c[0] / sv - 1) < 0.04
    ax.text(10, 20 + 14 * dl.index(c), "{}: line {:.0f} px -> {:.4f} m/px {}".format(
        c[1], c[3] * f, c[0], "AGREES" if ok else ""), color="green" if ok else "darkorange", fontsize=8)
ax.legend(handles=[Patch(color="red", label="dimension numbers"), Patch(color="green", label="room sizes"),
                   Patch(color="purple", label="labels")], loc="lower right", fontsize=8)
save(fig, "stage_07_ocr_scale.png", "7. Text read on the plan + dimension lines measured. Scale used: {}".format(
    plan["precision"]["scale_source"]))

# 08 stairs
mx, my = make_mapper(plan["mapping"], "x"), make_mapper(plan["mapping"], "y")
inv = lambda xm, ym: (np.interp(xm, [float(mx(q)) for q in (0, W)], [0, W]), np.interp(ym, [float(my(q)) for q in (0, H)], [0, H]))
fig, ax = canvas()
st_mask = np.zeros((H, W, 4)); st_mask[icon == STAIRS] = (1, 0, 1, 0.35); ax.imshow(st_mask)
for s in plan["stairs"]:
    for k, tr in enumerate(s["treads"]):
        pts = [inv(x, y) for x, y in tr["polygon"]]
        ax.add_patch(MPoly(pts, fc=plt.get_cmap("viridis")(k / max(len(s["treads"]) - 1, 1)), ec="k", lw=0.4, alpha=0.8))
    cx, cy = inv(*np.mean(s["outline"], 0))
    ax.text(cx, cy, "{} #{}\n{} risers x {:.0f} mm\n{:+.2f} -> {:+.2f} m".format(
        s["kind"], s["id"], s["risers"], s["riser"] * 1000, s["base_z"], s["top_z"]),
        fontsize=8, ha="center", bbox=dict(fc="white", ec="m"))
save(fig, "stage_08_stairs.png", "8. Stairs: magenta = model's stair pixels, coloured = treads built (dark = low)")

# 09 rooms
fig, ax = canvas()
o, hd = overlay(room, range(3, 12), ROOM_NAMES, skip=())
ax.imshow(o, alpha=0.5)
for r in plan["rooms"]:
    pts = [inv(x, y) for x, y in r["polygon"]]
    ax.add_patch(MPoly(pts, fill=False, ec="red", lw=2))
    ax.text(*inv(*r["point"]), "{}\n{:.1f} m2".format(r["name"], r["area_m2"]), color="red", fontsize=8, ha="center")
save(fig, "stage_09_rooms.png", "9. Rooms: red = closed rooms from walls ({}), colours = model room head".format(len(plan["rooms"])))

# 10 final plan.json, in metres
fig, ax = plt.subplots(figsize=(11, 11 * H / W))
for w in plan["walls"]:
    a_, d_, n_, L = wall_frame(w)
    h_ = w["thickness"] / 2
    q = [a_ - d_ * w["ext"][0] - n_ * h_, a_ + d_ * (L + w["ext"][1]) - n_ * h_, a_ + d_ * (L + w["ext"][1]) + n_ * h_,
         a_ - d_ * w["ext"][0] + n_ * h_]
    ax.add_patch(MPoly(q, fc="0.25" if w["external"] else "0.55", ec="k", lw=0.3))
for o_ in plan["openings"]:
    w = plan["walls"][o_["wall"]]; a_, d_, n_, L = wall_frame(w)
    c = np.array(o_["center"]); e = d_ * o_["width"] / 2
    ax.plot([c[0] - e[0], c[0] + e[0]], [c[1] - e[1], c[1] + e[1]], color="lime" if o_["type"] == "door" else "deepskyblue", lw=4)
for s in plan["stairs"]:
    for tr in s["treads"]:
        ax.add_patch(MPoly(tr["polygon"], fc="plum", ec="purple", lw=0.4))
for fu in plan.get("furniture", []):
    ax.add_patch(MPoly(fu["corners"], fc="tan", ec="saddlebrown", lw=0.6))
    ax.text(*fu["center"], fu["type"], fontsize=6, ha="center")
ax.set_aspect("equal"); ax.invert_yaxis(); ax.autoscale_view(); ax.grid(alpha=0.3)
ax.set_xlabel("metres"); ax.set_ylabel("metres")
save(fig, "stage_10_plan_json.png", "10. plan.json (what Revit / IFC / GLB are built from): {} walls, {} openings, {} stairs, {} furniture".format(
    len(plan["walls"]), len(plan["openings"]), len(plan["stairs"]), len(plan.get("furniture", []))))

# contact sheet
cols = 2
rows = (len(stages) + 1) // cols
fig, axs = plt.subplots(rows, cols, figsize=(20, 10 * rows * H / W))
for ax, (p, t) in zip(axs.ravel(), stages):
    ax.imshow(plt.imread(p)); ax.set_axis_off()
for ax in axs.ravel()[len(stages):]:
    ax.set_axis_off()
plt.tight_layout()
sheet = os.path.join(DBG, "all_stages.png")
plt.savefig(sheet, dpi=70)
plt.show()
print("saved:", ", ".join(os.path.basename(p) for p, _ in stages), "+ all_stages.png in", DBG)
