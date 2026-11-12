"""Edit the existing plan image: rewrite numbers to what is drawn, fix notes, renumber duplicates.
Nothing in the drawing (walls, doors, windows, furniture, stairs) is moved or redrawn."""
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont

S = 16.0 / 770.5                       # m per px, from the top 16.00 m overall dimension (kept as the reference)
src = Image.open("plan.png").convert("RGB")
im = src.copy()
dr = ImageDraw.Draw(im)
REG = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
BOLD = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
ocr = json.load(open("ocr.json"))


def box_of(text, near=None):
    c = [o for o in ocr if o["t"].replace(" ", "") == text.replace(" ", "")]
    if near:
        c.sort(key=lambda o: abs((o["x0"] + o["x1"]) / 2 - near[0]) + abs((o["y0"] + o["y1"]) / 2 - near[1]))
    return c[0]


def erase(o, pad=2):
    dr.rectangle([o["x0"] - pad, o["y0"] - pad, o["x1"] + pad, o["y1"] + pad], fill="white")


def write(xy, s, size, bold=False, anchor="mm"):
    dr.text(xy, s, fill="black", font=ImageFont.truetype(BOLD if bold else REG, size), anchor=anchor)


def replace(old, new, near=None, size=None, bold=True):
    o = box_of(old, near)
    erase(o)
    h = o["y1"] - o["y0"]
    write(((o["x0"] + o["x1"]) / 2, (o["y0"] + o["y1"]) / 2), new, size or int(h * 0.95), bold)


def m(px):
    return "{:.2f} m".format(px * S)


# ---- top chain (ticks measured at x 263.5 / 458.5 / 838.5 / 1034)
replace("3.66 m", m(458.5 - 263.5), near=(367, 44))
replace("8.68 m", m(838.5 - 458.5), near=(643, 44))
replace("3.66 m", m(1034 - 838.5), near=(935, 44))
# ---- bottom chain (ticks at x 110 / 415.5 and 856.5 / 1079; overall line 110 -> 1079)
replace("3.66 m", m(415.5 - 110), near=(248, 1040))
replace("3.66 m", m(1079 - 856.5), near=(966, 1040))
erase(box_of("8.68 m", near=(617, 1095)))                    # no 8.68 m segment exists at the bottom
dr.line([(110, 1101), (1079, 1101)], fill="black", width=2)  # restore the overall line under the erased text
replace("16.00 m", m(1079 - 110), near=(617, 1121))

# ---- the two slanted side dimensions (11.21 m written, ~594 px drawn on both sides)
side = m(np.hypot(289 - 6, 603 - 81))
lb = box_of("11.21 m")
quad = np.array(lb["box"])
for q, ang in ((quad, 61), (np.column_stack([1300 - quad[:, 0], quad[:, 1]]), -61)):
    c = q.mean(0)
    dr.polygon([tuple(p) for p in (c + (q - c) * 1.12)], fill="white")
    f = ImageFont.truetype(BOLD, 21)
    tw = int(f.getlength(side)) + 6
    t = Image.new("RGBA", (tw, 28), (255, 255, 255, 0))
    ImageDraw.Draw(t).text((tw / 2, 14), side, font=f, fill="black", anchor="mm")
    t = t.rotate(ang, expand=True, resample=Image.BICUBIC)
    im.paste(t, (int(c[0] - t.width / 2), int(c[1] - t.height / 2)), t)

# ---- stair note: what the flight actually shows (17 tread lines y 280..457, sides 613-620 / 677-686)
treads = 16
run_px, width_px = 457 - 280, 677 - 621
replace("16R@150,T300", "{}R @ 150, T {:.0f}".format(treads + 1, run_px / treads * S * 1000), size=17, bold=False)
replace("1200W.4800RUN", "{:.0f} W, {:.0f} RUN".format(width_px * S * 1000, run_px * S * 1000), size=17, bold=False)

# ---- entrance steps: 5 treads (y 1012 .. 1077), risers 150 mm -> plinth 0.75 m; level marks
n_e = 5
t_e = (1077 - 1012) / n_e * S * 1000
write((641, 1089), "{}R @ 150, T {:.0f}".format(n_e, t_e), 14, anchor="mm")
write((505, 1072), "±0.00 GL", 15, anchor="rm")
write((745, 945), "+{:.2f} FFL".format(n_e * 0.15), 15, anchor="mm")

# ---- duplicate room numbers: dining hall 2 -> 12, second multi-purpose 9 -> 10
for (cx, cy, r, num) in ((892, 844, 13, "12"), (1070, 416, 12, "10")):
    dr.ellipse([cx - r + 2, cy - r + 2, cx + r - 2, cy + r - 2], fill="white")
    write((cx, cy + 1), num, 15 if len(num) == 2 else 18, bold=True)

im.save("plan_edited.png")
diff = np.abs(np.asarray(im, int) - np.asarray(src, int)).sum(2) > 30
print("pixels changed: {:.2f}% of the image".format(100 * diff.mean()))
