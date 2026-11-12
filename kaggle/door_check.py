# ===================== DOOR CHECK: how sure is the model about doors / windows? =====================
# Run AFTER the plan_to_revit cell (it uses IMAGE, predict and LAST_ICON_PROB from that cell).
import numpy as np, matplotlib.pyplot as plt
img, room, icon = predict(IMAGE)
P_ = LAST_ICON_PROB.astype(np.float32)
door, win = P_[2], P_[1]
fig, axs = plt.subplots(1, 3, figsize=(24, 8.5))
for ax, (m, name) in zip(axs, [(door, "DOOR"), (win, "WINDOW")]):
    ax.imshow(img); h = ax.imshow(m, cmap="magma", alpha=0.65, vmin=0, vmax=1); ax.set_axis_off()
    ax.set_title("model {} probability (bright = sure)".format(name)); plt.colorbar(h, ax=ax, fraction=0.03)
for t, c in [(0.5, "lime"), (0.3, "yellow"), (0.15, "red")]:
    axs[2].contour(door >= t, levels=[0.5], colors=c, linewidths=1)
axs[2].imshow(img, alpha=0.6); axs[2].set_axis_off()
axs[2].set_title("door areas at probability >= 0.5 (green), 0.3 (yellow), 0.15 (red)")
plt.tight_layout(); plt.savefig(OUT_DIR + "/door_check.png", dpi=80); plt.show()
for t in (0.5, 0.3, 0.15, 0.05):
    print("door pixels with probability >= {:.2f}: {:.3f}% of the image".format(t, 100 * (door >= t).mean()))
print("max door probability anywhere: {:.2f} | used now (argmax): {:.3f}%".format(door.max(), 100 * (icon == 2).mean()))
