"""Render best-checkpoint predictions on a few validation plans and upload the image to HF."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from fp_common import CachedFloorplans, build_model, split_logits, MEAN, STD, ROOM_CLASSES, ICON_CLASSES


def main(cfg_path, n=4):
    cfg = json.load(open(cfg_path))
    best = os.path.join(cfg["ckpt_dir"], "best.pt")
    if not os.path.exists(best):
        print("No best.pt yet; skipping visualisation.")
        return
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ck = torch.load(best, map_location="cpu", weights_only=False)
    net = build_model(cfg["encoder"], None)
    net.load_state_dict(ck["ema"])
    net.to(dev).eval()
    ds = CachedFloorplans(cfg["cache_root"], "val", False, cfg["crop"], cfg["sigma"])
    idx = np.linspace(0, len(ds) - 1, n).astype(int)
    cmap_r = plt.get_cmap("tab20", len(ROOM_CLASSES))
    cmap_i = plt.get_cmap("tab20", len(ICON_CLASSES))
    fig, axes = plt.subplots(n, 5, figsize=(25, 5 * n))
    axes = np.atleast_2d(axes)
    for row, i in enumerate(idx):
        x, room, icon, heat = ds[int(i)]
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            out = net(x[None].to(dev)).float()
        r, ic, h = split_logits(out)
        img = (x.numpy().transpose(1, 2, 0) * STD + MEAN).clip(0, 255).astype(np.uint8)
        panels = [(img, None, "input"),
                  (room.numpy(), cmap_r, "rooms GT"), (r.argmax(1)[0].cpu().numpy(), cmap_r, "rooms pred"),
                  (ic.argmax(1)[0].cpu().numpy(), cmap_i, "icons pred (door/window/stairs...)"),
                  (torch.sigmoid(h)[0].max(0).values.cpu().numpy(), "magma", "junction heatmaps")]
        for ax, (im, cm, title) in zip(axes[row], panels):
            if cm is None:
                ax.imshow(im)
            elif isinstance(cm, str):
                ax.imshow(im, cmap=cm, vmin=0, vmax=1)
            else:
                ax.imshow(np.where(im == 255, 0, im), cmap=cm, vmin=0, vmax=cm.N - 1,
                          interpolation="nearest")
            ax.set_title(title)
            ax.axis("off")
    plt.tight_layout()
    out_png = os.path.join(cfg["ckpt_dir"], "val_predictions.png")
    plt.savefig(out_png, dpi=60)
    print("Saved", out_png, "| best epoch", ck["epoch"], "| score", round(ck["metrics"]["score"], 4))
    tok = os.environ.get("HF_TOKEN")
    if tok and cfg.get("hf_repo"):
        try:
            from huggingface_hub import HfApi
            HfApi(token=tok).upload_file(path_or_fileobj=out_png, path_in_repo="val_predictions.png",
                                         repo_id=cfg["hf_repo"], commit_message="validation samples")
            print("Uploaded val_predictions.png")
        except Exception as e:
            print("Upload of preview failed:", e)


if __name__ == "__main__":
    main(sys.argv[1])
