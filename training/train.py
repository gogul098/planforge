"""Multi-task floorplan training (rooms + icons/doors/windows/stairs + junction heatmaps).
Launch:  torchrun --standalone --nproc_per_node=2 train.py --config run_config.json
Probe :  python train.py --config run_config.json --probe   (finds the largest safe batch)
"""
import argparse
import datetime
import json
import math
import os
import random
import shutil
import threading
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from fp_common import (CachedFloorplans, MultiTaskLoss, EMA, build_model, split_logits,
                       update_confusion, iou_from_confusion, ROOM_CLASSES, ICON_CLASSES,
                       N_ROOM, N_ICON)


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ----------------------------------------------------------------------------- HF uploads
class Uploader:
    """Background, retrying uploads so training never waits on (or dies from) the network."""

    def __init__(self, repo, token):
        self.repo, self.token = repo, token
        self.thread = None
        self.staging = os.path.join(os.environ.get("TMPDIR", "/tmp"), "hf_staging")
        os.makedirs(self.staging, exist_ok=True)

    def _run(self, items, msg):
        from huggingface_hub import HfApi
        api = HfApi(token=self.token)
        for attempt in range(1, 6):
            try:
                from huggingface_hub import CommitOperationAdd
                ops = [CommitOperationAdd(path_in_repo=dst, path_or_fileobj=src) for src, dst in items]
                api.create_commit(repo_id=self.repo, operations=ops, commit_message=msg,
                                  repo_type="model")
                log(f"[hf] uploaded: {', '.join(d for _, d in items)}")
                return
            except Exception as e:
                log(f"[hf] upload attempt {attempt}/5 failed: {type(e).__name__}: {e}")
                time.sleep(min(120, 15 * attempt))
        log("[hf] WARNING: giving up on this upload; next checkpoint will retry.")

    def submit(self, files, msg):
        """files: list of (local_path, path_in_repo). Snapshots files before returning."""
        if not self.repo or not self.token:
            return
        self.wait()
        snap = []
        for i, (src, dst) in enumerate(files):
            if os.path.exists(src):
                tmp = os.path.join(self.staging, f"{i}_{os.path.basename(src)}")
                shutil.copyfile(src, tmp)
                snap.append((tmp, dst))
        if snap:
            self.thread = threading.Thread(target=self._run, args=(snap, msg), daemon=True)
            self.thread.start()

    def wait(self, timeout=None):
        if self.thread is not None:
            self.thread.join(timeout)
            if not self.thread.is_alive():
                self.thread = None


# ----------------------------------------------------------------------------- helpers
def seed_worker(_):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def set_lr(opt, step, steps_per_epoch, total_epochs, warmup_epochs, min_ratio=0.01):
    warm = max(1, int(warmup_epochs * steps_per_epoch))
    total = max(warm + 1, total_epochs * steps_per_epoch)
    if step < warm:
        f = (step + 1) / warm
    else:
        p = min(1.0, (step - warm) / max(1, total - warm))
        f = min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))
    for g in opt.param_groups:
        g["lr"] = g["base_lr"] * f
    return f


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def signature(cfg):
    return {k: cfg[k] for k in ("encoder", "crop", "max_side", "sigma")} | {
        "n_room": N_ROOM, "n_icon": N_ICON}


def make_optimizer(model, cfg):
    enc, rest = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (enc if n.startswith("encoder.") else rest).append(p)
    groups = [{"params": enc, "base_lr": cfg["lr"] * cfg["encoder_lr_mult"]},
              {"params": rest, "base_lr": cfg["lr"]}]
    for g in groups:
        g["lr"] = g["base_lr"]
    return torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])


# ----------------------------------------------------------------------------- probe
def probe(cfg):
    """Find the largest per-GPU batch that trains at the crop size with >=15% memory headroom."""
    dev = torch.device("cuda:0")
    total_mem = torch.cuda.get_device_properties(0).total_memory
    model = build_model(cfg["encoder"], None).to(dev).train()
    opt = make_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda")
    crit = MultiTaskLoss(*cfg["loss_weights"])
    best = None
    for bs in cfg["probe_batches"]:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            for _ in range(3):
                x = torch.randn(bs, 3, cfg["crop"], cfg["crop"], device=dev)
                room = torch.randint(0, N_ROOM, (bs, cfg["crop"], cfg["crop"]), device=dev)
                icon = torch.randint(0, N_ICON, (bs, cfg["crop"], cfg["crop"]), device=dev)
                heat = torch.rand(bs, 3, cfg["crop"], cfg["crop"], device=dev)
                with torch.autocast("cuda", dtype=torch.float16):
                    out = model(x)
                loss, _ = crit(out, room, icon, heat)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            peak = torch.cuda.max_memory_allocated()
            log(f"probe bs={bs}: peak {peak / 2 ** 30:.2f} GiB / {total_mem / 2 ** 30:.2f} GiB")
            if peak < 0.85 * total_mem:
                best = bs
                break
        except torch.cuda.OutOfMemoryError:
            log(f"probe bs={bs}: OOM")
        finally:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    best = best or min(cfg["probe_batches"])
    print(f"PROBE_BS={best}", flush=True)


# ----------------------------------------------------------------------------- validation
@torch.no_grad()
def validate(net, ds, rank, world, dev):
    net.eval()
    bench = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False  # every val plan has its own size
    cm_r = torch.zeros(N_ROOM, N_ROOM, dtype=torch.long, device=dev)
    cm_i = torch.zeros(N_ICON, N_ICON, dtype=torch.long, device=dev)
    idx = list(range(rank, len(ds), world))
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=1,
                                         shuffle=False, num_workers=2,
                                         pin_memory=dev.type == "cuda")
    for x, room, icon, _ in loader:
        x, room, icon = x.to(dev, non_blocking=True), room.to(dev), icon.to(dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            out = net(x)
        r, i, _h = split_logits(out.float())
        update_confusion(cm_r, r.argmax(1), room, N_ROOM)
        update_confusion(cm_i, i.argmax(1), icon, N_ICON)
    if world > 1:
        dist.all_reduce(cm_r)
        dist.all_reduce(cm_i)
    torch.backends.cudnn.benchmark = bench
    iou_r, iou_i = iou_from_confusion(cm_r).cpu(), iou_from_confusion(cm_i).cpu()
    acc_r = (cm_r.diag().sum() / cm_r.sum().clamp(min=1)).item()
    miou_r = float(np.nanmean(iou_r.numpy()))
    miou_i = float(np.nanmean(iou_i.numpy()))
    return {"room_mIoU": miou_r, "icon_mIoU": miou_i, "room_pixacc": acc_r,
            "score": (miou_r + miou_i) / 2,
            "room_IoU": {c: round(float(v), 4) for c, v in zip(ROOM_CLASSES, iou_r)},
            "icon_IoU": {c: round(float(v), 4) for c, v in zip(ICON_CLASSES, iou_i)}}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    torch.backends.cudnn.benchmark = True

    if args.probe:
        probe(cfg)
        return

    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo",
                                timeout=datetime.timedelta(minutes=60))
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local = 0, 1, 0
    if torch.cuda.is_available():
        dev = torch.device("cuda", local)
        torch.cuda.set_device(dev)
    else:  # CPU smoke-test path only
        dev = torch.device("cpu")
    is_main = rank == 0
    seed = cfg["seed"] + rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    def bcast(value):
        """Broadcast a float from rank 0 so every rank takes the same decision."""
        t = torch.tensor([float(value)], device=dev)
        if world > 1:
            dist.broadcast(t, 0)
        return t.item()

    # data
    train_ds = CachedFloorplans(cfg["cache_root"], "train", True, cfg["crop"], cfg["sigma"])
    val_ds = CachedFloorplans(cfg["cache_root"], "val", False, cfg["crop"], cfg["sigma"])
    sampler = torch.utils.data.distributed.DistributedSampler(
        train_ds, num_replicas=world, rank=rank, shuffle=True, seed=cfg["seed"], drop_last=True)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg["batch_per_gpu"], sampler=sampler, num_workers=cfg["workers"],
        pin_memory=dev.type == "cuda", drop_last=True, persistent_workers=True, prefetch_factor=4,
        worker_init_fn=seed_worker)
    spe = len(train_loader)

    # model (+ resume)
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)
    last_path = os.path.join(cfg["ckpt_dir"], "last.pt")
    best_path = os.path.join(cfg["ckpt_dir"], "best.pt")
    ckpt = None
    if os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        if ckpt.get("signature") != signature(cfg):
            raise RuntimeError(f"Checkpoint config {ckpt.get('signature')} != current "
                               f"{signature(cfg)}. Use a new HF repo name or matching settings.")
    pretrained = None if (ckpt or os.environ.get("FP_NO_PRETRAIN")) else "imagenet"
    model = build_model(cfg["encoder"], pretrained)
    if ckpt:
        model.load_state_dict(ckpt["model"])
    if world > 1 and dev.type == "cuda":
        from timm.layers import convert_sync_batchnorm
        model = convert_sync_batchnorm(model)
    model = model.to(dev)
    ema = EMA(model, cfg["ema_decay"])
    if ckpt:
        ema.module.load_state_dict(ckpt["ema"])
        ema.updates = ckpt.get("ema_updates", 0)
    ddp = (nn.parallel.DistributedDataParallel(
        model, device_ids=[local] if dev.type == "cuda" else None, gradient_as_bucket_view=True)
        if world > 1 else model)
    opt = make_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    crit = MultiTaskLoss(*cfg["loss_weights"])

    start_epoch, best_score, total_epochs, history = 0, -1.0, None, []
    if ckpt:
        opt.load_state_dict(ckpt["opt"])
        for g, bl in zip(opt.param_groups, ckpt["base_lrs"]):
            g["base_lr"] = bl
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best_score = ckpt["epoch"], ckpt["best_score"]
        total_epochs, history = ckpt["total_epochs"], ckpt.get("history", [])
        log(f"Resumed at epoch {start_epoch} (best score {best_score:.4f}, "
            f"schedule {total_epochs} epochs)") if is_main else None
    del ckpt
    if cfg.get("force_total_epochs"):
        total_epochs = int(cfg["force_total_epochs"])
    adaptive = total_epochs is None
    if adaptive:
        total_epochs = cfg["max_epochs"]
    uploader = Uploader(cfg.get("hf_repo"), os.environ.get("HF_TOKEN")) if is_main else None
    deadline = cfg["deadline_ts"]
    if is_main:
        log(f"world={world} batch/gpu={cfg['batch_per_gpu']} steps/epoch={spe} "
            f"train={len(train_ds)} val={len(val_ds)} epochs {start_epoch}->{total_epochs}")

    epoch_secs = None
    stopped_by_time = False
    epoch = start_epoch
    for epoch in range(start_epoch, total_epochs):
        # ---- time guard (rank 0 decides)
        if epoch_secs is not None:
            need = epoch_secs * 1.15 + cfg["final_margin_s"]
            stop = bcast(1.0 if time.time() + need > deadline else 0.0)
            if stop:
                stopped_by_time = True
                if is_main:
                    log(f"Time budget reached before epoch {epoch}; stopping cleanly.")
                break
        t_ep = time.time()
        ddp.train()
        sampler.set_epoch(epoch)
        run = {"loss": 0.0, "room": 0.0, "icon": 0.0, "heat": 0.0}
        n_seen = 0
        for it, (x, room, icon, heat) in enumerate(train_loader):
            step = epoch * spe + it
            set_lr(opt, step, spe, total_epochs, cfg["warmup_epochs"])
            x = x.to(dev, non_blocking=True)
            room, icon = room.to(dev, non_blocking=True), icon.to(dev, non_blocking=True)
            heat = heat.to(dev, non_blocking=True)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                out = ddp(x)
            loss, parts = crit(out, room, icon, heat)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            ema.update(model)
            if torch.isfinite(loss):
                run["loss"] += loss.item(); n_seen += 1
                for k in parts:
                    run[k] += parts[k].item()
            if is_main and ((it + 1) % cfg["log_every"] == 0 or it == 0 or it + 1 == spe):
                d = max(1, n_seen)
                el = time.time() - t_ep
                ips = (it + 1) * cfg["batch_per_gpu"] * world / max(el, 1e-6)
                ep_eta = el / (it + 1) * (spe - it - 1)
                per_ep = epoch_secs or (el / (it + 1) * spe * 1.15)
                run_eta = ep_eta + per_ep * (total_epochs - epoch - 1)
                frac = (it + 1) / spe
                bar = "#" * int(frac * 20) + "-" * (20 - int(frac * 20))
                log(f"ep {epoch + 1}/{total_epochs} [{bar}] {frac * 100:5.1f}% it {it + 1}/{spe} | "
                    f"loss {run['loss'] / d:.3f} (room {run['room'] / d:.3f} icon {run['icon'] / d:.3f} "
                    f"heat {run['heat'] / d:.3f}) | {ips:.1f} img/s | epoch ETA {_hms(ep_eta)} | "
                    f"run ETA {_hms(run_eta)} | best {max(best_score, 0):.4f}")
        train_secs = time.time() - t_ep

        # ---- validation (EMA weights)
        metrics = None
        if (epoch + 1) % cfg["val_every"] == 0 or epoch + 1 == total_epochs:
            if is_main:
                log(f"ep {epoch + 1}: training done in {_hms(time.time() - t_ep)}, validating {len(val_ds)} plans ...")
            metrics = validate(ema.module, val_ds, rank, world, dev)
        epoch_secs = time.time() - t_ep

        # ---- adapt the cosine schedule to the time budget (fresh runs, after epoch 0)
        if adaptive and epoch == start_epoch:
            remaining = deadline - time.time() - cfg["final_margin_s"]
            fit = epoch + 1 + int(remaining // (epoch_secs * 1.08))
            new_total = int(bcast(max(epoch + 2, min(cfg["max_epochs"], fit))))
            if is_main:
                log(f"Epoch takes {epoch_secs / 60:.1f} min -> schedule set to {new_total} epochs "
                    f"to finish inside the time budget.")
            total_epochs, adaptive = new_total, False

        # ---- checkpoint + upload (rank 0)
        if is_main:
            d = max(1, n_seen)
            rec = {"epoch": epoch, "train_loss": run["loss"] / d, "room_loss": run["room"] / d,
                   "icon_loss": run["icon"] / d, "heat_loss": run["heat"] / d,
                   "epoch_min": round(epoch_secs / 60, 2), "train_min": round(train_secs / 60, 2)}
            improved = False
            if metrics:
                rec.update(metrics)
                improved = metrics["score"] > best_score
                if improved:
                    best_score = metrics["score"]
                log(f"VAL ep {epoch}: room mIoU {metrics['room_mIoU']:.4f} icon mIoU "
                    f"{metrics['icon_mIoU']:.4f} | wall {metrics['room_IoU']['Wall']:.3f} "
                    f"door {metrics['icon_IoU']['Door']:.3f} window {metrics['icon_IoU']['Window']:.3f} "
                    f"stairs {metrics['icon_IoU']['Stairs']:.3f} {'*BEST*' if improved else ''}")
            history.append(rec)
            state = {"model": model.state_dict() if world == 1 else _unsync_state(model),
                     "ema": ema.module.state_dict(), "ema_updates": ema.updates,
                     "opt": opt.state_dict(), "base_lrs": [g["base_lr"] for g in opt.param_groups],
                     "scaler": scaler.state_dict(), "epoch": epoch + 1, "best_score": best_score,
                     "total_epochs": total_epochs, "history": history,
                     "signature": signature(cfg), "config": cfg | {"hf_token": None}}
            atomic_save(state, last_path)
            files = [(last_path, "checkpoints/last.pt")]
            if improved:
                atomic_save({"ema": ema.module.state_dict(), "epoch": epoch, "metrics": metrics,
                             "signature": signature(cfg), "room_classes": ROOM_CLASSES,
                             "icon_classes": ICON_CLASSES, "encoder": cfg["encoder"]}, best_path)
                files.append((best_path, "checkpoints/best.pt"))
            mpath = os.path.join(cfg["ckpt_dir"], "metrics.json")
            json.dump(history, open(mpath, "w"), indent=1)
            files.append((mpath, "metrics.json"))
            uploader.submit(files, f"epoch {epoch}" + (" (best)" if improved else ""))
        if world > 1:
            dist.barrier()

    # ---- finish
    if is_main:
        done = (not stopped_by_time) and len(history) > 0 and history[-1]["epoch"] + 1 >= total_epochs
        status = {"finished_schedule": bool(done), "stopped_by_time": stopped_by_time,
                  "epochs_done": (history[-1]["epoch"] + 1) if history else start_epoch,
                  "total_epochs": total_epochs, "best_score": best_score,
                  "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        spath = os.path.join(cfg["ckpt_dir"], "status.json")
        json.dump(status, open(spath, "w"), indent=1)
        uploader.submit([(spath, "status.json")], "run status")
        uploader.wait(timeout=cfg["final_margin_s"] - 60)
        log("STATUS", json.dumps(status))
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def _hms(sec):
    sec = int(max(0, sec))
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"


def _unsync_state(model):
    # SyncBatchNorm and BatchNorm share parameter/buffer names, so the state dict loads
    # straight into the plain (non-sync) model on resume / inference.
    return model.state_dict()


if __name__ == "__main__":
    main()
