"""Shared code: classes, cached dataset + augmentation, model, losses, metrics."""
import json
import math
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

cv2.setNumThreads(0)

# ----------------------------------------------------------------------------- classes
# Room head (from the official CubiCasa5K "rooms_selected" mapping)
ROOM_CLASSES = ["Background", "Outdoor", "Wall", "Kitchen", "LivingRoom", "Bedroom",
                "Bath", "Entry/Hallway", "Railing", "Storage", "Garage", "OtherRoom"]
# Icon head (official "icons_selected" mapping) + Stairs (parsed separately, id 11)
ICON_CLASSES = ["NoIcon", "Window", "Door", "Closet", "ElectricalAppliance", "Toilet",
                "Sink", "SaunaBench", "Fireplace", "Bathtub", "Chimney", "Stairs"]
# Junction heatmaps (orientation-free so flips/rotations stay valid)
HEAT_CLASSES = ["WallJunction", "OpeningEndpoint", "IconCorner"]

# v2 heads (appended after the heatmaps, so channels 0-26 keep their meaning and older engines still read them)
#   wall type: on wall pixels only. Learned from every source that labels the outside (rooms + background):
#              a wall the outside touches is exterior. Unknown (IGNORE) elsewhere and for stroke-only sources.
#   level:     on floor pixels only. CubiCasa5K marks raised floor ("Elevated") and voids over a lower floor
#              ("OpenToBelow"); Structured3D floors are all normal. Unknown for the other sources.
WALLTYPE_CLASSES = ["Interior", "Exterior"]
LEVEL_CLASSES = ["Normal", "Raised", "OpenToBelow"]

N_ROOM, N_ICON, N_HEAT = len(ROOM_CLASSES), len(ICON_CLASSES), len(HEAT_CLASSES)
N_WT, N_LV = len(WALLTYPE_CLASSES), len(LEVEL_CLASSES)
N_BASE = N_ROOM + N_ICON + N_HEAT           # 27: the v1 model
N_OUT = N_BASE + N_WT + N_LV                # 32
# sources whose room labels include the outside (background / outdoor), so wall types can be derived
WALLTYPE_SOURCES = {"cubicasa", "resplan", "synthetic", "structured3d"}
IGNORE = 255
NOT_WALL = 254       # room target: "known not wall, room type unknown" (sources that only label walls)
# Rotation augmentation: plans with angled wings mostly use these angles; any angle is also trained
ROT_ANGLES = (30.0, 45.0, 60.0)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0


# ----------------------------------------------------------------------------- heatmaps
def render_heatmaps(points, h, w, sigma):
    """points: (N,3) float array [x, y, type]. Returns (N_HEAT, h, w) float32, peak exactly 1."""
    heat = np.zeros((N_HEAT, h, w), dtype=np.float32)
    if points is None or len(points) == 0:
        return heat
    r = int(math.ceil(3 * sigma))
    ax = np.arange(-r, r + 1, dtype=np.float32)
    g = np.exp(-(ax[None, :] ** 2 + ax[:, None] ** 2) / (2 * sigma * sigma)).astype(np.float32)
    for x, y, t in points:
        cx, cy, t = int(round(float(x))), int(round(float(y))), int(t)
        if not (0 <= cx < w and 0 <= cy < h) or not (0 <= t < N_HEAT):
            continue
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        patch = g[y0 - (cy - r):y1 - (cy - r), x0 - (cx - r):x1 - (cx - r)]
        np.maximum(heat[t, y0:y1, x0:x1], patch, out=heat[t, y0:y1, x0:x1])
    return heat


# ----------------------------------------------------------------------------- dataset
class CachedFloorplans(torch.utils.data.Dataset):
    """Reads the preprocessed cache: <root>/data/<id>_img.png, _lbl.png (B=room, G=icon), _pts.npy"""

    def __init__(self, root, split, train, crop=768, sigma=3.0, scale_range=(0.6, 1.8), rot_prob=0.6,
                 fixed_angle=None, sources=None, limit=None):
        """rot_prob: share of training crops rotated by a non-90-degree angle (30/45/60 +-3 deg, or any angle).
        fixed_angle: evaluation only - every sample rotated by exactly this angle (rotated-plan validation)."""
        with open(os.path.join(root, "index.json")) as f:
            index = json.load(f)
        self.items = index[split]
        if sources is not None:                     # e.g. ["cubicasa"]; entries without "source" are CubiCasa5K
            self.items = [it for it in self.items if it.get("source", "cubicasa") in sources]
        if limit:
            self.items = self.items[:limit]
        self.root = os.path.join(root, "data")
        self.train, self.crop, self.sigma, self.scale_range = train, crop, sigma, scale_range
        self.rot_prob, self.fixed_angle = rot_prob, fixed_angle

    def __len__(self):
        return len(self.items)

    def source_of(self, i):
        return self.items[i].get("source", "cubicasa")

    def _load(self, i):
        it = self.items[i]
        sid = it["id"]
        img = cv2.imread(os.path.join(self.root, sid + "_img.png"), cv2.IMREAD_COLOR)
        lbl = cv2.imread(os.path.join(self.root, sid + "_lbl.png"), cv2.IMREAD_UNCHANGED)
        if img is None or lbl is None:
            raise RuntimeError(f"Corrupt cache entry {sid}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pp = os.path.join(self.root, sid + "_pts.npy")
        # sources without junction labels have no _pts file: their heat target is "unknown", not "none"
        pts = np.load(pp).astype(np.float32).reshape(-1, 3) if os.path.exists(pp) else None
        room, icon = lbl[:, :, 0].copy(), lbl[:, :, 1].copy()
        # third label channel = floor level classes, only for entries written with it (index flag "level")
        level = lbl[:, :, 2].copy() if it.get("level") else np.full(room.shape, IGNORE, np.uint8)
        wtype = wall_type_labels(room) if it.get("source", "cubicasa") in WALLTYPE_SOURCES \
            else np.full(room.shape, IGNORE, np.uint8)
        return img, [room, icon, wtype, level], pts

    def __getitem__(self, i):
        img, maps, pts = self._load(i)
        has_heat = pts is not None
        if pts is None:
            pts = np.zeros((0, 3), np.float32)
        if self.train:
            img, maps, pts = self._augment(img, maps, pts)
        elif self.fixed_angle is not None:
            img, maps, pts = self._rotate_whole(img, maps, pts, self.fixed_angle)
            img, maps = self._pad32(img, maps)
        else:
            img, maps = self._pad32(img, maps)
        h, w = maps[0].shape
        heat = render_heatmaps(pts, h, w, self.sigma) if has_heat else np.full((N_HEAT, h, w), -1.0, np.float32)
        x = (img.astype(np.float32) - MEAN) / STD
        t = [torch.from_numpy(np.ascontiguousarray(m)).long() for m in maps]
        return (torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))), t[0], t[1],
                torch.from_numpy(heat), t[2], t[3])

    @staticmethod
    def _pad32(img, maps):
        h, w = maps[0].shape
        ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32
        if ph or pw:
            img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            maps = [cv2.copyMakeBorder(m, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=IGNORE) for m in maps]
        return img, maps

    @staticmethod
    def _rotate_whole(img, maps, pts, angle):
        """Rotate a whole plan by `angle` degrees on a canvas that fits it (evaluation on rotated plans)."""
        h, w = maps[0].shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        c, s_ = abs(M[0, 0]), abs(M[0, 1])
        W2, H2 = int(math.ceil(h * s_ + w * c)), int(math.ceil(h * c + w * s_))
        M[0, 2] += W2 / 2 - w / 2
        M[1, 2] += H2 / 2 - h / 2
        img = cv2.warpAffine(img, M, (W2, H2), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
        maps = [cv2.warpAffine(m, M, (W2, H2), flags=cv2.INTER_NEAREST, borderValue=IGNORE) for m in maps]
        if len(pts):
            pts = pts.copy()
            xy = pts[:, :2] @ M[:, :2].T + M[:, 2]
            pts[:, 0], pts[:, 1] = xy[:, 0], xy[:, 1]
        return img, maps, pts

    def _augment(self, img, maps, pts):
        S = self.crop
        h, w = maps[0].shape
        lo, hi = self.scale_range
        s = math.exp(random.uniform(math.log(lo), math.log(hi)))
        src = S / s  # crop size in source pixels
        x0 = random.uniform(0, w - src) if w > src else -random.uniform(0, src - w)
        y0 = random.uniform(0, h - src) if h > src else -random.uniform(0, src - h)
        M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], dtype=np.float32)
        if random.random() < self.rot_prob:
            # angled walls: rotate the crop about its centre (the 90-degree turns / flips below add the rest,
            # so 30 also covers 120, 210, 300 and with the flip -30 ...). Corners outside the plan stay
            # white paper and are ignored in the loss.
            a = random.choice(ROT_ANGLES) + random.uniform(-3, 3) if random.random() < 0.7 else random.uniform(0, 90)
            R = cv2.getRotationMatrix2D((S / 2, S / 2), a, 1.0).astype(np.float32)
            M = (np.vstack([R, [0, 0, 1]]) @ np.vstack([M, [0, 0, 1]]))[:2].astype(np.float32)
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
        img = cv2.warpAffine(img, M, (S, S), flags=interp, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(255, 255, 255))
        maps = [cv2.warpAffine(m, M, (S, S), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=IGNORE) for m in maps]
        if len(pts):
            pts = pts.copy()
            xy = pts[:, :2] @ M[:, :2].T + M[:, 2]
            pts[:, 0], pts[:, 1] = xy[:, 0], xy[:, 1]
        # rotations / flips (all heads are orientation-free, so labels stay valid)
        k = random.randint(0, 3)
        for _ in range(k):  # 90 deg counter-clockwise
            img, maps = np.rot90(img), [np.rot90(m) for m in maps]
            if len(pts):
                pts[:, 0], pts[:, 1] = pts[:, 1].copy(), (S - 1) - pts[:, 0]
        if random.random() < 0.5:
            img, maps = img[:, ::-1], [m[:, ::-1] for m in maps]
            if len(pts):
                pts[:, 0] = (S - 1) - pts[:, 0]
        img = np.ascontiguousarray(img).astype(np.float32)
        # photometric
        if random.random() < 0.3:
            g = img.mean(axis=2, keepdims=True)
            img = np.repeat(g, 3, axis=2)
        if random.random() < 0.8:
            img = (img - 128.0) * random.uniform(0.75, 1.25) + 128.0 + random.uniform(-25, 25)
        if random.random() < 0.2:
            img = cv2.GaussianBlur(img, (0, 0), random.uniform(0.3, 1.0))
        if random.random() < 0.25:
            img = img + np.random.normal(0, random.uniform(2, 8), img.shape).astype(np.float32)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img, [np.ascontiguousarray(m) for m in maps], pts


def wall_type_labels(room):
    """Wall pixels -> 0 interior / 1 exterior; IGNORE off walls. A wall is exterior where the outside (background
    connected to the page edge, or an outdoor area) lies within about one wall thickness of it."""
    WALL, BG, OUT = ROOM_CLASSES.index("Wall"), ROOM_CLASSES.index("Background"), ROOM_CLASSES.index("Outdoor")
    out = np.full(room.shape, IGNORE, np.uint8)
    wall = room == WALL
    if not wall.any():
        return out
    bg = (room == BG).astype(np.uint8)
    n, cc = cv2.connectedComponents(bg, connectivity=4)
    edge = np.unique(np.concatenate([cc[0], cc[-1], cc[:, 0], cc[:, -1]]))
    outside = np.isin(cc, edge[edge > 0]) | (room == OUT)
    if not outside.any():
        out[wall] = IGNORE                       # a crop with no outside in it: no way to tell
        return out
    # local wall thickness at each pixel (twice the largest inscribed radius nearby) vs distance to the outside
    dt = cv2.distanceTransform(wall.astype(np.uint8), cv2.DIST_L2, 3)
    k = max(3, int(2 * np.percentile(dt[wall], 95)) * 2 + 1)
    thick = 2.0 * cv2.dilate(dt, np.ones((k, k), np.uint8))
    d_out = cv2.distanceTransform((~outside).astype(np.uint8), cv2.DIST_L2, 3)
    near = d_out <= thick + 1.5
    out[wall] = 0
    out[wall & near] = 1
    return out


# ----------------------------------------------------------------------------- model
def build_model(encoder="tu-efficientnet_b3", weights="imagenet"):
    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name=encoder, encoder_weights=weights, classes=N_OUT,
                     decoder_attention_type="scse")
    # focal-loss prior for heatmap logits (CenterNet init)
    head = [m for m in model.segmentation_head.modules() if isinstance(m, nn.Conv2d)][-1]
    with torch.no_grad():
        head.bias[N_ROOM + N_ICON:N_BASE].fill_(-2.19)
    # Some smp decoder blocks own parameters that never get gradients (e.g. the attention on
    # the missing skip of the last block). DDP aborts on those, so detect + freeze them.
    saved = {k: v.clone() for k, v in model.state_dict().items()}  # keep BN stats untouched
    model.train()
    model(torch.randn(2, 3, 64, 64)).float().sum().backward()
    for p in model.parameters():
        if p.grad is None:
            p.requires_grad_(False)
        p.grad = None
    model.load_state_dict(saved)
    return model


def split_logits(out):
    return out[:, :N_ROOM], out[:, N_ROOM:N_ROOM + N_ICON], out[:, N_ROOM + N_ICON:N_BASE]


def split_aux(out):
    """v2 heads: (wall type logits, level logits)."""
    return out[:, N_BASE:N_BASE + N_WT], out[:, N_BASE + N_WT:N_BASE + N_WT + N_LV]


def expand_head(state, model):
    """Weights of a model with fewer output channels (the 27-channel v1 model) -> this model's state dict:
    every layer copied, the segmentation head's first channels copied and the new channels left as initialised.
    Returns (state dict, number of channels copied) or (state, None) when nothing needed changing."""
    own = model.state_dict()
    out, grown = dict(state), None
    for k, v in state.items():
        if k in own and own[k].shape != v.shape and k.startswith("segmentation_head") and \
                own[k].shape[1:] == v.shape[1:] and own[k].shape[0] > v.shape[0]:
            w = own[k].clone()
            w[:v.shape[0]] = v
            out[k], grown = w, int(v.shape[0])
    return out, grown


# ----------------------------------------------------------------------------- losses
def dice_loss(logits, target, n_cls, ignore=IGNORE, eps=1.0):
    valid = (target != ignore)
    t = torch.where(valid, target, torch.zeros_like(target))
    prob = logits.softmax(1) * valid.unsqueeze(1)
    oh = F.one_hot(t, n_cls).permute(0, 3, 1, 2).float() * valid.unsqueeze(1)
    dims = (0, 2, 3)
    inter = (prob * oh).sum(dims)
    card = prob.sum(dims) + oh.sum(dims)
    present = oh.sum(dims) > 0  # only classes present in the batch
    dice = (2 * inter + eps) / (card + eps)
    if present.any():
        return 1 - dice[present].mean()
    return logits.sum() * 0.0


def focal_ce(logits, target, gamma=2.0, ignore=IGNORE):
    ce = F.cross_entropy(logits, target, ignore_index=ignore, reduction="none")
    valid = (target != ignore)
    if not valid.any():
        return logits.sum() * 0.0
    pt = torch.exp(-ce)
    return (((1 - pt) ** gamma) * ce)[valid].mean()


def heat_focal(logits, gt, alpha=2.0, beta=4.0):
    keep = gt[:, 0, 0, 0] >= 0                     # samples whose source labels junctions (else target is -1)
    if not keep.any():
        return logits.sum() * 0.0
    logits, gt = logits[keep], gt[keep]
    pred = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = gt.ge(0.999).float()
    neg = 1 - pos
    pos_loss = torch.log(pred) * (1 - pred) ** alpha * pos
    neg_loss = torch.log(1 - pred) * pred ** alpha * (1 - gt) ** beta * neg
    n_pos = pos.sum().clamp(min=1.0)
    return -(pos_loss.sum() + neg_loss.sum()) / n_pos


class MultiTaskLoss(nn.Module):
    def __init__(self, w_room=1.0, w_icon=1.0, w_heat=1.0, w_wtype=0.5, w_level=0.5):
        super().__init__()
        self.w = (w_room, w_icon, w_heat, w_wtype, w_level)

    def forward(self, out, room, icon, heat, wtype=None, level=None):
        r, i, h = [t.float() for t in split_logits(out)]
        not_wall = room == NOT_WALL
        room = torch.where(not_wall, torch.full_like(room, IGNORE), room)
        l_room = F.cross_entropy(r, room, ignore_index=IGNORE) + dice_loss(r, room, N_ROOM)
        if not_wall.any():                          # "not a wall, room type unknown": push only P(wall) down
            p_wall = torch.softmax(r, 1)[:, ROOM_CLASSES.index("Wall")]
            l_room = l_room + (-torch.log((1 - p_wall[not_wall]).clamp(min=1e-6))).mean()
        l_icon = focal_ce(i, icon) + dice_loss(i, icon, N_ICON)
        l_heat = heat_focal(h, heat)
        total = self.w[0] * l_room + self.w[1] * l_icon + self.w[2] * l_heat
        parts = {"room": l_room.detach(), "icon": l_icon.detach(), "heat": l_heat.detach()}
        if out.shape[1] >= N_OUT and wtype is not None:
            wt, lv = [t.float() for t in split_aux(out)]
            l_wt = focal_ce(wt, wtype)              # 0 when the batch has no labelled wall (IGNORE everywhere)
            l_lv = focal_ce(lv, level) + (dice_loss(lv, level, N_LV) if (level != IGNORE).any() else lv.sum() * 0)
            total = total + self.w[3] * l_wt + self.w[4] * l_lv
            parts.update(wtype=l_wt.detach(), level=l_lv.detach())
        return total, parts


# ----------------------------------------------------------------------------- metrics
def update_confusion(cm, pred, target, n_cls, ignore=IGNORE):
    valid = (target != ignore) & (target != NOT_WALL)
    idx = target[valid] * n_cls + pred[valid]
    cm += torch.bincount(idx, minlength=n_cls * n_cls).reshape(n_cls, n_cls)
    return cm


def iou_from_confusion(cm):
    cm = cm.double()
    tp = cm.diag()
    denom = cm.sum(0) + cm.sum(1) - tp
    iou = torch.where(denom > 0, tp / denom.clamp(min=1), torch.full_like(tp, float("nan")))
    return iou  # nan for classes absent from both pred and gt


class EMA:
    """Exponential moving average of weights (fp32), with warmup-aware decay."""

    def __init__(self, model, decay=0.999):
        import copy
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.updates = decay, 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


class SourceWeightedSampler(torch.utils.data.Sampler):
    """Each epoch draws `num_samples` items with per-source shares (e.g. CubiCasa 0.4, ResPlan 0.25, synthetic
    angled 0.2, FloorPlanCAD 0.15), so a big source does not drown the others; sharded across DDP ranks."""

    def __init__(self, ds, weights, num_samples, rank=0, world=1, seed=0):
        self.groups = {}
        for i in range(len(ds)):
            self.groups.setdefault(ds.source_of(i), []).append(i)
        w = {k: float(weights.get(k, 0.0)) for k in self.groups}
        tot = sum(w.values()) or 1.0
        self.share = {k: v / tot for k, v in w.items() if v > 0}
        self.num_samples = num_samples - num_samples % world
        self.rank, self.world, self.seed, self.epoch = rank, world, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = random.Random(self.seed + self.epoch)
        picks = []
        for k, sh in self.share.items():
            n = int(round(sh * self.num_samples))
            pool = self.groups[k]
            picks += [pool[g.randrange(len(pool))] for _ in range(n)] if n > len(pool) else g.sample(pool, n)
        g.shuffle(picks)
        picks = picks[:self.num_samples]
        picks += picks[:self.num_samples - len(picks)]
        return iter(picks[self.rank::self.world])

    def __len__(self):
        return self.num_samples // self.world
