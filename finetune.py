"""
finetune.py
Phase II supervised fine-tuning for brain tumour segmentation (manuscript Sec. 3.7, 3.8).

Architecture (Sec. 3.7.1)
  Encoder  the pre-trained ViT (patch embedding + transformer blocks) loaded from a
           pre-training checkpoint. The MAE decoder and the token sampler are discarded.
  Decoder  UNETR-style: features after blocks 3, 6, 9 and 12 are reshaped to 14x14 maps
           and progressively upsampled to 224x224, with skip connections at each scale
           and a final 1x1 convolution to 4 classes (background, NCR, ED, ET).

Loss (Sec. 3.7.2)
  L_seg = 0.5 * soft Dice (tumour classes 1-3) + 0.5 * cross-entropy

Evaluation (Sec. 3.8.4)
  Slice predictions are stacked back into each patient's volume, and metrics are computed
  per patient in 3D for the three BraTS regions, then averaged over patients:
    WT = labels {1,2,3}    TC = labels {1,3}    ET = label 3
  DSC and Recall always; HD95 with --hd95 (slower). When a region is absent from both the
  prediction and the ground truth, DSC is 1 (BraTS convention); absent from the ground
  truth only, DSC is 0.

Training uses a fixed epoch budget and the final checkpoint (Sec. 3.6.1). A checkpoint is
written every epoch, so an interrupted run resumes from its last completed epoch.

Usage
  python finetune.py --pretrained results/ckpt/<run>/pretrain.pt --cache ./cache --split dev
  python finetune.py --scratch --cache ./cache --split dev          # no pre-training
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import BraTSSliceDataset, load_split
from models import PatchEmbed, TransformerBlock, sincos_pos_embed

REGIONS = {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,)}
BACKBONES = {"small": dict(dim=384, depth=12, heads=6), "base": dict(dim=768, depth=12, heads=12)}


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


def up(cin, cout):
    return nn.ConvTranspose2d(cin, cout, kernel_size=2, stride=2)


class SegViT(nn.Module):
    def __init__(self, dim=384, depth=12, heads=6, n_classes=4, in_ch=4, ch=(256, 128, 64, 32)):
        super().__init__()
        assert depth == 12, "skip connections are taken after blocks 3, 6, 9, 12"
        self.patch_embed = PatchEmbed(224, 16, in_ch, dim)
        self.register_buffer("pos", sincos_pos_embed(dim, 14)[None])
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        c0, c1, c2, c3 = ch

        self.up12 = up(dim, c0)                                            # 14 -> 28
        self.sk9 = up(dim, c0)
        self.dec28 = conv_block(2 * c0, c0)
        self.up28 = up(c0, c1)                                             # 28 -> 56
        self.sk6 = nn.Sequential(up(dim, c1), up(c1, c1))
        self.dec56 = conv_block(2 * c1, c1)
        self.up56 = up(c1, c2)                                             # 56 -> 112
        self.sk3 = nn.Sequential(up(dim, c2), up(c2, c2), up(c2, c2))
        self.dec112 = conv_block(2 * c2, c2)
        self.up112 = up(c2, c3)                                            # 112 -> 224
        self.sk0 = conv_block(in_ch, c3)
        self.dec224 = conv_block(2 * c3, c3)
        self.head = nn.Conv2d(c3, n_classes, 1)

    def encoder_parameters(self):
        return [p for n, p in self.named_parameters()
                if n.startswith(("patch_embed", "blocks", "norm"))]

    def forward(self, img):
        B = img.shape[0]
        x = self.patch_embed(img) + self.pos
        feats = {}
        for i, blk in enumerate(self.blocks, 1):
            x = blk(x)
            if i in (3, 6, 9, 12):
                t = self.norm(x) if i == 12 else x
                feats[i] = t.transpose(1, 2).reshape(B, -1, 14, 14)

        y = self.dec28(torch.cat([self.up12(feats[12]), self.sk9(feats[9])], 1))
        y = self.dec56(torch.cat([self.up28(y), self.sk6(feats[6])], 1))
        y = self.dec112(torch.cat([self.up56(y), self.sk3(feats[3])], 1))
        y = self.dec224(torch.cat([self.up112(y), self.sk0(img)], 1))
        return self.head(y)


def load_pretrained_encoder(model, ckpt_path):
    """Copy patch_embed / blocks / norm weights from a pre-training checkpoint."""
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = sd.get("model", sd)
    enc = {k: v for k, v in sd.items() if k.startswith(("patch_embed.", "blocks.", "norm."))}
    missing, unexpected = model.load_state_dict(enc, strict=False)
    missing = [k for k in missing if k.startswith(("patch_embed.", "blocks.", "norm."))]
    if missing or unexpected:
        raise RuntimeError(f"encoder mismatch: missing {missing[:3]} unexpected {unexpected[:3]}")
    return len(enc)


# --------------------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------------------
def seg_loss(logits, target, w_dice=0.5, w_ce=0.5):
    ce = F.cross_entropy(logits, target)
    prob = logits.float().softmax(1)[:, 1:]                                  # tumour classes
    onehot = F.one_hot(target, 4).permute(0, 3, 1, 2).float()[:, 1:]
    inter = (prob * onehot).sum((0, 2, 3))
    denom = prob.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    dice = 1.0 - ((2 * inter + 1.0) / (denom + 1.0)).mean()
    return w_dice * dice + w_ce * ce


# --------------------------------------------------------------------------------------
# Evaluation: per-patient 3D metrics
# --------------------------------------------------------------------------------------
def hd95(pred, gt):
    """95th-percentile symmetric surface distance in voxels (1 mm isotropic in BraTS)."""
    from scipy.ndimage import binary_erosion, distance_transform_edt
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return float(np.sqrt(sum(s * s for s in gt.shape)))                # worst case
    sp = pred & ~binary_erosion(pred)
    sg = gt & ~binary_erosion(gt)
    d_to_g = distance_transform_edt(~sg)[sp]
    d_to_p = distance_transform_edt(~sp)[sg]
    return float(np.percentile(np.concatenate([d_to_g, d_to_p]), 95))


@torch.no_grad()
def evaluate(model, ds, device, batch_size=32, workers=2, with_hd95=False, amp=True):
    """Returns (summary dict, per-case list). ds must be un-augmented and un-shuffled."""
    model.eval()
    info = json.load(open(ds.dir / "cases.json"))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    preds, gts = [], []
    for b in loader:
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            logits = model(b["img"].to(device, non_blocking=True))
        preds.append(logits.argmax(1).to(torch.uint8).cpu().numpy())
        gts.append(b["seg"].to(torch.uint8).numpy())
    preds, gts = np.concatenate(preds), np.concatenate(gts)

    per_case, pos = [], 0
    for cid, n in zip(info["cases"], info["counts"]):
        p, g = preds[pos:pos + n], gts[pos:pos + n]
        pos += n
        row = {"case_id": cid}
        for r, labs in REGIONS.items():
            pr, gr = np.isin(p, labs), np.isin(g, labs)
            inter, sp, sg = int((pr & gr).sum()), int(pr.sum()), int(gr.sum())
            row[f"dsc_{r}"] = 1.0 if sp + sg == 0 else 2 * inter / (sp + sg)
            row[f"recall_{r}"] = float("nan") if sg == 0 else inter / sg
            if with_hd95:
                row[f"hd95_{r}"] = hd95(pr, gr)
        per_case.append(row)
    model.train()

    keys = [k for k in per_case[0] if k != "case_id"]
    summary = {k: float(np.nanmean([r[k] for r in per_case])) for k in keys}
    summary["dsc_mean"] = float(np.mean([summary[f"dsc_{r}"] for r in REGIONS]))
    return summary, per_case


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
def cosine_lr(step, total, base, warmup):
    if step < warmup:
        return base * (step + 1) / warmup
    return base * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def finetune(pretrained, cache, split, out_dir, backbone="small", epochs=30, batch_size=16,
             lr=3e-4, weight_decay=0.05, train_stride=2, workers=4, device="cuda",
             seed=42, with_hd95=False, amp=True, log=print):
    """Fine-tune one pre-trained encoder and evaluate it. Returns the summary metrics."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "finetune_result.json"
    if result_path.exists():
        log(f"  finetune already complete -> {result_path}")
        return json.load(open(result_path))["summary"]

    device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ids, val_ids = load_split(cache, split)
    tr = BraTSSliceDataset(cache, "finetune", train_ids, anat_kind=None, augment=True,
                           slice_stride=train_stride)
    va = BraTSSliceDataset(cache, "finetune", val_ids, anat_kind=None, augment=False,
                           slice_stride=1)
    loader = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=workers,
                        pin_memory=True, drop_last=True, persistent_workers=workers > 0)
    log(f"  finetune on '{split}': {len(train_ids)} train cases ({len(tr):,} slices, stride "
        f"{train_stride}) | {len(val_ids)} val cases ({len(va):,} slices)")

    model = SegViT(**BACKBONES[backbone]).to(device)
    if pretrained:
        n = load_pretrained_encoder(model, pretrained)
        log(f"  loaded {n} encoder tensors from {pretrained}")
    else:
        log("  encoder randomly initialised (--scratch)")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    total = epochs * len(loader)
    warmup = max(1, int(0.05 * total))

    ckpt_path, start_epoch, history = out_dir / "finetune_last.pt", 0, []
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, history = ck["epoch"], ck["history"]
        log(f"  resuming fine-tuning from epoch {start_epoch}")

    model.train()
    for epoch in range(start_epoch, epochs):
        t0, tot, n = time.time(), 0.0, 0
        for i, b in enumerate(loader):
            step = epoch * len(loader) + i
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, total, lr, warmup)
            img = b["img"].to(device, non_blocking=True)
            seg = b["seg"].to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=use_amp):
                loss = seg_loss(model(img), seg)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tot += loss.item(); n += 1
        history.append({"epoch": epoch + 1, "loss": tot / max(n, 1),
                        "seconds": time.time() - t0})
        log(f"    ft epoch {epoch+1}/{epochs}  loss {tot/max(n,1):.4f}  "
            f"({time.time()-t0:.0f}s, {n*batch_size/(time.time()-t0):.0f} img/s)")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch + 1, "history": history},
                   ckpt_path)

    summary, per_case = evaluate(model, va, device, batch_size=2 * batch_size,
                                 workers=workers, with_hd95=with_hd95, amp=amp)
    log("  val: " + "  ".join(f"{k} {v:.4f}" for k, v in summary.items() if k.startswith("dsc")))
    torch.save({"model": model.state_dict(), "backbone": backbone}, out_dir / "finetune_final.pt")
    json.dump({"summary": summary, "per_case": per_case, "history": history,
               "config": dict(pretrained=str(pretrained), split=split, backbone=backbone,
                              epochs=epochs, batch_size=batch_size, lr=lr,
                              train_stride=train_stride, seed=seed)},
              open(result_path, "w"), indent=1)
    ckpt_path.unlink(missing_ok=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pretrained", help="pre-training checkpoint (.pt)")
    src.add_argument("--scratch", action="store_true", help="random encoder initialisation")
    ap.add_argument("--cache", default="./cache")
    ap.add_argument("--split", default="dev", help="'dev' or 'fold_1'..'fold_5'")
    ap.add_argument("--out", default="./results/finetune_manual")
    ap.add_argument("--backbone", default="small", choices=list(BACKBONES))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--train_stride", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hd95", action="store_true")
    ap.add_argument("--no_amp", action="store_true")
    a = ap.parse_args()
    s = finetune(None if a.scratch else a.pretrained, a.cache, a.split, a.out, a.backbone,
                 a.epochs, a.batch_size, a.lr, train_stride=a.train_stride, workers=a.workers,
                 device=a.device, with_hd95=a.hd95, amp=not a.no_amp)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
