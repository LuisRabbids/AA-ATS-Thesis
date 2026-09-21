"""
run_pretrain.py
Phase I self-supervised pre-training (manuscript Sec. 3.5-3.6).

Trains one masked autoencoder configuration on the pre-training slices of a split,
saves the encoder checkpoint, and reports reconstruction MSE on the split's validation
cases under a FIXED random mask (Sec. 3.8.3): every model is scored on byte-identical
masked patches, independent of how it masked during training, and only on patches that
contain brain tissue.

Optimization (Sec. 3.6.1): AdamW, cosine schedule with linear warm-up, separate learning
rate for the token sampler, fixed epoch budget, final checkpoint. A checkpoint is written
every epoch so an interrupted run resumes where it stopped.

Single run:
  python run_pretrain.py --cache ./cache --mask_mode anatomical --fusion modulated \
      --direction visible --mask_ratio 0.75 --epochs 50

The Stage 1 grid is driven by run_stage1.py, which calls pretrain() below.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import BraTSSliceDataset, load_split
from models import MaskedAutoencoder

BACKBONES = {"small": dict(dim=384, depth=12, heads=6), "base": dict(dim=768, depth=12, heads=12)}
DEFAULTS = dict(
    mask_mode="anatomical", fusion="modulated", direction="visible", mask_ratio=0.75,
    backbone="small", epochs=50, batch_size=64, lr=1.5e-4, sampler_lr=1e-4,
    weight_decay=0.05, lambda_sampler=1.0, clip=1.0, split="dev", eval_ratio=0.75,
    workers=4, device="cuda", amp=True, seed=42,
)


def cosine_lr(step, total, base, warmup):
    if step < warmup:
        return base * (step + 1) / warmup
    return base * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def build(cfg):
    return MaskedAutoencoder(mask_mode=cfg["mask_mode"], fusion=cfg["fusion"],
                             direction=cfg["direction"], **BACKBONES[cfg["backbone"]])


@torch.no_grad()
def fixed_mask_mse(model, ds, device, ratio=0.75, seed=1234, batch_size=64, workers=4, amp=True):
    """Sec. 3.8.3 protocol: identical random mask for every model, brain patches only."""
    orig = model.mask_mode
    model.mask_mode = "random"
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    tot, cnt = 0.0, 0.0
    for i, b in enumerate(loader):
        torch.manual_seed(seed + i)
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            out = model(b["img"].to(device), b["anat"].to(device), ratio,
                        b["tissue"].to(device))
        n = float(out["n_scored"])
        tot += float(out["loss_recon"]) * n
        cnt += n
    model.mask_mode = orig
    model.train()
    return tot / max(cnt, 1.0)


class TimeUp(Exception):
    """Raised after a checkpoint when the next epoch would overrun the session deadline."""


def pretrain(cache, out_dir, log=print, should_stop=None, **overrides):
    """Pre-train one configuration. Returns dict(checkpoint=..., fixed_mask_mse=..., ...).
    should_stop(epoch_seconds) -> bool is checked after each saved epoch."""
    cfg = {**DEFAULTS, **overrides}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path, final_path = out_dir / "pretrain_result.json", out_dir / "pretrain.pt"
    if result_path.exists() and final_path.exists():
        log(f"  pretrain already complete -> {final_path}")
        return json.load(open(result_path))

    device = torch.device(cfg["device"] if (cfg["device"] != "cuda" or torch.cuda.is_available())
                          else "cpu")
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    train_ids, val_ids = load_split(cache, cfg["split"])
    anat_kind = "sobel" if cfg["mask_mode"] in {"anatomical", "hard_anat"} else None
    tr = BraTSSliceDataset(cache, "pretrain", train_ids, anat_kind=anat_kind, augment=True)
    va = BraTSSliceDataset(cache, "pretrain", val_ids, anat_kind=anat_kind, augment=False)
    loader = DataLoader(tr, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["workers"],
                        pin_memory=True, drop_last=True, persistent_workers=cfg["workers"] > 0)
    log(f"  pretrain {cfg['mask_mode']}/{cfg['fusion']}/{cfg['direction']}/p={cfg['mask_ratio']} "
        f"ViT-{cfg['backbone']}: {len(train_ids)} cases, {len(tr):,} slices, "
        f"{len(loader)} steps/epoch, device {device}")

    model = build(cfg).to(device)
    samp = [p for n, p in model.named_parameters() if n.startswith("sampler.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("sampler.")]
    groups = [{"params": rest, "base_lr": cfg["lr"]}]
    if samp:
        groups.append({"params": samp, "base_lr": cfg["sampler_lr"]})
    opt = torch.optim.AdamW(groups, lr=cfg["lr"], betas=(0.9, 0.95),
                            weight_decay=cfg["weight_decay"])
    use_amp = cfg["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    total = cfg["epochs"] * len(loader)
    warmup = max(1, int(0.1 * total))

    last, start, history = out_dir / "pretrain_last.pt", 0, []
    if last.exists():
        ck = torch.load(last, map_location="cpu")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        start, history = ck["epoch"], ck["history"]
        log(f"  resuming pre-training from epoch {start}")

    model.train()
    for epoch in range(start, cfg["epochs"]):
        t0, sr, ss, n = time.time(), 0.0, 0.0, 0
        for i, b in enumerate(loader):
            step = epoch * len(loader) + i
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, total, g["base_lr"], warmup)
            img = b["img"].to(device, non_blocking=True)
            anat = b["anat"].to(device, non_blocking=True)
            tis = b["tissue"].to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=use_amp):
                out = model(img, anat, cfg["mask_ratio"], tis)
                loss = out["loss_recon"] + cfg["lambda_sampler"] * out["loss_sampler"]
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            scaler.step(opt); scaler.update()
            sr += out["loss_recon"].item(); ss += out["loss_sampler"].item(); n += 1
        sec = time.time() - t0
        history.append({"epoch": epoch + 1, "train_mse": sr / max(n, 1),
                        "sampler_loss": ss / max(n, 1), "seconds": sec})
        log(f"    pt epoch {epoch+1}/{cfg['epochs']}  mse {sr/max(n,1):.5f}  "
            f"L_S {ss/max(n,1):+.5f}  ({sec:.0f}s, {n*cfg['batch_size']/sec:.0f} img/s)")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch + 1, "history": history}, last)
        if should_stop and epoch + 1 < cfg["epochs"] and should_stop(sec):
            raise TimeUp(f"pre-training saved at epoch {epoch+1}/{cfg['epochs']}")

    mse = fixed_mask_mse(model, va, device, cfg["eval_ratio"], workers=cfg["workers"],
                         amp=cfg["amp"])
    log(f"  fixed-mask reconstruction MSE (p={cfg['eval_ratio']}, brain patches): {mse:.6f}")
    torch.save({"model": model.state_dict(), "config": cfg, "history": history}, final_path)
    result = {"checkpoint": str(final_path), "fixed_mask_mse": mse, "config": cfg,
              "history": history,
              "minutes": sum(h["seconds"] for h in history) / 60}
    json.dump(result, open(result_path, "w"), indent=1)
    last.unlink(missing_ok=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="./cache")
    ap.add_argument("--out", default=None, help="output folder (default: results/manual/<tag>)")
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            ap.add_argument(f"--no_{k}", dest=k, action="store_false")
        else:
            ap.add_argument(f"--{k}", type=type(v), default=v)
    a = vars(ap.parse_args())
    cache, out = a.pop("cache"), a.pop("out")
    tag = f"{a['mask_mode']}_{a['fusion']}_{a['direction']}_p{a['mask_ratio']}"
    r = pretrain(cache, out or f"./results/manual/{tag}", **a)
    print(json.dumps({k: r[k] for k in ("checkpoint", "fixed_mask_mse", "minutes")}, indent=1))


if __name__ == "__main__":
    main()
