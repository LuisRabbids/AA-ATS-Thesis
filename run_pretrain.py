"""
run_pretrain.py
Phase I self-supervised pretraining + RQ1 reconstruction experiment
(manuscript Sec. 3.5-3.6, Sec. 3.8.1 / Table 3.1).

Trains one (model, masking-ratio) configuration and appends the final masked-patch
MSE to results/reconstruction.csv, which plot_results.py turns into Table 3.1.

Single run:
  python run_pretrain.py --cache ./cache --mask_mode anatomical --mask_ratio 0.75 --epochs 20

Full RQ1 sweep (4 strategies x 3 ratios = 12 runs):
  python run_pretrain.py --cache ./cache --sweep --epochs 20

Smoke test with no data at all (validates shapes and the loss in ~1 minute on CPU):
  python run_pretrain.py --synthetic --epochs 1 --steps_per_epoch 5 --batch_size 2 --dim 192 --depth 2
"""

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from models import build_model


# --------------------------------------------------------------------------------------
def get_device(pref="auto"):
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SyntheticDataset(Dataset):
    """Random brain-like phantoms, for validating the pipeline before BraTS is staged."""

    def __init__(self, n=64, img_size=224, patch=16):
        self.n, self.img_size, self.patch = n, img_size, patch

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        from anatomy import anatomical_features

        rng = np.random.RandomState(i)
        H = self.img_size
        yy, xx = np.mgrid[0:H, 0:H]
        cy, cx = H / 2 + rng.randn() * 6, H / 2 + rng.randn() * 6
        brain = (((yy - cy) / (H * 0.36)) ** 2 + ((xx - cx) / (H * 0.30)) ** 2) < 1.0
        img = np.stack([brain * rng.uniform(0.5, 0.9) for _ in range(4)]).astype(np.float32)
        ty, tx = rng.randint(70, 150, size=2)
        img[:, ty : ty + 22, tx : tx + 22] += 0.3  # bright "lesion"
        img = np.clip(img + rng.normal(0, 0.02, img.shape), 0, 1).astype(np.float32)
        A, _ = anatomical_features(img, kind="hybrid", patch=self.patch)
        return {"img": torch.from_numpy(img), "anat": torch.from_numpy(A)}


def build_loaders(args):
    if args.synthetic:
        tr = SyntheticDataset(args.batch_size * max(args.steps_per_epoch, 1) * 2,
                              args.img_size, args.patch)
        va = SyntheticDataset(args.batch_size * 2, args.img_size, args.patch)
    else:
        from data import BraTSSliceDataset, load_fold

        train_ids, val_ids = load_fold(args.cache, args.fold)
        anat_kind = None if args.mask_mode in {"random", "adaptive"} else args.anat_kind
        common = dict(cache_root=args.cache, stage="pretrain", anat_kind=anat_kind,
                      patch=args.patch, img_size=args.img_size,
                      max_slices_per_patient=args.max_slices_per_patient)
        tr = BraTSSliceDataset(split_patients=train_ids, augment=True, **common)
        va = BraTSSliceDataset(split_patients=val_ids, augment=False, **common)
        print(f"Fold {args.fold}: {len(train_ids)} train / {len(val_ids)} val patients | "
              f"{len(tr)} train / {len(va)} val slices")

    kw = dict(batch_size=args.batch_size, num_workers=args.workers,
              pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    return DataLoader(tr, shuffle=True, **kw), DataLoader(va, shuffle=False, **kw)


def cosine_lr(step, total, base_lr, warmup):
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * prog))


# --------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, mask_ratio, device, max_batches=0):
    """Mean masked-patch MSE over the validation split -> the Table 3.1 number."""
    model.eval()
    tot, n = 0.0, 0
    for i, b in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        out = model(b["img"].to(device), b["anat"].to(device), mask_ratio)
        tot += out["loss_recon"].item()
        n += 1
    model.train()
    return tot / max(n, 1)


def train_one(args):
    device = get_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_loader, val_loader = build_loaders(args)
    model = build_model(args).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)

    tag = f"{args.mask_mode}_{args.fusion}_{args.anat_kind}_p{args.mask_ratio}"
    print(f"\n=== {tag} | device={device} | {n_par/1e6:.1f}M params ===")

    # Sec. 3.6.1: AdamW (decoupled weight decay), cosine schedule, warmup.
    sampler_params = [p for n, p in model.named_parameters() if n.startswith("sampler.")]
    other_params = [p for n, p in model.named_parameters() if not n.startswith("sampler.")]
    groups = [{"params": other_params, "lr": args.lr}]
    if sampler_params:
        groups.append({"params": sampler_params, "lr": args.sampler_lr})
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.weight_decay)

    steps_per_epoch = args.steps_per_epoch or len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup = int(total_steps * 0.1)
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    history, step, t0 = [], 0, time.time()
    for epoch in range(1, args.epochs + 1):
        run_r = run_s = seen = 0.0
        for i, batch in enumerate(train_loader):
            if args.steps_per_epoch and i >= args.steps_per_epoch:
                break
            lr = cosine_lr(step, total_steps, args.lr, warmup)
            for gi, g in enumerate(opt.param_groups):
                g["lr"] = lr * (args.sampler_lr / args.lr if gi == 1 else 1.0)

            img = batch["img"].to(device, non_blocking=True)
            anat = batch["anat"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=amp):
                out = model(img, anat, args.mask_ratio)
                # Sec. 3.5.3 joint objective. loss_sampler already uses a detached
                # reward, so the two terms train their own parameters.
                loss = out["loss_recon"] + args.lambda_sampler * out["loss_sampler"]

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(opt)
            scaler.update()

            run_r += out["loss_recon"].item()
            run_s += float(out["loss_sampler"])
            seen += 1
            step += 1

        val = evaluate(model, val_loader, args.mask_ratio, device, args.eval_batches)
        history.append(dict(epoch=epoch, train_mse=run_r / max(seen, 1),
                            sampler_loss=run_s / max(seen, 1), val_mse=val))
        print(f"  epoch {epoch:3d}/{args.epochs}  train_mse {run_r/max(seen,1):.5f}  "
              f"val_mse {val:.5f}  L_S {run_s/max(seen,1):+.4f}  "
              f"[{time.time()-t0:.0f}s]")

    final_val = history[-1]["val_mse"]
    best_val = min(h["val_mse"] for h in history)

    out_dir = Path(args.out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    if not args.no_save:
        torch.save({"model": model.state_dict(), "args": vars(args), "history": history},
                   out_dir / "checkpoints" / f"{tag}.pt")
    with open(out_dir / "logs" / f"{tag}.json", "w") as f:
        json.dump({"args": vars(args), "history": history}, f, indent=2)

    row = dict(
        model=args.mask_mode, anat_kind=args.anat_kind, fusion=args.fusion,
        mask_ratio=args.mask_ratio, fold=args.fold, epochs=args.epochs,
        final_val_mse=round(final_val, 6), best_val_mse=round(best_val, 6),
        params_M=round(n_par / 1e6, 2), minutes=round((time.time() - t0) / 60, 2),
    )
    csv_path = out_dir / "reconstruction.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            w.writeheader()
        w.writerow(row)

    print(f"  -> final val MSE {final_val:.5f} (best {best_val:.5f}) appended to {csv_path}")
    return row


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--cache", default="./cache")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--max_slices_per_patient", type=int, default=0)
    # model
    ap.add_argument("--mask_mode", default="anatomical",
                    choices=["random", "adaptive", "anatomical", "hard_anat"])
    ap.add_argument("--anat_kind", default="hybrid", choices=["sobel", "canny", "hybrid"])
    ap.add_argument("--fusion", default="learnable", choices=["direct", "learnable"])
    ap.add_argument("--mask_ratio", type=float, default=0.75)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--dec_dim", type=int, default=384)
    ap.add_argument("--dec_depth", type=int, default=4)
    ap.add_argument("--norm_pix_loss", action="store_true")
    # optimization (Sec. 3.6.1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--sampler_lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--lambda_sampler", type=float, default=1.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--steps_per_epoch", type=int, default=0)
    ap.add_argument("--eval_batches", type=int, default=0)
    # runtime
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--no_save", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="./results")
    # sweep
    ap.add_argument("--sweep", action="store_true", help="RQ1: all strategies x all ratios")
    ap.add_argument("--sweep_modes", default="random,adaptive,hard_anat,anatomical")
    ap.add_argument("--sweep_ratios", default="0.50,0.75,0.80")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not args.sweep:
        train_one(args)
        return

    modes = args.sweep_modes.split(",")
    ratios = [float(r) for r in args.sweep_ratios.split(",")]
    print(f"RQ1 sweep: {len(modes)} strategies x {len(ratios)} ratios = "
          f"{len(modes)*len(ratios)} runs")
    rows = []
    for mode in modes:
        for r in ratios:
            args.mask_mode, args.mask_ratio = mode, r
            rows.append(train_one(args))

    print("\n=== RQ1 summary (final val MSE, lower is better) ===")
    print(f"{'model':<12}" + "".join(f"{int(r*100)}%".rjust(10) for r in ratios))
    for mode in modes:
        cells = [next((f"{x['final_val_mse']:.5f}" for x in rows
                       if x["model"] == mode and x["mask_ratio"] == r), "-") for r in ratios]
        print(f"{mode:<12}" + "".join(c.rjust(10) for c in cells))


if __name__ == "__main__":
    main()