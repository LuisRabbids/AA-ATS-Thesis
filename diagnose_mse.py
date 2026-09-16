"""
diagnose_mse.py
Checks whether a low masked-patch MSE reflects better representation learning or
simply an easier self-selected set of masked patches.

Motivation
----------
Each masking strategy chooses which patches it is evaluated on. In brain MRI most
of the frame is black background. A sampler that learns to keep brain tissue
visible therefore masks mostly air — and air reconstructs to near-zero error.
That produces a spectacular MSE without any improvement in learned features.

This script runs two diagnostics on every checkpoint:

  1. FIXED-MASK MSE
     Every model is re-evaluated under the *same* random mask (identical seed,
     identical patches). This is the apples-to-apples comparison: all models
     reconstruct exactly the same targets.

  2. MASK COMPOSITION
     Under each model's own masking, what fraction of the masked patches is
     background rather than brain tissue? If the proposed model masks far more
     background than the random baseline, its native MSE is inflated.

Usage
-----
  python diagnose_mse.py --cache ./cache --results ./results --mask_ratio 0.75

Interpretation
--------------
  Native MSE much lower, fixed-mask MSE comparable   -> metric artefact.
  Both lower                                          -> genuine improvement.
"""

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import BraTSSliceDataset, load_fold
from models import MaskedAutoencoder

LABELS = {
    "random": "MAE (random)",
    "adaptive": "AdaMAE (adaptive)",
    "hard_anat": "Rule-based hard anatomical",
    "anatomical": "AA-AdaMAE (ours)",
}


def get_device(pref="auto"):
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_ckpt(path, device):
    ck = torch.load(path, map_location="cpu")
    a = ck["args"]
    model = MaskedAutoencoder(
        img_size=a["img_size"], patch=a["patch"], in_ch=4, dim=a["dim"],
        depth=a["depth"], heads=a["heads"], dec_dim=a["dec_dim"],
        dec_depth=a["dec_depth"], mask_mode=a["mask_mode"], fusion=a["fusion"],
        norm_pix_loss=a["norm_pix_loss"],
    )
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), a


@torch.no_grad()
def native_mse(model, loader, ratio, device, max_batches=0):
    """MSE under the model's OWN masking strategy — what Table 3.1 reports."""
    tot, n = 0.0, 0
    for i, b in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        out = model(b["img"].to(device), b["anat"].to(device), ratio)
        tot += out["loss_recon"].item()
        n += 1
    return tot / max(n, 1)


@torch.no_grad()
def fixed_mask_mse(model, loader, ratio, device, seed=1234, max_batches=0):
    """
    MSE under an identical random mask for every model.

    Temporarily forces mask_mode='random' and re-seeds per batch, so the set of
    masked patches is byte-identical across checkpoints.
    """
    orig = model.mask_mode
    model.mask_mode = "random"
    tot, n = 0.0, 0
    for i, b in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        torch.manual_seed(seed + i)          # identical mask for every model
        out = model(b["img"].to(device), b["anat"].to(device), ratio)
        tot += out["loss_recon"].item()
        n += 1
    model.mask_mode = orig
    return tot / max(n, 1)


@torch.no_grad()
def mask_composition(model, loader, ratio, device, thresh=0.02, max_batches=0):
    """
    Fraction of BACKGROUND (non-brain) content among masked vs visible patches,
    under the model's own masking.

    Returns (bg_fraction_of_masked, bg_fraction_of_visible).
    A value near 1.0 for masked means the model is being graded almost entirely
    on empty air.
    """
    bg_m, bg_v, n = 0.0, 0.0, 0
    for i, b in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        img = b["img"].to(device)
        out = model(img, b["anat"].to(device), ratio)
        mask = out["mask"]                                  # [B, N], 1 = masked
        tok = model.patchify(img)                           # [B, N, p*p*C]
        brain = (tok > thresh).float().mean(-1)             # [B, N] in [0, 1]
        bg = 1.0 - brain
        vis = 1.0 - mask
        bg_m += (bg * mask).sum().item() / max(mask.sum().item(), 1)
        bg_v += (bg * vis).sum().item() / max(vis.sum().item(), 1)
        n += 1
    return bg_m / max(n, 1), bg_v / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="./cache")
    ap.add_argument("--results", default="./results")
    ap.add_argument("--mask_ratio", type=float, default=0.75)
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max_batches", type=int, default=0,
                    help="Limit batches for a quick check (0 = all)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = get_device(args.device)
    ck_dir = Path(args.results, "checkpoints")
    ckpts = sorted(ck_dir.glob(f"*_p{args.mask_ratio}.pt"))
    if not ckpts:
        raise SystemExit(f"No checkpoints matching *_p{args.mask_ratio}.pt in {ck_dir}")

    _, val_ids = load_fold(args.cache, args.fold)
    print(f"device={device} | {len(ckpts)} checkpoint(s) at ratio {args.mask_ratio}\n")

    rows = []
    for cp in ckpts:
        model, a = load_ckpt(cp, device)
        mode = a["mask_mode"]

        # anatomy is needed only for the anatomy-aware modes
        anat_kind = None if mode in {"random", "adaptive"} else a["anat_kind"]
        ds = BraTSSliceDataset(
            cache_root=args.cache, stage="pretrain", split_patients=val_ids,
            anat_kind=anat_kind, patch=a["patch"], img_size=a["img_size"],
            augment=False,
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, drop_last=True)

        nat = native_mse(model, loader, args.mask_ratio, device, args.max_batches)
        fix = fixed_mask_mse(model, loader, args.mask_ratio, device,
                             max_batches=args.max_batches)
        bgm, bgv = mask_composition(model, loader, args.mask_ratio, device,
                                    max_batches=args.max_batches)

        rows.append(dict(model=mode, native_mse=round(nat, 6),
                         fixed_mask_mse=round(fix, 6),
                         bg_frac_masked=round(bgm, 4),
                         bg_frac_visible=round(bgv, 4)))
        print(f"  {LABELS.get(mode, mode):<28} native {nat:.5f} | "
              f"fixed {fix:.5f} | bg(masked) {bgm:.1%} | bg(visible) {bgv:.1%}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = Path(args.results, f"diagnostic_p{int(args.mask_ratio*100)}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {out}")
    print("\n" + "=" * 74)
    print("READING THE RESULT")
    print("=" * 74)
    print("  native_mse       each model graded on the patches IT chose to mask")
    print("  fixed_mask_mse   every model graded on the SAME patches  <- fair")
    print("  bg_frac_masked   how much of what it masked was empty background")
    print()
    print("  If native MSE is far lower but fixed-mask MSE is comparable, the gap")
    print("  is a metric artefact: the sampler kept the hard patches visible and")
    print("  was graded on easy background. Report fixed-mask MSE instead.")
    print()

    base = next((r for r in rows if r["model"] == "random"), None)
    ours = next((r for r in rows if r["model"] == "anatomical"), None)
    if base and ours:
        nat_gain = (base["native_mse"] - ours["native_mse"]) / base["native_mse"]
        fix_gain = (base["fixed_mask_mse"] - ours["fixed_mask_mse"]) / base["fixed_mask_mse"]
        bg_gap = ours["bg_frac_masked"] - base["bg_frac_masked"]
        print(f"  ours vs. random — native gain {nat_gain:+.1%}, "
              f"fixed-mask gain {fix_gain:+.1%}")
        print(f"  background masked: ours {ours['bg_frac_masked']:.1%} vs "
              f"random {base['bg_frac_masked']:.1%}  (gap {bg_gap:+.1%})")
        if nat_gain > 0.3 and fix_gain < 0.05:
            print("\n  VERDICT: metric artefact. Do not present the native MSE gap.")
        elif fix_gain > 0.05:
            print("\n  VERDICT: improvement survives the fair comparison. Present both.")
        else:
            print("\n  VERDICT: inconclusive — differences are small either way.")
    print()


if __name__ == "__main__":
    main()
