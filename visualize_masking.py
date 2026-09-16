"""
visualize_masking.py
Qualitative figure generator -- this produces the single most useful slide visual:
a side-by-side of where each strategy chooses to look.

Panels: input slice | Sobel | Canny | Hybrid | sampling heatmap | visible patches |
        reconstruction

  # from a trained checkpoint
  python visualize_masking.py --ckpt results/checkpoints/anatomical_learnable_hybrid_p0.75.pt \
                              --cache ./cache --out results/figures/qualitative.png

  # anatomy panels only, no checkpoint needed (works as soon as prepare_data.py has run)
  python visualize_masking.py --cache ./cache --maps_only --out results/figures/anat_maps.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from anatomy import build_anatomical_map, patch_features


def load_slice(cache, patient=None, index=None):
    files = sorted(Path(cache, "pretrain").glob("*.npz"))
    if not files:
        raise SystemExit(f"No cached slices in {Path(cache,'pretrain')}")
    f = next((x for x in files if x.stem == patient), files[0]) if patient else files[0]
    with np.load(f) as z:
        vol = z["img"]
    i = index if index is not None else vol.shape[0] // 2
    return vol[min(i, vol.shape[0] - 1)].astype(np.float32) / 255.0, f.stem


def grid_overlay(ax, img_gray, keep_mask, grid, patch, title, color="#c0392b"):
    """Grey out masked patches; keep visible ones at full intensity."""
    alpha = keep_mask.reshape(grid, grid).repeat(patch, 0).repeat(patch, 1)
    shown = img_gray * (0.25 + 0.75 * alpha)
    ax.imshow(shown, cmap="gray", vmin=0, vmax=1)
    for gy in range(grid):
        for gx in range(grid):
            if keep_mask.reshape(grid, grid)[gy, gx] > 0.5:
                ax.add_patch(plt.Rectangle((gx * patch - 0.5, gy * patch - 0.5), patch, patch,
                                           fill=False, edgecolor=color, linewidth=0.45))
    ax.set_title(title, fontsize=8, fontweight="bold")
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="./cache")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--patient", default=None)
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--mask_ratio", type=float, default=0.75)
    ap.add_argument("--maps_only", action="store_true")
    ap.add_argument("--out", default="results/figures/qualitative.png")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    img, pid = load_slice(args.cache, args.patient, args.index)
    gray = img.mean(axis=0)
    grid = img.shape[-1] // args.patch
    N = grid * grid
    Nv = int(round(N * (1 - args.mask_ratio)))

    maps = {k: build_anatomical_map(img, kind=k) for k in ("sobel", "canny", "hybrid")}

    if args.maps_only:
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), dpi=200)
        axes[0].imshow(gray, cmap="gray"); axes[0].set_title("Input (4-modality mean)",
                                                            fontsize=9, fontweight="bold")
        for ax, (k, m) in zip(axes[1:], maps.items()):
            ax.imshow(m, cmap="inferno")
            ax.set_title(f"{k.capitalize()} map", fontsize=9, fontweight="bold")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"Anatomical map generation - {pid}", fontweight="bold")
        fig.tight_layout()
        fig.savefig(args.out, bbox_inches="tight")
        print(f"wrote {args.out}")
        return

    # ---- panels that need a model -------------------------------------------------
    import torch
    from models import MaskedAutoencoder

    probs_np, recon = None, None
    A = patch_features(maps["hybrid"], args.patch)

    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu")
        a = ck["args"]
        model = MaskedAutoencoder(
            img_size=a["img_size"], patch=a["patch"], in_ch=4, dim=a["dim"],
            depth=a["depth"], heads=a["heads"], dec_dim=a["dec_dim"],
            dec_depth=a["dec_depth"], mask_mode=a["mask_mode"], fusion=a["fusion"],
            norm_pix_loss=a["norm_pix_loss"],
        )
        model.load_state_dict(ck["model"])
        model.eval()
        with torch.no_grad():
            t_img = torch.from_numpy(img)[None]
            t_anat = torch.from_numpy(A)[None]
            out = model(t_img, t_anat, args.mask_ratio)
            if out["probs"] is not None:
                probs_np = out["probs"][0].numpy()
            keep = (1 - out["mask"][0]).numpy()
            rec = model.unpatchify(out["pred"])[0].numpy()
            recon = np.clip(rec, 0, 1).mean(axis=0)
    else:
        keep = np.zeros(N)
        keep[np.argsort(-A.mean(axis=1))[:Nv]] = 1.0

    if probs_np is None:
        probs_np = A.mean(axis=1) / max(A.mean(axis=1).sum(), 1e-6)

    fig, axes = plt.subplots(1, 6, figsize=(19, 3.6), dpi=200)
    axes[0].imshow(gray, cmap="gray"); axes[0].set_title("Input slice", fontsize=8,
                                                         fontweight="bold"); axes[0].axis("off")
    axes[1].imshow(maps["sobel"], cmap="inferno"); axes[1].set_title("Sobel (Eq. 3.6)",
                                                                    fontsize=8,
                                                                    fontweight="bold")
    axes[1].axis("off")
    axes[2].imshow(maps["hybrid"], cmap="inferno"); axes[2].set_title("Hybrid (Eq. 3.8)",
                                                                     fontsize=8,
                                                                     fontweight="bold")
    axes[2].axis("off")

    hm = probs_np.reshape(grid, grid).repeat(args.patch, 0).repeat(args.patch, 1)
    axes[3].imshow(gray, cmap="gray")
    axes[3].imshow(hm, cmap="jet", alpha=0.55)
    axes[3].set_title("Sampling probability P", fontsize=8, fontweight="bold")
    axes[3].axis("off")

    grid_overlay(axes[4], gray, keep, grid, args.patch,
                 f"Visible tokens ({int((1-args.mask_ratio)*100)}%)")

    if recon is not None:
        axes[5].imshow(recon, cmap="gray", vmin=0, vmax=1)
        axes[5].set_title("Reconstruction", fontsize=8, fontweight="bold")
    else:
        axes[5].text(0.5, 0.5, "no checkpoint", ha="center", va="center", fontsize=9)
    axes[5].axis("off")

    fig.suptitle(f"Anatomically-aware adaptive masking - {pid}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
