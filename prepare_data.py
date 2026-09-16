"""
prepare_data.py
Stage 1 of the pipeline: convert raw BraTS 2021 NIfTI volumes into cached 2D axial
slices, following Section 3.3 of the manuscript.

Steps implemented (manuscript Sec. 3.3):
  3.3.1 Reorientation + cleaning   -> nibabel as_closest_canonical (RAS+); BraTS is
                                      already skull-stripped and N4 bias-corrected.
  3.3.2 Slice extraction           -> axial slices; every Nth slice for pretraining,
                                      all brain-containing slices for fine-tuning.
  3.3.3 Normalization              -> per-slice min-max scaling to [0, 1].
  3.3.4 Resizing                   -> 224 x 224 (patching happens in the model).

Expected input layout (standard BraTS 2021 release):
  <braTS_root>/
    BraTS2021_00000/
      BraTS2021_00000_flair.nii.gz
      BraTS2021_00000_t1.nii.gz
      BraTS2021_00000_t1ce.nii.gz
      BraTS2021_00000_t2.nii.gz
      BraTS2021_00000_seg.nii.gz
    BraTS2021_00002/
      ...

Output layout:
  <out_root>/
    pretrain/<PatientID>.npz   ->  img: uint8 [S, 4, 224, 224]
    finetune/<PatientID>.npz   ->  img: uint8 [S, 4, 224, 224], seg: uint8 [S, 224, 224]
    manifest.csv               ->  patient_id, n_pretrain_slices, n_finetune_slices
    folds.json                 ->  patient-level 5-fold split (Sec. 3.2.1)

Usage:
  python prepare_data.py --brats_root /data/BraTS2021 --out_root ./cache
  python prepare_data.py --brats_root /data/BraTS2021 --out_root ./cache --limit 50
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

MODALITIES = ["t1", "t1ce", "t2", "flair"]  # channel order = C0..C3


# --------------------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------------------
def find_modality_file(case_dir: Path, pid: str, mod: str):
    """BraTS naming varies slightly between mirrors; try the common patterns."""
    for pattern in (f"{pid}_{mod}.nii.gz", f"{pid}_{mod}.nii", f"*_{mod}.nii.gz"):
        hits = sorted(case_dir.glob(pattern))
        if hits:
            return hits[0]
    return None


def load_canonical(path: Path) -> np.ndarray:
    """Load a NIfTI volume and reorient to closest canonical (RAS+) axes."""
    import nibabel as nib

    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)
    return np.asanyarray(img.dataobj).astype(np.float32)


# --------------------------------------------------------------------------------------
# Preprocessing (Sec. 3.3.3 / 3.3.4)
# --------------------------------------------------------------------------------------
def minmax_norm(slice_2d: np.ndarray, clip_percentile: float = 99.5) -> np.ndarray:
    """
    Per-slice min-max normalization into [0, 1] (manuscript Sec. 3.3.3).

    A high-percentile clip is applied first so that a handful of hyper-intense
    voxels do not compress the whole dynamic range. Set clip_percentile=100 to
    disable and get textbook min-max.
    """
    brain = slice_2d[slice_2d > 0]
    if brain.size == 0:
        return np.zeros_like(slice_2d, dtype=np.float32)
    hi = np.percentile(brain, clip_percentile) if clip_percentile < 100 else brain.max()
    lo = brain.min()
    if hi <= lo:
        return np.zeros_like(slice_2d, dtype=np.float32)
    out = (slice_2d - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def resize_img(slice_2d: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(slice_2d, (size, size), interpolation=cv2.INTER_LINEAR)


def resize_lbl(slice_2d: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(slice_2d, (size, size), interpolation=cv2.INTER_NEAREST)


def brain_fraction(slice_2d: np.ndarray) -> float:
    return float((slice_2d > 0).mean())


# --------------------------------------------------------------------------------------
# Per-case processing
# --------------------------------------------------------------------------------------
def process_case(case_dir: Path, args):
    pid = case_dir.name

    vols = {}
    for mod in MODALITIES:
        f = find_modality_file(case_dir, pid, mod)
        if f is None:
            print(f"  [skip] {pid}: missing modality '{mod}'")
            return None
        vols[mod] = load_canonical(f)

    seg_path = find_modality_file(case_dir, pid, "seg")
    seg_vol = load_canonical(seg_path) if seg_path is not None else None

    shapes = {v.shape for v in vols.values()}
    if len(shapes) != 1:
        print(f"  [skip] {pid}: inconsistent modality shapes {shapes}")
        return None

    n_axial = next(iter(vols.values())).shape[2]

    # Which slices actually contain brain? (avoids wasting capacity on empty air)
    ref = vols["flair"]
    valid = [z for z in range(n_axial) if brain_fraction(ref[:, :, z]) >= args.min_brain_frac]
    if not valid:
        print(f"  [skip] {pid}: no slices above min_brain_frac")
        return None

    # Sec. 3.3.2 differential slicing: stride for pretraining, all slices for fine-tuning
    pre_idx = valid[:: args.pretrain_stride]
    fine_idx = valid

    def build(indices, want_seg):
        imgs = np.zeros((len(indices), len(MODALITIES), args.size, args.size), np.uint8)
        segs = np.zeros((len(indices), args.size, args.size), np.uint8) if want_seg else None
        for i, z in enumerate(indices):
            for c, mod in enumerate(MODALITIES):
                s = minmax_norm(vols[mod][:, :, z], args.clip_percentile)
                imgs[i, c] = (resize_img(s, args.size) * 255.0).round().astype(np.uint8)
            if want_seg and seg_vol is not None:
                lab = seg_vol[:, :, z].astype(np.uint8)
                lab[lab == 4] = 3  # BraTS labels {0,1,2,4} -> {0,1,2,3}
                segs[i] = resize_lbl(lab, args.size)
        return imgs, segs

    pre_imgs, _ = build(pre_idx, want_seg=False)
    np.savez_compressed(Path(args.out_root, "pretrain", f"{pid}.npz"), img=pre_imgs)

    n_fine = 0
    if seg_vol is not None and not args.pretrain_only:
        fine_imgs, fine_segs = build(fine_idx, want_seg=True)
        np.savez_compressed(
            Path(args.out_root, "finetune", f"{pid}.npz"), img=fine_imgs, seg=fine_segs
        )
        n_fine = len(fine_idx)

    tumor_vox = int((seg_vol > 0).sum()) if seg_vol is not None else 0
    return dict(
        patient_id=pid,
        n_pretrain_slices=len(pre_idx),
        n_finetune_slices=n_fine,
        tumor_voxels=tumor_vox,
    )


# --------------------------------------------------------------------------------------
# Patient-level 5-fold split (Sec. 3.2.1)
# --------------------------------------------------------------------------------------
def make_folds(records, k=5, seed=42):
    """
    Patient-level K-fold split, stratified by total tumour burden so that each fold
    stays representative of the overall tumour-size distribution (manuscript Sec. 3.2.1).
    """
    rng = np.random.RandomState(seed)
    recs = [r for r in records if r["n_finetune_slices"] > 0] or list(records)
    order = np.argsort([r["tumor_voxels"] for r in recs])

    strata, folds = [], {i: [] for i in range(k)}
    for start in range(0, len(order), k):  # bins of k consecutive patients by burden
        strata.append([recs[j]["patient_id"] for j in order[start : start + k]])
    for stratum in strata:
        stratum = list(stratum)
        rng.shuffle(stratum)
        for i, pid in enumerate(stratum):
            folds[i % k].append(pid)

    all_pids = [r["patient_id"] for r in records]
    assigned = {p for v in folds.values() for p in v}
    for i, pid in enumerate(p for p in all_pids if p not in assigned):
        folds[i % k].append(pid)

    return {
        f"fold_{i+1}": {
            "val": sorted(folds[i]),
            "train": sorted(p for j in range(k) if j != i for p in folds[j]),
        }
        for i in range(k)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brats_root", required=True, help="Directory of BraTS2021_XXXXX folders")
    ap.add_argument("--out_root", default="./cache")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--pretrain_stride", type=int, default=5, help="Sec. 3.3.2: every Nth slice")
    ap.add_argument("--min_brain_frac", type=float, default=0.02)
    ap.add_argument("--clip_percentile", type=float, default=99.5)
    ap.add_argument("--limit", type=int, default=0, help="Process only the first N patients")
    ap.add_argument("--pretrain_only", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(Path(args.out_root, "pretrain"), exist_ok=True)
    os.makedirs(Path(args.out_root, "finetune"), exist_ok=True)

    cases = sorted(d for d in Path(args.brats_root).iterdir() if d.is_dir())
    if args.limit:
        cases = cases[: args.limit]
    print(f"Found {len(cases)} candidate cases under {args.brats_root}")

    records = []
    for i, case in enumerate(cases, 1):
        try:
            rec = process_case(case, args)
        except Exception as e:  # keep going; report at the end
            print(f"  [error] {case.name}: {type(e).__name__}: {e}")
            rec = None
        if rec:
            records.append(rec)
        if i % 25 == 0 or i == len(cases):
            print(f"  processed {i}/{len(cases)}  (kept {len(records)})")

    if not records:
        raise SystemExit("No cases processed successfully. Check --brats_root layout.")

    import pandas as pd

    pd.DataFrame(records).to_csv(Path(args.out_root, "manifest.csv"), index=False)
    with open(Path(args.out_root, "folds.json"), "w") as f:
        json.dump(make_folds(records, args.folds, args.seed), f, indent=2)

    tot_pre = sum(r["n_pretrain_slices"] for r in records)
    tot_fine = sum(r["n_finetune_slices"] for r in records)
    print(f"\nDone. {len(records)} patients | {tot_pre} pretrain slices | {tot_fine} finetune slices")
    print(f"Wrote manifest.csv and folds.json to {args.out_root}")


if __name__ == "__main__":
    main()
