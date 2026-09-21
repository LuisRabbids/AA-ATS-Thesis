"""
prepare_data.py
Convert raw BraTS 2021 volumes into cached 2D axial slices (manuscript Sec. 3.2-3.3).

Pipeline per case
-----------------
  3.3.1  Reorientation    nibabel as_closest_canonical (RAS+). BraTS 2021 is distributed
                          skull-stripped and co-registered, so no further cleaning is done.
  3.3.2  Slice selection  a slice is kept if ANY of the four modalities has a non-zero voxel.
                          Pre-training keeps every `--pretrain_stride`-th kept slice;
                          fine-tuning keeps all of them (sub-sampling happens at load time).
  3.3.3  Normalization    per volume, per modality: divide by the 99.5th percentile of brain
                          voxels, clip to [0, 1]. Background is 0 by construction.
  3.3.4  Cropping         center crop 240 -> 224 (indices 8..231). No resampling.
                          Verified on all 1,251 cases: no brain voxel falls outside the crop.
  Tissue index (set B)    per slice, a boolean [196] marking 16x16 patches that contain any
                          non-zero voxel in any modality (used by the loss, Sec. 3.5.1).

Splits (Sec. 3.2.2)
-------------------
  Reads metadata/partitions.json (200 development / 1,051 experimental cases) and
  metadata/tumor_volumes.json, and writes splits.json:
    dev      : one 80/20 train/val split of the development partition   (Stage 1)
    fold_1-5 : 5-fold CV over the experimental partition                 (Stage 2)
  Both are stratified by whole-tumour volume.

Input: either extracted case folders (--brats_root) or the Kaggle tar directly (--brats_tar).

Output
------
  <out>/pretrain/<case>.npz   img uint8 [S,4,224,224], tissue bool [S,196]
  <out>/finetune/<case>.npz   img uint8 [S,4,224,224], seg uint8 [S,224,224], tissue bool [S,196]
  <out>/manifest.csv          per-case slice counts and tumour volume
  <out>/splits.json

Labels: BraTS {0,1,2,4} are remapped to {0,1,2,3} so classes are contiguous for
cross-entropy. 1 = NCR/NET, 2 = ED, 3 = ET (originally 4).

Usage
-----
  python prepare_data.py --brats_tar  /path/BraTS2021_Training_Data.tar --out ./cache
  python prepare_data.py --brats_root /path/extracted_cases            --out ./cache
  add --limit 5 for a quick test run
"""

import argparse
import gzip
import io
import json
import os
import tarfile
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

MODALITIES = ["t1", "t1ce", "t2", "flair"]   # channel order C0..C3 (FLAIR = channel 3)
FULL = 240
SIZE = 224
LO = (FULL - SIZE) // 2                        # 8
HI = LO + SIZE                                 # 232 (exclusive)
PATCH = 16
GRID = SIZE // PATCH                           # 14 -> 196 patches


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def _canonical(img):
    import nibabel as nib
    return np.asanyarray(nib.as_closest_canonical(img).dataobj)


def _nifti_from_bytes(raw):
    import nibabel as nib
    fh = nib.FileHolder(fileobj=io.BytesIO(gzip.decompress(raw)))
    return nib.Nifti1Image.from_file_map({"header": fh, "image": fh})


def iter_cases_from_tar(tar_path, wanted):
    """Stream the tar once, yielding (case_id, {modality: array}) as each case completes."""
    buf, remaining = {}, set(wanted)
    with tarfile.open(tar_path, "r") as tf:
        for m in tf:
            if not remaining:
                break                      # every requested case already yielded
            if not m.isfile() or not m.name.endswith(".nii.gz"):
                continue
            fname = os.path.basename(m.name)
            parts = fname.replace(".nii.gz", "").split("_")
            cid, mod = f"{parts[0]}_{parts[1]}", parts[2]
            if cid not in wanted:
                continue
            arr = _canonical(_nifti_from_bytes(tf.extractfile(m).read()))
            buf.setdefault(cid, {})[mod] = arr
            if len(buf[cid]) == 5:
                remaining.discard(cid)
                yield cid, buf.pop(cid)
    for cid, mods in buf.items():
        print(f"  [warn] {cid}: incomplete in tar ({sorted(mods)}), skipped")


def iter_cases_from_folders(root, wanted):
    import nibabel as nib
    for cid in sorted(wanted):
        d = Path(root, cid)
        mods = {}
        for mod in MODALITIES + ["seg"]:
            f = d / f"{cid}_{mod}.nii.gz"
            if not f.exists():
                break
            mods[mod] = _canonical(nib.load(str(f)))
        if len(mods) == 5:
            yield cid, mods
        else:
            print(f"  [warn] {cid}: missing files, skipped")


# --------------------------------------------------------------------------------------
# Per-case processing
# --------------------------------------------------------------------------------------
def normalize_volume(vol, pct=99.5):
    """Sec. 3.3.3: per-volume scaling to [0, 1]. Background (0) stays 0."""
    vol = vol.astype(np.float32)
    brain = vol[vol > 0]
    if brain.size == 0:
        return np.zeros_like(vol)
    hi = np.percentile(brain, pct)
    return np.clip(vol / max(hi, 1e-6), 0.0, 1.0)


def tissue_index(nonzero_2d):
    """[224,224] bool -> [196] bool: patch contains any non-zero voxel."""
    return nonzero_2d.reshape(GRID, PATCH, GRID, PATCH).any(axis=(1, 3)).reshape(-1)


def process_case(job):
    cid, mods, out, stride = job

    shapes = {mods[m].shape for m in MODALITIES + ["seg"]}
    if shapes != {(FULL, FULL, mods["flair"].shape[2])}:
        return {"case_id": cid, "error": f"unexpected shapes {shapes}"}

    raw = np.stack([mods[m] for m in MODALITIES])                # [4, 240, 240, Z]
    nonzero = (raw != 0).any(axis=0)                               # [240, 240, Z]

    # Safety: the crop must never discard brain tissue.
    lost = int(nonzero.sum() - nonzero[LO:HI, LO:HI].sum())
    if lost:
        return {"case_id": cid, "error": f"crop would discard {lost} brain voxels"}

    keep = [z for z in range(raw.shape[3]) if nonzero[:, :, z].any()]
    if not keep:
        return {"case_id": cid, "error": "no brain-containing slices"}

    norm = np.stack([normalize_volume(raw[c]) for c in range(4)])  # [4, 240, 240, Z]
    norm = norm[:, LO:HI, LO:HI, :]
    nz = nonzero[LO:HI, LO:HI, :]

    seg = mods["seg"].astype(np.uint8)[LO:HI, LO:HI, :]
    seg[seg == 4] = 3

    def pack(zs):
        img = (np.transpose(norm[..., zs], (3, 0, 1, 2)) * 255.0).round().astype(np.uint8)
        tis = np.stack([tissue_index(nz[:, :, z]) for z in zs])
        return img, tis

    pre_z = keep[::stride]
    img, tis = pack(pre_z)
    np.savez_compressed(Path(out, "pretrain", f"{cid}.npz"), img=img, tissue=tis)

    img, tis = pack(keep)
    np.savez_compressed(Path(out, "finetune", f"{cid}.npz"),
                        img=img, seg=np.transpose(seg[..., keep], (2, 0, 1)), tissue=tis)

    return {
        "case_id": cid,
        "n_brain_slices": len(keep),
        "n_pretrain_slices": len(pre_z),
        "n_finetune_slices": len(keep),
        "tumor_voxels": int((seg > 0).sum()),
        "mean_tissue_frac": float(tis.mean()),
    }


# --------------------------------------------------------------------------------------
# Splits (Sec. 3.2.2)
# --------------------------------------------------------------------------------------
def stratified_kfold(ids, volumes, k, seed):
    """Sort by tumour volume, deal cases round-robin within shuffled strata of size k."""
    rng = np.random.RandomState(seed)
    order = sorted(ids, key=lambda c: volumes[c])
    folds = [[] for _ in range(k)]
    for start in range(0, len(order), k):
        stratum = order[start:start + k]
        rng.shuffle(stratum)
        for i, cid in enumerate(stratum):
            folds[i].append(cid)
    return [sorted(f) for f in folds]


def make_splits(partitions, volumes, seed):
    dev = [c for c in partitions["development"] if c in volumes]
    exp = [c for c in partitions["experimental"] if c in volumes]

    dev_folds = stratified_kfold(dev, volumes, 5, seed)     # 1 of 5 = 20% validation
    splits = {"dev": {"val": dev_folds[0],
                      "train": sorted(c for f in dev_folds[1:] for c in f)}}

    exp_folds = stratified_kfold(exp, volumes, 5, seed)
    for i in range(5):
        splits[f"fold_{i+1}"] = {
            "val": exp_folds[i],
            "train": sorted(c for j, f in enumerate(exp_folds) if j != i for c in f),
        }
    return splits


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--brats_tar", help="Path to BraTS2021_Training_Data.tar")
    src.add_argument("--brats_root", help="Directory of extracted BraTS2021_XXXXX folders")
    ap.add_argument("--out", default="./cache")
    ap.add_argument("--metadata", default="./metadata")
    ap.add_argument("--pretrain_stride", type=int, default=5)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0, help="Process only N cases (testing)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for sub in ("pretrain", "finetune"):
        os.makedirs(Path(args.out, sub), exist_ok=True)

    with open(Path(args.metadata, "partitions.json")) as f:
        partitions = json.load(f)
    wanted = set(partitions["development"]) | set(partitions["experimental"])

    # Resume: skip cases already written.
    done = {p.stem for p in Path(args.out, "finetune").glob("*.npz")}
    todo = wanted - done
    if args.limit:
        todo = set(sorted(todo)[: args.limit])
    print(f"{len(wanted)} cases in partitions | {len(done)} already cached | {len(todo)} to process")

    source = (iter_cases_from_tar(args.brats_tar, todo) if args.brats_tar
              else iter_cases_from_folders(args.brats_root, todo))
    jobs = ((cid, mods, args.out, args.pretrain_stride) for cid, mods in source)

    manifest_path = Path(args.out, "manifest.csv")
    records, errors, t0 = [], [], time.time()
    with Pool(args.workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(process_case, jobs), 1):
            (errors if "error" in rec else records).append(rec)
            if i % 25 == 0 or i == len(todo):
                rate = i / (time.time() - t0)
                eta = (len(todo) - i) / max(rate, 1e-9) / 60
                print(f"  {i}/{len(todo)}  {rate:.2f} cases/s  ETA {eta:.0f} min")

    import pandas as pd
    new = pd.DataFrame(records)
    if manifest_path.exists() and len(new):
        new = pd.concat([pd.read_csv(manifest_path), new]).drop_duplicates("case_id", keep="last")
    elif manifest_path.exists():
        new = pd.read_csv(manifest_path)
    if len(new):
        new.sort_values("case_id").to_csv(manifest_path, index=False)

    for e in errors:
        print(f"  [error] {e['case_id']}: {e['error']}")

    with open(Path(args.metadata, "tumor_volumes.json")) as f:
        volumes = json.load(f)
    splits = make_splits(partitions, volumes, args.seed)
    with open(Path(args.out, "splits.json"), "w") as f:
        json.dump(splits, f, indent=1)

    print(f"\nDone in {(time.time()-t0)/60:.1f} min. "
          f"{len(records)} processed, {len(errors)} errors, {len(new)} total in manifest.")
    if len(new):
        print(f"  pre-training slices: {int(new.n_pretrain_slices.sum()):,}")
        print(f"  fine-tuning slices:  {int(new.n_finetune_slices.sum()):,}")
        print(f"  mean tissue fraction per slice (rho, cropped): {new.mean_tissue_frac.mean():.3f}")
    print(f"  splits.json: dev {len(splits['dev']['train'])}/{len(splits['dev']['val'])} "
          f"train/val, fold_1 {len(splits['fold_1']['train'])}/{len(splits['fold_1']['val'])}")


if __name__ == "__main__":
    main()
