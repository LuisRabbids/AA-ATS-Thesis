"""
data.py
Datasets for pre-training and fine-tuning (manuscript Sec. 3.2-3.3, 3.6.3).

Reads the per-case cache written by prepare_data.py. On first use of a split, the
per-case .npz files are packed into uncompressed memory-mapped arrays under
<cache>/packed/, so every later epoch reads single slices directly from disk instead of
decompressing a whole case to fetch one slice.

Each sample is a dict:
  img    float32 [4, 224, 224] in [0, 1]    channels [T1, T1Gd, T2, FLAIR]
  anat   float32 [196, 4]                    A_i (zeros for random / adaptive)
  tissue bool    [196]                       patch contains brain tissue (set B, Sec. 3.5.1)
  seg    int64   [224, 224]                  fine-tuning only; 0 bg, 1 NCR, 2 ED, 3 ET

Augmentation (Sec. 3.6.3), training only:
  spatial   random resized crop (scale 0.8-1.0) + horizontal flip, applied identically to
            the image, the brain mask and the labels
  intensity brightness/contrast jitter and Gaussian noise, applied INSIDE the brain only,
            so the background stays exactly 0 (otherwise Sobel would fire on noise and
            the tissue index would be meaningless)
  anatomy   the Sobel map is computed after augmentation, from the augmented image
"""

import hashlib
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from anatomy import anatomical_features

PATCH = 16
GRID = 14
N_PATCH = GRID * GRID


# --------------------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------------------
def load_split(cache_root, name="dev"):
    """name: 'dev' (Stage 1) or 'fold_1'..'fold_5' (Stage 2). Returns (train_ids, val_ids)."""
    with open(Path(cache_root, "splits.json")) as f:
        splits = json.load(f)
    if name not in splits:
        raise KeyError(f"'{name}' not in splits.json (available: {list(splits)})")
    return splits[name]["train"], splits[name]["val"]


def load_fold(cache_root, fold=1):
    """Backwards-compatible alias used by diagnose_mse.py."""
    return load_split(cache_root, fold if isinstance(fold, str) else f"fold_{fold}")


# --------------------------------------------------------------------------------------
# Packing .npz -> memory-mapped .npy
# --------------------------------------------------------------------------------------
def _packed_root(cache_root):
    """Where packed arrays go: $AATS_PACKED_DIR if set, else <cache>/packed.
    Set AATS_PACKED_DIR when the cache is read-only (e.g. a Kaggle input)."""
    return Path(os.environ.get("AATS_PACKED_DIR") or Path(cache_root, "packed"))


def _pack(cache_root, stage, case_ids, stride):
    key = hashlib.md5(("|".join(sorted(case_ids)) + f"|{stage}|{stride}").encode()).hexdigest()[:10]
    out = _packed_root(cache_root) / f"{stage}_s{stride}_{key}"
    if (out / "done").exists():
        return out

    manifest = pd.read_csv(Path(cache_root, "manifest.csv")).set_index("case_id")
    col = "n_pretrain_slices" if stage == "pretrain" else "n_finetune_slices"
    ids = [c for c in sorted(case_ids) if c in manifest.index]
    missing = sorted(set(case_ids) - set(ids))
    if missing:
        raise RuntimeError(f"{len(missing)} cases not in manifest.csv, e.g. {missing[:3]}")
    counts = [len(range(0, int(manifest.loc[c, col]), stride)) for c in ids]
    total = sum(counts)

    tmp = Path(str(out) + f".tmp{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    img = np.lib.format.open_memmap(tmp / "img.npy", "w+", np.uint8, (total, 4, 224, 224))
    tis = np.lib.format.open_memmap(tmp / "tissue.npy", "w+", bool, (total, N_PATCH))
    seg = (np.lib.format.open_memmap(tmp / "seg.npy", "w+", np.uint8, (total, 224, 224))
           if stage == "finetune" else None)

    print(f"  packing {stage} ({len(ids)} cases, stride {stride}) -> {total:,} slices ...",
          flush=True)
    pos = 0
    for cid, n in zip(ids, counts):
        with np.load(Path(cache_root, stage, f"{cid}.npz")) as z:
            img[pos:pos + n] = z["img"][::stride]
            tis[pos:pos + n] = z["tissue"][::stride]
            if seg is not None:
                seg[pos:pos + n] = z["seg"][::stride]
        pos += n
    img.flush(); tis.flush()
    if seg is not None:
        seg.flush()
    del img, tis, seg
    with open(tmp / "cases.json", "w") as f:
        json.dump({"cases": ids, "counts": counts}, f)
    (tmp / "done").touch()

    try:
        tmp.rename(out)
    except OSError:                       # another process finished packing first
        shutil.rmtree(tmp, ignore_errors=True)
    return out


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------
class BraTSSliceDataset(Dataset):
    """
    stage          'pretrain' | 'finetune'
    split_patients list of case IDs (from load_split)
    anat_kind      'sobel' (default) or None to skip anatomy (random / adaptive baselines)
    slice_stride   keep every k-th slice; fine-tuning uses 2 (Sec. 3.3.2)
    """

    def __init__(self, cache_root, stage="pretrain", split_patients=None, anat_kind="sobel",
                 anat_reduce="flair", patch=16, img_size=224, augment=True,
                 slice_stride=1, max_slices_per_patient=0):
        assert stage in {"pretrain", "finetune"}
        assert patch == PATCH and img_size == 224, "cache is fixed at 224x224, 16x16 patches"
        if split_patients is None:
            split_patients = [p.stem for p in Path(cache_root, stage).glob("*.npz")]
        self.stage, self.augment = stage, augment
        self.anat_kind, self.anat_reduce = anat_kind, anat_reduce
        self.dir = _pack(cache_root, stage, split_patients, slice_stride)
        self.n = int(np.load(self.dir / "img.npy", mmap_mode="r").shape[0])
        self._arrays = None               # opened lazily inside each worker

    def __len__(self):
        return self.n

    def _open(self):
        if self._arrays is None:
            a = {k: np.load(self.dir / f"{k}.npy", mmap_mode="r") for k in ("img", "tissue")}
            if self.stage == "finetune":
                a["seg"] = np.load(self.dir / "seg.npy", mmap_mode="r")
            self._arrays = a
        return self._arrays

    # ---------------------------------------------------------------------------------
    @staticmethod
    def _augment(img, brain, seg):
        S = img.shape[-1]
        s = np.random.uniform(0.8, 1.0)
        h = int(round(S * np.sqrt(s)))
        if h < S:
            t, l = np.random.randint(0, S - h + 1, size=2)
            img = np.stack([cv2.resize(c[t:t + h, l:l + h], (S, S),
                                       interpolation=cv2.INTER_LINEAR) for c in img])
            brain = cv2.resize(brain[t:t + h, l:l + h].astype(np.uint8), (S, S),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            if seg is not None:
                seg = cv2.resize(seg[t:t + h, l:l + h], (S, S), interpolation=cv2.INTER_NEAREST)

        if np.random.rand() < 0.5:
            img, brain = img[:, :, ::-1], brain[:, ::-1]
            if seg is not None:
                seg = seg[:, ::-1]

        if np.random.rand() < 0.5:
            img = np.where(brain, img * np.random.uniform(0.9, 1.1)
                           + np.random.uniform(-0.05, 0.05), 0.0)
        if np.random.rand() < 0.3:
            img = np.where(brain, img + np.random.normal(0, 0.02, img.shape), 0.0)

        img = np.clip(np.where(brain, img, 0.0), 0.0, 1.0).astype(np.float32)
        return np.ascontiguousarray(img), np.ascontiguousarray(brain), seg

    def __getitem__(self, i):
        a = self._open()
        img = a["img"][i].astype(np.float32) / 255.0
        seg = np.array(a["seg"][i]) if self.stage == "finetune" else None

        if self.augment:
            brain = (img > 0).any(axis=0)
            img, brain, seg = self._augment(img, brain, seg)
            tissue = brain.reshape(GRID, PATCH, GRID, PATCH).any(axis=(1, 3)).reshape(-1)
        else:
            tissue = np.array(a["tissue"][i])   # exact index computed from raw volumes

        out = {"img": torch.from_numpy(img), "tissue": torch.from_numpy(tissue)}
        if self.anat_kind is not None:
            A, _ = anatomical_features(img, kind=self.anat_kind, reduce=self.anat_reduce)
            out["anat"] = torch.from_numpy(A)
        else:
            out["anat"] = torch.zeros(N_PATCH, 4)
        if seg is not None:
            out["seg"] = torch.from_numpy(np.ascontiguousarray(seg).astype(np.int64))
        return out
