"""
data.py
Dataset and augmentation for self-supervised pretraining (manuscript Sec. 3.3, 3.6.3).

Reads the slice cache written by prepare_data.py and returns, per sample:
    img  : float32 [4, 224, 224] in [0, 1]
    anat : float32 [N, 4]        patch-level anatomical features A_i (Eq. 3.9)

The anatomical map is recomputed *after* augmentation so that structural cues stay
spatially aligned with the augmented image.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from anatomy import anatomical_features


class BraTSSliceDataset(Dataset):
    """
    split_patients : list of patient IDs (from folds.json), or None for every cached file
    anat_kind      : 'sobel' | 'canny' | 'hybrid' | None  (None skips anatomy entirely,
                     used by the MAE and AdaMAE baselines to save CPU time)
    """

    def __init__(
        self,
        cache_root,
        stage="pretrain",
        split_patients=None,
        anat_kind="hybrid",
        anat_reduce="mean",
        patch=16,
        img_size=224,
        augment=True,
        max_slices_per_patient=0,
    ):
        self.dir = Path(cache_root, stage)
        self.anat_kind = anat_kind
        self.anat_reduce = anat_reduce
        self.patch = patch
        self.img_size = img_size
        self.augment = augment
        self.stage = stage

        files = sorted(self.dir.glob("*.npz"))
        if split_patients is not None:
            keep = set(split_patients)
            files = [f for f in files if f.stem in keep]
        if not files:
            raise RuntimeError(f"No .npz files found in {self.dir} for the requested split")

        # Build a flat (file, slice) index without loading pixel data into RAM.
        self.index, self.files = [], files
        for fi, f in enumerate(files):
            with np.load(f) as z:
                n = z["img"].shape[0]
            if max_slices_per_patient:
                n = min(n, max_slices_per_patient)
            self.index.extend((fi, si) for si in range(n))

        self._cache = {}  # lazy per-worker file cache

    def __len__(self):
        return len(self.index)

    def _get_volume(self, fi):
        if fi not in self._cache:
            with np.load(self.files[fi]) as z:
                self._cache[fi] = z["img"]
        return self._cache[fi]

    # ---------------------------------------------------------------------------------
    # Augmentation (Sec. 3.6.3): light spatial + intensity transforms that preserve
    # anatomical integrity. No vertical flips or large rotations.
    # ---------------------------------------------------------------------------------
    def _augment(self, img):
        # Random resized crop, constrained scale in [0.8, 1.0]
        s = np.random.uniform(0.8, 1.0)
        h = w = int(round(self.img_size * np.sqrt(s)))
        if h < self.img_size:
            top = np.random.randint(0, self.img_size - h + 1)
            left = np.random.randint(0, self.img_size - w + 1)
            img = img[:, top : top + h, left : left + w]
            img = np.stack(
                [cv2.resize(c, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                 for c in img]
            )

        # Random horizontal flip, p = 0.5 (left-right brain symmetry makes this safe)
        if np.random.rand() < 0.5:
            img = img[:, :, ::-1].copy()

        # Intensity jitter: brightness / contrast, simulating scanner variability
        if np.random.rand() < 0.5:
            img = img * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.05, 0.05)

        # Low-variance Gaussian noise
        if np.random.rand() < 0.3:
            img = img + np.random.normal(0, 0.02, img.shape).astype(np.float32)

        return np.clip(img, 0.0, 1.0).astype(np.float32)

    def __getitem__(self, i):
        fi, si = self.index[i]
        img = self._get_volume(fi)[si].astype(np.float32) / 255.0  # [4, H, W]

        if self.augment:
            img = self._augment(img)

        out = {"img": torch.from_numpy(np.ascontiguousarray(img))}

        if self.anat_kind is not None:
            A, _ = anatomical_features(
                img, kind=self.anat_kind, patch=self.patch, reduce=self.anat_reduce
            )
            out["anat"] = torch.from_numpy(A)  # [N, 4]
        else:
            n = (self.img_size // self.patch) ** 2
            out["anat"] = torch.zeros(n, 4, dtype=torch.float32)

        return out


def load_fold(cache_root, fold=1):
    """Return (train_patient_ids, val_patient_ids) for the requested fold."""
    with open(Path(cache_root, "folds.json")) as f:
        folds = json.load(f)
    key = f"fold_{fold}"
    if key not in folds:
        raise KeyError(f"{key} not in folds.json (available: {list(folds)})")
    return folds[key]["train"], folds[key]["val"]
