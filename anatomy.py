"""
anatomy.py
Anatomical Feature Extraction Module (manuscript Sec. 3.4.2).

Default configuration used for all experiments (manuscript Sec. 3.4.2):
  - Map          : Sobel gradient magnitude (Eq. 3.6)
  - Source       : FLAIR channel only (index 3 in the cached [T1, T1Gd, T2, FLAIR] order)
  - Normalization: per-slice min-max to [0, 1]
  - Descriptors  : mean, max, std, density per 16x16 patch -> A_i in R^4 (Eq. 3.7)
  - Density      : fraction of patch pixels above tau, where tau is the mean normalized
                   gradient magnitude over the non-zero pixels of the slice's map

Canny and hybrid maps are retained only for the qualitative figures produced by
visualize_masking.py. They are not used in any experiment.

All maps are computed with OpenCV on the CPU inside DataLoader workers, after
augmentation, so structural cues stay aligned with the augmented image.
"""

import cv2
import numpy as np

N_ANAT_FEATURES = 4          # F in Eq. 3.7
FLAIR = 3                    # channel index of FLAIR in the cache


# --------------------------------------------------------------------------------------
# 1. Anatomical map generation
# --------------------------------------------------------------------------------------
def sobel_map(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Eq. 3.6: M_sobel = sqrt((I * S_x)^2 + (I * S_y)^2), then per-slice min-max to [0, 1].
    img: float32 [H, W]. Returns float32 [H, W].
    """
    img = img.astype(np.float32)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = np.sqrt(gx * gx + gy * gy)
    lo, hi = float(mag.min()), float(mag.max())
    if hi - lo < 1e-8:
        return np.zeros_like(mag, dtype=np.float32)
    return ((mag - lo) / (hi - lo)).astype(np.float32)


def canny_map(img: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    """Figures only. Binary edge map as float32 {0, 1}."""
    u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return (cv2.Canny(u8, low, high) > 0).astype(np.float32)


def hybrid_map(img: np.ndarray, w1: float = 0.5, w2: float = 0.5) -> np.ndarray:
    """Figures only. 0.5 * Sobel + 0.5 * Canny."""
    return (w1 * sobel_map(img) + w2 * canny_map(img)).astype(np.float32)


MAP_FUNCS = {"sobel": sobel_map, "canny": canny_map, "hybrid": hybrid_map}


def build_anatomical_map(img_chw: np.ndarray, kind: str = "sobel",
                         reduce: str = "flair") -> np.ndarray:
    """
    img_chw : float32 [C, H, W] in [0, 1], channel order [T1, T1Gd, T2, FLAIR]
    reduce  : 'flair' (default, used in all experiments) | 'mean' | 'max'
    returns : float32 [H, W]
    """
    if kind not in MAP_FUNCS:
        raise ValueError(f"unknown map '{kind}', expected one of {list(MAP_FUNCS)}")
    fn = MAP_FUNCS[kind]
    if reduce == "flair":
        return fn(img_chw[FLAIR])
    if reduce == "mean":
        return fn(img_chw.mean(axis=0))
    if reduce == "max":
        return np.max(np.stack([fn(c) for c in img_chw]), axis=0)
    raise ValueError(f"unknown reduce '{reduce}'")


# --------------------------------------------------------------------------------------
# 2. Patch-level feature extraction (Eq. 3.7)
# --------------------------------------------------------------------------------------
def patchify_map(m: np.ndarray, patch: int = 16) -> np.ndarray:
    """[H, W] -> [N, patch*patch], row-major, matching the ViT tokenizer order."""
    h, w = m.shape
    assert h % patch == 0 and w % patch == 0, f"map {m.shape} not divisible by {patch}"
    gh, gw = h // patch, w // patch
    return m.reshape(gh, patch, gw, patch).transpose(0, 2, 1, 3).reshape(gh * gw, patch * patch)


def patch_features(m: np.ndarray, patch: int = 16, tau: float = None) -> np.ndarray:
    """
    A_i in R^4: [mean, max, std, density] per patch. Returns float32 [N, 4].

    Features are NOT re-normalized across patches: the map is already in [0, 1], so
    mean/max/density lie in [0, 1] and std in [0, 0.5]. Re-normalizing per slice would
    make a structurally flat slice look as salient as a highly structured one.

    tau defaults to the mean of the non-zero map pixels (i.e. over brain tissue; the
    skull-stripped background has zero gradient).
    """
    if tau is None:
        nz = m[m > 0]
        tau = float(nz.mean()) if nz.size else 1.0
    p = patchify_map(m, patch)
    return np.stack(
        [p.mean(axis=1), p.max(axis=1), p.std(axis=1), (p > tau).mean(axis=1)], axis=1
    ).astype(np.float32)


def anatomical_features(img_chw: np.ndarray, kind: str = "sobel", patch: int = 16,
                        reduce: str = "flair"):
    """Multi-modal slice -> (A [N, 4], anatomical map [H, W])."""
    m = build_anatomical_map(img_chw, kind=kind, reduce=reduce)
    return patch_features(m, patch=patch), m


if __name__ == "__main__":
    # Self-test: bright disc (FLAIR channel) on a dark field.
    H = 224
    yy, xx = np.mgrid[0:H, 0:H]
    disc = (((yy - 112) ** 2 + (xx - 112) ** 2) < 60 ** 2).astype(np.float32)
    img = np.stack([disc * 0.8, disc * 0.9, disc * 0.7, disc * 1.0])

    A, m = anatomical_features(img)
    edge = A[:, 1] > 0.5
    print(f"map range [{m.min():.2f}, {m.max():.2f}]  A {A.shape}  "
          f"patches with a strong edge: {edge.sum()}/{len(A)}")
    assert A.shape == (196, 4) and m.min() == 0.0 and abs(m.max() - 1.0) < 1e-6
    assert A[0].sum() == 0.0, "corner patch (pure background) should have zero features"
    assert 20 <= edge.sum() <= 80, "edge patches should form a ring, not the whole image"
    print("self-test passed")
