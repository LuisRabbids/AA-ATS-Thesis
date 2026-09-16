"""
anatomy.py
Anatomical Feature Extraction Module (manuscript Sec. 3.4.2).

Implements:
  1. Anatomical map generation in the pixel domain, prior to tokenization:
       - Sobel gradient magnitude   (Eq. 3.6)
       - Canny edge map             (Eq. 3.7)
       - Hybrid weighted fusion     (Eq. 3.8), equal weights w1 = w2 = 0.5
  2. Patch-level feature extraction: mean, max, standard deviation, density
     -> anatomical feature vector A_i in R^F with F = 4  (Eq. 3.9)

All maps are computed with OpenCV on the CPU inside the DataLoader workers, so the
GPU never waits on edge detection.
"""

import cv2
import numpy as np

N_ANAT_FEATURES = 4  # F in Eq. 3.9: mean, max, std, density


# --------------------------------------------------------------------------------------
# 1. Anatomical map generation (Sec. 3.4.2 part 1)
# --------------------------------------------------------------------------------------
def sobel_map(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Eq. 3.6:  M_sobel = sqrt( (I * S_x)^2 + (I * S_y)^2 )

    img: float32 [H, W] in [0, 1]. Returns float32 [H, W] rescaled to [0, 1].
    """
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = np.sqrt(gx * gx + gy * gy)
    m = mag.max()
    return (mag / m).astype(np.float32) if m > 0 else mag.astype(np.float32)


def canny_map(img: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    """
    Eq. 3.7:  M_canny = C(I)

    Binary edge map returned as float32 {0.0, 1.0}. The image is converted to uint8
    because OpenCV's Canny requires 8-bit input.
    """
    u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    edges = cv2.Canny(u8, low, high)
    return (edges > 0).astype(np.float32)


def hybrid_map(img: np.ndarray, w1: float = 0.5, w2: float = 0.5, **kw) -> np.ndarray:
    """
    Eq. 3.8:  M_hybrid = w1 * M_sobel + w2 * M_canny

    Equal weights are the manuscript default (Sec. 3.4.2): no consensus in the
    literature favours either operator, equal weighting avoids biasing toward one
    representation, and tuning w1/w2 would turn this into a hyperparameter study.
    """
    return (w1 * sobel_map(img, **kw) + w2 * canny_map(img)).astype(np.float32)


MAP_FUNCS = {"sobel": sobel_map, "canny": canny_map, "hybrid": hybrid_map}


def build_anatomical_map(img_chw: np.ndarray, kind: str = "hybrid",
                         reduce: str = "mean") -> np.ndarray:
    """
    Compute one anatomical map for a multi-modal slice.

    img_chw : float32 [C, H, W] in [0, 1]   (C = 4: T1, T1Gd, T2, FLAIR)
    reduce  : 'mean'  -> average the modalities into one image, then run the operator
              'flair' -> run the operator on FLAIR only (channel index 3)
              'max'   -> run per modality and keep the per-pixel maximum response
    returns : float32 [H, W]
    """
    if kind not in MAP_FUNCS:
        raise ValueError(f"unknown anatomical map '{kind}', expected one of {list(MAP_FUNCS)}")
    fn = MAP_FUNCS[kind]

    if reduce == "mean":
        return fn(img_chw.mean(axis=0))
    if reduce == "flair":
        return fn(img_chw[3])
    if reduce == "max":
        return np.max(np.stack([fn(img_chw[c]) for c in range(img_chw.shape[0])]), axis=0)
    raise ValueError(f"unknown reduce mode '{reduce}'")


# --------------------------------------------------------------------------------------
# 2. Patch-level feature extraction (Sec. 3.4.2 part 2, Eq. 3.9)
# --------------------------------------------------------------------------------------
def patchify_map(m: np.ndarray, patch: int = 16) -> np.ndarray:
    """[H, W] -> [N, patch*patch] with N = (H/patch) * (W/patch), row-major order.

    The ordering matches the ViT tokenizer in models.py so that anatomical feature
    A_i always corresponds to patch embedding x_i.
    """
    h, w = m.shape
    assert h % patch == 0 and w % patch == 0, f"map {m.shape} not divisible by patch {patch}"
    gh, gw = h // patch, w // patch
    return (
        m.reshape(gh, patch, gw, patch)     # [gh, p, gw, p]
        .transpose(0, 2, 1, 3)              # [gh, gw, p, p]
        .reshape(gh * gw, patch * patch)    # [N, p*p]
    )


def patch_features(m: np.ndarray, patch: int = 16, density_thresh: float = 0.1) -> np.ndarray:
    """
    Eq. 3.9: A_i in R^F, F = 4.

      mean    - average gradient / edge strength in the patch
      max     - strongest local response
      std     - variability within the patch
      density - fraction of pixels above `density_thresh` (proportion of active pixels)

    Returns float32 [N, 4], each column min-max normalized across patches so the
    descriptors sit on a comparable scale before fusion.
    """
    p = patchify_map(m, patch)
    feats = np.stack(
        [
            p.mean(axis=1),
            p.max(axis=1),
            p.std(axis=1),
            (p > density_thresh).mean(axis=1),
        ],
        axis=1,
    ).astype(np.float32)

    lo = feats.min(axis=0, keepdims=True)
    hi = feats.max(axis=0, keepdims=True)
    return (feats - lo) / np.maximum(hi - lo, 1e-6)


def anatomical_features(img_chw: np.ndarray, kind: str = "hybrid", patch: int = 16,
                        reduce: str = "mean", density_thresh: float = 0.1):
    """Convenience wrapper: multi-modal slice -> (A [N, 4], anatomical map [H, W])."""
    m = build_anatomical_map(img_chw, kind=kind, reduce=reduce)
    return patch_features(m, patch=patch, density_thresh=density_thresh), m


if __name__ == "__main__":
    # Self-test on a synthetic phantom: a bright disc on a dark field.
    H = W = 224
    yy, xx = np.mgrid[0:H, 0:W]
    disc = (((yy - 112) ** 2 + (xx - 112) ** 2) < 60**2).astype(np.float32)
    img = np.stack([disc * 0.8, disc * 0.9, disc * 0.7, disc * 1.0]).astype(np.float32)

    for kind in ("sobel", "canny", "hybrid"):
        A, m = anatomical_features(img, kind=kind)
        boundary = A[:, 0] > 0.2
        print(f"{kind:7s} map[{m.min():.3f},{m.max():.3f}]  A{A.shape}  "
              f"active patches: {boundary.sum():3d}/{A.shape[0]}")
