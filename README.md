# AA-AdaMAE — Preliminary Results Pipeline

Reference implementation of Chapter III of *Anatomically-Guided Adaptive Masking for
Masked Autoencoder Pretraining in Brain MRI Segmentation* (Felipe & Sulay, 2026),
scoped to **Phase I pretraining and the RQ1 reconstruction experiment (Table 3.1)** —
i.e. exactly the numbers you need for the "Preliminary Results" slide.

## Files

| File | Manuscript section | Purpose |
|---|---|---|
| `prepare_data.py` | 3.2, 3.3 | BraTS NIfTI → cached 224×224 axial slices, patient-level 5-fold split |
| `anatomy.py` | 3.4.2 | Sobel / Canny / hybrid maps (Eq. 3.6–3.8), patch features `A_i` (Eq. 3.9) |
| `data.py` | 3.3, 3.6.3 | Dataset + anatomy-preserving augmentation |
| `models.py` | 3.4, 3.5 | ViT-Base MAE, AA-ATS sampler, dual-objective losses (Eq. 3.21–3.22) |
| `run_pretrain.py` | 3.6, 3.8.1 | Training loop, RQ1 sweep, writes `results/reconstruction.csv` |
| `plot_results.py` | Table 3.1 | Table + presentation figures |
| `visualize_masking.py` | — | Qualitative "where does it look" figure |

## Install

```bash
pip install torch numpy opencv-python nibabel pandas matplotlib
```

No `timm` or `einops` — the ViT is implemented directly so nothing breaks on MPS.

## Run order

**0. Smoke test (no data needed, ~1 min on CPU).** Do this first to confirm the
shapes and losses are wired correctly.

```bash
python run_pretrain.py --synthetic --epochs 1 --steps_per_epoch 5 \
  --batch_size 2 --dim 192 --depth 2 --workers 0 --no_save
```

**1. Preprocess.** Start with `--limit 50` to sanity-check before the full run.

```bash
python prepare_data.py --brats_root /path/to/BraTS2021 --out_root ./cache --limit 50
python prepare_data.py --brats_root /path/to/BraTS2021 --out_root ./cache   # full
```

**2. Single configuration** (the proposed model at the standard ratio):

```bash
python run_pretrain.py --cache ./cache --mask_mode anatomical \
  --anat_kind hybrid --fusion learnable --mask_ratio 0.75 --epochs 20
```

**3. Full RQ1 sweep** — 4 strategies × 3 ratios = 12 runs, appended to one CSV:

```bash
python run_pretrain.py --cache ./cache --sweep --epochs 20 --batch_size 32
```

**4. Table and figures:**

```bash
python plot_results.py --results ./results
python visualize_masking.py --cache ./cache \
  --ckpt results/checkpoints/anatomical_learnable_hybrid_p0.75.pt \
  --out results/figures/qualitative.png
```

## Masking strategies

| `--mask_mode` | What it does | Role |
|---|---|---|
| `random` | Uniform random masking | MAE baseline |
| `adaptive` | Learned sampler over patch embeddings only | AdaMAE baseline |
| `hard_anat` | Deterministic top-`Nv` by anatomical response, no learning | Rule-based masking **proxy** |
| `anatomical` | AA-ATS: sampler over embeddings fused with `A_i` | **Proposed** |

> **Be careful how you label `hard_anat` on the slide.** It is a stand-in for the
> hard-prior family, *not* a reimplementation of API-MAE or AMAP — both require atlas
> registration (SRI-24) and tumour-occurrence statistics that this pipeline does not
> build. Call it "rule-based hard anatomical masking (proxy)" and note that the true
> API-MAE / AMAP rows in Table 3.1 are still outstanding.

## Runtime guidance

Per-epoch cost scales with the number of cached slices. With `--pretrain_stride 5`,
expect roughly 25–30 slices per patient, so ~1,251 patients ≈ 33k slices.

Rough per-run estimates at ViT-Base, batch 32, 75% masking:

- **A100 / 4090 (AMP on):** minutes per epoch on the full set; a 20-epoch run is
  comfortably an overnight job, and the 12-run sweep is best split across sessions.
- **M1 MacBook Pro (MPS):** viable only for the smoke test and small subsets. Use
  `--max_slices_per_patient 5 --dim 384 --depth 6 --batch_size 8`.

For a progress presentation you do **not** need the full sweep. A defensible
preliminary result is a subset run — e.g. 50–100 patients, 10–20 epochs, all four
strategies at 75% masking — as long as the slide labels it as a subset and states the
patient count, epoch count, and fold. Reserve full-dataset numbers for the final paper.

Useful throttles: `--max_slices_per_patient`, `--steps_per_epoch`, `--eval_batches`.

## Implementation notes worth flagging in the defense

1. **Eq. 3.22 as written is not standard REINFORCE.** The manuscript defines
   `L_S = -Σ p_i · L_R(i)`, which is implemented here verbatim. The usual policy-gradient
   estimator uses `log p_i` rather than `p_i`. The literal form still pushes probability
   mass toward high-error tokens, but its gradient scale differs. Flag this as a known
   design decision, not an oversight — expect a panel question.
2. **The reward is detached** (`Sec. 3.5.2`), so sampler gradients never reach the
   encoder-decoder. The sampler also gets its own learning rate (`--sampler_lr`).
3. **Masking ratio conflict.** Sec. 3.4.1/3.4.3 fix p = 0.75 while Sec. 3.6.2 sweeps
   {0.50, 0.75, 0.80}. Here 0.75 is the *default* and the sweep overrides it — reword
   the manuscript to match.
4. **Optimizer.** Sec. 3.6.1 says "Adam … decoupled weight decay," which describes
   AdamW. This code uses AdamW; update the text.
5. **Straight comparison caveat.** Reconstruction MSE is *not* directly comparable
   across masking strategies, because each strategy masks a different, self-selected
   subset of patches. A sampler that learns to keep easy patches visible will look
   better on MSE without learning better representations. Report MSE alongside a
   fixed-random-mask evaluation, or lean on the RQ2 segmentation numbers as the real
   verdict. This is the most likely line of attack from the panel.
