# AA-ATS: Anatomically-Aware Adaptive Token Sampler

Self-supervised masked-autoencoder pre-training for brain tumour segmentation on
BraTS 2021, with a token sampler guided by self-derived Sobel anatomical features.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Preprocess BraTS once | `prepare_data.py` | `cache/` (per-case slices, `manifest.csv`, `splits.json`) |
| 2. Stage 1 / RQ1 grid | `run_stage1.py` | `results/stage1/stage1.csv` (Table 3.1) |

`run_stage1.py` runs 12 configurations (fusion x direction x masking ratio). For each it
pre-trains (`run_pretrain.py`), fine-tunes for segmentation (`finetune.py`) and appends one
row to the CSV. Everything is resumable: rerun the same command after any interruption.

## Setup (lab workstation)

```bash
git clone https://github.com/LuisRabbids/AA-ATS-Thesis.git
cd AA-ATS-Thesis
python -m venv env
source env/bin/activate
pip install -r requirements.txt
nvidia-smi                        # note the GPU index you are allowed to use
```

Put the preprocessed cache in `./cache` (from the Kaggle preprocessing notebook output),
or build it from the raw tar:

```bash
python prepare_data.py --brats_tar /path/BraTS2021_Training_Data.tar --out ./cache
```

## Running Stage 1

Always inside tmux (`tmux`, then `Ctrl+b d` to detach, `tmux attach -t 0` to return).

```bash
source env/bin/activate
python run_stage1.py --cache ./cache --device cuda:0 --smoke   # ~10 min: test + time estimate
python run_stage1.py --cache ./cache --device cuda:0           # full grid
python run_stage1.py --cache ./cache --list                    # progress and ranking
```

**Lab slot (ends 4:30 PM):** add `--stop_at 16:15`. The run finishes its current epoch,
saves, and exits before 16:15. Next session, rerun the same command to continue.

**Kaggle background run (12-hour limit):** add `--max_hours 11` to exit cleanly in time.

**Two GPUs** (e.g. Kaggle T4 x2): run `--shard 0/2 --device cuda:0` and
`--shard 1/2 --device cuda:1` side by side. Both write to the same CSV safely. The lab
workstation has one GPU, so there the default (`--shard 0/1`) runs all 12 in sequence.

Defaults: ViT-Small, 50 pre-training epochs, 30 fine-tuning epochs, fine-tuning on every
2nd slice. Change with `--backbone`, `--pt_epochs`, `--ft_epochs`, `--ft_stride`.

## Files

| File | Role (manuscript section) |
|---|---|
| `prepare_data.py` | crop 240->224, per-volume normalization, tissue index, dev/experimental splits (3.2-3.3) |
| `anatomy.py` | Sobel map from FLAIR, patch descriptors A_i (3.4.2) |
| `data.py` | datasets, brain-masked augmentation (3.6.3) |
| `models.py` | MAE, AA-ATS sampler, sampling direction, tissue-restricted losses (3.4-3.5) |
| `run_pretrain.py` | pre-training + fixed-mask reconstruction MSE (3.6, 3.8.3) |
| `finetune.py` | segmentation fine-tuning, per-patient WT/TC/ET metrics (3.7, 3.8.4) |
| `run_stage1.py` | Stage 1 configuration grid (3.8.2) |
| `metadata/` | case partition, tumour volumes, bounding boxes |

Legacy from the first prototype (not updated to the new checkpoint format):
`diagnose_mse.py`, `visualize_masking.py`, `plot_results.py`, `plot_results(comp).py`,
`check_progress.py`.
