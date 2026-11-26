"""
run_seedstudy.py
How much does an identical configuration move between training runs?

Stage 2 reports differences of 0.33 to 0.42 Dice points between masking strategies, and
the step-3 replicate on the development partition differed by 0.78 points between two
runs of one configuration. That replicate cannot settle the question: with 40 validation
patients, evaluation noise alone is worth about 0.6 points, so it bounds training
variability without measuring it.

This script measures it where it matters. It re-runs the proposed model on Stage 2
fold 1 under additional training seeds, changing nothing else. Combined with the
existing seed-42 run in results/stage2/ours_f1, that gives several independent training
runs of one configuration evaluated on the same 350 held-out patients, where evaluation
noise is roughly 0.2 points and the spread that remains is training variability.

The comparison is only valid if everything except the seed is identical, so the folds,
masking configuration, mask ratio, epoch budget, stride, batch sizes and backbone are
imported from or defaulted to run_stage2's own values rather than restated here. The
seed is threaded into both phases: pre-training (weight init, mask sampling, data order)
and fine-tuning (weight init, data order). Note that run_stage2's --seed controls the
fold partitioning, not training; this script leaves that at its Stage 2 value of 42 so
the folds are bit-identical, and varies the training seed separately.

Results go to results/seedstudy/, never results/stage2/, so that analyze_stage2.py and
analyze_factorial.py keep seeing exactly the four models they are meant to compare.

Usage
  python run_seedstudy.py --cache ./cache --seeds 43,44 --smoke      # sanity check
  python run_seedstudy.py --cache ./cache --seeds 43,44              # ~4.4 h per seed
  python run_seedstudy.py --cache ./cache --seeds 43,44 --list
  two GPUs:  --device cuda:0 --shard 0/2   and   --device cuda:1 --shard 1/2
  Kaggle:    --max_hours 5
"""

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path

from run_stage2 import (MASK_RATIO, MODELS, build_splits, done_tags,
                        make_logger, make_should_stop)

MODEL = "ours"          # the configuration whose run-to-run spread we need
FOLD = 1                # the fold whose seed-42 run already exists in results/stage2
FOLD_SEED = 42          # Stage 2's fold-partitioning seed; must not change
BASELINE = Path("results/stage2/ours_f1/finetune_result.json")

COLUMNS = ["tag", "model", "fold", "seed", "dsc_mean", "dsc_WT", "dsc_TC", "dsc_ET",
           "recall_WT", "recall_TC", "recall_ET", "fixed_mask_mse",
           "pt_epochs", "ft_epochs", "ft_stride", "backbone",
           "pt_minutes", "ft_minutes", "finished"]


def run_job(seed, a, out_root, log):
    from finetune import finetune
    from run_pretrain import pretrain
    cfg = MODELS[MODEL]
    split = f"fold_{FOLD}"
    run_dir = out_root / f"{MODEL}_f{FOLD}_s{seed}"

    pt = pretrain(a.cache, run_dir, log=log, mask_ratio=MASK_RATIO, backbone=a.backbone,
                  epochs=a.pt_epochs, batch_size=a.pt_batch, split=split, seed=seed,
                  device=a.device, workers=a.workers, should_stop=a.should_stop, **cfg)
    ft = finetune(pt["checkpoint"], a.cache, split, run_dir, backbone=a.backbone,
                  epochs=a.ft_epochs, batch_size=a.ft_batch, train_stride=a.ft_stride,
                  device=a.device, workers=a.workers, log=log, with_hd95=False,
                  seed=seed, should_stop=a.should_stop)
    hist = json.load(open(run_dir / "finetune_result.json"))["history"]
    return {"tag": f"{MODEL}_f{FOLD}_s{seed}", "model": MODEL, "fold": FOLD, "seed": seed,
            **{k: round(v, 5) for k, v in ft.items()},
            "fixed_mask_mse": round(pt["fixed_mask_mse"], 6),
            "pt_epochs": a.pt_epochs, "ft_epochs": a.ft_epochs, "ft_stride": a.ft_stride,
            "backbone": a.backbone, "pt_minutes": round(pt["minutes"], 1),
            "ft_minutes": round(sum(h["seconds"] for h in hist) / 60, 1),
            "finished": f"{datetime.now():%Y-%m-%d %H:%M}"}


def summarize(csv_path):
    runs = []
    if BASELINE.exists():
        s = json.load(open(BASELINE))["summary"]
        runs.append(("42 (Stage 2)", s["dsc_mean"]))
    if csv_path.exists():
        runs += [(r["seed"], float(r["dsc_mean"])) for r in csv.DictReader(open(csv_path))]
    if not runs:
        return

    print(f"\n{MODEL} on fold {FOLD}, identical in everything but the training seed\n")
    print(f"{'seed':<16}{'DSC mean':>10}")
    for s, d in runs:
        print(f"{str(s):<16}{d:>10.4f}")

    if len(runs) > 1:
        vals = [d for _, d in runs]
        spread = 100 * (max(vals) - min(vals))
        mean = sum(vals) / len(vals)
        sd = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
        print(f"\nrange {spread:.2f} Dice points across {len(vals)} runs, "
              f"sd {100*sd:.2f}")
        print("Stage 2 reported ours - mae = -0.33 and ours - fixed_sobel = -0.42 "
              "Dice points.")
        if spread >= 0.33:
            print("The spread between identical runs is as large as those effects, so "
                  "they cannot be attributed to the masking strategy on this evidence.")
        else:
            print("The spread between identical runs is smaller than those effects, "
                  "which supports reading them as real rather than as training noise.")
        print("Run the per-patient comparison with analyze_seedstudy.py for intervals.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="./cache")
    ap.add_argument("--metadata", default="./metadata")
    ap.add_argument("--out", default="./results")
    ap.add_argument("--seeds", default="43,44", help="comma-separated training seeds")
    ap.add_argument("--n_folds", type=int, default=3)
    ap.add_argument("--backbone", default="small", choices=["small", "base"])
    ap.add_argument("--pt_epochs", type=int, default=100)      # Stage 2 values
    ap.add_argument("--ft_epochs", type=int, default=20)
    ap.add_argument("--pt_batch", type=int, default=64)
    ap.add_argument("--ft_batch", type=int, default=16)
    ap.add_argument("--ft_stride", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--stop_at", default=None)
    ap.add_argument("--max_hours", type=float, default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="one seed, 1 epoch each")
    a = ap.parse_args()

    out_root = Path(a.out, "smoke_seed" if a.smoke else "seedstudy")
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "seedstudy.csv"
    log = make_logger(out_root / "log.txt")

    # Identical folds to Stage 2: same metadata, same fold count, same partition seed.
    splits_path = out_root / f"splits_{a.n_folds}fold.json"
    build_splits(a.metadata, a.n_folds, FOLD_SEED, splits_path)
    os.environ["AATS_SPLITS_JSON"] = str(splits_path)

    seeds = [int(s) for s in a.seeds.split(",")]
    if a.smoke:
        seeds = seeds[:1]
        a.pt_epochs = a.ft_epochs = 1

    if a.list:
        d = done_tags(csv_path)
        for s in seeds:
            tag = f"{MODEL}_f{FOLD}_s{s}"
            state = "done" if tag in d else ("in progress" if (out_root / tag).exists()
                                             else "pending")
            print(f"  {tag:<24} {state}")
        summarize(csv_path)
        return

    i, n = map(int, a.shard.split("/"))
    mine = seeds[i::n]
    todo = [s for s in mine if f"{MODEL}_f{FOLD}_s{s}" not in done_tags(csv_path)]
    log(f"Seed study {'SMOKE ' if a.smoke else ''}| shard {a.shard}: seeds {mine}, "
        f"{len(mine)-len(todo)} done, {len(todo)} to run | {MODEL} fold {FOLD}, "
        f"ViT-{a.backbone}, {a.pt_epochs}pt/{a.ft_epochs}ft ep, stride {a.ft_stride}, "
        f"p={MASK_RATIO}, device {a.device}")

    a.should_stop = make_should_stop(a.stop_at, a.max_hours, log)
    from run_pretrain import TimeUp
    for k, s in enumerate(todo, 1):
        log(f"=== [{k}/{len(todo)}] {MODEL} fold {FOLD} seed {s} ===")
        if a.should_stop and a.should_stop(0):
            log("Deadline reached before starting the next seed. Rerun to continue.")
            break
        try:
            row = run_job(s, a, out_root, log)
        except TimeUp as e:
            log(f"Stopping cleanly for the session deadline ({e}). Rerun to resume.")
            break
        append_row_cols(csv_path, row)
        log(f"  -> seed {s}: DSC mean {row['dsc_mean']:.4f} "
            f"({row['pt_minutes'] + row['ft_minutes']:.1f} min)")

    summarize(csv_path)


def append_row_cols(csv_path, row):
    """append_row from run_stage2, but with this script's column set."""
    import fcntl
    with open(csv_path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        if f.tell() == 0:
            csv.DictWriter(f, COLUMNS).writeheader()
        csv.DictWriter(f, COLUMNS, extrasaction="ignore").writerow(row)
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
