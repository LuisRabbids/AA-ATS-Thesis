"""
run_stage2.py
Stage 2 evaluation / RQ2 and RQ3 (manuscript Sec. 3.8.3, 3.8.4).

Compares the configuration selected in Stage 1 against three baselines on the
experimental partition, under three-fold cross-validation:

  ours         anatomical guidance, learnable scaling, structure-masked   [proposed]
  mae          uniform random masking                                      [baseline]
  adamae       adaptive sampling, no anatomical features                   [baseline]
  fixed_sobel  patches ranked by mean Sobel response, no learning          [baseline]

Every model shares backbone, preprocessing, optimiser, schedule, augmentation, epoch
budget and checkpoint policy; only the masking strategy differs. The adaptive-family
baselines use the same sampling direction as the proposed method, so the comparison
isolates the anatomical guidance rather than the direction (Stage 1 found no measurable
effect of direction: -0.03 Dice points, favoured in three of six matched pairs).

Self-supervised pre-training is performed independently within each fold, on that fold's
training cases only, so no validation patient is seen in either phase (Sec. 3.2.2). The
folds are regenerated here from metadata/partitions.json and metadata/tumor_volumes.json
rather than read from the cache, because the cache may carry a different fold count and
may be read-only.

Everything is resumable: finished (model, fold) pairs are skipped, and an interrupted one
resumes from its last completed epoch. Rerun the same command.

Usage
  python run_stage2.py --cache ./cache --folds 1 --smoke        # ~20 min sanity check
  python run_stage2.py --cache ./cache --folds 1                # one fold, four models
  python run_stage2.py --cache ./cache --folds 1,2,3            # the full reported run
  python run_stage2.py --cache ./cache --list                   # progress and ranking
  two GPUs:  --device cuda:0 --shard 0/2   and   --device cuda:1 --shard 1/2
  Kaggle:    --max_hours 11                                     # 12-hour session limit
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

MODELS = {
    "ours":        dict(mask_mode="anatomical", fusion="modulated", direction="masked"),
    "mae":         dict(mask_mode="random",     fusion="direct",    direction="visible"),
    "adamae":      dict(mask_mode="adaptive",   fusion="direct",    direction="masked"),
    "fixed_sobel": dict(mask_mode="hard_anat",  fusion="direct",    direction="masked"),
}
# Scheduling order: interleaved so that a 2-shard split gives each GPU one model with a
# learned sampler (ours, adamae) and one without (mae, fixed_sobel).
ORDER = ["ours", "adamae", "mae", "fixed_sobel"]
# Reporting order: the ablation ladder, each step adding one component.
DISPLAY = ["mae", "adamae", "fixed_sobel", "ours"]
MASK_RATIO = 0.75                                        # selected in Stage 1
COLUMNS = ["tag", "model", "fold", "dsc_mean", "dsc_WT", "dsc_TC", "dsc_ET",
           "recall_WT", "recall_TC", "recall_ET", "hd95_WT", "hd95_TC", "hd95_ET",
           "fixed_mask_mse", "pt_epochs", "ft_epochs", "ft_stride", "backbone",
           "pt_minutes", "ft_minutes", "finished"]


# --------------------------------------------------------------------------------------
# Folds, regenerated from metadata so the cache is never written to
# --------------------------------------------------------------------------------------
def build_splits(metadata, n_folds, seed, out_path):
    from prepare_data import stratified_kfold
    with open(Path(metadata, "partitions.json")) as f:
        partitions = json.load(f)
    with open(Path(metadata, "tumor_volumes.json")) as f:
        volumes = json.load(f)
    exp = [c for c in partitions["experimental"] if c in volumes]
    folds = stratified_kfold(exp, volumes, n_folds, seed)
    splits = {f"fold_{i+1}": {"val": folds[i],
                              "train": sorted(c for j, f_ in enumerate(folds) if j != i
                                              for c in f_)}
              for i in range(n_folds)}
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=1)
    assert not set().union(*[set(v["val"]) for v in splits.values()]) ^ set(exp)
    assert sum(len(v["val"]) for v in splits.values()) == len(exp)
    return splits


# --------------------------------------------------------------------------------------
def done_tags(csv_path):
    if not csv_path.exists():
        return set()
    with open(csv_path) as f:
        return {r["tag"] for r in csv.DictReader(f)}


def append_row(csv_path, row):
    import fcntl
    with open(csv_path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        if f.tell() == 0:
            csv.DictWriter(f, COLUMNS).writeheader()
        csv.DictWriter(f, COLUMNS, extrasaction="ignore").writerow(row)
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


def make_logger(path):
    def log(msg):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        with open(path, "a") as f:
            f.write(line + "\n")
    return log


def make_should_stop(stop_at, max_hours, log):
    deadlines = []
    if stop_at:
        h, m = map(int, stop_at.split(":"))
        d = datetime.now().replace(hour=h, minute=m, second=0, microsecond=0)
        if d <= datetime.now():
            d += timedelta(days=1)
        deadlines.append(d)
    if max_hours:
        deadlines.append(datetime.now() + timedelta(hours=max_hours))
    if not deadlines:
        return None
    deadline = min(deadlines)
    log(f"Will stop cleanly before {deadline:%Y-%m-%d %H:%M}.")

    def should_stop(epoch_seconds):
        return time.time() + 1.1 * epoch_seconds > deadline.timestamp()
    return should_stop


def run_job(model, fold, a, out_root, log):
    from finetune import finetune
    from run_pretrain import pretrain
    cfg = MODELS[model]
    split = f"fold_{fold}"
    run_dir = out_root / f"{model}_f{fold}"

    pt = pretrain(a.cache, run_dir, log=log, mask_ratio=MASK_RATIO, backbone=a.backbone,
                  epochs=a.pt_epochs, batch_size=a.pt_batch, split=split,
                  device=a.device, workers=a.workers, should_stop=a.should_stop, **cfg)
    ft = finetune(pt["checkpoint"], a.cache, split, run_dir, backbone=a.backbone,
                  epochs=a.ft_epochs, batch_size=a.ft_batch, train_stride=a.ft_stride,
                  device=a.device, workers=a.workers, log=log, with_hd95=a.hd95,
                  should_stop=a.should_stop)
    hist = json.load(open(run_dir / "finetune_result.json"))["history"]
    return {"tag": f"{model}_f{fold}", "model": model, "fold": fold,
            **{k: round(v, 5) for k, v in ft.items()},
            "fixed_mask_mse": round(pt["fixed_mask_mse"], 6),
            "pt_epochs": a.pt_epochs, "ft_epochs": a.ft_epochs, "ft_stride": a.ft_stride,
            "backbone": a.backbone, "pt_minutes": round(pt["minutes"], 1),
            "ft_minutes": round(sum(h["seconds"] for h in hist) / 60, 1),
            "finished": f"{datetime.now():%Y-%m-%d %H:%M}"}


def summarize(csv_path):
    if not csv_path.exists():
        return
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    print(f"\n{'model':<14}{'folds':>6}{'DSC mean':>10}{'WT':>9}{'TC':>9}{'ET':>9}{'MSE':>10}")
    for m in DISPLAY:
        rs = by_model.get(m, [])
        if not rs:
            continue
        mean = lambda k: sum(float(r[k]) for r in rs) / len(rs)
        print(f"{m:<14}{len(rs):>6}{mean('dsc_mean'):>10.4f}{mean('dsc_WT'):>9.4f}"
              f"{mean('dsc_TC'):>9.4f}{mean('dsc_ET'):>9.4f}{mean('fixed_mask_mse'):>10.6f}")
    if "ours" in by_model:
        base = sum(float(r["dsc_mean"]) for r in by_model["ours"]) / len(by_model["ours"])
        print("\nDice points vs ours (same folds only):")
        for m in [x for x in DISPLAY if x != "ours"]:
            shared = [r for r in by_model.get(m, [])
                      if r["fold"] in {x["fold"] for x in by_model["ours"]}]
            if shared:
                d = base - sum(float(r["dsc_mean"]) for r in shared) / len(shared)
                print(f"  ours - {m:<12} {100*d:+.2f}")


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="./cache")
    ap.add_argument("--metadata", default="./metadata")
    ap.add_argument("--out", default="./results")
    ap.add_argument("--folds", default="1", help="comma-separated, e.g. 1 or 1,2,3")
    ap.add_argument("--n_folds", type=int, default=3)
    ap.add_argument("--models", default="all", help="comma-separated, or 'all'")
    ap.add_argument("--backbone", default="small", choices=["small", "base"])
    ap.add_argument("--pt_epochs", type=int, default=100)
    ap.add_argument("--ft_epochs", type=int, default=20)
    ap.add_argument("--pt_batch", type=int, default=64)
    ap.add_argument("--ft_batch", type=int, default=16)
    ap.add_argument("--ft_stride", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--stop_at", default=None)
    ap.add_argument("--max_hours", type=float, default=None)
    ap.add_argument("--hd95", action="store_true", help="also compute HD95 (slow)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="one model, one fold, 1 epoch each, into results/smoke2")
    a = ap.parse_args()

    out_root = Path(a.out, "smoke2" if a.smoke else "stage2")
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "stage2.csv"
    log = make_logger(out_root / "log.txt")

    # Folds are rebuilt here and handed to data.load_split through the environment.
    splits_path = out_root / f"splits_{a.n_folds}fold.json"
    build_splits(a.metadata, a.n_folds, a.seed, splits_path)
    os.environ["AATS_SPLITS_JSON"] = str(splits_path)

    folds = [int(x) for x in a.folds.split(",")]
    models = ORDER if a.models == "all" else [m for m in ORDER if m in a.models.split(",")]
    if a.smoke:
        folds, models = folds[:1], models[:1]
        a.pt_epochs = a.ft_epochs = 1

    jobs = [(m, k) for k in folds for m in models]

    if a.list:
        d = done_tags(csv_path)
        for m, k in jobs:
            tag = f"{m}_f{k}"
            state = "done" if tag in d else (
                "in progress" if (out_root / tag).exists() else "pending")
            print(f"  {tag:<20} {state}")
        summarize(csv_path)
        return

    i, n = map(int, a.shard.split("/"))
    mine = jobs[i::n]
    todo = [j for j in mine if f"{j[0]}_f{j[1]}" not in done_tags(csv_path)]
    log(f"Stage 2 {'SMOKE ' if a.smoke else ''}| shard {a.shard}: {len(mine)} jobs, "
        f"{len(mine)-len(todo)} done, {len(todo)} to run | ViT-{a.backbone}, "
        f"{a.pt_epochs}pt/{a.ft_epochs}ft ep, stride {a.ft_stride}, p={MASK_RATIO}, "
        f"{a.n_folds}-fold, device {a.device}")

    a.should_stop = make_should_stop(a.stop_at, a.max_hours, log)
    from run_pretrain import TimeUp
    for k, (model, fold) in enumerate(todo, 1):
        log(f"=== [{k}/{len(todo)}] {model} fold {fold} ===")
        if a.should_stop and a.should_stop(0):
            log("Deadline reached before starting the next job. Rerun to continue.")
            break
        try:
            row = run_job(model, fold, a, out_root, log)
        except TimeUp as e:
            log(f"Stopping cleanly for the session deadline ({e}). Rerun to resume.")
            break
        append_row(csv_path, row)
        log(f"  -> DSC mean {row['dsc_mean']:.4f} (WT {row['dsc_WT']:.4f} "
            f"TC {row['dsc_TC']:.4f} ET {row['dsc_ET']:.4f}) | "
            f"MSE {row['fixed_mask_mse']:.6f} | "
            f"{row['pt_minutes'] + row['ft_minutes']:.1f} min")

    summarize(csv_path)


if __name__ == "__main__":
    main()
