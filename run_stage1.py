"""
run_stage1.py
Stage 1 configuration search / RQ1 (manuscript Sec. 3.8.2, Table 3.1).

Grid: fusion {direct, modulated} x direction {visible, masked} x ratio {0.50, 0.75, 0.90}
      = 12 configurations of the proposed model, all on the development split.
Each configuration is pre-trained, then fine-tuned for segmentation, and one row is
appended to <out>/stage1.csv. Configurations are ranked by mean Dice (WT/TC/ET);
fixed-mask reconstruction MSE is recorded as a secondary column.

Everything is resumable: finished configurations are skipped, and an interrupted one
resumes from its last completed epoch. Just rerun the same command.

Usage
  python run_stage1.py --cache ./cache --smoke          # 1-epoch test + time estimate
  python run_stage1.py --cache ./cache                  # the full grid
  python run_stage1.py --cache ./cache --list           # what's done / pending
  two GPUs:  --device cuda:0 --shard 0/2   and   --device cuda:1 --shard 1/2
  lab slot:  --stop_at 16:15   finish the current epoch, save, and exit before 16:15
  Kaggle:    --max_hours 11   exit cleanly before the 12-hour background-run limit
"""

import argparse
import csv
import itertools
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

from finetune import finetune
from run_pretrain import TimeUp, pretrain

FUSIONS = ["direct", "modulated"]
DIRECTIONS = ["visible", "masked"]
RATIOS = [0.50, 0.75, 0.90]
COLUMNS = ["tag", "fusion", "direction", "mask_ratio", "dsc_mean", "dsc_WT", "dsc_TC",
           "dsc_ET", "recall_WT", "recall_TC", "recall_ET", "fixed_mask_mse",
           "pt_epochs", "ft_epochs", "backbone", "pt_minutes", "ft_minutes", "finished"]


def grid():
    return [dict(fusion=f, direction=d, mask_ratio=r, tag=f"{f}_{d}_p{r:.2f}")
            for f, d, r in itertools.product(FUSIONS, DIRECTIONS, RATIOS)]


def done_tags(csv_path):
    if not csv_path.exists():
        return set()
    with open(csv_path) as f:
        return {row["tag"] for row in csv.DictReader(f)}


def append_row(csv_path, row):
    """Append one result row; safe when two processes (two GPUs) write at once."""
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
    """
    Stop if starting another epoch of the same length would pass the deadline.
    stop_at   'HH:MM' clock time; if that time has already passed today, it means tomorrow
    max_hours hours from now
    If both are given, the earlier deadline wins.
    """
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


def run_config(c, a, out_root, log):
    run_dir = out_root / c["tag"]
    t0 = time.time()
    pt = pretrain(a.cache, run_dir, log=log, mask_mode="anatomical", fusion=c["fusion"],
                  direction=c["direction"], mask_ratio=c["mask_ratio"], backbone=a.backbone,
                  epochs=a.pt_epochs, batch_size=a.pt_batch, split="dev",
                  device=a.device, workers=a.workers, should_stop=a.should_stop)
    t1 = time.time()
    ft = finetune(pt["checkpoint"], a.cache, "dev", run_dir, backbone=a.backbone,
                  epochs=a.ft_epochs, batch_size=a.ft_batch, train_stride=a.ft_stride,
                  device=a.device, workers=a.workers, log=log, should_stop=a.should_stop)
    ft_hist = json.load(open(run_dir / "finetune_result.json"))["history"]
    return {**c, **{k: round(v, 5) for k, v in ft.items()},
            "fixed_mask_mse": round(pt["fixed_mask_mse"], 6), "pt_epochs": a.pt_epochs,
            "ft_epochs": a.ft_epochs, "backbone": a.backbone,
            "pt_minutes": round(pt["minutes"], 1),
            "ft_minutes": round(sum(h["seconds"] for h in ft_hist) / 60, 1),
            "finished": f"{datetime.now():%Y-%m-%d %H:%M}"}


def summarize(csv_path):
    if not csv_path.exists():
        return
    rows = sorted(csv.DictReader(open(csv_path)), key=lambda r: -float(r["dsc_mean"]))
    print(f"\n{'rank':<5}{'configuration':<28}{'DSC mean':>9}{'WT':>8}{'TC':>8}{'ET':>8}{'MSE':>10}")
    for i, r in enumerate(rows, 1):
        print(f"{i:<5}{r['tag']:<28}{float(r['dsc_mean']):>9.4f}{float(r['dsc_WT']):>8.4f}"
              f"{float(r['dsc_TC']):>8.4f}{float(r['dsc_ET']):>8.4f}"
              f"{float(r['fixed_mask_mse']):>10.6f}")
    if len(rows) == len(grid()):
        print(f"\nAll 12 complete. Selected configuration: {rows[0]['tag']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="./cache")
    ap.add_argument("--out", default="./results")
    ap.add_argument("--backbone", default="small", choices=["small", "base"])
    ap.add_argument("--pt_epochs", type=int, default=50)
    ap.add_argument("--ft_epochs", type=int, default=30)
    ap.add_argument("--pt_batch", type=int, default=64)
    ap.add_argument("--ft_batch", type=int, default=16)
    ap.add_argument("--ft_stride", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th config starting at i")
    ap.add_argument("--stop_at", default=None, help="HH:MM: exit cleanly before this time")
    ap.add_argument("--max_hours", type=float, default=None,
                    help="exit cleanly before this many hours from now")
    ap.add_argument("--list", action="store_true", help="show status and exit")
    ap.add_argument("--smoke", action="store_true",
                    help="1 pre-train + 1 fine-tune epoch on one config, then estimate the grid")
    a = ap.parse_args()

    out_root = Path(a.out, "smoke" if a.smoke else "stage1")
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "stage1.csv"
    log = make_logger(out_root / "log.txt")

    configs = grid()
    if a.smoke:
        configs = [c for c in configs if c["tag"] == "modulated_visible_p0.75"]
        full_pt, full_ft = a.pt_epochs, a.ft_epochs
        a.pt_epochs = a.ft_epochs = 1

    if a.list:
        done = done_tags(csv_path)
        for c in configs:
            state = "done" if c["tag"] in done else (
                "in progress" if (out_root / c["tag"]).exists() else "pending")
            print(f"  {c['tag']:<28} {state}")
        summarize(csv_path)
        return

    i, n = map(int, a.shard.split("/"))
    mine = configs[i::n]
    done = done_tags(csv_path)
    todo = [c for c in mine if c["tag"] not in done]
    log(f"Stage 1 {'SMOKE TEST ' if a.smoke else ''}| shard {a.shard}: {len(mine)} configs, "
        f"{len(mine)-len(todo)} already done, {len(todo)} to run | ViT-{a.backbone}, "
        f"{a.pt_epochs} pt + {a.ft_epochs} ft epochs, device {a.device}")

    a.should_stop = make_should_stop(a.stop_at, a.max_hours, log)
    for k, c in enumerate(todo, 1):
        log(f"=== [{k}/{len(todo)}] {c['tag']} ===")
        if a.should_stop and a.should_stop(0):
            log("Deadline reached before starting the next configuration. Rerun to continue.")
            break
        try:
            row = run_config(c, a, out_root, log)
        except TimeUp as e:
            log(f"Stopping cleanly for the session deadline ({e}). "
                f"Rerun the same command to resume.")
            break
        append_row(csv_path, row)
        log(f"  -> DSC mean {row['dsc_mean']:.4f} (WT {row['dsc_WT']:.4f} TC {row['dsc_TC']:.4f} "
            f"ET {row['dsc_ET']:.4f}) | MSE {row['fixed_mask_mse']:.6f} | "
            f"{row['pt_minutes'] + row['ft_minutes']:.1f} min")

    if a.smoke and todo:
        run_dir = out_root / todo[0]["tag"]
        pt_s = json.load(open(run_dir / "pretrain_result.json"))["history"][0]["seconds"]
        ft_s = json.load(open(run_dir / "finetune_result.json"))["history"][0]["seconds"]
        per_cfg = (pt_s * full_pt + ft_s * full_ft) / 3600
        log(f"\nMeasured: {pt_s:.0f}s per pre-train epoch, {ft_s:.0f}s per fine-tune epoch.")
        log(f"Projected: {per_cfg:.2f} h per configuration x 12 = {12 * per_cfg:.1f} h "
            f"on one GPU ({6 * per_cfg:.1f} h on two), at {full_pt} pt + {full_ft} ft epochs.")
    summarize(csv_path)


if __name__ == "__main__":
    main()
