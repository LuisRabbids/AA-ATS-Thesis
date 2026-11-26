"""
run_step3.py
Follow-up experiments on the two limitations named in the results discussion:

  1. Guidance strength (alpha). The proposed sampler draws the whole masked set from the
     learned distribution P, so uniform random masking is unreachable even when it would
     be better. Here a fraction alpha of the masked set is drawn from P and the remaining
     (1 - alpha) uniformly, interpolating between random masking (alpha = 0) and the
     method as submitted (alpha = 1).

  2. Sampling objective. Eq. 3.21 maximises sum p_i L_R(i), whose gradient w.r.t. a
     logit is proportional to p_i and therefore vanishes exactly on the tokens the
     sampler has already learned to ignore. The REINFORCE form sum log p_i L_R(i) has
     gradient proportional to 1, i.e. roughly N times larger near uniform P, so its
     weight is reduced by the same factor (lambda 1.0 -> ~1/N = 0.005) and bracketed.
     An entropy bonus on P is available to discourage collapse.

These are configuration-search experiments, so they run on the DEVELOPMENT partition,
under the same recipe as Stage 1 (50 pre-train epochs, 30 fine-tune epochs, stride 2,
ViT-Small, p = 0.75). The experimental partition is untouched: nothing here may inform
the Stage 2 numbers reported for RQ2 and RQ3.

The alpha = 1.0 / prob reference is the Stage 1 cell 'modulated_masked_p0.75'
(DSC 0.8173) and is not re-run; --list and the summary print it for comparison.

Everything is resumable: finished configurations are skipped, and an interrupted one
resumes from its last completed epoch. Rerun the same command.

Usage
  python run_step3.py --cache ./cache --smoke        # 1 config, 1 epoch each
  python run_step3.py --cache ./cache               # all seven configurations
  python run_step3.py --cache ./cache --list        # progress and the alpha curve
  two GPUs:  --device cuda:0 --shard 0/2   and   --device cuda:1 --shard 1/2
  Kaggle:    --max_hours 11
"""

import argparse
import csv
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

BASE = dict(mask_mode="anatomical", fusion="modulated", direction="masked")
MASK_RATIO = 0.75

# Every configuration below is the Stage 1 winner with one thing changed.
CONFIGS = {
    # --- guidance sweep, Eq. 3.21 objective unchanged -------------------------------
    # The grid is weighted towards small alpha on purpose. Only about 29% of patches
    # contain brain tissue (~57 of 196), the masked set at p = 0.75 holds 147 tokens,
    # and the Stage 1 diagnostic shows the sampler concentrating 9.6-18.5x above uniform
    # on scored tokens. Drawing even a quarter of the masked set from P therefore
    # already covers most tissue patches, so alpha = 0.75 would be indistinguishable
    # from alpha = 1.0. The 'tissue_masked' column records what each alpha actually did.
    #
    # alpha = 0: the mask is uniform and the sampler is switched off (lambda 0), which
    # makes this the random-masking endpoint under an otherwise identical recipe.
    "alpha000":     dict(guidance=0.00, lambda_sampler=0.0),
    "alpha010":     dict(guidance=0.10),
    "alpha025":     dict(guidance=0.25),
    "alpha050":     dict(guidance=0.50),
    # alpha100 = Stage 1 'modulated_masked_p0.75', reused rather than re-run.

    # --- REINFORCE objective, full guidance ----------------------------------------
    "logp_lam005":  dict(sampler_obj="logprob", lambda_sampler=0.005),
    "logp_lam05":   dict(sampler_obj="logprob", lambda_sampler=0.05),
    "logp_ent01":   dict(sampler_obj="logprob", lambda_sampler=0.005, ent_coef=0.01),
}
# Interleaved so a 2-shard split puts roughly half of each family on each GPU.
ORDER = ["alpha050", "logp_lam005", "alpha025", "logp_lam05",
         "alpha010", "logp_ent01", "alpha000"]
REFERENCE = ("alpha100", "results/stage1/stage1.csv", "modulated_masked_p0.75")

COLUMNS = ["tag", "guidance", "sampler_obj", "lambda_sampler", "ent_coef",
           "dsc_mean", "dsc_WT", "dsc_TC", "dsc_ET",
           "recall_WT", "recall_TC", "recall_ET", "fixed_mask_mse", "tissue_masked",
           "pt_epochs", "ft_epochs", "ft_stride", "backbone",
           "pt_minutes", "ft_minutes", "finished"]


def full_cfg(name):
    c = dict(BASE, guidance=1.0, sampler_obj="prob", ent_coef=0.0, lambda_sampler=1.0)
    c.update(CONFIGS[name])
    return c


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


def run_job(name, a, out_root, log):
    from finetune import finetune
    from run_pretrain import pretrain
    cfg = full_cfg(name)
    run_dir = out_root / name

    pt = pretrain(a.cache, run_dir, log=log, mask_ratio=MASK_RATIO, backbone=a.backbone,
                  epochs=a.pt_epochs, batch_size=a.pt_batch, split="dev",
                  device=a.device, workers=a.workers, should_stop=a.should_stop, **cfg)
    ft = finetune(pt["checkpoint"], a.cache, "dev", run_dir, backbone=a.backbone,
                  epochs=a.ft_epochs, batch_size=a.ft_batch, train_stride=a.ft_stride,
                  device=a.device, workers=a.workers, log=log, with_hd95=False,
                  should_stop=a.should_stop)
    hist = json.load(open(run_dir / "finetune_result.json"))["history"]
    return {"tag": name, "guidance": cfg["guidance"], "sampler_obj": cfg["sampler_obj"],
            "lambda_sampler": cfg["lambda_sampler"], "ent_coef": cfg["ent_coef"],
            **{k: round(v, 5) for k, v in ft.items()},
            "fixed_mask_mse": round(pt["fixed_mask_mse"], 6),
            "tissue_masked": round(pt["history"][-1].get("tissue_masked", float("nan")), 4),
            "pt_epochs": a.pt_epochs, "ft_epochs": a.ft_epochs, "ft_stride": a.ft_stride,
            "backbone": a.backbone, "pt_minutes": round(pt["minutes"], 1),
            "ft_minutes": round(sum(h["seconds"] for h in hist) / 60, 1),
            "finished": f"{datetime.now():%Y-%m-%d %H:%M}"}


def reference_row():
    """The Stage 1 cell that is already the alpha = 1.0 / prob configuration."""
    name, path, tag = REFERENCE
    if not Path(path).exists():
        return None
    for r in csv.DictReader(open(path)):
        if r["tag"] == tag:
            return {"tag": name, "guidance": 1.0, "sampler_obj": "prob",
                    "lambda_sampler": 1.0, "ent_coef": 0.0,
                    "dsc_mean": r["dsc_mean"], "fixed_mask_mse": r["fixed_mask_mse"],
                    "finished": "Stage 1"}
    return None


def summarize(csv_path):
    rows = list(csv.DictReader(open(csv_path))) if csv_path.exists() else []
    ref = reference_row()
    if ref:
        rows.append(ref)
    if not rows:
        return
    by_tag = {r["tag"]: r for r in rows}
    show = lambda r, k: f"{float(r[k]):.4f}" if r.get(k) not in (None, "") else "   -  "

    print(f"\n{'configuration':<16}{'alpha':>7}{'objective':>11}{'lambda':>8}"
          f"{'ent':>6}{'DSC mean':>10}{'MSE':>10}{'tissue masked':>15}")
    order = ["alpha000", "alpha010", "alpha025", "alpha050", "alpha100",
             "logp_lam005", "logp_lam05", "logp_ent01"]
    for t in order:
        r = by_tag.get(t)
        if not r:
            continue
        print(f"{t:<16}{float(r['guidance']):>7.2f}{r['sampler_obj']:>11}"
              f"{float(r['lambda_sampler']):>8.3f}{float(r['ent_coef']):>6.2f}"
              f"{show(r, 'dsc_mean'):>10}{show(r, 'fixed_mask_mse'):>10}"
              f"{show(r, 'tissue_masked'):>15}")

    base = by_tag.get("alpha100")
    if base and base.get("dsc_mean"):
        b = float(base["dsc_mean"])
        print(f"\nDice points vs alpha = 1.0 / prob ({b:.4f}, the submitted method):")
        for t in order:
            r = by_tag.get(t)
            if r and t != "alpha100" and r.get("dsc_mean"):
                print(f"  {t:<16}{100 * (float(r['dsc_mean']) - b):+.2f}")
        print("\nStage 1 spread across its 12 cells was 1.85 Dice points with 11 of 12 "
              "paired comparisons non-significant; read differences below ~0.5 points "
              "as noise until the paired test says otherwise.")


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="./cache")
    ap.add_argument("--out", default="./results")
    ap.add_argument("--configs", default="all", help="comma-separated, or 'all'")
    ap.add_argument("--backbone", default="small", choices=["small", "base"])
    ap.add_argument("--pt_epochs", type=int, default=50)
    ap.add_argument("--ft_epochs", type=int, default=30)
    ap.add_argument("--pt_batch", type=int, default=64)
    ap.add_argument("--ft_batch", type=int, default=16)
    ap.add_argument("--ft_stride", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--stop_at", default=None)
    ap.add_argument("--max_hours", type=float, default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="one configuration, 1 epoch each, into results/smoke3")
    a = ap.parse_args()

    out_root = Path(a.out, "smoke3" if a.smoke else "step3")
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "step3.csv"
    log = make_logger(out_root / "log.txt")

    jobs = ORDER if a.configs == "all" else [c for c in ORDER if c in a.configs.split(",")]
    if a.smoke:
        jobs = jobs[:1]
        a.pt_epochs = a.ft_epochs = 1

    if a.list:
        d = done_tags(csv_path)
        for t in jobs:
            state = "done" if t in d else ("in progress" if (out_root / t).exists()
                                           else "pending")
            print(f"  {t:<16} {state}")
        summarize(csv_path)
        return

    i, n = map(int, a.shard.split("/"))
    mine = jobs[i::n]
    todo = [t for t in mine if t not in done_tags(csv_path)]
    log(f"Step 3 {'SMOKE ' if a.smoke else ''}| shard {a.shard}: {len(mine)} jobs, "
        f"{len(mine)-len(todo)} done, {len(todo)} to run | development split, "
        f"ViT-{a.backbone}, {a.pt_epochs}pt/{a.ft_epochs}ft ep, stride {a.ft_stride}, "
        f"p={MASK_RATIO}, device {a.device}")

    a.should_stop = make_should_stop(a.stop_at, a.max_hours, log)
    from run_pretrain import TimeUp
    for k, name in enumerate(todo, 1):
        log(f"=== [{k}/{len(todo)}] {name} {full_cfg(name)} ===")
        if a.should_stop and a.should_stop(0):
            log("Deadline reached before starting the next job. Rerun to continue.")
            break
        try:
            row = run_job(name, a, out_root, log)
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
