"""
analyze_seedstudy.py
Is the Stage 2 ranking larger than the noise between identical training runs?

run_seedstudy.py trains the proposed model on Stage 2 fold 1 several times, changing
only the training seed. This script compares those runs against one another patient by
patient, and puts the resulting spread next to the differences Stage 2 reported between
masking strategies on that same fold and those same patients.

The comparison is the point. A paired test between two models answers "given these two
trained networks, do they differ across patients", which is not the question a reader
has. The question is whether re-running the pipeline would reproduce the ranking, and
only a seed replicate can answer it. If two identical runs differ by as much as ours
differs from MAE, the Stage 2 ordering is not attributable to the masking strategy.

Reported:
  1. Each seed pair, as a paired per-patient difference with a bootstrap interval.
  2. The Stage 2 fold-1 differences between models, computed on the same patients.
  3. A verdict comparing the largest seed-pair gap against the model differences.

Usage
  python analyze_seedstudy.py [--seedstudy ./results/seedstudy] [--stage2 ./results/stage2]
"""

import argparse
import itertools
import json
import re
from pathlib import Path

import pandas as pd

from analyze_stage2 import LABEL, paired

REGIONS = ["WT", "TC", "ET"]
FOLD = 1
MODEL = "ours"


def dsc_series(path):
    df = pd.DataFrame(json.load(open(path))["per_case"]).set_index("case_id")
    return df[[f"dsc_{r}" for r in REGIONS]].mean(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seedstudy", default="./results/seedstudy")
    ap.add_argument("--stage2", default="./results/stage2")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out or a.seedstudy)

    # ---------------------------------------------------------------- the seed runs
    runs = {}
    base = Path(a.stage2, f"{MODEL}_f{FOLD}", "finetune_result.json")
    if base.exists():
        runs[42] = dsc_series(base)
    for f in sorted(Path(a.seedstudy).glob(f"{MODEL}_f{FOLD}_s*/finetune_result.json")):
        m = re.search(r"_s(\d+)$", f.parent.name)
        runs[int(m.group(1))] = dsc_series(f)
    if len(runs) < 2:
        raise SystemExit(f"need at least two seed runs, found {sorted(runs)}")

    cases = sorted(set.intersection(*[set(s.index) for s in runs.values()]))
    runs = {k: v.loc[cases] for k, v in runs.items()}
    print(f"{MODEL} on fold {FOLD}: {len(runs)} training seeds, {len(cases)} patients\n")
    print(f"{'seed':<8}{'DSC mean':>10}")
    for s in sorted(runs):
        print(f"{s:<8}{runs[s].mean():>10.4f}")

    # ---------------------------------------------------------------- seed vs seed
    print(f"\nSame configuration, different seed")
    print(f"{'pair':<16}{'diff':>8}{'95% CI':>22}{'p':>10}")
    seed_gaps, rows = [], []
    for x, y in itertools.combinations(sorted(runs), 2):
        r = paired(runs[x] - runs[y])
        ci = "[{:+.2f}, {:+.2f}]".format(100 * r["ci_low"], 100 * r["ci_high"])
        print(f"{f'seed {x} - {y}':<16}{100*r['mean_diff']:>8.2f}{ci:>22}"
              f"{r['p_wilcoxon']:>10.2e}")
        seed_gaps.append(abs(100 * r["mean_diff"]))
        rows.append({"kind": "seed", "contrast": f"seed {x} - seed {y}",
                     "diff_x100": round(100 * r["mean_diff"], 3),
                     "ci_low_x100": round(100 * r["ci_low"], 3),
                     "ci_high_x100": round(100 * r["ci_high"], 3),
                     "p_wilcoxon": r["p_wilcoxon"]})

    # ---------------------------------------------------------------- model vs model
    print(f"\nDifferent masking strategies, same seed (Stage 2, fold {FOLD}, "
          f"these patients)")
    print(f"{'pair':<16}{'diff':>8}{'95% CI':>22}{'p':>10}")
    model_gaps = {}
    for other in ("mae", "adamae", "fixed_sobel"):
        p = Path(a.stage2, f"{other}_f{FOLD}", "finetune_result.json")
        if not p.exists():
            continue
        s = dsc_series(p).reindex(cases)
        if s.isna().any():
            continue
        r = paired(runs[42] - s)
        ci = "[{:+.2f}, {:+.2f}]".format(100 * r["ci_low"], 100 * r["ci_high"])
        print(f"{f'ours - {other}':<16}{100*r['mean_diff']:>8.2f}{ci:>22}"
              f"{r['p_wilcoxon']:>10.2e}")
        model_gaps[other] = abs(100 * r["mean_diff"])
        rows.append({"kind": "model", "contrast": f"ours - {other}",
                     "diff_x100": round(100 * r["mean_diff"], 3),
                     "ci_low_x100": round(100 * r["ci_low"], 3),
                     "ci_high_x100": round(100 * r["ci_high"], 3),
                     "p_wilcoxon": r["p_wilcoxon"]})

    # ---------------------------------------------------------------- the verdict
    worst_seed = max(seed_gaps)
    print(f"\nLargest gap between identical runs: {worst_seed:.2f} Dice points.")
    if model_gaps:
        swamped = [m for m, g in model_gaps.items() if g <= worst_seed]
        if swamped:
            print(f"That is at least as large as the measured difference between ours "
                  f"and {', '.join(LABEL.get(m, m) for m in swamped)}, so those "
                  f"differences cannot be attributed to the masking strategy: changing "
                  f"only the seed moves the score as much.")
        if len(swamped) < len(model_gaps):
            kept = [m for m in model_gaps if m not in swamped]
            print(f"The difference against {', '.join(LABEL.get(m, m) for m in kept)} "
                  f"is larger than the seed spread and survives this check.")
    print("With a handful of seeds this is a rough bound, not a variance estimate; "
          "report it as the range observed, not as a standard error.")

    path = Path(out, "seedstudy_paired.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
