"""
analyze_stage2.py
Paired per-patient analysis of the Stage 2 comparison (manuscript Sec. 3.8.5).

Every model is evaluated on every case in the folds that were run, so the proposed
method and each baseline can be compared case by case. Per-patient differences are
pooled across folds, which gives one paired observation per case regardless of the
number of folds.

Reports, for the proposed method against each baseline:
  mean difference in Dice, a 95% bootstrap confidence interval over cases, a Wilcoxon
  signed-rank p-value Holm-corrected across the baselines, and how many cases favour
  each side -- for the mean over subregions and for WT, TC and ET separately.

Qualification: cases held out in the same fold are evaluated by the same pair of models,
so their paired differences are not mutually independent and the effective sample size
is smaller than the case count. The intervals describe variation across patients, not
across training runs; the design uses a single seed.

Usage
  python analyze_stage2.py --results ./results/stage2 [--out ./results/stage2]
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REGIONS = ["WT", "TC", "ET"]
METRICS = ["dsc_mean"] + [f"dsc_{r}" for r in REGIONS]
LABEL = {"ours": "Ours", "mae": "MAE (random)", "adamae": "AdaMAE",
         "fixed_sobel": "Fixed-Sobel"}
DISPLAY = ["mae", "adamae", "fixed_sobel", "ours"]


def load(results_dir):
    """Returns {model: DataFrame indexed by case_id}, pooling folds."""
    per = {}
    for f in sorted(Path(results_dir).glob("*/finetune_result.json")):
        m = re.fullmatch(r"(.+)_f(\d+)", f.parent.name)
        if not m:
            print(f"  [skip] {f.parent.name}: not <model>_f<fold>")
            continue
        model = m.group(1)
        df = pd.DataFrame(json.load(open(f))["per_case"]).set_index("case_id")
        df = df[[f"dsc_{r}" for r in REGIONS]]
        df["dsc_mean"] = df.mean(axis=1)
        per.setdefault(model, []).append(df)
    if not per:
        raise SystemExit(f"no */finetune_result.json under {results_dir}")
    out = {k: pd.concat(v).sort_index() for k, v in per.items()}
    for k, v in out.items():
        if v.index.duplicated().any():
            raise SystemExit(f"{k}: a case appears in more than one fold")
    return out


def paired(diff, n_boot=10000, seed=0):
    diff = np.asarray(diff, float)
    rng = np.random.default_rng(seed)
    boots = diff[rng.integers(0, len(diff), (n_boot, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    nz = diff[np.abs(diff) > 1e-12]
    p = float(wilcoxon(nz).pvalue) if len(nz) >= 5 else float("nan")
    return {"mean_diff": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_wilcoxon": p, "n_better": int((diff > 1e-12).sum()),
            "n_worse": int((diff < -1e-12).sum())}


def holm(pvals):
    p = np.asarray(pvals, float)
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="./results/stage2")
    ap.add_argument("--out", default=None)
    ap.add_argument("--reference", default="ours")
    a = ap.parse_args()
    out = Path(a.out or a.results)
    out.mkdir(parents=True, exist_ok=True)

    per = load(a.results)
    if a.reference not in per:
        raise SystemExit(f"reference '{a.reference}' not among {list(per)}")
    cases = sorted(set.intersection(*[set(d.index) for d in per.values()]))
    per = {k: d.loc[cases] for k, d in per.items()}
    print(f"{len(per)} models x {len(cases)} cases evaluated by all of them\n")

    print(f"{'model':<16}" + "".join(f"{m.replace('dsc_',''):>10}" for m in METRICS))
    for k in [m for m in DISPLAY if m in per] + [m for m in per if m not in DISPLAY]:
        print(f"{LABEL.get(k,k):<16}" + "".join(f"{per[k][m].mean():>10.4f}" for m in METRICS))

    others = [m for m in DISPLAY if m in per and m != a.reference]
    others += [m for m in per if m not in DISPLAY and m != a.reference]

    rows = []
    for m in others:
        r = {"comparison": f"{a.reference} - {m}"}
        for metric in METRICS:
            s = paired(per[a.reference][metric] - per[m][metric])
            for k, v in s.items():
                r[f"{metric}_{k}" if metric != "dsc_mean" else k] = v
        rows.append(r)
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df["p_wilcoxon"])

    print(f"\nPaired over {len(cases)} cases - {LABEL.get(a.reference, a.reference)} "
          f"minus each baseline, in Dice points")
    print(f"{'vs':<16}{'diff':>8}{'95% CI':>20}{'p':>9}{'p_holm':>9}{'better/worse':>15}")
    for _, r in df.iterrows():
        m = r["comparison"].split(" - ")[1]
        print(f"{LABEL.get(m,m):<16}{100*r.mean_diff:>+7.2f} "
              f"[{100*r.ci_low:+6.2f},{100*r.ci_high:+6.2f}]"
              f"{r.p_wilcoxon:>9.2e}{r.p_holm:>9.2e}{r.n_better:>8}/{r.n_worse:<6}")
    print("   negative = the baseline is ahead; CI excluding 0 = unlikely to be chance")

    print(f"\nBy subregion ({LABEL.get(a.reference, a.reference)} minus baseline)")
    print(f"{'vs':<16}" + "".join(f"{r:>22}" for r in REGIONS))
    for _, r in df.iterrows():
        m = r["comparison"].split(" - ")[1]
        cells = "".join(
            f"{100*r[f'dsc_{reg}_mean_diff']:>+8.2f} "
            f"[{100*r[f'dsc_{reg}_ci_low']:+5.2f},{100*r[f'dsc_{reg}_ci_high']:+5.2f}]"
            for reg in REGIONS)
        print(f"{LABEL.get(m,m):<16}{cells}")

    df.to_csv(out / "stage2_paired.csv", index=False)
    pd.DataFrame({LABEL.get(k, k): {m.replace("dsc_", ""): per[k][m].mean()
                                    for m in METRICS} for k in per}).T.to_csv(
        out / "stage2_means.csv")
    print(f"\nSaved {out/'stage2_paired.csv'} and {out/'stage2_means.csv'}")


if __name__ == "__main__":
    main()
