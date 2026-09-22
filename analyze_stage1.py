"""
analyze_stage1.py
Paired per-patient analysis of the Stage 1 grid (manuscript Sec. 3.8.2, 3.8.5).

Every configuration was evaluated on the same development validation patients, so
configurations can be compared patient by patient. For each comparison this reports the
mean difference in Dice, a 95% bootstrap confidence interval over patients, a Wilcoxon
signed-rank p-value, and how many patients favour each side.

  1. Selected configuration vs each of the other 11 (Holm-corrected p-values)
  2. Design factors, each averaged over its matched pairs:
       fusion     modulated - direct            (6 pairs)
       direction  masked - visible              (6 pairs)
       ratio      0.75 - 0.50, 0.75 - 0.90      (4 pairs each)

The unit of analysis is the patient; the intervals describe uncertainty over which
patients were in the validation set, not over training randomness (single seed).

Usage
  python analyze_stage1.py --results ./results/stage1 --out ./results/stage1
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REGIONS = ["WT", "TC", "ET"]


def load(results_dir):
    """Returns {tag: DataFrame indexed by case_id with dsc_WT, dsc_TC, dsc_ET, dsc_mean}."""
    per = {}
    for f in sorted(Path(results_dir).glob("*/finetune_result.json")):
        rows = json.load(open(f))["per_case"]
        df = pd.DataFrame(rows).set_index("case_id")[[f"dsc_{r}" for r in REGIONS]]
        df["dsc_mean"] = df.mean(axis=1)
        per[f.parent.name] = df
    if not per:
        raise SystemExit(f"no */finetune_result.json under {results_dir}")
    cases = sorted(set.intersection(*[set(d.index) for d in per.values()]))
    if any(len(d) != len(cases) for d in per.values()):
        raise SystemExit("configurations were not evaluated on the same patients")
    return {t: d.loc[cases] for t, d in per.items()}, cases


def paired(diff, n_boot=10000, seed=0):
    """diff: per-patient differences. Returns summary dict."""
    diff = np.asarray(diff, float)
    rng = np.random.default_rng(seed)
    boots = diff[rng.integers(0, len(diff), (n_boot, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    nz = diff[np.abs(diff) > 1e-12]
    p = float(wilcoxon(nz).pvalue) if len(nz) >= 5 else float("nan")
    return {"mean_diff": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_wilcoxon": p, "n_better": int((diff > 1e-12).sum()),
            "n_worse": int((diff < -1e-12).sum()), "n_tied": int((np.abs(diff) <= 1e-12).sum())}


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, same order as input."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def parse(tag):
    fusion, direction, ratio = tag.split("_")
    return fusion, direction, ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="./results/stage1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--metric", default="dsc_mean")
    a = ap.parse_args()
    out = Path(a.out or a.results)
    out.mkdir(parents=True, exist_ok=True)

    per, cases = load(a.results)
    means = {t: d[a.metric].mean() for t, d in per.items()}
    best = max(means, key=means.get)
    print(f"{len(per)} configurations x {len(cases)} validation patients | metric: {a.metric}")
    print(f"Selected (highest mean): {best}  {a.metric} = {means[best]:.4f}\n")

    # 1. selected vs every other configuration
    rows = []
    for t in sorted(means, key=means.get, reverse=True):
        if t == best:
            continue
        r = {"comparison": f"{best} - {t}", **paired(per[best][a.metric] - per[t][a.metric])}
        for reg in REGIONS:
            r[f"mean_diff_{reg}"] = float((per[best][f"dsc_{reg}"] - per[t][f"dsc_{reg}"]).mean())
        rows.append(r)
    vs = pd.DataFrame(rows)
    vs["p_holm"] = holm(vs["p_wilcoxon"])

    print("1. Selected configuration vs each alternative (paired over patients)")
    print(f"{'vs':<28}{'diff':>8}{'95% CI':>20}{'p':>8}{'p_holm':>8}{'better/worse':>14}")
    for _, r in vs.iterrows():
        other = r["comparison"].split(" - ")[1]
        print(f"{other:<28}{100*r.mean_diff:>+7.2f} "
              f"[{100*r.ci_low:+6.2f},{100*r.ci_high:+6.2f}]"
              f"{r.p_wilcoxon:>8.3f}{r.p_holm:>8.3f}{r.n_better:>8}/{r.n_worse:<5}")
    print("   (differences in Dice points; CI excluding 0 = difference unlikely to be chance)\n")

    # 2. design factors, averaged over matched pairs, per patient
    tags = list(per)
    groups = {
        "fusion: modulated - direct": [
            (t, t.replace("modulated", "direct", 1)) for t in tags if t.startswith("modulated")],
        "direction: masked - visible": [
            (t, t.replace("_masked_", "_visible_")) for t in tags if "_masked_" in t],
        "ratio: 0.75 - 0.50": [
            (t, t.replace("p0.75", "p0.50")) for t in tags if t.endswith("p0.75")],
        "ratio: 0.75 - 0.90": [
            (t, t.replace("p0.75", "p0.90")) for t in tags if t.endswith("p0.75")],
    }
    frows = []
    print("2. Design factors (per-patient difference averaged over matched pairs)")
    print(f"{'factor':<30}{'pairs':>6}{'diff':>8}{'95% CI':>20}{'p':>8}{'pairs favouring':>17}")
    for name, pairs in groups.items():
        pairs = [(x, y) for x, y in pairs if y in per]
        diff = np.mean([per[x][a.metric].values - per[y][a.metric].values for x, y in pairs], axis=0)
        s = paired(diff)
        favour = sum(means[x] > means[y] for x, y in pairs)
        frows.append({"factor": name, "pairs": len(pairs), "pairs_favouring": favour, **s})
        print(f"{name:<30}{len(pairs):>6}{100*s['mean_diff']:>+7.2f} "
              f"[{100*s['ci_low']:+6.2f},{100*s['ci_high']:+6.2f}]{s['p_wilcoxon']:>8.3f}"
              f"{favour:>9}/{len(pairs)}")
    factors = pd.DataFrame(frows)

    vs.to_csv(out / "stage1_paired_vs_selected.csv", index=False)
    factors.to_csv(out / "stage1_factor_effects.csv", index=False)
    print(f"\nSaved {out/'stage1_paired_vs_selected.csv'} and {out/'stage1_factor_effects.csv'}")


if __name__ == "__main__":
    main()
