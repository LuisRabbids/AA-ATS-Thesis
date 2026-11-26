"""
analyze_step3.py
Paired per-patient analysis of the step-3 follow-up configurations.

Every step-3 configuration and the Stage 1 reference cell 'modulated_masked_p0.75'
(alpha = 1.0, Eq. 3.21 objective) were fine-tuned and evaluated on the same development
validation patients, so they can be compared patient by patient. For each configuration
this reports the mean Dice difference against that reference, a 95% bootstrap confidence
interval over patients, a Wilcoxon signed-rank p-value, and how many patients favour
each side. P-values are Holm-corrected across the configurations compared.

The unit of analysis is the patient. The intervals describe uncertainty over which
patients were in the development validation set, not over training randomness
(single seed), so a configuration that merely lands inside the Stage 1 noise band is
not evidence of an improvement.

Usage
  python analyze_step3.py --step3 ./results/step3 --stage1 ./results/stage1
"""

import argparse
from pathlib import Path

import pandas as pd

from analyze_stage1 import holm, load, paired

REFERENCE = "modulated_masked_p0.75"
ORDER = ["alpha000", "alpha010", "alpha025", "alpha050",
         "logp_lam005", "logp_lam05", "logp_ent01"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step3", default="./results/step3")
    ap.add_argument("--stage1", default="./results/stage1")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out or a.step3)
    out.mkdir(parents=True, exist_ok=True)

    per3, cases3 = load(a.step3)
    per1, _ = load(a.stage1)
    if REFERENCE not in per1:
        raise SystemExit(f"{REFERENCE} not found under {a.stage1}")
    ref = per1[REFERENCE]

    shared = sorted(set(ref.index) & set(cases3))
    if len(shared) < len(cases3):
        print(f"note: {len(cases3) - len(shared)} step-3 patients are absent from the "
              f"Stage 1 reference and are dropped")
    ref = ref.loc[shared]
    per3 = {t: d.loc[shared] for t, d in per3.items()}

    print(f"{len(shared)} development validation patients, reference {REFERENCE} "
          f"(DSC {ref['dsc_mean'].mean():.4f})\n")

    tags = [t for t in ORDER if t in per3] + [t for t in sorted(per3) if t not in ORDER]
    rows = []
    for t in tags:
        r = paired(per3[t]["dsc_mean"] - ref["dsc_mean"])
        rows.append({"config": t, "dsc_mean": round(per3[t]["dsc_mean"].mean(), 5),
                     **{k: round(v, 5) if isinstance(v, float) else v
                        for k, v in r.items()}})
    if rows:
        for row, p in zip(rows, holm([r["p_wilcoxon"] for r in rows])):
            row["p_holm"] = round(float(p), 5)

    df = pd.DataFrame(rows)
    print(f"{'config':<14}{'DSC':>9}{'diff x100':>11}{'95% CI x100':>20}"
          f"{'p_holm':>9}{'better/worse':>15}")
    for r in rows:
        ci = "[{:+.2f}, {:+.2f}]".format(100 * r["ci_low"], 100 * r["ci_high"])
        bw = "{}/{}".format(r["n_better"], r["n_worse"])
        print(f"{r['config']:<14}{r['dsc_mean']:>9.4f}{100*r['mean_diff']:>11.2f}"
              f"{ci:>20}{r['p_holm']:>9.4f}{bw:>15}")

    path = out / "step3_paired.csv"
    df.to_csv(path, index=False)
    print(f"\nwrote {path}")
    print("A configuration counts as an improvement only if its interval excludes zero "
          "after correction; anything inside +/- 0.5 Dice points is within the spread "
          "Stage 1 measured across unrelated design choices.")
    print(f"With {len(shared)} patients this comparison can only resolve differences of "
          f"roughly a Dice point or more, so a promising-looking configuration here is a "
          f"candidate for the experimental partition, not a result.")


if __name__ == "__main__":
    main()
