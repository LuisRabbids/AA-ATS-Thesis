"""
analyze_factorial.py
Factorial reading of the Stage 2 comparison (manuscript Sec. 3.8.5).

analyze_stage2.py compares the proposed method against each baseline in turn. That
answers RQ2 and RQ3 as posed, but it understates what the four models actually form:
a complete 2 x 2 design crossing the two components the method introduces.

                        no anatomical guidance    anatomical guidance
  no learned sampler    MAE (random)              Fixed-Sobel
  learned sampler       AdaMAE                    Ours (proposed)

Reading it as a factorial separates the two components, which the pairwise comparisons
confound. Reported here:

  1. All six pairwise comparisons, Holm-corrected within that family, which show
     whether the four models separate into tiers.
  2. The two main effects and their interaction, each as a per-patient contrast:
       learning     ((MAE + Fixed-Sobel) - (AdaMAE + Ours)) / 2
       anatomy      ((Fixed-Sobel + Ours) - (MAE + AdaMAE)) / 2
       interaction  (Ours - AdaMAE) - (Fixed-Sobel - MAE)
  3. The learning effect fold by fold, since a pooled effect that flips sign between
     folds is not evidence of anything.

Every contrast is a within-patient difference, so the same bootstrap and Wilcoxon
procedure used elsewhere applies. The qualification from analyze_stage2.py carries
over: patients held out in the same fold are scored by the same trained models, so
their differences are not mutually independent and the effective sample size is
smaller than the patient count. A pooled interval is therefore read alongside the
per-fold signs, not in place of them.

Usage
  python analyze_factorial.py --results ./results/stage2 [--out ./results/stage2]
"""

import argparse
import itertools
import json
import re
from pathlib import Path

import pandas as pd

from analyze_stage2 import LABEL, holm, load, paired

REGIONS = ["WT", "TC", "ET"]
# (model, has_learned_sampler, has_anatomical_guidance)
CELLS = [("mae", False, False), ("fixed_sobel", False, True),
         ("adamae", True, False), ("ours", True, True)]


def per_fold_means(results_dir):
    out = {}
    for f in sorted(Path(results_dir).glob("*/finetune_result.json")):
        m = re.fullmatch(r"(.+)_f(\d+)", f.parent.name)
        if not m:
            continue
        df = pd.DataFrame(json.load(open(f))["per_case"]).set_index("case_id")
        out.setdefault(int(m.group(2)), {})[m.group(1)] = \
            df[[f"dsc_{r}" for r in REGIONS]].mean(axis=1).mean()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="./results/stage2")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out or a.results)

    per = load(a.results)
    missing = [m for m, _, _ in CELLS if m not in per]
    if missing:
        raise SystemExit(f"missing from {a.results}: {', '.join(missing)}")
    cases = sorted(set.intersection(*[set(d.index) for d in per.values()]))
    g = {m: per[m].loc[cases]["dsc_mean"] for m in per}
    print(f"{len(CELLS)} models x {len(cases)} patients scored by all of them\n")

    # ---------------------------------------------------------------- 1. all pairs
    pairs = list(itertools.combinations([m for m, _, _ in CELLS], 2))
    res = [(x, y, paired(g[x] - g[y])) for x, y in pairs]
    ps = holm([r["p_wilcoxon"] for _, _, r in res])
    print("All six pairwise comparisons, Holm-corrected within this family")
    print(f"{'comparison':<34}{'diff':>8}{'95% CI':>22}{'p_holm':>11}  verdict")
    rows = []
    for (x, y, r), p in zip(res, ps):
        ci = "[{:+.2f}, {:+.2f}]".format(100 * r["ci_low"], 100 * r["ci_high"])
        verdict = "significant" if p < 0.05 else "no difference"
        print(f"{LABEL[x] + ' - ' + LABEL[y]:<34}{100*r['mean_diff']:>8.2f}{ci:>22}"
              f"{float(p):>11.2e}  {verdict}")
        rows.append({"kind": "pair", "contrast": f"{x} - {y}",
                     "diff_x100": round(100 * r["mean_diff"], 3),
                     "ci_low_x100": round(100 * r["ci_low"], 3),
                     "ci_high_x100": round(100 * r["ci_high"], 3),
                     "p_wilcoxon": r["p_wilcoxon"], "p_holm": float(p),
                     "n_better": r["n_better"], "n_worse": r["n_worse"]})

    # ---------------------------------------------------------------- 2. the cells
    print(f"\n2 x 2 cell means (Dice)")
    print(f"{'':<22}{'no anatomy':>12}{'anatomy':>12}")
    for learned, label in [(False, "no learned sampler"), (True, "learned sampler")]:
        cells = {an: g[m].mean() for m, lr, an in CELLS if lr == learned}
        print(f"{label:<22}{cells[False]:>12.4f}{cells[True]:>12.4f}")

    # ---------------------------------------------------------------- 3. effects
    learn = (g["mae"] + g["fixed_sobel"]) / 2 - (g["adamae"] + g["ours"]) / 2
    anat = (g["fixed_sobel"] + g["ours"]) / 2 - (g["mae"] + g["adamae"]) / 2
    inter = (g["ours"] - g["adamae"]) - (g["fixed_sobel"] - g["mae"])
    print(f"\nMain effects and interaction, pooled over {len(cases)} patients")
    print(f"{'contrast':<34}{'diff':>8}{'95% CI':>22}{'p':>11}")
    for name, d in [("no learned sampler - learned", learn),
                    ("anatomical guidance - none", anat),
                    ("interaction", inter)]:
        r = paired(d)
        ci = "[{:+.2f}, {:+.2f}]".format(100 * r["ci_low"], 100 * r["ci_high"])
        print(f"{name:<34}{100*r['mean_diff']:>8.2f}{ci:>22}{r['p_wilcoxon']:>11.2e}")
        rows.append({"kind": "effect", "contrast": name,
                     "diff_x100": round(100 * r["mean_diff"], 3),
                     "ci_low_x100": round(100 * r["ci_low"], 3),
                     "ci_high_x100": round(100 * r["ci_high"], 3),
                     "p_wilcoxon": r["p_wilcoxon"], "p_holm": "",
                     "n_better": r["n_better"], "n_worse": r["n_worse"]})

    # ---------------------------------------------------------------- 4. per fold
    folds = per_fold_means(a.results)
    print(f"\nThe learning effect fold by fold")
    print(f"{'fold':<6}{'MAE':>9}{'FixSobel':>10}{'AdaMAE':>9}{'Ours':>9}"
          f"{'no-learn':>10}{'learned':>9}{'effect':>9}")
    signs = []
    for k in sorted(folds):
        d = folds[k]
        nl = (d["mae"] + d["fixed_sobel"]) / 2
        lr = (d["adamae"] + d["ours"]) / 2
        signs.append(nl - lr)
        print(f"{k:<6}{d['mae']:>9.4f}{d['fixed_sobel']:>10.4f}{d['adamae']:>9.4f}"
              f"{d['ours']:>9.4f}{nl:>10.4f}{lr:>9.4f}{100*(nl-lr):>+9.2f}")
    same = sum(s > 0 for s in signs)
    print(f"\nThe learned sampler costs accuracy in {same} of {len(signs)} folds "
          f"({', '.join(f'{100*s:+.2f}' for s in signs)} Dice points).")
    if same != len(signs):
        print("The effect changes sign between folds, so the pooled interval above "
              "overstates the evidence.")

    path = Path(out, "stage2_factorial.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
