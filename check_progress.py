"""
check_progress.py
Inspect a partial RQ1 sweep and report what is usable for the results slide.

Run this right after you Ctrl+C the sweep:

    python check_progress.py --results ./results

It reports which (strategy, ratio) cells completed, flags duplicate rows, and
tells you which comparison is currently defensible to present.
"""

import argparse
from pathlib import Path

import pandas as pd

LABELS = {
    "random": "MAE (random)",
    "adaptive": "AdaMAE (adaptive)",
    "hard_anat": "Rule-based hard anatomical",
    "anatomical": "AA-AdaMAE (ours)",
}
ORDER = ["random", "adaptive", "hard_anat", "anatomical"]
ALL_RATIOS = [0.50, 0.75, 0.80]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="./results")
    args = ap.parse_args()

    csv_path = Path(args.results, "reconstruction.csv")
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found — no completed runs yet.")

    df = pd.read_csv(csv_path)
    print(f"\n{len(df)} completed run(s) in {csv_path}\n")

    # ---- duplicates ------------------------------------------------------------
    dupes = df[df.duplicated(["model", "mask_ratio"], keep=False)]
    if len(dupes):
        print("NOTE: duplicate (strategy, ratio) rows found — expected if you ran a")
        print("      single configuration before starting the sweep. Plotting keeps")
        print("      the better MSE per cell.\n")
        for (m, r), grp in dupes.groupby(["model", "mask_ratio"]):
            vals = ", ".join(f"{v:.5f}" for v in grp["final_val_mse"])
            print(f"      {m} @ {int(r*100)}%  ->  {vals}")
        print()

    # ---- coverage grid ---------------------------------------------------------
    have = {(r["model"], round(r["mask_ratio"], 2)) for _, r in df.iterrows()}
    print("Coverage grid  (X = done, . = missing)\n")
    print("  " + "strategy".ljust(30) + "".join(f"{int(r*100)}%".rjust(7)
                                                for r in ALL_RATIOS))
    for m in ORDER:
        cells = "".join(("X" if (m, r) in have else ".").rjust(7) for r in ALL_RATIOS)
        print("  " + LABELS[m].ljust(30) + cells)

    # ---- which ratios are fully covered ---------------------------------------
    print()
    complete_ratios = [r for r in ALL_RATIOS
                       if all((m, r) in have for m in ORDER)]
    present_models = [m for m in ORDER if any((m, r) in have for r in ALL_RATIOS)]

    if complete_ratios:
        rl = ", ".join(f"{int(r*100)}%" for r in complete_ratios)
        print(f"USABLE: all four strategies completed at {rl}.")
        print("        Present that ratio as a like-for-like comparison:")
        print(f"          python plot_results.py --results {args.results} "
              f"--only_ratio {complete_ratios[0]:.2f}")
    else:
        print("NOT YET COMPLETE: no masking ratio has all four strategies.\n")

        # best partial comparison = most strategies at one ratio, must include ours
        scored = []
        for r in ALL_RATIOS:
            ms = [m for m in ORDER if (m, r) in have]
            if "anatomical" in ms and len(ms) >= 2:
                scored.append((len(ms), r, ms))
        scored.sort(reverse=True)

        if scored:
            n, r, ms = scored[0]
            print(f"BEST AVAILABLE COMPARISON: {int(r*100)}% masking, "
                  f"{n} of 4 strategies")
            print("        " + "  vs.  ".join(LABELS[m] for m in ms))
            print(f"\n          python plot_results.py --results {args.results} "
                  f"--only_ratio {r:.2f}")
            missing_here = [LABELS[m] for m in ORDER if m not in ms]
            if missing_here:
                print(f"\n        Say on the slide that {', '.join(missing_here)} "
                      "is still running.")
            print("\n        To complete this ratio, run only what is missing:")
            for m in ORDER:
                if m not in ms:
                    print(f"          python run_pretrain.py --cache ./cache "
                          f"--mask_mode {m} --mask_ratio {r:.2f} --epochs 20")
        elif "anatomical" not in present_models:
            print("        WARNING: the proposed method has no completed run.")
            print("        Run it before presenting anything:")
            print("          python run_pretrain.py --cache ./cache "
                  "--mask_mode anatomical --mask_ratio 0.75 --epochs 20")
        else:
            print("        Only the proposed method has completed. Run at least")
            print("        one baseline at the same ratio before presenting:")
            print("          python run_pretrain.py --cache ./cache "
                  "--mask_mode random --mask_ratio 0.75 --epochs 20")

    # ---- run settings sanity ---------------------------------------------------
    print()
    for col, name in (("epochs", "epochs"), ("fold", "fold")):
        if col in df.columns and df[col].nunique() > 1:
            print(f"WARNING: runs used different {name} values "
                  f"({sorted(df[col].unique())}) — not a fair comparison.")

    if "minutes" in df.columns:
        print(f"Total compute so far: {df['minutes'].sum():.1f} min "
              f"({df['minutes'].sum()/60:.1f} h)")

    print("\nPer-run detail:")
    cols = [c for c in ["model", "mask_ratio", "epochs", "final_val_mse",
                        "best_val_mse", "minutes"] if c in df.columns]
    print(df[cols].sort_values(["mask_ratio", "final_val_mse"]).to_string(index=False))
    print()


if __name__ == "__main__":
    main()
