"""
plot_results.py
Turns results/reconstruction.csv and results/logs/*.json into (a) the Table 3.1 grid
and (b) presentation-ready figures.

  python plot_results.py --results ./results
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LABELS = {
    "random": "MAE (random)",
    "adaptive": "AdaMAE (adaptive)",
    "hard_anat": "Hard anatomical (rule-based)",
    "anatomical": "AA-AdaMAE (ours)",
}
COLORS = {
    "random": "#9aa5b1",
    "adaptive": "#4c78a8",
    "hard_anat": "#f2a900",
    "anatomical": "#c0392b",
}
ORDER = ["random", "adaptive", "hard_anat", "anatomical"]


def table_31(df, out_dir):
    """Pivot into the Table 3.1 layout: rows = model, cols = masking ratio."""
    piv = (
        df.sort_values("final_val_mse")
        .drop_duplicates(["model", "mask_ratio"])
        .pivot(index="model", columns="mask_ratio", values="final_val_mse")
        .reindex([m for m in ORDER if m in df["model"].unique()])
    )
    piv.index = [LABELS.get(i, i) for i in piv.index]
    piv.columns = [f"{int(c*100)}%" for c in piv.columns]
    piv.to_csv(Path(out_dir, "table_3_1_reconstruction.csv"))
    print("\nTable 3.1 - Reconstruction Loss (MSE, lower is better)\n")
    print(piv.round(5).to_string())
    return piv


def plot_ratio_chart(df, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
    ratios = sorted(df["mask_ratio"].unique())
    width = 0.8 / max(len(df["model"].unique()), 1)

    for i, m in enumerate([m for m in ORDER if m in df["model"].unique()]):
        sub = df[df["model"] == m].groupby("mask_ratio")["final_val_mse"].min()
        xs = [ratios.index(r) + i * width - 0.4 + width / 2 for r in sub.index]
        bars = ax.bar(xs, sub.values, width * 0.92, label=LABELS.get(m, m),
                      color=COLORS.get(m, None), edgecolor="white", linewidth=0.8)
        for b, v in zip(bars, sub.values):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center",
                    va="bottom", fontsize=6.5, rotation=90)

    ax.set_xticks(range(len(ratios)))
    ax.set_xticklabels([f"{int(r*100)}%" for r in ratios])
    ax.set_xlabel("Masking ratio")
    ax.set_ylabel("Masked-patch MSE  (lower is better)")
    ax.set_title("Reconstruction loss by masking strategy and ratio",
                 fontweight="bold", pad=32)
    ax.set_ylim(0, df["final_val_mse"].max() * 1.28)  # headroom for value labels
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, 1.005), columnspacing=1.2, handlelength=1.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(Path(out_dir, "fig_reconstruction_by_ratio.png"))
    print(f"wrote {Path(out_dir,'fig_reconstruction_by_ratio.png')}")


def plot_curves(results_dir, out_dir, ratio=0.75):
    logs = sorted(Path(results_dir, "logs").glob("*.json"))
    if not logs:
        return
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
    plotted = 0
    for lg in logs:
        with open(lg) as f:
            d = json.load(f)
        a, h = d["args"], d["history"]
        if abs(a["mask_ratio"] - ratio) > 1e-6:
            continue
        m = a["mask_mode"]
        ax.plot([x["epoch"] for x in h], [x["val_mse"] for x in h],
                label=LABELS.get(m, m), color=COLORS.get(m), linewidth=2)
        plotted += 1
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Pretraining epoch")
    ax.set_ylabel("Validation masked-patch MSE")
    ax.set_title(f"Reconstruction convergence at {int(ratio*100)}% masking", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(Path(out_dir, f"fig_convergence_p{int(ratio*100)}.png"))
    print(f"wrote {Path(out_dir, f'fig_convergence_p{int(ratio*100)}.png')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="./results")
    ap.add_argument("--curve_ratio", type=float, default=0.75)
    args = ap.parse_args()

    csv_path = Path(args.results, "reconstruction.csv")
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found - run run_pretrain.py first.")
    df = pd.read_csv(csv_path)

    out = Path(args.results, "figures")
    out.mkdir(parents=True, exist_ok=True)

    table_31(df, out)
    plot_ratio_chart(df, out)
    plot_curves(args.results, out, args.curve_ratio)


if __name__ == "__main__":
    main()
