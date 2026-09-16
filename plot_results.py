"""
plot_results.py  (partial-sweep tolerant)

Turns results/reconstruction.csv and results/logs/*.json into (a) the Table 3.1 grid
and (b) presentation-ready figures.

Missing (strategy, ratio) cells render as "—" instead of NaN, so the table and
charts stay presentable while the sweep is still incomplete.

  python plot_results.py --results ./results
  python plot_results.py --results ./results --only_ratio 0.75   # like-for-like slide
  python plot_results.py --results ./results --subtitle "50 patients · 20 epochs · fold 1"
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
    "hard_anat": "Rule-based hard anatomical",
    "anatomical": "AA-AdaMAE (ours)",
}
COLORS = {
    "random": "#9aa5b1",
    "adaptive": "#4c78a8",
    "hard_anat": "#f2a900",
    "anatomical": "#c0392b",
}
ORDER = ["random", "adaptive", "hard_anat", "anatomical"]


def dedupe(df):
    """Keep the best (lowest MSE) row per (strategy, ratio).

    A duplicate is expected if you ran a single configuration before launching
    the sweep — both write to the same CSV.
    """
    return df.sort_values("final_val_mse").drop_duplicates(["model", "mask_ratio"])


def table_31(df, out_dir):
    piv = (
        dedupe(df)
        .pivot(index="model", columns="mask_ratio", values="final_val_mse")
        .reindex([m for m in ORDER if m in df["model"].unique()])
    )
    piv.index = [LABELS.get(i, i) for i in piv.index]
    piv.columns = [f"{int(c*100)}%" for c in piv.columns]
    piv.to_csv(Path(out_dir, "table_3_1_reconstruction.csv"))

    disp = piv.copy().astype(object)
    for c in disp.columns:
        disp[c] = piv[c].map(lambda v: "—" if pd.isna(v) else f"{v:.5f}")

    print("\nTable 3.1 — Reconstruction Loss (MSE, lower is better)\n")
    print(disp.to_string())
    n_missing = int(piv.isna().sum().sum())
    if n_missing:
        print(f"\n  {n_missing} cell(s) still missing — partial sweep.")
    return piv


def plot_ratio_chart(df, out_dir, subtitle=None, only_ratio=None):
    d = dedupe(df)
    if only_ratio is not None:
        d = d[abs(d["mask_ratio"] - only_ratio) < 1e-6]
        if d.empty:
            print(f"no runs at ratio {only_ratio}, skipping chart")
            return

    ratios = sorted(d["mask_ratio"].unique())
    models = [m for m in ORDER if m in d["model"].unique()]

    # single ratio -> one bar per strategy; multi-ratio -> grouped bars
    single = len(ratios) == 1
    fig, ax = plt.subplots(figsize=(8.4, 5.0), dpi=200)

    if single:
        vals = [d[d["model"] == m]["final_val_mse"].min() for m in models]
        bars = ax.bar(range(len(models)), vals, 0.58,
                      color=[COLORS[m] for m in models],
                      edgecolor="white", linewidth=1.0)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([LABELS[m].replace(" (", "\n(") for m in models],
                           fontsize=9)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.5f}", ha="center",
                    va="bottom", fontsize=9, fontweight="bold")
        ax.set_xlabel(f"Masking ratio {int(ratios[0]*100)}%", fontsize=10)
        ax.set_ylim(0, max(vals) * 1.20)
    else:
        width = 0.8 / max(len(models), 1)
        for i, m in enumerate(models):
            sub = d[d["model"] == m].groupby("mask_ratio")["final_val_mse"].min()
            xs = [ratios.index(r) + i * width - 0.4 + width / 2 for r in sub.index]
            bars = ax.bar(xs, sub.values, width * 0.92, label=LABELS[m],
                          color=COLORS[m], edgecolor="white", linewidth=0.8)
            for b, v in zip(bars, sub.values):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center",
                        va="bottom", fontsize=6.5, rotation=90)
        ax.set_xticks(range(len(ratios)))
        ax.set_xticklabels([f"{int(r*100)}%" for r in ratios])
        ax.set_xlabel("Masking ratio", fontsize=10)
        ax.set_ylim(0, d["final_val_mse"].max() * 1.28)
        ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower center",
                  bbox_to_anchor=(0.5, 1.005), columnspacing=1.2, handlelength=1.2)

    ax.set_ylabel("Masked-patch MSE  (lower is better)", fontsize=10)
    title = "Reconstruction loss by masking strategy"
    ax.set_title(title, fontweight="bold", fontsize=13,
                 pad=30 if not single else 14)
    if subtitle:
        ax.text(0.5, 1.005 if single else 1.115, subtitle, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=9, color="#5a5a5a")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    fig.tight_layout()

    name = ("fig_reconstruction_p%d.png" % int(ratios[0] * 100)) if single \
        else "fig_reconstruction_by_ratio.png"
    fig.savefig(Path(out_dir, name))
    plt.close(fig)
    print(f"wrote {Path(out_dir, name)}")


def plot_curves(results_dir, out_dir, ratio=0.75, subtitle=None):
    logs = sorted(Path(results_dir, "logs").glob("*.json"))
    if not logs:
        return
    fig, ax = plt.subplots(figsize=(8.4, 5.0), dpi=200)
    plotted = 0
    for lg in logs:
        with open(lg) as f:
            d = json.load(f)
        a, h = d["args"], d["history"]
        if abs(a["mask_ratio"] - ratio) > 1e-6:
            continue
        m = a["mask_mode"]
        ax.plot([x["epoch"] for x in h], [x["val_mse"] for x in h],
                label=LABELS.get(m, m), color=COLORS.get(m), linewidth=2.2)
        plotted += 1
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Pretraining epoch", fontsize=10)
    ax.set_ylabel("Validation masked-patch MSE", fontsize=10)
    ax.set_title(f"Reconstruction convergence at {int(ratio*100)}% masking",
                 fontweight="bold", fontsize=13, pad=14)
    if subtitle:
        ax.text(0.5, 1.005, subtitle, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=9, color="#5a5a5a")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    out = Path(out_dir, f"fig_convergence_p{int(ratio*100)}.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="./results")
    ap.add_argument("--curve_ratio", type=float, default=0.75)
    ap.add_argument("--only_ratio", type=float, default=None,
                    help="Restrict the bar chart to one masking ratio")
    ap.add_argument("--subtitle", default=None,
                    help='e.g. "50 patients · 20 epochs · fold 1"')
    args = ap.parse_args()

    csv_path = Path(args.results, "reconstruction.csv")
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found — run run_pretrain.py first.")
    df = pd.read_csv(csv_path)

    out = Path(args.results, "figures")
    out.mkdir(parents=True, exist_ok=True)

    table_31(df, out)
    plot_ratio_chart(df, out, args.subtitle, args.only_ratio)
    if args.only_ratio is None:
        plot_ratio_chart(df, out, args.subtitle, args.curve_ratio)
    plot_curves(args.results, out, args.curve_ratio, args.subtitle)


if __name__ == "__main__":
    main()
