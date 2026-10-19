"""
inspect_sampler.py
Diagnostic for the adaptive sampling objective (manuscript Sec. 3.5.2, Eq. 3.21).

Reads the pre-training histories written by run_pretrain.py and asks whether the
sampler actually concentrated probability mass, as opposed to staying near uniform.

  L_S = -sum_{i in I_m and B} p_i * err_i        (Eq. 3.21)
  L_R = mean_{i in I_m and B} err_i              (Eq. 3.20)

so |L_S| / L_R is the probability mass the sampler places on scored tokens, weighted by
each token's reconstruction error relative to the mean. It is comparable across epochs
even as reconstruction error falls, and it may exceed 1 when the sampler concentrates on
tokens that are harder than average. Under a uniform distribution over tokens of equal
difficulty it equals n_scored / N, where n_scored is the number of masked tokens
containing brain tissue:

  n_scored ~= N * p * rho,  N = 196, rho ~= 0.29  ->  uniform ratio ~= p * rho

  ratio >> uniform  the sampler concentrates on tokens it is scored on
  ratio ~= uniform  the sampler is close to uniform: the objective taught it little
  ratio << uniform  mass sits on tokens that are NOT scored (expected when direction
                    is 'visible', since high-probability tokens are drawn as visible)

Usage
  python inspect_sampler.py --results ./results/stage1 [--rho 0.291]
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="./results/stage1")
    ap.add_argument("--rho", type=float, default=0.291,
                    help="mean fraction of patches containing brain tissue")
    a = ap.parse_args()

    files = sorted(Path(a.results).glob("*/pretrain_result.json"))
    if not files:
        raise SystemExit(f"no */pretrain_result.json under {a.results}")

    print(f"{'configuration':<28}{'L_S first':>10}{'L_S last':>10}"
          f"{'MSE first':>11}{'MSE last':>10}{'mass':>8}{'unif':>7}{'  verdict'}")

    rows = []
    for f in files:
        r = json.load(open(f))
        h, cfg = r["history"], r.get("config", {})
        if not h or h[0].get("sampler_loss") is None:
            continue
        tag = f.parent.name
        ls0, ls1 = h[0]["sampler_loss"], h[-1]["sampler_loss"]
        m0, m1 = h[0]["train_mse"], h[-1]["train_mse"]
        if abs(ls1) < 1e-12 or m1 <= 0:
            continue                                   # baselines with no sampler
        mass = abs(ls1) / m1                           # share of mass on scored tokens
        unif = cfg.get("mask_ratio", 0.75) * a.rho
        ratio = mass / unif
        verdict = ("concentrates on scored" if ratio > 1.3 else
                   "avoids scored" if ratio < 0.7 else "near uniform")
        rows.append((tag, ratio, verdict))
        print(f"{tag:<28}{ls0:>10.5f}{ls1:>10.5f}{m0:>11.5f}{m1:>10.5f}"
              f"{mass:>8.3f}{unif:>7.3f}  {verdict} ({ratio:.2f}x)")

    if rows:
        near = sum(1 for _, _, v in rows if v == "near uniform")
        print(f"\n{len(rows)} configurations with a learned sampler | "
              f"{near} near uniform, "
              f"{sum(1 for _, _, v in rows if v == 'concentrates on scored')} concentrating, "
              f"{sum(1 for _, _, v in rows if v == 'avoids scored')} avoiding")
        if near == len(rows):
            print("Every configuration is near uniform: the objective of Eq. 3.21 provided "
                  "little usable learning signal.")
        else:
            print("Departure from the uniform reference indicates the objective of Eq. 3.21 "
                  "produced a usable learning signal; the sign of the departure follows the "
                  "sampling direction, since 'visible' draws high-probability tokens out of "
                  "the scored set and 'masked' draws them into it.")


if __name__ == "__main__":
    main()
