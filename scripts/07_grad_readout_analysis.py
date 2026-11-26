"""Offline analysis of the gradient readout probe (script 03). No Tinker calls.

Reads artifacts/grad_readout/per_layer_grad_norms.json, which holds the Frobenius norm of the
LoRA B gradient per layer and module type for 3 data seeds (eps = 1e3 setting), and writes
  - a per-layer table (mean over the 3 seeds and the seed range), printed and saved as CSV
  - artifacts/grad_readout/per_layer_grad_norms.png
  - artifacts/grad_readout/analysis_summary.json with the boundary and trend statistics

Boundary statistics, all on log gradient norms:
  - loop/prelude and loop/coda: mean over layers 6-17 divided by mean over 0-5 or 18-23
    (script 03's "ratio" pools prelude and coda in one denominator).
  - adjacent steps: log(v[l+1] / v[l]) at the two boundaries (5->6, 17->18) compared with the
    20 non-boundary adjacent steps of the same module type.
  - loop offset: fit log v = a + b*layer + c*looped on all 24 layers (and with a quadratic depth
    term). exp(c) is the looped/unlooped factor after removing a smooth depth trend.

Usage: .venv/bin/python scripts/07_grad_readout_analysis.py
"""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mlstyle import mpl as ms

OUT = Path("artifacts/grad_readout")
RUNS = ("c_eps1e3", "c_eps1e3_data1", "c_eps1e3_data2")
LOOP = np.arange(6, 18)
PRE, CODA = np.arange(0, 6), np.arange(18, 24)
L = np.arange(24)
T975 = {21: 2.080, 20: 2.086}  # Student-t 0.975 quantiles for the regression residual df

d = json.loads((OUT / "per_layer_grad_norms.json").read_text())
MODS = list(d[RUNS[0]]["per_layer"])
V = np.array([[d[r]["per_layer"][m] for m in MODS] for r in RUNS], dtype=float)  # seed x module x layer
mean, lo, hi = V.mean(0), V.min(0), V.max(0)
short = {m: m.split(".")[-1] for m in MODS}


def loop_offset(y, quad=False):
    """exp(c) and its 95% interval from log y = a + b*l (+ q*l^2) + c*looped."""
    cols = [np.ones(24), L, np.isin(L, LOOP).astype(float)]
    if quad:
        cols.insert(2, (L - 11.5) ** 2)
    X = np.stack(cols, 1)
    ly = np.log(y)
    beta, *_ = np.linalg.lstsq(X, ly, rcond=None)
    res = ly - X @ beta
    df = 24 - X.shape[1]
    se = np.sqrt(res @ res / df * np.linalg.inv(X.T @ X)[-1, -1])
    c = beta[-1]
    return [float(np.exp(c)), float(np.exp(c - T975[df] * se)), float(np.exp(c + T975[df] * se))]


summary = {}
for i, m in enumerate(MODS):
    per_seed = lambda f: [float(f(V[s, i])) for s in range(len(RUNS))]
    steps = np.diff(np.log(mean[i]))                       # step l -> l+1 at index l
    null = np.abs(np.delete(steps, [5, 17]))               # 20 non-boundary steps
    summary[m] = dict(
        ratio_pooled=per_seed(lambda v: v[LOOP].mean() / np.r_[v[PRE], v[CODA]].mean()),
        loop_over_prelude=per_seed(lambda v: v[LOOP].mean() / v[PRE].mean()),
        loop_over_coda=per_seed(lambda v: v[LOOP].mean() / v[CODA].mean()),
        j6=per_seed(lambda v: v[6] / v[5]),
        j18=per_seed(lambda v: v[18] / v[17]),
        j6_rank_among_steps=int((null >= abs(steps[5])).sum()),    # how many ordinary steps are as large
        j18_rank_among_steps=int((null >= abs(steps[17])).sum()),
        median_abs_ordinary_step=float(np.exp(np.median(null))),
        loop_offset_linear=loop_offset(mean[i]),
        loop_offset_quadratic=loop_offset(mean[i], quad=True),
        seed_range_pct_median=float(np.median((hi[i] - lo[i]) / mean[i] * 100)),
    )
summary["_lm_head"] = [d[r]["other"]["lm_head"] for r in RUNS]
(OUT / "analysis_summary.json").write_text(json.dumps(summary, indent=1))

# ---- compact printout --------------------------------------------------------------------------
with open(OUT / "per_layer_table.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["layer"] + [f"{short[m]}_{k}" for m in MODS for k in ("mean", "min", "max")])
    for l in L:
        w.writerow([l] + [f"{x:.2f}" for i in range(len(MODS)) for x in (mean[i, l], lo[i, l], hi[i, l])])
print("Mean |g_B| over 3 seeds, (max-min)/mean in % after the slash")
print("L  " + " ".join(f"{short[m]:>10s}" for m in MODS))
for l in L:
    tag = "*" if l in LOOP else " "
    print(f"{l:2d}{tag}" + " ".join(f"{mean[i, l]:6.0f}/{(hi[i, l] - lo[i, l]) / mean[i, l] * 100:3.0f}"
                                     for i in range(len(MODS))))
fmt = lambda xs: "-".join(f"{x:.2f}" for x in (min(xs), max(xs)))
print("\nmodule      pooled     loop/pre   loop/coda  j6         j18        #steps>=|j6| #>=|j18| offset_lin (95%)    offset_quad")
for m in MODS:
    s = summary[m]
    ol, oq = s["loop_offset_linear"], s["loop_offset_quadratic"]
    print(f"{short[m]:10s}  {fmt(s['ratio_pooled']):9s}  {fmt(s['loop_over_prelude']):9s}  {fmt(s['loop_over_coda']):9s}  "
          f"{fmt(s['j6']):9s}  {fmt(s['j18']):9s}  {s['j6_rank_among_steps']:2d}/20        {s['j18_rank_among_steps']:2d}/20   "
          f"{ol[0]:.2f} ({ol[1]:.2f}-{ol[2]:.2f})  {oq[0]:.2f} ({oq[1]:.2f}-{oq[2]:.2f})")
print("lm_head |g_B| per seed:", " ".join(f"{x:.0f}" for x in summary["_lm_head"]))

# ---- figure --------------------------------------------------------------------------------------
ms.use("blog")
blue = ms.color_map(MODS)[MODS[0]]
panel = {m: ("Attention " if m.startswith("attn") else "Expert ") + short[m] for m in MODS}
series = {panel[m]: (L, mean[i]) for i, m in enumerate(MODS)}


def setup(ax):
    ax.axvspan(5.5, 17.5, color="#eceef1", lw=0, zorder=0)
    ms.log_axis(ax, "y", [10, 30, 100, 300, 1000])
    ax.set_xticks([0, 6, 12, 18, 23])


fig, axs = ms.small_multiples(series, ncols=3, width="blog_wide", aspect=0.62, color=blue, setup=setup)
for ax, (i, m) in zip(axs.ravel(), enumerate(MODS)):
    ax.fill_between(L, lo[i], hi[i], color=blue, alpha=0.25, lw=0, zorder=2)
for ax in axs[-1]:
    ax.set_xlabel("Layer")
for ax in axs[:, 0]:
    ax.set_ylabel("Gradient norm of B")
ms.fig_titles(fig, "LoRA B gradient norm by layer, looped layers 6 to 17 shaded",
              "Mean of 3 batches, band from minimum to maximum. Grey lines are the other module types")
ms.save(fig, OUT / "per_layer_grad_norms")
print("wrote", OUT / "per_layer_grad_norms.png")
