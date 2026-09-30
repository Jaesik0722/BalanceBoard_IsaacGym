"""Sensitivity panels for the robustness study.

Every panel reports the same quantity as the headline result -- the fraction of 60 s
trials completed -- so panels can be read against Figure 1 and against each other.
Error bars are the standard deviation across seeds, the replicate unit used throughout.
Only the restricted observation condition is plotted: that is the condition the paper's
claim is about, and PD-assisted IK is at zero across all of it, which is the point.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERIES = [("BalanceBoardPID", "PD-assisted IK", "#c0392b", "o", "-"),
          ("BalanceBoardIK", "IK", "#2471a3", "s", "-"),
          ("BalanceBoardJoint", "Joint-space", "#1e8449", "^", "-")]


def load(fn):
    d = {}
    for r in csv.DictReader(open(os.path.join(ROOT, "results", fn))):
        if r["obs_mode"] != "restricted" or r["controller"] != "policy":
            continue
        d[(r["task"], r["setting"])] = (float(r["success_60s_mean"]) * 100,
                                        float(r["success_60s_std"]) * 100)
    return d


def panel(ax, data, settings, xs, xlabel, title):
    for task, label, colour, marker, ls in SERIES:
        ys, es, xv = [], [], []
        for s, x in zip(settings, xs):
            if (task, s) in data:
                m, sd = data[(task, s)]
                ys.append(m); es.append(sd); xv.append(x)
        if xv:
            # A success rate cannot leave [0, 100], so the whiskers are clipped at the
            # bounds rather than drawn through them.
            lo = [min(e, y) for y, e in zip(ys, es)]
            hi = [min(e, 100 - y) for y, e in zip(ys, es)]
            ax.errorbar(xv, ys, yerr=[lo, hi], label=label, color=colour, marker=marker,
                        linestyle=ls, capsize=3, markersize=5, linewidth=1.6)
    ax.set_xlabel(xlabel); ax.set_title(title, fontsize=10)
    ax.set_ylim(-5, 105); ax.grid(alpha=0.3, linewidth=0.5)


e1, e2 = load("e1_summary.csv"), load("e2_summary.csv")
fig, axes = plt.subplots(2, 2, figsize=(9, 6.6))

panel(axes[0][0], e1, ["N0", "N1", "N2", "N3", "N4"], [0, 0.25, 0.5, 1.0, 2.0],
      "Sensor model level (labelled by $\\sigma_{att}$, deg)", "(a) Sensor model (N0--N4)")
panel(axes[0][1], e1, ["N0", "N0_L1", "N0_L2", "N0_L3"], [0, 10, 20, 30],
      "Policy-input delay (ms)", "(b) Full-input delay, noise-free")
panel(axes[1][0], e2, ["stiffness_0.5", "stiffness_0.75", "nominal", "stiffness_1.5",
                       "stiffness_2.0"], [0.5, 0.75, 1.0, 1.5, 2.0],
      "Joint stiffness scale", "(c) Joint stiffness")
panel(axes[1][1], e2, ["roller_0.030", "roller_0.035", "roller_0.037", "roller_0.038",
                       "roller_0.040", "roller_0.041", "roller_0.043", "roller_0.045",
                       "roller_0.050"], [30, 35, 37, 38, 40, 41, 43, 45, 50],
      "Roller radius (mm)", "(d) Contact geometry")

for row in axes:
    row[0].set_ylabel("60 s trials completed (%)")
axes[0][0].legend(fontsize=8, loc="upper right", framealpha=0.9)
fig.tight_layout()
out = os.path.join(ROOT, "Applied Sciences", "Figures", "sensitivity.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=300, bbox_inches="tight")
print("wrote", out)
