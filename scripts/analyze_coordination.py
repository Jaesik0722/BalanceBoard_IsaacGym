"""Aggregate the coordination sweep and draw the ankle/hip figure.

The reported quantity is the ankle share of ankle-plus-hip torque activity. It is a
description of where torque is being applied, not a decomposition of who is responsible
for keeping the robot upright: a single ratio cannot establish that, and the text says so.

Seeds are the replicate unit, as everywhere else in this paper. Trials are pooled within a
seed and the mean and standard deviation are taken across the three seeds. Trials that
never met the pre-perturbation criterion, or that fell before the analysis window closed,
are excluded from the ratio and counted separately, because a ratio computed over a falling
robot describes the fall rather than the response.
"""
import csv
import glob
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODE_NAME = {"restricted": "restricted", "full": "privileged"}
SERIES = [("PID", "PD-assisted IK", "#c0392b", "o"),
          ("IK", "IK", "#2471a3", "s"),
          ("Joint", "Joint-space", "#1e8449", "^")]
FORCES = [0, 1, 3, 5]

rows = []
for f in glob.glob(os.path.join(ROOT, "results", "coordination", "*.csv")):
    rows += list(csv.DictReader(open(f)))

# key -> seed -> list of per-trial ankle shares, for each of the two torque measures.
# MEASURES maps a short name to the (ankle, hip) column pair it is computed from.
MEASURES = {"sensor": ("ankle_torque_activity", "hip_torque_activity"),
            "drive": ("ankle_drive_activity", "hip_drive_activity")}
acc = {m: defaultdict(lambda: defaultdict(list)) for m in MEASURES}
counts = defaultdict(lambda: [0, 0, 0, 0])   # analysable, trials, stabilised, survived
for r in rows:
    key = (r["task"], r["obs_mode"], r["axis"], float(r["force_N"]))
    counts[key][1] += 1
    counts[key][2] += int(r["stabilised"])
    counts[key][3] += int(r["survived_window"])
    if r["analysable"] != "1":
        continue
    counts[key][0] += 1
    for m, (ac, hc) in MEASURES.items():
        a, h = float(r[ac]), float(r[hc])
        if a + h > 0:
            acc[m][key][r["seed_index"]].append(a / (a + h))


def stat(key, measure="sensor"):
    """Mean and SD across seeds of the per-seed mean ankle share.

    Trials are pooled within a seed first, so a seed with more surviving trials does not
    carry more weight than one with fewer; the seed is the replicate throughout the paper.
    A condition analysed in fewer than two seeds is not summarised.
    """
    per_seed = [np.mean(v) for v in acc[measure][key].values() if v]
    if len(per_seed) < 2:
        return None, None, len(per_seed)
    return float(np.mean(per_seed)), float(np.std(per_seed)), len(per_seed)


with open(os.path.join(ROOT, "results", "coordination_summary.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["task", "obs_mode", "axis", "force_N", "n_seeds", "n_stabilised",
                "n_survived_window", "n_analysable", "n_trials",
                "ankle_share_sensor_mean", "ankle_share_sensor_sd",
                "ankle_share_drive_mean", "ankle_share_drive_sd"])
    for key in sorted(counts):
        row = list(key) + [stat(key)[2], counts[key][2], counts[key][3],
                           counts[key][0], counts[key][1]]
        for meas in ("sensor", "drive"):
            m, sd, _ = stat(key, meas)
            row += ["" if m is None else round(m, 4),
                    "" if sd is None else round(sd, 4)]
        w.writerow(row)

plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 10.5,
                     "xtick.labelsize": 10, "ytick.labelsize": 10})
# Rows are the two torque definitions. They agree in the sagittal plane and diverge sharply
# in the frontal plane, so presenting only one of them would assert a result the data does
# not support.
ROWS = [("sensor", "Simulator DOF force sensor"),
        ("drive", "Position drive law")]
fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), sharey=True, sharex=True)
for row, (meas, mlabel) in enumerate(ROWS):
    for col, axis in enumerate(("pitch", "roll")):
        ax = axes[row][col]
        for short, label, colour, marker in SERIES:
            for mode, ls, fill in (("restricted", "-", "full"), ("full", "--", "none")):
                xs, ys, es = [], [], []
                for f in FORCES:
                    m, sd, ns = stat((short, mode, axis, float(f)), meas)
                    if m is None:
                        continue
                    xs.append(f); ys.append(m * 100); es.append(sd * 100)
                if xs:
                    ax.errorbar(xs, ys, yerr=es, color=colour, marker=marker, capsize=3,
                                markersize=5, linewidth=1.6, linestyle=ls, fillstyle=fill,
                                label=f"{label}, {MODE_NAME[mode]}")
        plane = "Sagittal (pitch)" if axis == "pitch" else "Frontal (roll)"
        ax.set_title(f"({'abcd'[row * 2 + col]}) {plane} -- {mlabel}", fontsize=10)
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.set_ylim(0, 70)
        ax.axhline(50, color="0.4", linestyle=":", linewidth=1)
        ax.set_xticks(FORCES)
        if row == 1:
            ax.set_xlabel("Perturbation force (N, applied for 0.1 s)")
        if col == 0:
            ax.set_ylabel("Ankle share of ankle+hip\ntorque activity (%)")
axes[0][0].legend(fontsize=8, loc="upper left", framealpha=0.95, ncol=2)
fig.tight_layout()
out = os.path.join(ROOT, "Applied Sciences", "build", "Figures", "coordination.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print("wrote", out)
print("wrote results/coordination_summary.csv")
