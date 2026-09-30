"""Redraw the random-search balance durations without the spurious regression line.

The original figure was produced with seaborn's lmplot, whose default draws a regression
line and a confidence band against the trial index. The trial index is an arbitrary sample
label with no order, so that fit described nothing. The distribution of durations is the
only content, so it is shown as a sorted scatter with the median and range marked.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Path to the random-search log of the earlier experiment. It is not part of this
# repository; set PD_SEARCH_CSV to point at it.
SRC = os.environ.get("PD_SEARCH_CSV", "avg_pid.csv")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

d = pd.read_csv(SRC)
t = np.sort(d["Time(sec)"].to_numpy())

fig, (ax, bx) = plt.subplots(1, 2, figsize=(9, 3.6),
                             gridspec_kw={"width_ratios": [2, 1]})
ax.plot(np.arange(1, len(t) + 1), t, "o", color="#2471a3", markersize=5,
        alpha=0.85)
ax.axhline(np.median(t), color="#c0392b", lw=1.3,
           label=f"median {np.median(t):.2f} s")
ax.set_xlabel("Sampled parameter set (sorted by duration)")
ax.set_ylabel("Balance duration (s)")
ax.grid(alpha=0.3, lw=0.5)
ax.legend(fontsize=9)

bx.hist(t, bins=12, color="#2471a3", alpha=0.85)
bx.set_xlabel("Balance duration (s)")
bx.set_ylabel("Parameter sets")
bx.grid(alpha=0.3, lw=0.5)

fig.tight_layout()
out = os.path.join(ROOT, "Applied Sciences", "build", "Figures",
                   "baseline_PD_random_search.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"n={len(t)}  min={t.min():.2f}  median={np.median(t):.2f}  max={t.max():.2f}")
print("wrote", out)
