"""Training convergence, redrawn from the tensorboard logs of the 3000-iteration runs.

The earlier version of this figure carried the internal run labels ("Ideal", "Sim2Real",
"PID") and an "Epochs" axis. PPO iterations are not epochs, and "Ideal"/"Sim2Real" name
an intention rather than what was varied, which is which observations the policy receives.
The terminology here matches the rest of the paper.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERIES = [("PID", "PD-assisted IK", "#c0392b"),
          ("IK", "IK", "#2471a3"),
          ("Joint", "Joint-space", "#1e8449")]
MODES = [("full", "privileged", "-"), ("restricted", "restricted", "--")]


CURVES = np.load(os.path.join(ROOT, "results", "training_curves.npz"))


def curve(short, mode):
    key = f"{short}_{mode}"
    if key not in CURVES:
        return None, None
    a = CURVES[key]
    return a.mean(axis=0), a.std(axis=0)


def smooth(y, k=15):
    return np.convolve(y, np.ones(k) / k, mode="valid")


# Split by observation condition rather than overlaying all six curves: the contrast the
# paper is about is between the two panels, and six shaded bands on one axis hide it.
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, (mode, mlabel, _) in zip(axes, MODES):
    for short, label, colour in SERIES:
        m, sd = curve(short, mode)
        if m is None:
            print("missing", short, mode)
            continue
        ms, sds = smooth(m), smooth(sd)
        x = np.arange(len(ms))
        ax.plot(x, ms, color=colour, linewidth=1.5, label=label)
        ax.fill_between(x, ms - sds, ms + sds, color=colour, alpha=0.15, linewidth=0)
    ax.set_xlabel("PPO training iteration")
    ax.set_title(f"({'ab'[MODES.index((mode, mlabel, _))]}) {mlabel.capitalize()} observations",
                 fontsize=10)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.set_xlim(0, 3000)

axes[0].set_ylabel("Episode return")
axes[0].legend(fontsize=8, loc="lower right", framealpha=0.9)
fig.tight_layout()
out = os.path.join(ROOT, "Applied Sciences", "build", "Figures",
                   "training_convergence_results.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print("wrote", out)
