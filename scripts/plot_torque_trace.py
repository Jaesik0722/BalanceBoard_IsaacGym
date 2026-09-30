"""Torque and tilt against a common time axis, for one representative trial per condition.

The ratio plots say nothing about how torque evolves, which is what a postural response is.
The representative trial is chosen by a stated rule rather than by eye: among trials that met
the pre-perturbation criterion and survived the whole analysis window, the one whose ankle
share over the window is closest to the median of that set. Absolute activity is plotted, not
only the ratio, because a constant ratio is compatible with both groups growing together.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONDS = [("pitch", 0, "Sagittal, no push"), ("pitch", 3, "Sagittal, 3 N push"),
         ("roll", 0, "Frontal, no push"), ("roll", 3, "Frontal, 3 N push")]


def pick(z):
    a, h = z["ank"], z["hip"]                      # (steps, envs)
    ps, we = int(z["push_start"]), int(z["win_end"])
    # The eligible set must be exactly the analysable set of the table: stabilised at push
    # onset and alive for at least half the analysis window. Using a different criterion here
    # would select the representative trial from a different population than the one the
    # reported statistics describe.
    stab = z["stab"][ps].astype(bool)
    win_n = (z["alive"][ps:we].astype(bool) & stab[None, :]).sum(0)
    idx = np.flatnonzero(stab & (win_n >= 0.5 * (we - ps)))
    if idx.size == 0:
        return None, 0
    tot = a[ps:we, idx].sum(0) + h[ps:we, idx].sum(0)
    share = np.where(tot > 0, a[ps:we, idx].sum(0) / np.maximum(tot, 1e-9), np.nan)
    return int(idx[np.argmin(np.abs(share - np.nanmedian(share)))]), idx.size


# The two measures differ by roughly sixfold in magnitude, so they get their own rows;
# overlaying them on one axis hides the smaller one entirely.
fig, axes = plt.subplots(3, 4, figsize=(13.5, 7.2), sharex=True,
                         gridspec_kw={"height_ratios": [1.5, 1.5, 1]})
for col, (axis, force, title) in enumerate(CONDS):
    z = np.load(os.path.join(ROOT, "results", f"trace_IK_{axis}_{force}N.npz"))
    e, n = pick(z)
    ps, pe, we = int(z["push_start"]), int(z["push_end"]), int(z["win_end"])
    t = (np.arange(z["ank"].shape[0]) - ps) * float(z["dt"])

    for row, (ak, hk, lab, top_y) in enumerate([
            # Limits cover the largest peak in any plotted panel, so no trace is clipped:
            # the sensor row reaches 5.8 N*m (frontal, 3 N) and the drive row 13.7.
            ("ank", "hip", "Sensor", 6.5), ("ank_drive", "hip_drive", "Drive law", 15.0)]):
        ax = axes[row][col]
        ax.plot(t, z[ak][:, e], color="#2471a3", lw=1.1, label="ankle")
        ax.plot(t, z[hk][:, e], color="#c0392b", lw=1.1, label="hip")
        ax.set_ylim(0, top_y)
        if col == 0:
            ax.set_ylabel(f"{lab}\n(N$\\cdot$m)")

    ax = axes[2][col]
    ax.plot(t, np.degrees(z["tilt"][:, e]), color="#1e8449", lw=1.2)
    ax.axhline(10, color="0.4", ls="--", lw=0.9)
    ax.axhline(-10, color="0.4", ls="--", lw=0.9)
    ax.set_ylim(-12, 12)
    ax.set_xlabel("Time from push onset (s)")
    if col == 0:
        ax.set_ylabel("Body tilt\n(deg)")

    for r in range(3):
        a = axes[r][col]
        a.axvspan(0, (pe - ps) * float(z["dt"]), color="#d9534f", alpha=0.20, lw=0)
        a.axvline((we - ps) * float(z["dt"]), color="0.5", ls=":", lw=1)
        a.set_xlim(-1.0, 2.5)
        a.grid(alpha=0.3, lw=0.5)
    axes[0][col].set_title(f"{title}\n(median trial of {n} analysable)", fontsize=9)

axes[0][0].legend(fontsize=8, loc="upper left", ncol=2, framealpha=0.95)
fig.tight_layout()
out = os.path.join(ROOT, "Applied Sciences", "build", "Figures", "torque_trace.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print("wrote", out)
