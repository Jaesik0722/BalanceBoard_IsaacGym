"""Redraw the three action-representation pipelines.

The earlier diagram showed the policy feeding the PD stage, which misrepresents what the PD
stage consumes. The policy supplies gains and posture weights; the error the PD stage acts on
is the measured attitude, which reaches it directly from the IMU and never passes through the
policy. That path is what the paper's central argument turns on, so it is drawn explicitly.
The shared joint position servo is also drawn, because all three representations terminate in
the same one and differ only in how the target reaching it is produced.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY, FIXED, SHARED = "#d6e6f4", "#fae3e0", "#e8e8e8"


def box(ax, x, y, w, h, title, sub=None, fc="white"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec="#444", lw=1.2))
    ax.text(x, y + (0.012 if sub else 0), title, ha="center", va="center",
            fontsize=8.2, fontweight="bold")
    if sub:
        ax.text(x, y - 0.022, sub, ha="center", va="center", fontsize=6.9, color="#333")
    return (x, y - h / 2), (x, y + h / 2)


def arrow(ax, a, b, style="-|>", color="#444", ls="-", lw=1.2, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, color=color, lw=lw,
                                 linestyle=ls, mutation_scale=11,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1, shrinkB=1))


fig, axes = plt.subplots(1, 3, figsize=(11.2, 6.4))
W, H = 0.62, 0.085
COLS = [
    ("(a) PD-assisted IK",
     [("PD gains and posture weights", "$k_p$, $k_d$ per axis; limb weights; LPF $\\alpha$", POLICY),
      ("Attitude-error PD", "fixed structure; gains set by policy", FIXED),
      ("Low-pass filter", None, FIXED),
      ("Inverse kinematics", "foot pose $\\rightarrow$ joint angles", FIXED)]),
    ("(b) IK",
     [("Foot pose offsets", "per-foot pose; LPF $\\alpha$", POLICY),
      ("Low-pass filter", None, FIXED),
      ("Inverse kinematics", "foot pose $\\rightarrow$ joint angles", FIXED)]),
    ("(c) Joint-space",
     [("Joint angle offsets", "14 joint targets; LPF $\\alpha$", POLICY),
      ("Low-pass filter", None, FIXED)]),
]

for ax, (title, stages) in zip(axes, COLS):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title(title, fontsize=10, pad=6)
    ys = [0.86 - 0.125 * i for i in range(len(stages) + 2)]

    bot, _ = box(ax, 0.5, ys[0], W, H, "Observations",
                 "IMU attitude and rate; joint encoders", SHARED)
    prev = bot
    obs_bottom = bot
    for i, (name, sub, fc) in enumerate(stages):
        b, top = box(ax, 0.5, ys[i + 1], W, H, name, sub, fc)
        arrow(ax, prev, top)
        prev = b
        if name == "Attitude-error PD":
            # The error enters here from the IMU, not from the policy.
            # Routed outside the column so it does not cross the policy-output box.
            xl = 0.5 - W / 2
            ax.plot([xl, 0.075, 0.075, xl], 
                    [obs_bottom[1] - 0.005, obs_bottom[1] - 0.005, ys[i + 1], ys[i + 1]],
                    color="#b03a2e", lw=1.4, solid_joinstyle="round", zorder=0)
            arrow(ax, (0.075, ys[i + 1]), (xl + 0.006, ys[i + 1]), color="#b03a2e", lw=1.4)
            ax.text(0.075, (obs_bottom[1] + ys[i + 1]) / 2, " measured\n attitude\n error",
                    fontsize=6.6, color="#b03a2e", ha="left", va="center")

    b, top = box(ax, 0.5, ys[len(stages) + 1], W, H, "Joint position servo",
                 "$k=80$, $d=2$, $|\\tau| \\leq 9.9$ N$\\cdot$m", SHARED)
    arrow(ax, prev, top)
    ax.text(0.5, ys[len(stages) + 1] - H / 2 - 0.045, "robot and board",
            ha="center", va="center", fontsize=7.6, style="italic", color="#333")
    arrow(ax, b, (0.5, ys[len(stages) + 1] - H / 2 - 0.028))

fig.text(0.5, 0.025,
         "Blue: produced by the policy.   Red: fixed structure the policy does not alter.   "
         "Grey: shared by all three representations.",
         ha="center", fontsize=7.8, color="#333")
fig.tight_layout(rect=[0, 0.045, 1, 1])
out = os.path.join(ROOT, "Applied Sciences", "build", "Figures", "action_representations.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print("wrote", out)
