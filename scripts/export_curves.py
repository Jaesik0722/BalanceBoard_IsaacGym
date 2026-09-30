"""Export the 3000-iteration training curves to a npz, so plotting needs no tensorboard.

tensorboard lives in the isaacgym environment and matplotlib in the base one; keeping the
extraction separate from the drawing avoids installing into the environment that is
running experiments.
"""
import glob
import os

import numpy as np
from tensorboard.backend.event_processing import event_accumulator as ea

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = {}
for short in ("PID", "IK", "Joint"):
    for mode in ("full", "restricted"):
        runs = []
        for s in (1, 2, 3):
            d = sorted(glob.glob(os.path.join(ROOT, "runs", f"{short}_{mode}_3k_seed{s}_*")))
            if not d:
                continue
            ev = glob.glob(os.path.join(d[-1], "summaries", "events*"))
            if not ev:
                continue
            acc = ea.EventAccumulator(ev[0], size_guidance={"scalars": 0})
            acc.Reload()
            runs.append(np.array([e.value for e in acc.Scalars("rewards/iter")]))
        if not runs:
            print("missing", short, mode)
            continue
        n = min(len(r) for r in runs)
        out[f"{short}_{mode}"] = np.stack([r[:n] for r in runs])
        print(f"{short}_{mode}: {len(runs)} seeds x {n} iters")
np.savez(os.path.join(ROOT, "results", "training_curves.npz"), **out)
print("wrote results/training_curves.npz")
