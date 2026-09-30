"""Aggregate the robustness sweep CSVs into the numbers the paper reports.

Statistics follow the protocol: aggregate within a seed first, then report the
mean and standard deviation *across the five seeds*. A condition has 150
episodes, but only five independently trained policies, so the spread that
matters is the spread between seeds -- pooling all 150 episodes would quote a
precision the experiment does not have.

    python scripts/analyze_results.py --experiment E1
    python scripts/analyze_results.py --experiment E2 --format markdown

Writes results/<exp>_summary.csv and prints a table.
"""
import argparse
import csv
import glob
import os
import statistics as st
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PER_EPISODE_METRICS = [
    "survived_s", "body_roll_rms", "body_pitch_rms",
    "board_roll_rms", "board_pitch_rms",
    "body_roll_max", "body_pitch_max",
    "board_roll_max", "board_pitch_max",
    "board_edge_contact_ratio",
]


def survived_at_least(row, seconds: float) -> int:
    """Did this episode reach `seconds` without failing?

    At the full horizon the test is whether the episode ended by timeout rather
    than by a failure condition. Counting steps there is brittle: the
    environment stops at episodeLength - 1 and the harness spends one further
    step settling the observation buffer, so a complete run records two steps
    short of the nominal horizon and a ">= horizon" test rejects every success.
    """
    duration = float(row["duration_s"])
    if seconds >= duration - 1e-9:
        return int(row["failure_cause"] == "timeout")
    return int(int(row["survived_steps"]) >= round(seconds / 0.01))


def seed_level(rows):
    """Collapse one seed's episodes into a single record."""
    n = len(rows)
    rec = {
        "success_10s": sum(survived_at_least(r, 10.0) for r in rows) / n,
        "success_60s": sum(survived_at_least(r, 60.0) for r in rows) / n,
    }
    for m in PER_EPISODE_METRICS:
        rec[m] = st.mean(float(r[m]) for r in rows)
    causes = defaultdict(int)
    for r in rows:
        causes[r["failure_cause"]] += 1
    rec["_causes"] = {k: v / n for k, v in causes.items()}
    return rec


def across_seeds(seed_recs):
    out = {}
    for key in ["success_10s", "success_60s"] + PER_EPISODE_METRICS:
        vals = [r[key] for r in seed_recs]
        out[key + "_mean"] = st.mean(vals)
        out[key + "_std"] = st.stdev(vals) if len(vals) > 1 else 0.0
    causes = defaultdict(list)
    for r in seed_recs:
        for k, v in r["_causes"].items():
            causes[k].append(v)
    # A cause absent for a seed counts as zero for that seed, not as missing.
    out["failure_causes"] = {
        k: st.mean(v + [0.0] * (len(seed_recs) - len(v))) for k, v in causes.items()
    }
    out["n_seeds"] = len(seed_recs)
    return out


def load(experiment: str):
    """Pool every checkpoint file for a cell, then aggregate within seed, then across seeds.

    One cell is spread over several files, one per evaluated checkpoint. The protocol
    aggregates a seed's checkpoints and trials together before comparing seeds, because the
    independent replicate is the training run, not the checkpoint.
    """
    merged = defaultdict(lambda: defaultdict(list))
    for path in sorted(glob.glob(os.path.join(ROOT, "results", experiment.lower(), "*.csv"))):
        rows = list(csv.DictReader(open(path)))
        if not rows:
            continue
        meta = rows[0]
        key = (meta["task"], meta["obs_mode"], meta["controller"], meta["setting"])
        for r in rows:
            merged[key][r["seed_index"]].append(r)

    cells = {}
    for key, by_seed in merged.items():
        cells[key] = across_seeds([seed_level(v) for v in by_seed.values()])
        cells[key]["n_episodes"] = sum(len(v) for v in by_seed.values())
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, choices=["E1", "E2"])
    ap.add_argument("--format", choices=["plain", "markdown"], default="plain")
    args = ap.parse_args()

    cells = load(args.experiment)
    if not cells:
        raise SystemExit(f"no results found for {args.experiment}")

    out_path = os.path.join(ROOT, "results", f"{args.experiment.lower()}_summary.csv")
    fields = ["task", "obs_mode", "controller", "setting", "n_seeds", "n_episodes",
              "success_10s_mean", "success_10s_std",
              "success_60s_mean", "success_60s_std",
              "survived_s_mean", "survived_s_std",
              "body_roll_rms_mean", "body_pitch_rms_mean",
              "board_roll_rms_mean", "board_pitch_rms_mean",
              "board_edge_contact_ratio_mean", "dominant_failure"]
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for (task, obs, ctrl, setting), v in sorted(cells.items()):
            dom = max(v["failure_causes"].items(), key=lambda kv: kv[1])[0] if v["failure_causes"] else ""
            w.writerow({
                "task": task, "obs_mode": obs, "controller": ctrl, "setting": setting,
                "n_seeds": v["n_seeds"], "n_episodes": v.get("n_episodes", 0), "dominant_failure": dom,
                **{k: round(v[k], 4) for k in fields
                   if k.endswith(("_mean", "_std")) and k in v},
            })
    print(f"wrote {out_path}")

    sep = " | " if args.format == "markdown" else "  "
    header = ["task", "obs", "setting", "10s%", "60s%", "survived_s"]
    print(sep.join(header))
    if args.format == "markdown":
        print(sep.join(["---"] * len(header)))
    for (task, obs, ctrl, setting), v in sorted(cells.items()):
        label = task.replace("BalanceBoard", "") if ctrl == "policy" else "PD"
        print(sep.join([
            label, obs, setting,
            f"{100*v['success_10s_mean']:.0f}+-{100*v['success_10s_std']:.0f}",
            f"{100*v['success_60s_mean']:.0f}+-{100*v['success_60s_std']:.0f}",
            f"{v['survived_s_mean']:.1f}+-{v['survived_s_std']:.1f}",
        ]))


if __name__ == "__main__":
    main()
