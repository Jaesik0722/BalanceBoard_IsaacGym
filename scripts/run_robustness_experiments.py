"""Driver for the robustness sweeps (E1 sensor, E2 model mismatch).

Enumerates the sweep cells and runs `eval_harness.py` once per
(task, observation mode, setting) in its own process, because Isaac Gym only
supports one simulation per process.

Every sweep includes its nominal level, so each factor's baseline comes from
the same batch as its perturbed levels.

    python scripts/run_robustness_experiments.py --experiment E1 --dry-run
    python scripts/run_robustness_experiments.py --experiment E1
"""
import argparse
import json
import glob
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

TASKS = ["BalanceBoardPID", "BalanceBoardIK", "BalanceBoardJoint"]
OBS_MODES = ["restricted", "full"]
SEEDS = [1, 2, 3, 4, 5]

# Applied to every cell of every sweep so the termination rule is identical
# across experiments: foot displacement is measured from each env's own spawn
# position rather than the shipped reference, which sits ~2 cm away and leaves
# only ~1 cm of travel before the condition fires.
GLOBAL_OVERRIDES = ["eval.footReferenceFromSpawn=true"]

# Fixed-gain PD baseline. No tuned gains survive from the original controller,
# so the baseline is the mean command of the trained PD-assisted-IK policies
# while balancing (scripts/extract_pd_gains.py).
_PD_JSON = os.path.join(ROOT, "results", "pd_baseline_action.json")
PD_BASELINE_ACTION = (
    ",".join(f"{v:.6f}" for v in json.load(open(_PD_JSON))["action"])
    if os.path.exists(_PD_JSON) else None
)

# Dynamixel XH540: 4096 counts/rev.
ENCODER_RESOLUTION = 2 * math.pi / 4096  # 0.00153 rad (0.088 deg)


def deg(x):
    return math.radians(x)


# --- E1: sensor realism -----------------------------------------------------
NOISE_LEVELS = {
    "N0": dict(attitudeStd=0.0, gyroStd=0.0, encoderStd=0.0, encoderResolution=0.0),
    "N1": dict(attitudeStd=deg(0.25), gyroStd=0.005, encoderStd=0.0,
               encoderResolution=ENCODER_RESOLUTION),
    "N2": dict(attitudeStd=deg(0.5), gyroStd=0.01, encoderStd=deg(0.05),
               encoderResolution=ENCODER_RESOLUTION),
    "N3": dict(attitudeStd=deg(1.0), gyroStd=0.02, encoderStd=deg(0.1),
               encoderResolution=ENCODER_RESOLUTION),
    "N4": dict(attitudeStd=deg(2.0), gyroStd=0.05, encoderStd=deg(0.2),
               encoderResolution=ENCODER_RESOLUTION),
}


def e1_settings():
    settings = []
    # Noise sweep at zero latency.
    for name, noise in NOISE_LEVELS.items():
        settings.append((name, noise, 0))
    # Latency sweep at the mid noise level and at zero noise, so latency and
    # noise can be read apart. N2+L2 is the representative "realistic sensor"
    # cell and is not duplicated.
    for delay in (1, 2, 3):
        settings.append((f"N2_L{delay}", NOISE_LEVELS["N2"], delay))
    for delay in (1, 2, 3):
        settings.append((f"N0_L{delay}", NOISE_LEVELS["N0"], delay))
    return settings


def e1_overrides(noise: dict, delay: int):
    ov = ["eval.sensorModelActive=true", f"eval.observationDelaySteps={delay}"]
    for key, value in noise.items():
        ov.append(f"eval.sensorNoise.{key}={value}")
    return ov


# --- E2: model mismatch -----------------------------------------------------
ROLLER_ASSETS = {
    # Fine steps near nominal: a +-1 mm change is still fully survivable while
    # +-4 mm is not, so the tolerance edge is what this axis has to resolve.
    0.030: "urdf/board_r030.urdf",
    0.035: "urdf/board_r035.urdf",
    0.037: "urdf/board_r037.urdf",
    0.038: "urdf/board_r038.urdf",
    0.039: "urdf/board.urdf",       # nominal
    0.040: "urdf/board_r040.urdf",
    0.041: "urdf/board_r041.urdf",
    0.043: "urdf/board_r043.urdf",
    0.045: "urdf/board_r045.urdf",
    0.050: "urdf/board_r050.urdf",
}


def e2_settings():
    settings = [("nominal", [])]
    for radius, asset in ROLLER_ASSETS.items():
        if radius == 0.039:
            continue
        settings.append((f"roller_{radius:.3f}", [
            f"eval.boardAsset={asset}", f"eval.rollerRadius={radius}"]))
    for mu in (0.4, 0.6, 1.4):
        settings.append((f"friction_{mu}", [
            f"env.plane.staticFriction={mu}", f"env.plane.dynamicFriction={mu}"]))
    for scale in (0.8, 0.9, 1.1, 1.2):
        settings.append((f"robotmass_{scale}", [f"eval.physicsScale.robotMass={scale}"]))
    for scale in (0.5, 0.75, 1.5, 2.0):
        settings.append((f"stiffness_{scale}", [f"eval.physicsScale.jointStiffness={scale}"]))
    for scale in (0.5, 0.75, 1.5, 2.0):
        settings.append((f"damping_{scale}", [f"eval.physicsScale.jointDamping={scale}"]))
    for scale in (0.7, 1.3):
        settings.append((f"boardmass_{scale}", [f"eval.physicsScale.boardMass={scale}"]))
    # NOTE: a combined worst-case cell was specified, but the balance-board tasks
    # never call apply_randomizations, so task.randomize=true is inert -- it
    # reproduced the nominal numbers exactly. Left out until the randomization
    # hook is actually wired into the tasks.
    return settings


# --- checkpoint discovery ---------------------------------------------------
def find_checkpoints(task: str, obs_mode: str, which: str = "final",
                     run_tag: str = "", seeds=None, epoch: str = ""):
    """Locate one checkpoint per seed.

    `epoch` pins a specific training iteration. Evaluating several late
    checkpoints and averaging is what the protocol calls for: a single endpoint
    is unreliable here because training is non-monotonic, and picking the
    highest-training-reward checkpoint does not predict evaluation performance
    (it was 67 points worse for one condition and 24 better for another).
    """
    short = {"BalanceBoardPID": "PID", "BalanceBoardIK": "IK", "BalanceBoardJoint": "Joint"}[task]
    found = []
    for seed in (seeds or SEEDS):
        pattern = os.path.join(ROOT, "runs", f"{short}_{obs_mode}{run_tag}_seed{seed}_*", "nn")
        dirs = sorted(glob.glob(pattern))
        if not dirs:
            found.append(None)
            continue
        nn_dir = dirs[-1]
        if epoch:
            hit = glob.glob(os.path.join(nn_dir, f"last_*_ep_{epoch}_*.pth"))
            found.append(hit[0] if hit else None)
            continue
        if which == "final":
            cands = sorted(glob.glob(os.path.join(nn_dir, "last_*_ep_*.pth")))
            # highest epoch wins
            cands.sort(key=lambda p: int(p.split("_ep_")[1].split("_")[0]))
            pick = cands[-1] if cands else None
        else:  # best-reward checkpoint rl_games keeps under the plain name
            pick = os.path.join(nn_dir, f"{short}_{obs_mode}_seed{seed}.pth")
            pick = pick if os.path.exists(pick) else None
        found.append(pick)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=["E1", "E2"], required=True)
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--obs-modes", nargs="*", default=OBS_MODES)
    ap.add_argument("--checkpoint", choices=["final", "best"], default="final")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--duration-s", type=float, default=60.0)
    ap.add_argument("--results-dir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--include-pd-baseline", action="store_true", default=True)
    ap.add_argument("--run-tag", default="", help='e.g. "_3k" to target the extended-budget runs')
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--checkpoint-epoch", default="", help="pin a training iteration, e.g. 2800")
    ap.add_argument("--out-suffix", default="", help="appended to each result filename")
    args = ap.parse_args()

    settings = e1_settings() if args.experiment == "E1" else e2_settings()
    out_root = os.path.join(args.results_dir, args.experiment.lower())

    jobs = []
    for task in args.tasks:
        for obs_mode in args.obs_modes:
            checkpoints = find_checkpoints(task, obs_mode, args.checkpoint,
                                           args.run_tag, args.seeds, args.checkpoint_epoch)
            missing = [s for s, c in zip(args.seeds or SEEDS, checkpoints) if c is None]
            if missing:
                print(f"!! {task}/{obs_mode}: missing checkpoints for seeds {missing} -- skipped")
                continue

            for entry in settings:
                if args.experiment == "E1":
                    name, noise, delay = entry
                    overrides = e1_overrides(noise, delay)
                else:
                    name, overrides = entry

                out = os.path.join(out_root, f"{task}_{obs_mode}_{name}{args.out_suffix}.csv")
                cmd = [sys.executable, os.path.join(HERE, "eval_harness.py"),
                       "--task", task, "--obs-mode", obs_mode,
                       "--experiment", args.experiment, "--setting-name", name,
                       "--trials", str(args.trials), "--duration-s", str(args.duration_s),
                       "--checkpoints", *checkpoints, "--out", out]
                for ov in overrides + GLOBAL_OVERRIDES:
                    cmd += ["--eval-override", ov]
                jobs.append((out, cmd))

    # The PD baseline runs under the same settings, on BalanceBoardPID only.
    if args.include_pd_baseline and "BalanceBoardPID" in args.tasks:
        for obs_mode in args.obs_modes:
            for entry in settings:
                if args.experiment == "E1":
                    name, noise, delay = entry
                    overrides = e1_overrides(noise, delay)
                else:
                    name, overrides = entry
                out = os.path.join(out_root, f"PDbaseline_{obs_mode}_{name}{args.out_suffix}.csv")
                cmd = [sys.executable, os.path.join(HERE, "eval_harness.py"),
                       "--task", "BalanceBoardPID", "--obs-mode", obs_mode,
                       "--experiment", args.experiment, "--setting-name", name,
                       "--trials", str(args.trials), "--duration-s", str(args.duration_s),
                       "--pd-baseline", "--out", out]
                if PD_BASELINE_ACTION:
                    # "=" form: the value starts with "-", which argparse would
                    # otherwise read as the next option flag.
                    cmd += [f"--pd-action={PD_BASELINE_ACTION}"]
                for ov in overrides + GLOBAL_OVERRIDES:
                    cmd += ["--eval-override", ov]
                jobs.append((out, cmd))

    print(f"{args.experiment}: {len(jobs)} evaluation cells")
    if args.dry_run:
        for out, cmd in jobs[:5]:
            print("  " + " ".join(cmd))
        if len(jobs) > 5:
            print(f"  ... and {len(jobs) - 5} more")
        return

    for i, (out, cmd) in enumerate(jobs, 1):
        if os.path.exists(out):
            print(f"[{i}/{len(jobs)}] skip (exists): {os.path.basename(out)}")
            continue
        print(f"[{i}/{len(jobs)}] {os.path.basename(out)}")
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            print(f"!! failed ({result.returncode}): {' '.join(cmd)}")


if __name__ == "__main__":
    main()
