# BalanceBoard_IsaacGym

Isaac Gym-based reinforcement learning environments for humanoid balance control on a balance board.

This repository provides custom IsaacGymEnvs tasks for training and evaluating a humanoid robot balance controller using multiple action representations, including PD-based control, inverse-kinematics-based control, and direct joint-space control.

## Overview

The goal of this project is to train a humanoid robot to maintain balance on an unstable balance board in simulation. The environment is implemented using NVIDIA Isaac Gym and IsaacGymEnvs.

The repository includes:

- Humanoid and balance board URDF assets
- Custom IsaacGymEnvs task files
- PPO training configurations
- PID and low-pass filter utilities
- The fixed-gain PD reference used for comparison

## Main Features

- Isaac Gym simulation environment for humanoid balance control
- Balance board task using a humanoid robot model
- Three task/control variants:
  - `BalanceBoardPID`
  - `BalanceBoardIK`
  - `BalanceBoardJoint`
- PPO training configuration for each task
- GPU-based parallel simulation through Isaac Gym
- Fixed-gain PD reference, obtained from the time-averaged gains commanded by the trained
  PD-assisted IK policies (`scripts/extract_pd_gains.py`)

## Task Variants

### 1. BalanceBoardPID

`BalanceBoardPID` uses a policy to tune PID-related control parameters and motion weights. The controller converts balance errors into joint targets through PD control, low-pass filtering, and inverse kinematics.

### 2. BalanceBoardIK

`BalanceBoardIK` uses a policy that outputs foot-position, foot-orientation, arm-motion, and filtering parameters. These commands are converted into lower-body joint commands through inverse kinematics.

### 3. BalanceBoardJoint

`BalanceBoardJoint` uses a policy that directly outputs joint-space commands for the humanoid robot. This provides a more direct action representation compared with the PID and IK variants.

## Reproducing the paper

The experiments reported in *Learning Humanoid Balance on an Unstable Support: Effects of
Action Representation and Sensory Constraints* are driven by the scripts in `scripts/`.
Copy this repository's `tasks/`, `utils/`, `cfg/`, `scripts/` and `assets/` over a checkout of
[IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs), then run the commands below from
the `isaacgymenvs` directory.

### Training

```bash
# one run; repeat with seed=1,2,3 for the 3000-iteration budget
python train.py task=BalanceBoardJoint max_iterations=3000 seed=1
```

Set `task.env.fullStateObservation` to `true` for the privileged observation condition and
`false` for the restricted one.

### Evaluation

All reported numbers come from `eval_harness.py`, which fixes the initial conditions, pools a
seed's checkpoints and trials, and reports the seed as the replicate.

```bash
# nominal comparison and the sensor / model-mismatch sweeps
python scripts/run_robustness_experiments.py --run-tag _3k --seeds 1 2 3 \
    --checkpoint-epoch 2600 2800 3000
python scripts/analyze_results.py                 # writes results/e1_summary.csv, e2_summary.csv

# perturbation experiment (3 seeds x 10 trials, final checkpoint)
./scripts/run_coordination_sweep.sh
python scripts/analyze_coordination.py            # writes results/coordination_summary.csv

# action-selection diagnostic
python scripts/eval_harness.py --task BalanceBoardJoint --obs-mode full \
    --checkpoints <ckpt...> --stochastic --out results/stochastic/...csv
```

### Figures

`plot_sensitivity.py`, `plot_convergence.py`, `plot_torque_trace.py` and
`plot_action_representations.py` regenerate the figures from the summary files in `results/`.

### Trained checkpoints

The checkpoints are not tracked here. They are available on Google Drive:

https://drive.google.com/drive/folders/1knRNT0W7uzJ45eYwY0E5CAe41VPK0JHg?usp=sharing

The folder holds the `runs/` directory the scripts expect, with the checkpoints the evaluation
reads and the `config.yaml` of each run: the 3000-iteration matrix at iterations 2600, 2800
and 3000 (three action representations, two observation conditions, three seeds) and the
1000-iteration budget comparison at iterations 600, 800 and 1000 (five seeds). Download
`runs/` and place it directly in the `isaacgymenvs` directory, alongside `scripts/` and
`tasks/`, so that paths read `runs/<Task>_<obs-mode>_3k_seed<N>_<timestamp>/nn/*.pth`.

Training the 3000-iteration matrix from scratch instead takes roughly 16 h on a single
RTX 2070 (18 runs).

### Notes

- `verify_torque_signal.py` reproduces the check reported in the paper that the simulator's
  DOF force sensor and an explicit evaluation of the position drive law are not the same
  signal on this model.
- `results/` here contains only the summary files. The per-episode CSVs and the training
  checkpoints are large and are not tracked.
- The nominal roller radius is 39 mm (`assets/urdf/board.urdf`); `board_r0NN.urdf` are the
  variants used for the contact-geometry sweep.
