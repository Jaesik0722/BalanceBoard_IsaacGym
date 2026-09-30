"""Derive the fixed-gain PD baseline from what the trained policies actually command.

No tuned gains survive for the original baseline controller, so the baseline is
defined as the best fixed approximation of the learned gain schedule: the mean
action the PD-assisted-IK policies emit while they are balancing, averaged over
the five seeds under the nominal condition.

Extracted from the full-state policies: the restricted ones fall within about a
second, so their commands describe a falling robot rather than a balancing one.

    python scripts/extract_pd_gains.py --out results/pd_baseline_action.json
"""
# isaacgym must be imported before torch.
from isaacgym import gymapi  # noqa: F401

import argparse
import glob
import json
import os

import torch

from eval_harness import (apply_initial_conditions, build_task_cfg,
                          fixed_initial_conditions, load_players)
from isaacgymenvs.tasks import isaacgym_task_map

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACTION_LABELS = ["kp_roll", "kd_roll", "kp_pitch", "kd_pitch",
                 "w_leg_roll", "w_leg_pitch", "w_arm_roll_l", "w_arm_pitch_l",
                 "w_arm_roll_r", "w_arm_pitch_r", "lpf_roll", "lpf_pitch"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs-mode", default="full")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--duration-s", type=float, default=30.0)
    ap.add_argument("--condition-seed", type=int, default=20260919)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "pd_baseline_action.json"))
    args = ap.parse_args()

    checkpoints = []
    for seed in range(1, 6):
        nn = sorted(glob.glob(os.path.join(ROOT, "runs", f"PID_{args.obs_mode}_seed{seed}_*", "nn")))
        cands = sorted(glob.glob(os.path.join(nn[-1], "last_*_ep_*.pth")))
        cands.sort(key=lambda p: int(p.split("_ep_")[1].split("_")[0]))
        checkpoints.append(cands[-1])

    horizon = int(round(args.duration_s / 0.01))
    n_blocks = len(checkpoints)
    cfg, expected = build_task_cfg("BalanceBoardPID", args.obs_mode,
                                   args.trials * n_blocks,
                                   [("eval.footReferenceFromSpawn", True)], horizon)
    env = isaacgym_task_map["BalanceBoardPID"](
        cfg=cfg, rl_device="cuda:0", sim_device="cuda:0", graphics_device_id=0,
        headless=True, virtual_screen_capture=False, force_render=False)

    run_config = os.path.join(os.path.dirname(os.path.dirname(checkpoints[0])), "config.yaml")
    players = load_players(checkpoints, run_config, expected, env.num_actions, "cuda:0")

    offsets = fixed_initial_conditions(args.trials, cfg["env"]["offsetRange"], args.condition_seed)
    apply_initial_conditions(env, offsets, args.trials, n_blocks)

    obs_dict = env.reset()
    # One settling step first: reset() returns the still-zero obs_buf, and acting
    # on it produces a garbage command that can destabilise the robot outright.
    settle, _, _, _ = env.step(torch.zeros(env.num_envs, env.num_actions, device=env.device))
    obs = settle["obs"] if isinstance(settle, dict) else settle

    alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    total = torch.zeros(env.num_actions, device=env.device)
    count = 0

    for _ in range(horizon):
        if not alive.any():
            break
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        for i, player in enumerate(players):
            block = slice(i * args.trials, (i + 1) * args.trials)
            with torch.no_grad():
                actions[block] = player.get_action(obs[block], is_deterministic=True)
        # Only count commands issued while the robot is still balancing.
        total += actions[alive].sum(dim=0)
        count += int(alive.sum())

        obs_dict, _, dones, _ = env.step(actions)
        obs = obs_dict["obs"] if isinstance(obs_dict, dict) else obs_dict
        alive = alive & ~(alive & (dones.to(env.device) > 0))

    mean_action = (total / max(count, 1)).cpu().tolist()
    payload = {
        "source": f"PID_{args.obs_mode} policies, {n_blocks} seeds, nominal condition",
        "samples": count,
        "action": [round(v, 6) for v in mean_action],
        "labelled": {k: round(v, 6) for k, v in zip(ACTION_LABELS, mean_action)},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(payload, open(args.out, "w"), indent=2)
    print(json.dumps(payload["labelled"], indent=2))
    print(f"\nsamples={count}")
    print("--pd-action " + ",".join(f"{v:.6f}" for v in mean_action))


if __name__ == "__main__":
    main()
