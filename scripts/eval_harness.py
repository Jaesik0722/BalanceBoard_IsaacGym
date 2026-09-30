"""Common evaluation harness for the robustness experiments (E1 / E2 / E3).

Evaluates already-trained policies. No training happens here.

One invocation = one (task, observation mode, setting) cell of a sweep,
evaluated for every seed at once: the environment holds
``trials x n_policies`` envs, policy i drives the i-th block of ``trials``
envs, and every block starts from the *same* fixed initial conditions so the
seeds are compared on identical trials.

Protocol (fixed before any result is looked at):
  * the evaluation initial-condition set is derived from --condition-seed only,
    never from a policy or a gain choice;
  * every sweep cell includes the nominal level, so the baseline comes out of
    the same run as the perturbed levels;
  * metrics are computed from ground-truth simulator state, never from the
    (possibly noisy) observation the policy sees;
  * raw per-trial rows are written to CSV; aggregation across seeds happens in
    the analysis step, so the raw data survives.

Example:
    python scripts/eval_harness.py \
        --task BalanceBoardPID --obs-mode restricted \
        --setting-name N2 --eval-override eval.sensorNoise.attitudeStd=0.0087 \
        --out results/e1/PID_restricted_N2.csv
"""
# isaacgym must be imported before torch.
from isaacgym import gymapi  # noqa: F401

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime

import numpy as np
import torch
import yaml
from gym import spaces

from isaacgymenvs.tasks import isaacgym_task_map
from isaacgymenvs.utils.reformat import omegaconf_to_dict

TASK_OBS_DIMS = {
    "BalanceBoardPID": {"restricted": 43, "full": 80},
    "BalanceBoardIK": {"restricted": 42, "full": 79},
    "BalanceBoardJoint": {"restricted": 47, "full": 84},
}

FAILURE_CAUSES = [
    "imu_angle",
    "board_ground_contact",
    "torso_displacement",
    "foot_displacement",
    "timeout",
    "survived",
]


def set_nested(d: dict, dotted_key: str, value):
    keys = dotted_key.split(".")
    node = d
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def parse_override(text: str):
    key, _, raw = text.partition("=")
    raw = raw.strip()
    if raw.lower() in ("true", "false"):
        value = raw.lower() == "true"
    else:
        try:
            value = int(raw) if raw.isdigit() or (raw.startswith("-") and raw[1:].isdigit()) else float(raw)
        except ValueError:
            value = raw
    return key.strip(), value


def build_task_cfg(task: str, obs_mode: str, num_envs: int, overrides, episode_steps: int):
    from hydra import compose, initialize_config_dir

    cfg_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cfg")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name="config", overrides=[f"task={task}"])
    cfg_dict = omegaconf_to_dict(cfg.task)

    cfg_dict["env"]["numEnvs"] = num_envs
    cfg_dict["env"]["fullStateObservation"] = (obs_mode == "full")
    cfg_dict["env"]["episodeLength"] = episode_steps
    cfg_dict.setdefault("eval", {})
    for key, value in overrides:
        set_nested(cfg_dict, key, value)

    expected = TASK_OBS_DIMS[task][obs_mode]
    return cfg_dict, expected


def load_players(checkpoints, run_config_path, num_obs, num_actions, device):
    """Build rl_games players without letting them create their own env."""
    from rl_games.algos_torch.players import PpoPlayerContinuous

    run_cfg = yaml.safe_load(open(run_config_path))
    base_params = run_cfg["train"]["params"]

    env_info = {
        "observation_space": spaces.Box(-np.inf, np.inf, (num_obs,), dtype=np.float32),
        "action_space": spaces.Box(-1.0, 1.0, (num_actions,), dtype=np.float32),
        "agents": 1,
        "value_size": 1,
    }

    players = []
    for ckpt in checkpoints:
        params = json.loads(json.dumps(base_params))  # deep copy
        params["config"]["env_info"] = env_info
        params["config"]["device_name"] = device
        params["config"]["num_actors"] = 1
        player = PpoPlayerContinuous(params)
        player.restore(ckpt)
        player.has_batch_dimension = True
        if hasattr(player, "model"):
            player.model.eval()
        players.append(player)
    return players


def fixed_initial_conditions(trials: int, offset_range, condition_seed: int):
    """The evaluation initial-condition set. Depends only on condition_seed."""
    rng = np.random.RandomState(condition_seed)
    lo, hi = offset_range
    return rng.uniform(lo, hi, size=(trials, 2)).astype(np.float32)


def apply_initial_conditions(env, offsets, trials: int, n_blocks: int):
    """Write the fixed conditions into every seed block, then push to the sim.

    Each block of `trials` envs receives the identical offset table, so seed i
    and seed j are evaluated on the same trials.
    """
    from isaacgym import gymtorch

    tiled = np.tile(offsets, (n_blocks, 1))
    tiled_t = torch.from_numpy(tiled).to(env.device)

    robot_ids = env.robinion2s_indexes
    env.reset_positions[robot_ids, 0] = env.robinion2s_pose.p.x + tiled_t[:, 0]
    env.reset_positions[robot_ids, 1] = env.robinion2s_pose.p.y + tiled_t[:, 1]

    all_ids = torch.arange(0, env.num_envs, dtype=torch.long, device=env.device)
    agent_int32 = env.robinion2s_indexes[all_ids].to(torch.int32)
    board_int32 = env.board_indexes[all_ids].to(torch.int32)
    actor_ids = torch.cat([agent_int32, board_int32])

    env.dof_pos[all_ids, :] = env.torch_initial_pos
    env.dof_vel[all_ids, :] = 0.0
    env.gym.set_dof_state_tensor_indexed(
        env.sim, gymtorch.unwrap_tensor(env.dof_states),
        gymtorch.unwrap_tensor(actor_ids), len(actor_ids))

    env.dof_target_tensor[all_ids, :] = env.torch_initial_pos
    env.gym.set_dof_position_target_tensor_indexed(
        env.sim, gymtorch.unwrap_tensor(env.dof_target_tensor),
        gymtorch.unwrap_tensor(actor_ids), len(actor_ids))

    env.gym.set_actor_root_state_tensor_indexed(
        env.sim, gymtorch.unwrap_tensor(env.reset_root_tensor),
        gymtorch.unwrap_tensor(actor_ids), len(actor_ids))

    env.progress_buf[:] = 0
    env.reset_buf[:] = 0
    env.tick[:] = 0


def ground_truth_state(env):
    """Body/board attitude and displacement, read from the simulator, not obs."""
    from isaacgymenvs.utils.torch_jit_utils import get_euler_xyz

    # Refresh here rather than trusting the task to have done it: the tensors are
    # only updated by an explicit refresh, and a task that skips one would hand
    # back spawn-time values for the whole episode.
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)

    root = env.root_tensor
    robot_q = root[env.robinion2s_indexes, 3:7]
    board_q = root[env.board_indexes, 3:7]

    def wrapped(q):
        r, p, y = get_euler_xyz(q)
        r = torch.where(r > np.pi, r - 2 * np.pi, r)
        p = torch.where(p > np.pi, p - 2 * np.pi, p)
        y = torch.where(y > np.pi, y - 2 * np.pi, y)
        return r, p, y

    body_roll, body_pitch, body_yaw = wrapped(robot_q)
    board_roll, board_pitch, _ = wrapped(board_q)

    robot_pos = root[env.robinion2s_indexes, :3]
    torso = torch.sqrt(robot_pos[:, 0] ** 2 + robot_pos[:, 1] ** 2)

    convert = 10.0
    l_foot = env.rigid_body_state[:, 19, 0:2]
    r_foot = env.rigid_body_state[:, 25, 0:2]
    l_dist = torch.norm(l_foot - env.l_foot_init_position, p=2, dim=1) * convert
    r_dist = torch.norm(r_foot - env.r_foot_init_position, p=2, dim=1) * convert

    return {
        "body_roll": body_roll, "body_pitch": body_pitch, "body_yaw": body_yaw,
        "board_roll": board_roll, "board_pitch": board_pitch,
        "torso": torso, "l_foot": l_dist, "r_foot": r_dist,
    }


def classify_failure(state, env, idx: int) -> str:
    """Recompute the termination conditions to say *why* a trial ended."""
    max_imu = math.radians(10)
    if (abs(state["body_roll"][idx].item()) > max_imu
            or abs(state["body_pitch"][idx].item()) > max_imu
            or abs(state["body_yaw"][idx].item()) > max_imu):
        return "imu_angle"
    if env.tick[idx].item() >= env.max_touch_ground_time - 1:
        return "board_ground_contact"
    if state["torso"][idx].item() > 0.20:
        return "torso_displacement"
    if state["l_foot"][idx].item() > 0.30 or state["r_foot"][idx].item() > 0.30:
        return "foot_displacement"
    return "timeout"


STOCHASTIC = [False]


def rollout(env, players, trials, horizon_steps, board_reset_angle_rad, pd_baseline_action=None):
    num_envs = env.num_envs
    device = env.device

    alive = torch.ones(num_envs, dtype=torch.bool, device=device)
    survived = torch.zeros(num_envs, dtype=torch.long, device=device)
    sum_sq = {k: torch.zeros(num_envs, device=device) for k in
              ("body_roll", "body_pitch", "board_roll", "board_pitch")}
    peak = {k: torch.zeros(num_envs, device=device) for k in
            ("body_roll", "body_pitch", "board_roll", "board_pitch")}
    edge_contact = torch.zeros(num_envs, device=device)
    causes = ["survived"] * num_envs

    obs_dict = env.reset()
    obs = obs_dict["obs"] if isinstance(obs_dict, dict) else obs_dict

    # VecTask.reset() hands back obs_buf without ever calling compute_observations,
    # so it is still all zeros, and reset_idx only *queues* the new state -- the
    # simulator applies it on the next simulate(). One settling step is therefore
    # required before the policy sees a real observation. Acting on the zero
    # vector produces a garbage first command that some policies survive and
    # others do not, which reads exactly like seed-to-seed variance.
    settle_dict, _, _, _ = env.step(torch.zeros(num_envs, env.num_actions, device=device))
    obs = settle_dict["obs"] if isinstance(settle_dict, dict) else settle_dict

    for _ in range(horizon_steps):
        if not alive.any():
            break

        if pd_baseline_action is not None:
            actions = pd_baseline_action.repeat(num_envs, 1)
        else:
            actions = torch.zeros(num_envs, env.num_actions, device=device)
            for i, player in enumerate(players):
                block = slice(i * trials, (i + 1) * trials)
                with torch.no_grad():
                    actions[block] = player.get_action(obs[block], is_deterministic=not STOCHASTIC[0])

        obs_dict, _, dones, _ = env.step(actions)
        obs = obs_dict["obs"] if isinstance(obs_dict, dict) else obs_dict

        state = ground_truth_state(env)
        live = alive.clone()

        survived += live.long()
        for key in sum_sq:
            v = state[key]
            sum_sq[key] += torch.where(live, v ** 2, torch.zeros_like(v))
            peak[key] = torch.where(live & (v.abs() > peak[key]), v.abs(), peak[key])

        board_angle = torch.sqrt(state["board_roll"] ** 2 + state["board_pitch"] ** 2)
        edge_contact += torch.where(live & (board_angle >= board_reset_angle_rad),
                                    torch.ones_like(edge_contact), torch.zeros_like(edge_contact))

        newly_done = live & (dones.to(device) > 0)
        for idx in torch.nonzero(newly_done).flatten().tolist():
            causes[idx] = classify_failure(state, env, idx)
        alive = alive & ~newly_done

    rows = []
    for idx in range(num_envs):
        n = max(int(survived[idx].item()), 1)
        rows.append({
            "trial": idx % trials,
            "policy_block": idx // trials,
            "survived_steps": int(survived[idx].item()),
            "survived_s": round(int(survived[idx].item()) * 0.01, 3),
            "success_10s": int(survived[idx].item() >= 1000),
            "success_60s": int(survived[idx].item() >= 6000),
            "body_roll_rms": round(math.sqrt(sum_sq["body_roll"][idx].item() / n), 6),
            "body_pitch_rms": round(math.sqrt(sum_sq["body_pitch"][idx].item() / n), 6),
            "board_roll_rms": round(math.sqrt(sum_sq["board_roll"][idx].item() / n), 6),
            "board_pitch_rms": round(math.sqrt(sum_sq["board_pitch"][idx].item() / n), 6),
            "body_roll_max": round(peak["body_roll"][idx].item(), 6),
            "body_pitch_max": round(peak["body_pitch"][idx].item(), 6),
            "board_roll_max": round(peak["board_roll"][idx].item(), 6),
            "board_pitch_max": round(peak["board_pitch"][idx].item(), 6),
            "board_edge_contact_ratio": round(edge_contact[idx].item() / n, 6),
            "failure_cause": causes[idx],
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASK_OBS_DIMS))
    ap.add_argument("--obs-mode", required=True, choices=["restricted", "full"])
    ap.add_argument("--checkpoints", nargs="*", default=None,
                    help="one checkpoint per seed; omit with --pd-baseline")
    ap.add_argument("--run-config", default=None,
                    help="runs/<exp>/config.yaml providing the rl_games params")
    ap.add_argument("--experiment", default="E1")
    ap.add_argument("--setting-name", default="nominal")
    ap.add_argument("--eval-override", action="append", default=[],
                    metavar="dotted.key=value")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--duration-s", type=float, default=60.0)
    ap.add_argument("--condition-seed", type=int, default=20260919)
    ap.add_argument("--stochastic", action="store_true", help="sample actions instead of using the mean")
    ap.add_argument("--pd-baseline", action="store_true",
                    help="evaluate fixed-gain PD instead of a policy (BalanceBoardPID only)")
    ap.add_argument("--pd-action", default=None,
                    help="comma-separated action vector for the PD baseline; default all zeros (mid-range gains)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    overrides = [parse_override(o) for o in args.eval_override]
    horizon_steps = int(round(args.duration_s / 0.01))

    if args.pd_baseline:
        if args.task != "BalanceBoardPID":
            sys.exit("--pd-baseline only applies to BalanceBoardPID")
        n_blocks = 1
    else:
        if not args.checkpoints:
            sys.exit("--checkpoints is required unless --pd-baseline is set")
        n_blocks = len(args.checkpoints)

    num_envs = args.trials * n_blocks
    cfg_dict, expected_obs = build_task_cfg(
        args.task, args.obs_mode, num_envs, overrides, horizon_steps)

    env = isaacgym_task_map[args.task](
        cfg=cfg_dict, rl_device="cuda:0", sim_device="cuda:0",
        graphics_device_id=0, headless=True,
        virtual_screen_capture=False, force_render=False,
    )
    assert env.obs_buf.shape[1] == expected_obs, (
        f"observation dim {env.obs_buf.shape[1]} != expected {expected_obs}")

    players = []
    pd_action = None
    if args.pd_baseline:
        if args.pd_action:
            values = [float(x) for x in args.pd_action.split(",")]
        else:
            values = [0.0] * env.num_actions
        pd_action = torch.tensor(values, device=env.device).unsqueeze(0)
    else:
        run_config = args.run_config or os.path.join(
            os.path.dirname(os.path.dirname(args.checkpoints[0])), "config.yaml")
        players = load_players(args.checkpoints, run_config, expected_obs,
                               env.num_actions, "cuda:0")

    offsets = fixed_initial_conditions(
        args.trials, cfg_dict["env"]["offsetRange"], args.condition_seed)
    apply_initial_conditions(env, offsets, args.trials, n_blocks)

    STOCHASTIC[0] = args.stochastic
    board_reset_angle_rad = math.radians(cfg_dict["env"]["boardResetAngle"])
    rows = rollout(env, players, args.trials, horizon_steps,
                   board_reset_angle_rad, pd_baseline_action=pd_action)

    meta = {
        "experiment": args.experiment,
        "setting": args.setting_name,
        "task": args.task,
        "obs_mode": args.obs_mode,
        "controller": "pd_baseline" if args.pd_baseline else "policy",
        "condition_seed": args.condition_seed,
        "trials": args.trials,
        "duration_s": args.duration_s,
        "overrides": ";".join(f"{k}={v}" for k, v in overrides) or "none",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    for i, ckpt in enumerate(args.checkpoints or []):
        meta.setdefault("checkpoints", "")
        meta["checkpoints"] += ("" if i == 0 else ";") + os.path.basename(ckpt)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fieldnames = list(meta) + ["seed_index"] + list(rows[0])
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            record = dict(meta)
            record["seed_index"] = row["policy_block"]
            record.update(row)
            writer.writerow(record)

    # The per-row success_60s column predates the horizon fix and undercounts by one step;
    # the reported statistics come from analyze_results.py, which tests failure_cause.
    # This console line uses the same test so the two cannot disagree.
    n_ok = sum(r["failure_cause"] == "timeout" for r in rows)
    print(f"[{args.task}/{args.obs_mode}/{args.setting_name}] "
          f"{len(rows)} trials, 60s success {n_ok}/{len(rows)} -> {args.out}")


if __name__ == "__main__":
    main()
