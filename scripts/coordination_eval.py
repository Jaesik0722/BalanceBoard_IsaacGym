"""Joint coordination under a perturbation applied to an already-balancing robot.

This is the measurement that lets the paper say something about postural coordination
rather than only about success rates. The protocol below is fixed before the run.

  1. Perturbation is applied *after* stabilisation, not by starting the board tilted.
     Starting from a tilt asks how a policy recovers from a bad initial condition, which
     is a different question from how a standing system answers a push, and it is the
     latter that the human postural literature measures.
  2. Pitch and roll are analysed separately. The joints, and the meaning of a response,
     differ between the sagittal and frontal planes; collapsing them into one ratio would
     mix two mechanisms.
  3. Two torque measures are recorded, because the conclusion must not depend on which one
     is chosen. The first is the simulator's DOF force sensor, which reports the generalised
     force along the joint axis produced by the articulation solver. The second is the
     explicit evaluation of the position drive law,
     clip(k*(target - q) - d*qdot, -effort, +effort), computed from the state at the start of
     the step. These are NOT the same signal on this model: the sensor reading correlates
     with the drive-law evaluation at r = 0.58 and is about six times smaller in magnitude,
     so calling either one "the torque the servo applied" would be an unverified claim. Both
     are reported as joint torque activity, and neither is a causal share of balance control,
     which a single ratio cannot establish.
  4. Failure handling is decided here, not after seeing the numbers. A trial that never
     reached the stabilisation criterion is "not analysable", not a zero. A trial that
     stabilised and then fell is kept in the success statistics and excluded from the
     coordination window only if it did not survive that window; the counts are reported.

Ankle and hip joint indices follow the DOF layout: left leg 10--15 and right leg 16--21,
ordered hip_yaw, hip_roll, hip_pitch, knee_pitch, ankle_pitch, ankle_roll.
"""
# isaacgym must be imported before torch.
from isaacgym import gymapi, gymtorch

import argparse
import csv
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_harness import build_task_cfg, load_players  # noqa: E402

from isaacgymenvs.tasks import isaacgym_task_map  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = {"PID": "BalanceBoardPID", "IK": "BalanceBoardIK", "Joint": "BalanceBoardJoint"}

# Sagittal (pitch) and frontal (roll) joints, left and right.
HIP_PITCH, ANKLE_PITCH = [12, 18], [14, 20]
HIP_ROLL, ANKLE_ROLL = [11, 17], [15, 21]

# The policies do not settle into a quiet stance. Measured over surviving trials of a
# competent policy, body tilt sits at a median of 4-5 deg with a 90th percentile near 8 deg
# (termination is at 10 deg) and body angular rate at a median of ~0.7 rad/s. They hold
# balance by oscillating inside the failure envelope rather than by converging to an
# equilibrium. The pre-perturbation criterion is therefore "inside its normal operating
# envelope and not already failing", not "quiet": any comparison with human perturbation
# studies, which perturb from quiet stance, has to be read with that difference in mind.
STABILISE_S = 3.0
STABLE_TILT = 0.140    # rad, 8 deg -- below the 10 deg termination bound
STABLE_RATE = 1.50     # rad/s -- above the 90th percentile of normal operation
# Uniform drive parameters, identical on every DOF (set in the task), used to evaluate the
# drive law as the second torque measure.
STIFFNESS, DAMPING, EFFORT = 80.0, 2.0, 9.9
PUSH_DURATION_S = 0.1
WINDOW_S = 2.0         # analysis window after push onset


def find_checkpoints(short, mode, seeds, epoch):
    cks = []
    for s in seeds:
        nn = sorted(glob.glob(os.path.join(ROOT, "runs", f"{short}_{mode}_3k_seed{s}_*", "nn")))
        if not nn:
            return None
        hit = glob.glob(os.path.join(nn[-1], f"last_*_ep_{epoch}_*.pth"))
        if not hit:
            return None
        cks.append(hit[0])
    return cks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--obs-mode", default="restricted")
    ap.add_argument("--axis", default="pitch", choices=["pitch", "roll"])
    ap.add_argument("--force", type=float, required=True, help="push magnitude in N")
    ap.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3])
    ap.add_argument("--epoch", default="3000")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trace", help="npz path for per-step ankle/hip torque and body tilt")
    args = ap.parse_args()

    cks = find_checkpoints(args.task, args.obs_mode, args.seeds, args.epoch)
    if cks is None:
        sys.exit(f"no checkpoints for {args.task}/{args.obs_mode}")

    per = args.trials
    n_env = per * len(args.seeds)
    horizon = int(round((STABILISE_S + WINDOW_S + 1.0) / 0.01))
    cfg, dim = build_task_cfg(TASKS[args.task], args.obs_mode, n_env,
                              [("eval.logJointTorque", True),
                               ("eval.footReferenceFromSpawn", True)], horizon)
    env = isaacgym_task_map[TASKS[args.task]](
        cfg=cfg, rl_device="cuda:0", sim_device="cuda:0", graphics_device_id=0,
        headless=True, virtual_screen_capture=False, force_render=False)
    players = load_players(cks, os.path.join(os.path.dirname(os.path.dirname(cks[0])), "config.yaml"),
                           dim, env.num_actions, "cuda:0")

    dof_force = gymtorch.wrap_tensor(env.gym.acquire_dof_force_tensor(env.sim))

    hip = HIP_PITCH if args.axis == "pitch" else HIP_ROLL
    ankle = ANKLE_PITCH if args.axis == "pitch" else ANKLE_ROLL
    push_dir = 0 if args.axis == "pitch" else 1          # +x is sagittal, +y frontal
    torso = env.rigid_body_state.shape[1]                # bodies per env
    forces = torch.zeros((env.num_envs, torso, 3), device=env.device)

    obs = env.reset()["obs"]
    o, _, _, _ = env.step(torch.zeros(env.num_envs, env.num_actions, device=env.device))
    obs = o["obs"] if isinstance(o, dict) else o

    alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    stabilised = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    hip_t = torch.zeros(env.num_envs, device=env.device)
    ank_t = torch.zeros(env.num_envs, device=env.device)
    hip_d = torch.zeros(env.num_envs, device=env.device)
    ank_d = torch.zeros(env.num_envs, device=env.device)
    hip_k = torch.zeros(env.num_envs, device=env.device)
    ank_k = torch.zeros(env.num_envs, device=env.device)
    win_n = torch.zeros(env.num_envs, device=env.device)
    trace = {"ank": [], "hip": [], "ank_drive": [], "hip_drive": [],
             "tilt": [], "alive": [], "stab": []} if args.trace else None
    push_start = int(STABILISE_S / 0.01)
    push_end = push_start + int(PUSH_DURATION_S / 0.01)
    win_end = push_start + int(WINDOW_S / 0.01)

    for step in range(horizon):
        if step == push_start:
            env.gym.refresh_actor_root_state_tensor(env.sim)
            from isaacgymenvs.utils.torch_jit_utils import get_euler_xyz
            q = env.root_tensor[env.robinion2s_indexes, 3:7]
            r, p, _ = get_euler_xyz(q)
            import numpy as np
            r = torch.where(r > np.pi, r - 2 * np.pi, r)
            p = torch.where(p > np.pi, p - 2 * np.pi, p)
            w = env.root_tensor[env.robinion2s_indexes, 10:13]
            stabilised = alive & (r.abs() < STABLE_TILT) & (p.abs() < STABLE_TILT) \
                & (w.abs().max(dim=1).values < STABLE_RATE)

        a = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        for i, pl in enumerate(players):
            with torch.no_grad():
                a[i*per:(i+1)*per] = pl.get_action(obs[i*per:(i+1)*per], is_deterministic=True)

        if push_start <= step < push_end:
            forces.zero_()
            forces[:, 0, push_dir] = args.force
            env.gym.apply_rigid_body_force_tensors(
                env.sim, gymtorch.unwrap_tensor(forces.view(-1, 3)), None, gymapi.ENV_SPACE)

        env.gym.refresh_dof_state_tensor(env.sim)
        q0, v0 = env.dof_pos.clone(), env.dof_vel.clone()

        o, _, dones, _ = env.step(a)
        obs = o["obs"] if isinstance(o, dict) else o

        if push_start <= step < win_end:
            env.gym.refresh_dof_force_tensor(env.sim)
            tq = dof_force.view(env.num_envs, -1).abs()
            m = (alive & stabilised).float()
            hip_t += m * tq[:, hip].sum(dim=1)
            ank_t += m * tq[:, ankle].sum(dim=1)
            drive = (STIFFNESS * (env.dof_target_tensor - q0)
                     - DAMPING * v0).clamp(-EFFORT, EFFORT).abs()
            hip_d += m * drive[:, hip].sum(dim=1)
            ank_d += m * drive[:, ankle].sum(dim=1)
            dq = (env.dof_pos - env.torch_initial_pos).abs()
            hip_k += m * dq[:, hip].sum(dim=1)
            ank_k += m * dq[:, ankle].sum(dim=1)
            win_n += m

        if trace is not None:
            env.gym.refresh_dof_force_tensor(env.sim)
            env.gym.refresh_actor_root_state_tensor(env.sim)
            from isaacgymenvs.utils.torch_jit_utils import get_euler_xyz as _ge
            import numpy as _np
            _q = env.root_tensor[env.robinion2s_indexes, 3:7]
            _r, _p, _ = _ge(_q)
            _r = torch.where(_r > _np.pi, _r - 2 * _np.pi, _r)
            _p = torch.where(_p > _np.pi, _p - 2 * _np.pi, _p)
            _tq = dof_force.view(env.num_envs, -1).abs()
            _dr = (STIFFNESS * (env.dof_target_tensor - q0)
                   - DAMPING * v0).clamp(-EFFORT, EFFORT).abs()
            trace["ank"].append(_tq[:, ankle].sum(dim=1).cpu().numpy())
            trace["hip"].append(_tq[:, hip].sum(dim=1).cpu().numpy())
            trace["ank_drive"].append(_dr[:, ankle].sum(dim=1).cpu().numpy())
            trace["hip_drive"].append(_dr[:, hip].sum(dim=1).cpu().numpy())
            trace["tilt"].append((_p if args.axis == "pitch" else _r).cpu().numpy())
            trace["alive"].append(alive.cpu().numpy().copy())
            trace["stab"].append(stabilised.cpu().numpy().copy())

        alive = alive & ~(alive & (dones.to(env.device) > 0))
        if os.environ.get("COORD_DEBUG") and step >= push_start and (step - push_start) % 10 == 0:
            print(f"  step +{step-push_start:3d} ({(step-push_start)*0.01:.2f}s) "
                  f"alive {int(alive.sum())}/{env.num_envs} "
                  f"alive&stab {int((alive & stabilised).sum())}", flush=True)

    rows = []
    for i in range(env.num_envs):
        n = max(float(win_n[i]), 1.0)
        analysable = bool(stabilised[i]) and float(win_n[i]) >= (win_end - push_start) * 0.5
        rows.append({
            "task": args.task, "obs_mode": args.obs_mode, "axis": args.axis,
            "force_N": args.force, "seed_index": i // per, "trial": i % per,
            "stabilised": int(bool(stabilised[i])),
            "survived_window": int(float(win_n[i]) >= (win_end - push_start) * 0.5),
            "analysable": int(analysable),
            "hip_torque_activity": round(float(hip_t[i]) / n, 6),
            "ankle_torque_activity": round(float(ank_t[i]) / n, 6),
            "hip_drive_activity": round(float(hip_d[i]) / n, 6),
            "ankle_drive_activity": round(float(ank_d[i]) / n, 6),
            "hip_excursion_rad": round(float(hip_k[i]) / n, 6),
            "ankle_excursion_rad": round(float(ank_k[i]) / n, 6),
        })
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if trace is not None:
        import numpy as _np
        _np.savez(args.trace, push_start=push_start, push_end=push_end,
                  win_end=win_end, dt=0.01,
                  **{k: _np.array(v) for k, v in trace.items()})
        print("trace ->", args.trace)

    ok = sum(r["analysable"] for r in rows)
    print(f"[{args.task}/{args.obs_mode}/{args.axis}/{args.force:.0f}N] "
          f"stabilised {sum(r['stabilised'] for r in rows)}/{len(rows)}, "
          f"analysable {ok}/{len(rows)} -> {args.out}")


if __name__ == "__main__":
    main()
