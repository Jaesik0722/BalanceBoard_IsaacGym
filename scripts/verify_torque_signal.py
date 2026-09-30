"""Check what acquire_dof_force_tensor actually reports on this model.

The coordination analysis calls the signal "the torque the position servo applied". That
name is only justified if the sensor reading matches the drive law the servo implements.
Every DOF here carries the same uniform gains, so the drive torque is computable in closed
form:

    tau_drive = clip(stiffness * (target - q) - damping * qdot, -effort, +effort)

If the sensor tracks this, the signal is drive torque. If it departs from it, the reading
also carries constraint, contact or inertial terms projected onto the DOF axis, and the
analysis has to be renamed and requalified.
"""
from isaacgym import gymapi, gymtorch  # noqa: F401  (must precede torch)

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_harness import build_task_cfg, load_players  # noqa: E402
from coordination_eval import TASKS, find_checkpoints, STABILISE_S  # noqa: E402

from isaacgymenvs.tasks import isaacgym_task_map  # noqa: E402

STIFFNESS, DAMPING, EFFORT = 80.0, 2.0, 9.9

cks = find_checkpoints("IK", "restricted", [1], "3000")
assert cks, "no checkpoint"
n_env, horizon = 8, 500
cfg, dim = build_task_cfg(TASKS["IK"], "restricted", n_env,
                          [("eval.logJointTorque", True),
                           ("eval.footReferenceFromSpawn", True)], horizon)
env = isaacgym_task_map[TASKS["IK"]](cfg=cfg, rl_device="cuda:0", sim_device="cuda:0",
                                     graphics_device_id=0, headless=True,
                                     virtual_screen_capture=False, force_render=False)
players = load_players(cks, os.path.join(os.path.dirname(os.path.dirname(cks[0])), "config.yaml"),
                       dim, env.num_actions, "cuda:0")
dof_force = gymtorch.wrap_tensor(env.gym.acquire_dof_force_tensor(env.sim))

obs = env.reset()["obs"]
o, _, _, _ = env.step(torch.zeros(env.num_envs, env.num_actions, device=env.device))
obs = o["obs"] if isinstance(o, dict) else o

sens, pred = [], []
for step in range(300):
    a = players[0].get_action(obs, is_deterministic=True)
    # The drive acts on the state at the START of the step against the target set during
    # this step, so both have to be sampled with that timing, not after the solve.
    env.gym.refresh_dof_state_tensor(env.sim)
    q0, v0 = env.dof_pos.clone(), env.dof_vel.clone()
    with torch.no_grad():
        o, _, dones, _ = env.step(a)
    obs = o["obs"] if isinstance(o, dict) else o
    env.gym.refresh_dof_force_tensor(env.sim)
    if step > 50:
        tau = (STIFFNESS * (env.dof_target_tensor - q0)
               - DAMPING * v0).clamp(-EFFORT, EFFORT)
        sens.append(dof_force.view(env.num_envs, -1).clone())
        pred.append(tau.clone())

S = torch.cat(sens).flatten().double()
P = torch.cat(pred).flatten().double()
keep = S.abs() + P.abs() > 1e-6
S, P = S[keep], P[keep]
corr = torch.corrcoef(torch.stack([S, P]))[0, 1].item()
print(f"samples                 {S.numel()}")
print(f"sensor  |tau|  mean/max {S.abs().mean():.4f} / {S.abs().max():.4f}")
print(f"PD pred |tau|  mean/max {P.abs().mean():.4f} / {P.abs().max():.4f}")
print(f"Pearson r               {corr:.6f}")
print(f"mean |sensor - pred|    {(S - P).abs().mean():.6f}")
print(f"ratio sensor/pred       {(S.abs().sum() / P.abs().sum()).item():.6f}")
print(f"frac |sensor| > effort  {(S.abs() > EFFORT + 1e-3).double().mean():.6f}")
