"""Evaluation-only: measure foot displacement from where the foot actually started.

The task ships fixed reference positions (l/r_foot_init_position = [-0.04, +-0.055]),
but the feet actually settle at y ~= +-0.075 at spawn. The termination threshold is
3 cm, so in the nominal configuration a foot is already ~2 cm from its reference
before the episode begins and only ~1 cm of travel is left. Any perturbation that
shifts the stance slightly -- a different roller radius, for instance -- trips the
condition on the first step, which makes a model-mismatch sweep unreadable.

Turning this on replaces the fixed reference with each environment's own measured
spawn position, so the criterion becomes "this foot moved 3 cm from where it
started", which is what the manuscript describes.

Default off: with the flag absent, the fixed references are used and training
behaviour is unchanged. Training and evaluation then differ in this one respect,
which has to be stated in the paper.

Capture is deferred by one step on purpose. reset_idx only *queues* the new state
via set_*_tensor_indexed; the simulator applies it during the next
gym.simulate(). Reading rigid_body_state in the same post_physics_step would
capture the pre-reset pose. progress_buf == 1 is the first moment the reset pose
is actually in the tensor.
"""
import torch

L_FOOT_BODY_INDEX = 19
R_FOOT_BODY_INDEX = 25


class FootReference:
    def __init__(self, num_envs: int, device, cfg: dict):
        # `cfg` is already the eval sub-dict, as passed by the tasks.
        self.enabled = bool((cfg or {}).get("footReferenceFromSpawn", False))
        self.device = device
        self.pending = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def mark_reset(self, env_ids: torch.Tensor):
        if not self.enabled:
            return
        self.pending[env_ids] = True

    def maybe_capture(self, rigid_body_state: torch.Tensor, progress_buf: torch.Tensor,
                      l_ref: torch.Tensor, r_ref: torch.Tensor):
        """Write the measured spawn foot positions into the reference tensors in place."""
        if not self.enabled or not bool(self.pending.any()):
            return
        ready = self.pending & (progress_buf == 1)
        if not bool(ready.any()):
            return
        idx = torch.nonzero(ready).flatten()
        l_ref[idx] = rigid_body_state[idx, L_FOOT_BODY_INDEX, 0:2]
        r_ref[idx] = rigid_body_state[idx, R_FOOT_BODY_INDEX, 0:2]
        self.pending[idx] = False
