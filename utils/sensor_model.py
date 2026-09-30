"""Evaluation-only sensor model: measurement noise, bias, quantization, latency.

Opt-in layer. With the default config (all zeros) every method is a
pass-through, so training behaviour is unchanged by this file existing -- the
training matrix does not need to be redone.

Two properties matter for the experiment to be valid:

1. The up vector is NOT perturbed independently. It is re-derived from the
   already-noisy attitude via the same quaternion helpers the observation
   builder uses, because on a real robot the same IMU error rides on both
   signals. Perturbing them independently yields a physically impossible
   observation that a policy can average out.
2. Whatever reads a "measurement" reads it from here. For BalanceBoardPID the
   PD layer and the policy consume the same noisy sample, otherwise the
   PD-vs-policy comparison runs under two different information conditions --
   the same class of mistake as the original observation-index bug.

Observation layout (identical across the three tasks; only the trailing action
block differs in width):

  restricted: roll,pitch,yaw 0:3 | angvel 3:6 | upvec 6:9 | dofpos 9:31 | actions 31:
  full-state: pos 0:3 | roll,pitch,yaw 3:6 | linvel 6:9 | angvel 9:12 | upvec 12:15
              | dofpos 15:37 | dofvel 37:59 | boardpos 59:62 | boardroll 62
              | boardpitch 63 | boardup 64:67 | tick 67 | actions 68:

Only the onboard channels (attitude, body-frame angular velocity, joint
positions) are perturbed. The full-state-only channels are privileged signals
that no onboard sensor produces, so a "sensor realism" sweep leaves them alone.
"""
import torch

from isaacgym.torch_utils import quat_from_euler_xyz

from isaacgymenvs.utils.torch_jit_utils import get_basis_vector

NUM_DOF = 22


def channel_slices(full_state: bool):
    """Index ranges of the onboard channels for the given observation mode."""
    if full_state:
        att = slice(3, 6)
        angvel = slice(9, 12)
        upvec = slice(12, 15)
        dofpos = slice(15, 15 + NUM_DOF)
    else:
        att = slice(0, 3)
        angvel = slice(3, 6)
        upvec = slice(6, 9)
        dofpos = slice(9, 9 + NUM_DOF)
    return att, angvel, upvec, dofpos


class SensorModel:
    def __init__(self, num_envs: int, device, cfg: dict, full_state: bool):
        cfg = cfg or {}
        noise_cfg = cfg.get("sensorNoise", {}) or {}

        self.num_envs = num_envs
        self.device = device
        self.full_state = full_state

        self.attitude_std = float(noise_cfg.get("attitudeStd", 0.0))
        self.gyro_std = float(noise_cfg.get("gyroStd", 0.0))
        self.encoder_std = float(noise_cfg.get("encoderStd", 0.0))
        self.encoder_resolution = float(noise_cfg.get("encoderResolution", 0.0))
        self.bias_per_episode = bool(noise_cfg.get("biasPerEpisode", True))
        self.delay_steps = int(cfg.get("observationDelaySteps", 0))

        self.noise_enabled = (
            self.attitude_std > 0.0
            or self.gyro_std > 0.0
            or self.encoder_std > 0.0
            or self.encoder_resolution > 0.0
        )
        # `sensorModelActive` keeps the zero-noise baseline (N0/L0) on the *same*
        # code path as the perturbed levels. Without it the baseline would route
        # the PD input through the ground-truth read while N1..N4 route it
        # through obs_buf, so the sweep would confound "noise added" with
        # "PD input became one step stale".
        self.active = bool(cfg.get("sensorModelActive", False))
        self.enabled = self.active or self.noise_enabled or self.delay_steps > 0

        self.attitude_bias = torch.zeros(num_envs, 3, device=device)
        self.gyro_bias = torch.zeros(num_envs, 3, device=device)

        self._slices = channel_slices(full_state)

        self._obs_buffer = None
        self._buffer_len = self.delay_steps + 1
        self._cursor = 0
        self._pending_refill = None

        if self.bias_per_episode and self.noise_enabled:
            self._resample_bias(torch.arange(num_envs, device=device))

    def describe(self) -> str:
        if not self.enabled:
            return "sensor model: disabled (nominal)"
        return (
            f"sensor model: att_std={self.attitude_std:.5f} rad, "
            f"gyro_std={self.gyro_std:.5f} rad/s, enc_std={self.encoder_std:.5f} rad, "
            f"enc_res={self.encoder_resolution:.5f} rad, delay={self.delay_steps} step"
        )

    def _resample_bias(self, env_ids: torch.Tensor):
        n = len(env_ids)
        if n == 0:
            return
        if self.attitude_std > 0.0:
            self.attitude_bias[env_ids] = (
                torch.rand((n, 3), device=self.device) * 2.0 - 1.0
            ) * self.attitude_std
        if self.gyro_std > 0.0:
            self.gyro_bias[env_ids] = (
                torch.rand((n, 3), device=self.device) * 2.0 - 1.0
            ) * self.gyro_std

    def reset(self, env_ids: torch.Tensor):
        """Resample per-episode bias; mark the delay buffer for refill on these envs."""
        if not self.enabled:
            return
        if self.bias_per_episode:
            self._resample_bias(env_ids)
        if self.delay_steps > 0:
            if self._pending_refill is None:
                self._pending_refill = env_ids.clone()
            else:
                self._pending_refill = torch.unique(
                    torch.cat([self._pending_refill, env_ids])
                )

    def apply(self, obs: torch.Tensor, basis_z_vec: torch.Tensor) -> torch.Tensor:
        """Return a noisy, delayed copy of `obs`. Pass-through when disabled."""
        if not self.enabled:
            return obs

        out = obs.clone()
        att_s, angvel_s, upvec_s, dofpos_s = self._slices

        if self.noise_enabled:
            if self.attitude_std > 0.0:
                noise = torch.randn((self.num_envs, 3), device=self.device) * self.attitude_std
                out[:, att_s] = obs[:, att_s] + self.attitude_bias + noise
                # Re-derive the up vector from the noisy attitude rather than
                # perturbing it independently.
                noisy_quat = quat_from_euler_xyz(
                    out[:, att_s.start], out[:, att_s.start + 1], out[:, att_s.start + 2]
                )
                out[:, upvec_s] = get_basis_vector(noisy_quat, basis_z_vec)

            if self.gyro_std > 0.0:
                # Gyro bias/noise live in the body frame, which is what this
                # channel already holds (quat_rotate_inverse was applied upstream).
                noise = torch.randn((self.num_envs, 3), device=self.device) * self.gyro_std
                out[:, angvel_s] = obs[:, angvel_s] + self.gyro_bias + noise

            if self.encoder_resolution > 0.0 or self.encoder_std > 0.0:
                joints = obs[:, dofpos_s]
                if self.encoder_resolution > 0.0:
                    joints = torch.round(joints / self.encoder_resolution) * self.encoder_resolution
                if self.encoder_std > 0.0:
                    joints = joints + torch.randn_like(joints) * self.encoder_std
                out[:, dofpos_s] = joints

        return self._delay(out)

    def _delay(self, obs: torch.Tensor) -> torch.Tensor:
        if self.delay_steps <= 0:
            return obs

        if self._obs_buffer is None:
            self._obs_buffer = obs.unsqueeze(0).repeat(self._buffer_len, 1, 1).clone()
            self._cursor = 0
            self._pending_refill = None

        if self._pending_refill is not None and len(self._pending_refill) > 0:
            # Fresh episode: every slot holds the current measurement, so the
            # first step of an episode never reads a pre-reset observation.
            self._obs_buffer[:, self._pending_refill] = obs[self._pending_refill].unsqueeze(0)
            self._pending_refill = None

        self._obs_buffer[self._cursor] = obs
        self._cursor = (self._cursor + 1) % self._buffer_len
        # After advancing, the cursor points at the oldest retained entry.
        return self._obs_buffer[self._cursor].clone()

    def sensed_attitude(self, obs: torch.Tensor):
        """roll, pitch as the control layer should see them (post-noise, post-delay).

        Read off the same observation the policy receives, so the PD layer and
        the policy cannot diverge in what they are measuring.
        """
        att_s = self._slices[0]
        return obs[:, att_s.start], obs[:, att_s.start + 1]
