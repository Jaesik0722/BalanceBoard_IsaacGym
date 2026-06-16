import numpy as np
import os
import torch
import math
import random

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask

import enum
from isaacgymenvs.utils.filter import *

START_NUM = 0
LEG_LENGTH = 190.0
HIP_LENGTH = 110.0
SIT_VALUE = 30.0

SPACING = 1.0

NP_INITIAL_POS = np.array([0., 0., 
    0.0873, -1.3439, 0., -0.2618, 
    0.0873, 1.3439, 0., 0.2618, 
    0., 0., -0.4, 0.8, -0.4, 0.,
    0., 0., -0.4, 0.8, -0.4, 0.], dtype=np.float32)

class LEG(enum.Enum):
    HIP_YAW = START_NUM
    HIP_ROLL = enum.auto()
    HIP_PITCH = enum.auto()
    KNEE_PITCH = enum.auto()
    ANKLE_PITCH = enum.auto()
    ANKLE_ROLL = enum.auto()


class BalanceBoardJoint(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.max_touch_ground_time = self.cfg["env"]["touchGroundTime"]
        self.num_rewards = self.cfg["env"]["numRewards"]
        self.reward_weight = self.cfg["env"]["rewardWeight"]
        self.board_reset_angle = self.cfg["env"]["boardResetAngle"]
        self._offset_range = self.cfg["env"]["offsetRange"]

        # Ideal Observations: 84
        # Real World Possible Observations: 47
        self.cfg["env"]["numObservations"] = 84
        
        self.cfg["env"]["numActions"] = 16

        self.dt = self.cfg["sim"]["dt"]

        self.torch_initial_pos = to_torch(NP_INITIAL_POS)

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.sit_value = torch.ones(self.num_envs, device=self.device) * SIT_VALUE
        
        self.filter_arm = FILTER_TORCH(self.num_envs, device=self.device)
        self.filter_leg = FILTER_TORCH(self.num_envs, device=self.device)

        self.joint_position_upper = torch.ones((self.num_envs, 14), dtype=torch.float32, device=self.device) * (math.pi/2)
        self.joint_position_lower = torch.ones((self.num_envs, 14), dtype=torch.float32, device=self.device) * -(math.pi/2)
        
        self.lpf_alpha_upper = torch.ones((self.num_envs, 2), dtype=torch.float32, device=self.device) * 0.1
        self.lpf_alpha_lower = torch.ones((self.num_envs, 2), dtype=torch.float32, device=self.device) * 0.00001

        self._range_offset = [-0.005, 0.005]# Set torch variables for IK

        self.up_vec = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))
        
        self.basis_x_vec = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.basis_y_vec = to_torch([0, 1, 0], device=self.device).repeat((self.num_envs, 1))
        self.basis_z_vec = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))
        
        self.l_foot_init_position = to_torch([-0.04, 0.055], device=self.device).repeat((self.num_envs, 1))
        self.r_foot_init_position = to_torch([-0.04, -0.055], device=self.device).repeat((self.num_envs, 1))
        
        self.tick = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.count = 0

        self._create_view()
        self._generate_indexes()
        self._allocate_tensors()
        self.reset_idx(self.all_env_indexes)

    
    def create_sim(self):
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs)))

    
    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    
    def _create_view(self):
        if self.viewer != None:
            pose = self.cfg["env"]["viewer"]["pose"]
            target = self.cfg["env"]["viewer"]["target"]
            cam_pos = gymapi.Vec3(pose[0], pose[1], pose[2])
            cam_target = gymapi.Vec3(target[0], target[1], target[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    
    def _load_asset(self):
        robinion2s_asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        robinion2s_asset_path = "urdf/robinion2s.urdf"

        robinion2s_asset_options = gymapi.AssetOptions()
        robinion2s_asset_options.fix_base_link = False

        # Load all Robinon2S asset
        self.robinion2s_asset = self.gym.load_asset(
            self.sim, robinion2s_asset_root, robinion2s_asset_path, robinion2s_asset_options
        )

        self.robinion2s_pose = gymapi.Transform()
        self.robinion2s_pose.p.y = 0.0245
        self.robinion2s_pose.p.x = 0.02
        self.robinion2s_pose.p.z = 0.612
        self.robinion2s_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Create Board asset
        board_asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        board_asset_path = "urdf/board.urdf"

        board_asset_options = gymapi.AssetOptions()
        board_asset_options.fix_base_link = False

        # Load all Board asset
        self.board_asset = self.gym.load_asset(self.sim, board_asset_root, board_asset_path, board_asset_options)

        self.board_pose = gymapi.Transform()
        self.board_pose.p.z = 0.064
        x, y, z, w = self.quat_from_euler_deg(roll_deg=0.0, pitch_deg=0.0)
        self.board_pose.r = gymapi.Quat(x, y, z, w)
        
        
    def _create_envs(self, num_envs, spacing, num_per_row):
        self._load_asset()

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        envs = []
        robinion2s_handles = []
        board_handles = []

        for i in range(num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

            # Create Robinion2S
            robinion2s_handle = self.gym.create_actor(
                env_ptr, self.robinion2s_asset, self.robinion2s_pose, "robinion2s", 0, 0
            )

            props = self.gym.get_actor_dof_properties(env_ptr, robinion2s_handle)
            props["driveMode"] = gymapi.DOF_MODE_POS
            props["stiffness"] = 80.0
            props["damping"] = 2.0
            self.gym.set_actor_dof_properties(env_ptr, robinion2s_handle, props)
            self.num_dofs = self.gym.get_actor_dof_count(env_ptr, robinion2s_handle)

            # Create Balance board
            board_handle = self.gym.create_actor(env_ptr, self.board_asset, self.board_pose, "board", 0, 0)

            robinion2s_handles.append(robinion2s_handle)
            board_handles.append(board_handle)
            envs.append(env_ptr)

    
    def _allocate_tensors(self):
        self._root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_tensor = gymtorch.wrap_tensor(self._root_tensor)
        self.root_positions = self.root_tensor[:, 0:3]
        self.root_orientations = self.root_tensor[:, 3:7]
        self.root_linvels = self.root_tensor[:, 7:10]
        self.root_angvels = self.root_tensor[:, 10:13]

        self._dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(self._dof_states)
        self.dof_pos = self.dof_states.view(self.num_envs, self.num_dofs, 2)[:, :, 0]
        self.dof_vel = self.dof_states.view(self.num_envs, self.num_dofs, 2)[:, :, 1]

        self._rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)        
        self.rigid_body_state = gymtorch.wrap_tensor(self._rigid_body_tensor).view(self.num_envs, -1, 13)
        
        # Refresh tensors.
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.reset_root_tensor = self.root_tensor.clone()
        self.reset_positions = self.reset_root_tensor[:, 0:3]
        self.reset_root_tensor[:, 7:] = 0.0

        self.dof_target_tensor = self.dof_pos.clone()

    
    def _generate_indexes(self):
        self.all_env_indexes = torch.arange(0, self.num_envs, dtype=torch.long, device=self.device)
        self.robinion2s_indexes = torch.arange(0, self.num_envs * 2, 2, dtype=torch.long, device=self.device)
        self.board_indexes = self.robinion2s_indexes + 1


    def reset_idx(self, reset_env_ids):
        self.filter_arm.reset_lowpass_filter(reset_env_ids)
        self.filter_leg.reset_lowpass_filter(reset_env_ids)
        # Convert to int32 for tensor API calls.
        reset_agent_inds_int32 = self.robinion2s_indexes[reset_env_ids].to(torch.int32)
        reset_board_inds_int32 = self.board_indexes[reset_env_ids].to(torch.int32)
        reset_actor_ids_int32 = torch.cat([reset_agent_inds_int32, reset_board_inds_int32])

        self.dof_pos[reset_env_ids, :] = self.torch_initial_pos
        self.dof_vel[reset_env_ids, :] = 0.0
        
        self.reset_positions[self.robinion2s_indexes[reset_env_ids],0] = self.robinion2s_pose.p.x + random.uniform(self._offset_range[0], self._offset_range[1])
        self.reset_positions[self.robinion2s_indexes[reset_env_ids],1] = self.robinion2s_pose.p.y + random.uniform(self._offset_range[0], self._offset_range[1])
        
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_states),
            gymtorch.unwrap_tensor(reset_actor_ids_int32),
            len(reset_actor_ids_int32))

        self.dof_target_tensor[reset_env_ids, :] = self.torch_initial_pos
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_target_tensor),
            gymtorch.unwrap_tensor(reset_actor_ids_int32),
            len(reset_actor_ids_int32))

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.reset_root_tensor),
            gymtorch.unwrap_tensor(reset_actor_ids_int32),
            len(reset_actor_ids_int32))

        self.progress_buf[reset_env_ids] = 0
        self.reset_buf[reset_env_ids] = 0
        self.tick[reset_env_ids] = 0
        self.count = self.count + 1
        
    
    def pre_physics_step(self, actions):
        self.actions = actions[:]

        joint_position = unscale_transform(actions[:, 0:14], self.joint_position_lower, self.joint_position_upper)
        lpf_alpha = unscale_transform(actions[:, 14:16], self.lpf_alpha_lower, self.lpf_alpha_upper)
        
        for index in range(4):
            joint_position[:, index] = self.filter_arm.update_lowpass_filter(joint_position[:, index], lpf_alpha[:, 0], self.all_env_indexes)
            
        for index in range(4, 14):
            joint_position[:, index] = self.filter_leg.update_lowpass_filter(joint_position[:, index], lpf_alpha[:, 1], self.all_env_indexes)
        
        self._send_dof_targets(joint_position)

    
    def _send_dof_targets(self, joint_position):
        
        self.dof_target_tensor[:, 2] = self.torch_initial_pos[2] + joint_position[:, 0]
        self.dof_target_tensor[:, 6] = self.torch_initial_pos[6] + joint_position[:, 1]
        self.dof_target_tensor[:, 3] = self.torch_initial_pos[3] + joint_position[:, 2]
        self.dof_target_tensor[:, 7] = self.torch_initial_pos[7] + joint_position[:, 3]
        
        # Left leg
        self.dof_target_tensor[:, 11] = self.torch_initial_pos[11] + joint_position[:, 4]
        self.dof_target_tensor[:, 12] = self.torch_initial_pos[12] + joint_position[:, 5]
        self.dof_target_tensor[:, 13] = self.torch_initial_pos[13] + joint_position[:, 6]
        self.dof_target_tensor[:, 14] = self.torch_initial_pos[14] + joint_position[:, 7]
        self.dof_target_tensor[:, 15] = self.torch_initial_pos[15] + joint_position[:, 8]
        
        # Right leg
        self.dof_target_tensor[:, 17] = self.torch_initial_pos[17] + joint_position[:, 9]
        self.dof_target_tensor[:, 18] = self.torch_initial_pos[18] + joint_position[:, 10]
        self.dof_target_tensor[:, 19] = self.torch_initial_pos[19] + joint_position[:, 11]
        self.dof_target_tensor[:, 20] = self.torch_initial_pos[20] + joint_position[:, 12]
        self.dof_target_tensor[:, 21] = self.torch_initial_pos[21] + joint_position[:, 13]
        
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_target_tensor))

    
    def post_physics_step(self):
        self.progress_buf += 1

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.compute_observations()
        self.compute_reward()

    
    def compute_reward(self):
        self.rew_buf, self.reset_buf[:], self.tick[:] = compute_reward(
            self.robinion2s_indexes,
            self.board_indexes,
            self.obs_buf,
            self.root_tensor,
            self.rigid_body_state,
            self.progress_buf,
            self.tick,
            self.basis_x_vec,
            self.basis_y_vec,
            self.basis_z_vec,
            self.l_foot_init_position,
            self.r_foot_init_position,
            self.max_episode_length,
            self.max_touch_ground_time,
            self.num_rewards,
            self.reward_weight,
            self.board_reset_angle
        )

    
    def compute_observations(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)       

        self.obs_buf[:] = compute_observations(
            self.root_tensor, 
            self.dof_pos, 
            self.dof_vel, 
            self.actions, 
            self.robinion2s_indexes, 
            self.board_indexes,
            self.basis_x_vec,
            self.basis_y_vec,
            self.basis_z_vec,
            self.tick
        )

    def quat_from_euler_deg(self, roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0):        
        roll = math.radians(roll_deg)
        pitch = math.radians(pitch_deg)
        yaw = math.radians(yaw_deg)

        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return x, y, z, w


#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def compute_reward(
    robinion2s_ids: torch.Tensor, 
    board_ids: torch.Tensor,
    obs_buf: torch.Tensor,
    root_tensor : torch.Tensor,
    rigid_body_state: torch.Tensor,
    progress_buf: torch.Tensor,
    tick: torch.Tensor,
    basis_x_vec: torch.Tensor,
    basis_y_vec: torch.Tensor,
    basis_z_vec: torch.Tensor,
    l_foot_init_position: torch.Tensor,
    r_foot_init_position: torch.Tensor,
    max_episode_length: int,
    max_touch_ground_time: int,
    num_rewards: int,
    reward_weight: float,
    board_reset_angle: int
):
    max_imu_angle = math.radians(10)
    max_board_angle = math.radians(board_reset_angle)
    max_torso_position = 0.20
    max_foot_position = 0.30
    convert_mm_to_cm = 10
    
    robinion2s_position = root_tensor[robinion2s_ids, :3]
    robinion2s_orientation = root_tensor[robinion2s_ids, 3:7]
    robinion2s_velocity = root_tensor[robinion2s_ids, 7:10]
    robinion2s_ang_velocity = root_tensor[robinion2s_ids, 10:13]
    robinion2s_velocity = quat_rotate(robinion2s_orientation, robinion2s_velocity)
    robinion2s_ang_velocity = quat_rotate_inverse(robinion2s_orientation, robinion2s_ang_velocity)    
    robinion2s_up_vec = get_basis_vector(robinion2s_orientation, basis_z_vec)
    
    board_position = root_tensor[board_ids, :3]
    board_orientation = root_tensor[board_ids, 3:7]
    
    # TODO: Add r_foot_position for reward
    robinion2s_torso_position = torch.sqrt(robinion2s_position[:, 0]**2 + robinion2s_position[:, 1]**2)
    l_foot_position = rigid_body_state[:, 19, 0:2]
    l_foot_dist = torch.norm(l_foot_position - l_foot_init_position, p=2, dim=1) * convert_mm_to_cm
    r_foot_position = rigid_body_state[:, 25, 0:2]
    r_foot_dist = torch.norm(r_foot_position - r_foot_init_position, p=2, dim=1) * convert_mm_to_cm

    robinion2s_roll, robinion2s_pitch, robinion2s_yaw = get_euler_xyz(robinion2s_orientation)
    robinion2s_roll = torch.where(robinion2s_roll > np.pi, robinion2s_roll - (2 * np.pi), robinion2s_roll)
    robinion2s_pitch = torch.where(robinion2s_pitch > np.pi, robinion2s_pitch - (2 * np.pi), robinion2s_pitch)
    robinion2s_yaw = torch.where(robinion2s_yaw > np.pi, robinion2s_yaw - (2 * np.pi), robinion2s_yaw)
    
    board_roll, board_pitch, _ = get_euler_xyz(board_orientation)
    board_roll = torch.where(board_roll > np.pi, board_roll - (2 * np.pi), board_roll)
    board_pitch = torch.where(board_pitch > np.pi, board_pitch - (2 * np.pi), board_pitch)
    board_angle = torch.sqrt(board_roll ** 2 + board_pitch ** 2)
    
    curr_tick = torch.where(board_angle>=max_board_angle, tick + 1, tick)
    curr_tick = torch.where(board_angle<max_board_angle, torch.zeros_like(tick), curr_tick)
    
    rew_robinion2s_torso_position = torch.exp(reward_weight * robinion2s_torso_position ** 2)
    rew_robinion2s_roll = torch.exp(reward_weight * robinion2s_roll ** 2)
    rew_robinion2s_pitch = torch.exp(reward_weight * robinion2s_pitch ** 2)
    rew_robinion2s_yaw = torch.exp(reward_weight * robinion2s_yaw ** 2)
    
    rew_board_angle = torch.exp(reward_weight * board_angle ** 2)
    
    rew_l_foot_position = torch.exp(reward_weight * l_foot_dist ** 2)
    rew_r_foot_position = torch.exp(reward_weight * r_foot_dist ** 2)
    
    rew_robinion2s_angvel = torch.exp(reward_weight/5 * torch.sum(robinion2s_ang_velocity ** 2, dim=1))

    reward = (rew_robinion2s_torso_position + rew_robinion2s_roll + rew_robinion2s_pitch + rew_robinion2s_yaw + rew_l_foot_position + rew_r_foot_position + rew_board_angle) / num_rewards

    reset_buf = (torch.abs(robinion2s_roll) > max_imu_angle) | (torch.abs(robinion2s_pitch) > max_imu_angle) | (torch.abs(robinion2s_yaw) > max_imu_angle)
    reset_buf = torch.where(curr_tick >= max_touch_ground_time - 1, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(robinion2s_torso_position > max_torso_position, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(l_foot_dist > max_foot_position, torch.ones_like(reset_buf), reset_buf)
    reward = torch.where(reset_buf == 1, -1.0, reward)
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)
    
    # TODO: Accelaration

    return reward, reset_buf, curr_tick


@torch.jit.script
def compute_observations(
    root_tensor: torch.Tensor, 
    dof_pos_tensor : torch.Tensor, 
    dof_vel_tensor : torch.Tensor, 
    actions : torch.Tensor,
    robinion2s_ids: torch.Tensor, 
    board_ids: torch.Tensor, 
    basis_x_vec: torch.Tensor,
    basis_y_vec: torch.Tensor,
    basis_z_vec: torch.Tensor,
    tick: torch.Tensor
):
    
    robinion2s_position = root_tensor[robinion2s_ids, :3]
    robinion2s_orientation = root_tensor[robinion2s_ids, 3:7]
    robinion2s_velocity = root_tensor[robinion2s_ids, 7:10]
    robinion2s_ang_velocity = root_tensor[robinion2s_ids, 10:13]

    robinion2s_velocity = quat_rotate(robinion2s_orientation, robinion2s_velocity)
    robinion2s_ang_velocity = quat_rotate_inverse(robinion2s_orientation, robinion2s_ang_velocity)

    board_position = root_tensor[board_ids, :3]
    board_orientation = root_tensor[board_ids, 3:7]

    robinion2s_roll, robinion2s_pitch, robinion2s_yaw = get_euler_xyz(robinion2s_orientation)
    board_roll, board_pitch, _ = get_euler_xyz(board_orientation)

    robinion2s_roll = torch.where(robinion2s_roll > np.pi, robinion2s_roll - (2 * np.pi), robinion2s_roll)
    robinion2s_pitch = torch.where(robinion2s_pitch > np.pi, robinion2s_pitch - (2 * np.pi), robinion2s_pitch)
    robinion2s_yaw = torch.where(robinion2s_yaw > np.pi, robinion2s_yaw - (2 * np.pi), robinion2s_yaw)

    board_roll = torch.where(board_roll > np.pi, board_roll - (2 * np.pi), board_roll)
    board_pitch = torch.where(board_pitch > np.pi, board_pitch - (2 * np.pi), board_pitch)

    robinion2s_up_vec = get_basis_vector(robinion2s_orientation, basis_z_vec)
    board_up_vec = get_basis_vector(board_orientation, basis_z_vec)

    obs = torch.cat(
        [
            robinion2s_position,
            robinion2s_roll.unsqueeze(1),
            robinion2s_pitch.unsqueeze(1),
            robinion2s_yaw.unsqueeze(1),
            robinion2s_velocity,
            robinion2s_ang_velocity,
            robinion2s_up_vec,
            dof_pos_tensor,
            dof_vel_tensor,
            board_position,
            board_roll.unsqueeze(1),
            board_pitch.unsqueeze(1),
            board_up_vec,
            tick.unsqueeze(1),
            actions
        ],
        dim=1,
    )

    return obs