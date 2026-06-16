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
from isaacgymenvs.utils.pid import *
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


class BalanceBoardPID(VecTask):
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

        # Ideal Observations: 80
        # Real World Possible Observations: 43
        self.cfg["env"]["numObservations"] = 43 
        
        self.cfg["env"]["numActions"] = 12

        self.dt = self.cfg["sim"]["dt"]

        self.torch_initial_pos = to_torch(NP_INITIAL_POS)

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.sit_value = torch.ones(self.num_envs, device=self.device) * SIT_VALUE

        # Set PID & Low pass filter
        self.pid_gains = torch.ones(self.num_envs, 3, device=self.device)
        self.lpf_alpha = torch.ones(self.num_envs, device=self.device) * 0.01

        self.pid_roll = PID_TORCH(self.num_envs, self.pid_gains, dt=self.dt, set_point=0.0, device=self.device)
        self.pid_pitch = PID_TORCH(self.num_envs, self.pid_gains, dt=self.dt, set_point=0.0, device=self.device)

        self.filter_roll = FILTER_TORCH(self.num_envs, device=self.device)
        self.filter_pitch = FILTER_TORCH(self.num_envs, device=self.device)

        self.p_gain_roll_upper = torch.ones((self.num_envs), dtype=torch.float32, device=self.device) * 10.0
        self.p_gain_roll_lower = torch.ones((self.num_envs), dtype=torch.float32, device=self.device) * 3.0
        self.d_gain_roll_upper = torch.ones((self.num_envs), dtype=torch.float32, device=self.device) * 3.0
        self.d_gain_roll_lower = torch.ones((self.num_envs), dtype=torch.float32, device=self.device) * 0.01
        self.p_gain_pitch_upper = torch.ones((self.num_envs), dtype=torch.float32, device=self.device) * 10.0
        self.p_gain_pitch_lower = torch.ones((self.num_envs), dtype=torch.float32, device=self.device) * 3.0
        self.d_gain_pitch_upper = torch.ones((self.num_envs), dtype=torch.float32, device=self.device) * 3.0
        self.d_gain_pitch_lower = torch.ones((self.num_envs), dtype=torch.float32, device=self.device) * 0.01
        self.weights_upper = torch.ones((self.num_envs, 6), dtype=torch.float32, device=self.device) * 0.5
        self.weights_lower = torch.ones((self.num_envs, 6), dtype=torch.float32, device=self.device) * 0.001
        self.lpf_alpha_upper = torch.ones((self.num_envs, 2), dtype=torch.float32, device=self.device) * 0.1
        self.lpf_alpha_lower = torch.ones((self.num_envs, 2), dtype=torch.float32, device=self.device) * 0.00001
        
        # Set torch variables for IK
        self.theta_hip_yaw = torch.zeros(self.num_envs, 1, device=self.device)
        self.leg_z_pos_left = torch.zeros(self.num_envs, device=self.device)
        self.leg_z_pos_right = torch.zeros(self.num_envs, device=self.device)

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
        
        self.robinion2s_rigid_body_dict = self.gym.get_asset_rigid_body_dict(self.robinion2s_asset)        

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
        
        self.board_rigid_body_dict = self.gym.get_asset_rigid_body_dict(self.board_asset)

    
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

    
    def calculate_leg_inverse_kinematics(self, pose: torch.Tensor, env_ids: torch.Tensor):
        x = pose[:, 0]
        y = pose[:, 1]
        z = pose[:, 2]
        theta = pose[:, 3]

        height = z + self.sit_value[env_ids]

        y = torch.where(
            torch.abs(y) > (2 * LEG_LENGTH) - height,
            torch.where(y > 0, (2 * LEG_LENGTH) - height, -((2 * LEG_LENGTH) - height)),
            y)

        hip_ankle_dist = torch.sqrt((x * x) + ((2 * LEG_LENGTH) - height) * ((2 * LEG_LENGTH) - height))

        hip_ankle_dist = torch.where(hip_ankle_dist > (2 * LEG_LENGTH), 2 * LEG_LENGTH, hip_ankle_dist)

        hip_angle_0 = torch.atan(x / ((2 * LEG_LENGTH) - height))
        hip_angle_1 = torch.acos((hip_ankle_dist / (2 * LEG_LENGTH)))

        hip_yaw = theta
        hip_pitch = hip_angle_0 + hip_angle_1
        hip_roll = torch.asin(y / ((2 * LEG_LENGTH) - height))

        knee_pitch = 2 * hip_angle_1

        ankle_pitch = hip_angle_1 - hip_angle_0
        ankle_roll = torch.asin(y / ((2 * LEG_LENGTH) - height))

        return hip_yaw, -hip_roll, -hip_pitch, knee_pitch, -ankle_pitch, ankle_roll

    
    def calculate_joint_position(self, pid_roll: torch.Tensor, pid_pitch: torch.Tensor, weights: torch.Tensor, env_ids: torch.Tensor):
        leg_gain_weight_roll = weights[:, 0]
        leg_gain_weight_pitch = weights[:, 1]
        arm_gain_weight_roll_left = weights[:, 2]
        arm_gain_weight_pitch_left = weights[:, 3]
        arm_gain_weight_roll_right = weights[:, 4]
        arm_gain_weight_pitch_right = weights[:, 5]

        leg_x_pos_left = -torch.rad2deg(pid_pitch)
        leg_x_pos_right = -torch.rad2deg(pid_pitch)

        leg_y_pos_left = torch.rad2deg(pid_roll)
        leg_y_pos_right = torch.rad2deg(pid_roll)

        arm_roll_left = self.torch_initial_pos[3] - pid_roll * arm_gain_weight_roll_left
        arm_pitch_left = self.torch_initial_pos[2] - pid_pitch * arm_gain_weight_pitch_left
        arm_roll_right = self.torch_initial_pos[7] - pid_roll * arm_gain_weight_roll_right
        arm_pitch_right = self.torch_initial_pos[6] - pid_pitch * arm_gain_weight_pitch_right

        pid_roll = pid_roll * leg_gain_weight_roll
        pid_pitch = pid_pitch * leg_gain_weight_pitch

        leg_y_pos_left = leg_y_pos_left + ((HIP_LENGTH - (HIP_LENGTH * torch.cos(pid_roll))) / 2)
        leg_y_pos_right = leg_y_pos_right + ((HIP_LENGTH - (HIP_LENGTH * torch.cos(pid_roll))) / 2)

        leg_z_pos_left = torch.where(pid_roll <= 0, torch.abs(torch.sin(pid_roll) * HIP_LENGTH), 0.0)
        leg_z_pos_right = torch.where(pid_roll > 0, torch.sin(pid_roll) * HIP_LENGTH, 0.0)

        leg_left = torch.cat((
                leg_x_pos_left.unsqueeze(1),
                -leg_y_pos_left.unsqueeze(1),
                leg_z_pos_left.unsqueeze(1),
                self.theta_hip_yaw[env_ids]),1)
        leg_right = torch.cat((
                leg_x_pos_right.unsqueeze(1),
                -leg_y_pos_right.unsqueeze(1),
                leg_z_pos_right.unsqueeze(1),
                self.theta_hip_yaw[env_ids]),1)

        leg_values_left = self.calculate_leg_inverse_kinematics(leg_left, env_ids)
        leg_values_right = self.calculate_leg_inverse_kinematics(leg_right, env_ids)

        leg_values_left, leg_values_right = list(leg_values_left), list(leg_values_right)
        
        leg_values_left[LEG.ANKLE_PITCH.value] = leg_values_left[LEG.ANKLE_PITCH.value] - pid_pitch
        leg_values_left[LEG.ANKLE_ROLL.value] = leg_values_left[LEG.ANKLE_ROLL.value] - pid_roll

        leg_values_right[LEG.ANKLE_PITCH.value] = leg_values_right[LEG.ANKLE_PITCH.value] - pid_pitch
        leg_values_right[LEG.ANKLE_ROLL.value] = leg_values_right[LEG.ANKLE_ROLL.value] - pid_roll

        leg_values_left, leg_values_right = tuple(leg_values_left), tuple(leg_values_right)
        
        left_arm = arm_roll_left, arm_pitch_left
        right_arm = arm_roll_right, arm_pitch_right

        return leg_values_left, leg_values_right, left_arm, right_arm

    
    def control_balance_board_with_pid(self, 
            roll: torch.Tensor, pitch: torch.Tensor,
            p_gain_roll: torch.Tensor, d_gain_roll: torch.Tensor, 
            p_gain_pitch: torch.Tensor, d_gain_pitch: torch.Tensor,
            weights: torch.Tensor, lpf_alpha: torch.Tensor,
            env_ids: torch.Tensor):
        kp_roll = p_gain_roll
        kd_roll = d_gain_roll
        kp_pitch = p_gain_pitch
        kd_pitch = d_gain_pitch
        lpf_alpha_roll = lpf_alpha[:, 0]
        lpf_alpha_pitch = lpf_alpha[:, 1]

        self.pid_roll.tune_pd_gains(kp_roll, kd_roll, env_ids)
        self.pid_pitch.tune_pd_gains(kp_pitch, kd_pitch, env_ids)

        output_roll = -self.pid_roll.update_pd_control(roll, env_ids)
        output_pitch = -self.pid_pitch.update_pd_control(pitch, env_ids)

        filtered_pid_roll = self.filter_roll.update_lowpass_filter(output_roll, lpf_alpha_roll, env_ids)
        filtered_pid_pitch = self.filter_pitch.update_lowpass_filter(output_pitch, lpf_alpha_pitch, env_ids)

        left_leg, right_leg, left_arm, right_arm = self.calculate_joint_position(filtered_pid_roll, filtered_pid_pitch, weights, env_ids)

        return left_leg, right_leg, left_arm, right_arm
    

    def reset_idx(self, reset_env_ids):
        self.pid_roll.reset_pid(reset_env_ids)
        self.pid_pitch.reset_pid(reset_env_ids)
        self.filter_roll.reset_lowpass_filter(reset_env_ids)
        self.filter_pitch.reset_lowpass_filter(reset_env_ids)

        self.leg_z_pos_left[reset_env_ids] = 0.0
        self.leg_z_pos_right[reset_env_ids] = 0.0

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
        
    
    def pre_physics_step(self, actions):
        self.actions = actions[:]

        roll = self.obs_buf[:, 3]
        pitch = self.obs_buf[:, 4]

        p_gain_roll = unscale_transform(actions[:, 0], self.p_gain_roll_lower, self.p_gain_roll_upper)
        d_gain_roll = unscale_transform(actions[:, 1], self.d_gain_roll_lower, self.d_gain_roll_upper)
        p_gain_pitch = unscale_transform(actions[:, 2], self.p_gain_pitch_lower, self.p_gain_pitch_upper)
        d_gain_pitch = unscale_transform(actions[:, 3], self.d_gain_pitch_lower, self.d_gain_pitch_upper)

        weights = unscale_transform(actions[:, 4:10], self.weights_lower, self.weights_upper)
        lpf_alpha = unscale_transform(actions[:, 10:12], self.lpf_alpha_lower, self.lpf_alpha_upper)
        
        left_leg, right_leg, left_arm, right_arm = self.control_balance_board_with_pid(
            -roll[:],
            -pitch[:],
            p_gain_roll,
            d_gain_roll,
            p_gain_pitch,
            d_gain_pitch,
            weights,
            lpf_alpha,
            self.all_env_indexes
        )
        
        self._send_dof_targets(left_leg, right_leg, left_arm, right_arm)

    
    def _send_dof_targets(self, left_leg, right_leg, left_arm, right_arm):
        left_arm_roll, left_arm_pitch = left_arm
        right_arm_roll, right_arm_pitch = right_arm
        left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee_pitch, left_ankle_pitch, left_ankle_roll = left_leg
        right_hip_yaw,right_hip_roll, right_hip_pitch, right_knee_pitch, right_ankle_pitch, right_ankle_roll = right_leg
        
        self.dof_target_tensor[:, 2] = left_arm_pitch
        self.dof_target_tensor[:, 6] = right_arm_pitch
        self.dof_target_tensor[:, 3] = left_arm_roll
        self.dof_target_tensor[:, 7] = right_arm_roll
        self.dof_target_tensor[:, 10] = left_hip_yaw
        self.dof_target_tensor[:, 11] = left_hip_roll
        self.dof_target_tensor[:, 12] = left_hip_pitch
        self.dof_target_tensor[:, 13] = left_knee_pitch
        self.dof_target_tensor[:, 14] = left_ankle_pitch
        self.dof_target_tensor[:, 15] = left_ankle_roll
        self.dof_target_tensor[:, 16] = right_hip_yaw
        self.dof_target_tensor[:, 17] = right_hip_roll
        self.dof_target_tensor[:, 18] = right_hip_pitch
        self.dof_target_tensor[:, 19] = right_knee_pitch
        self.dof_target_tensor[:, 20] = right_ankle_pitch
        self.dof_target_tensor[:, 21] = right_ankle_roll

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
    max_imu_angle = math.radians(25)
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
    # reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)
    
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
            # robinion2s_position,
            robinion2s_roll.unsqueeze(1),
            robinion2s_pitch.unsqueeze(1),
            robinion2s_yaw.unsqueeze(1),
            # robinion2s_velocity,
            robinion2s_ang_velocity,
            robinion2s_up_vec,
            dof_pos_tensor,
            # dof_vel_tensor,
            # board_position,
            # board_roll.unsqueeze(1),
            # board_pitch.unsqueeze(1),
            # board_up_vec, 
            # tick.unsqueeze(1),
            actions
        ],
        dim=1,
    )

    return obs
