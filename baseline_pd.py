from operator import index
import os
import numpy as np
 
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym.torch_utils import *
from math import *
import torch
import enum

from isaacgymenvs.utils.pid import *
from isaacgymenvs.utils.filter import *

import random

START_NUM = 0
LEG_LENGTH = 190.0
HIP_LENGTH = 110.0
SIT_VALUE = 30.0

SPACING = 1.0

MAX_IMU_ANGLE = radians(25.0)

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
    
class BalanceBoard:
    def __init__(self):
        
        self.device = "cuda:0"
        pid_dt = 1/100
        self.dt = 1/100
        # self.substeps = 2
        self.num_envs = 1
        
        self.check_pid = int(pid_dt / self.dt)
        self.tick = 0
        
        self.sit_value = torch.ones(self.num_envs, device=self.device) * SIT_VALUE
        
        # Set PID & Low pass filter
        self.pid_gains = torch.ones(self.num_envs, 3, device=self.device)
        
        self.pid_roll = PID_TORCH(self.num_envs, self.pid_gains, dt=pid_dt, set_point=0.0, device=self.device)
        self.pid_pitch = PID_TORCH(self.num_envs, self.pid_gains, dt=pid_dt, set_point=0.0, device=self.device)
        
        self.filter_roll = FILTER_TORCH(self.num_envs, device=self.device)
        self.filter_pitch = FILTER_TORCH(self.num_envs, device=self.device)
        self.alpha = torch.ones(self.num_envs, device=self.device) * 0.09226090461015701
        self.filter_roll_dt = 20.0
        self.filter_roll_fc = 100.0
        self.filter_pitch_dt = 20.0
        self.filter_pitch_fc = 100.0
        
        # Set torch variables
        self.theta_hip_yaw = torch.zeros(self.num_envs,1, device=self.device)
        
        self.pd_gain_roll = torch.ones(self.num_envs, 2, device=self.device)
        self.pd_gain_pitch = torch.ones(self.num_envs, 2, device=self.device)
        # Weights: leg_roll, leg_pitch, arm_roll, arm_pitch
        self.weights = torch.ones(self.num_envs, 6, device=self.device)
        
        self.leg_z_pos_left = torch.zeros(self.num_envs, device=self.device)
        self.leg_z_pos_right = torch.zeros(self.num_envs, device=self.device)
        
        self.torch_initial_pos = to_torch(NP_INITIAL_POS)
        
        self.progress_buf = torch.zeros((self.num_envs), dtype=torch.long, device=self.device)
        
        # Initialize Isaac Gym Envs
        self.gym = gymapi.acquire_gym()
        self.args = gymutil.parse_arguments(description="New Balance Baord on Isaac Gym",
                               headless=True)
        
        spacing = SPACING
        num_per_row = int(np.sqrt(self.num_envs))
        
        self._range_offset = [-0.002, 0.002]
        
        self.create_sim()
        self.create_ground_plance()
        self.create_envs(self.num_envs, spacing, num_per_row)
        self.gym.prepare_sim(self.sim)
        self.generate_indexes()
        self.allocate_tensors()
        self.reset(self.all_env_indexes)
        
        self._range_roll_kp = [0.0, 10.0]
        self._range_roll_kd = [0.0, 5.0]
        self._range_pitch_kp = [0.0, 10.0]
        self._range_pitch_kd = [0.0, 5.0]
        self._range_leg_roll = [0.0, 0.5]
        self._range_leg_pitch = [0.0, 0.5]
        self._range_arm_roll_left = [-1.0, 1.0]
        self._range_arm_pitch_left = [-1.0, 1.0]
        self._range_arm_roll_right = [-1.0, 1.0]
        self._range_arm_pitch_right = [-1.0, 1.0]
        self._range_lpf_alpha = [0.0, 0.1]
        self.best_progress = 1
    
    
    def resample_parameters(self, reset_env_ids):
        self.pd_gain_roll[reset_env_ids, 0] = torch_rand_float(self._range_roll_kp[0], self._range_roll_kp[1],
                (len(reset_env_ids), 1), device=self.device)
        self.pd_gain_roll[reset_env_ids, 1] = torch_rand_float(self._range_roll_kd[0], self._range_roll_kd[1],
                (len(reset_env_ids), 1), device=self.device)
        self.pd_gain_pitch[reset_env_ids, 0] = torch_rand_float(self._range_pitch_kp[0], self._range_pitch_kp[1],
                (len(reset_env_ids), 1), device=self.device)
        self.pd_gain_pitch[reset_env_ids, 1] = torch_rand_float(self._range_pitch_kd[0], self._range_pitch_kd[1],
                (len(reset_env_ids), 1), device=self.device)
        self.weights[reset_env_ids, 0] = torch_rand_float(self._range_leg_roll[0], self._range_leg_roll[1],
                (len(reset_env_ids), 1), device=self.device)
        self.weights[reset_env_ids, 1] = torch_rand_float(self._range_leg_pitch[0], self._range_leg_pitch[1],
                (len(reset_env_ids), 1), device=self.device)
        self.weights[reset_env_ids, 2] = torch_rand_float(self._range_arm_roll_left[0], self._range_arm_roll_left[1],
                (len(reset_env_ids), 1), device=self.device)
        self.weights[reset_env_ids, 3] = torch_rand_float(self._range_arm_pitch_left[0], self._range_arm_pitch_left[1],
                (len(reset_env_ids), 1), device=self.device)
        self.weights[reset_env_ids, 4] = torch_rand_float(self._range_arm_roll_right[0], self._range_arm_roll_right[1],
                (len(reset_env_ids), 1), device=self.device)
        self.weights[reset_env_ids, 5] = torch_rand_float(self._range_arm_pitch_right[0], self._range_arm_pitch_right[1],
                (len(reset_env_ids), 1), device=self.device)
        self.alpha[reset_env_ids] = torch_rand_float(self._range_lpf_alpha[0], self._range_lpf_alpha[1],
                (len(reset_env_ids), 1), device=self.device)
    
    def create_sim(self):
        sim_params = gymapi.SimParams()
 
        sim_params.dt = self.dt
        # sim_params.substeps = self.substeps
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = self.args.num_threads
        if self.device == "cuda:0":
            sim_params.physx.use_gpu = True
            sim_params.use_gpu_pipeline = True
        else:
            sim_params.physx.use_gpu = False
            sim_params.use_gpu_pipeline = False
        
        self.sim = self.gym.create_sim(self.args.compute_device_id, self.args.graphics_device_id, gymapi.SIM_PHYSX, sim_params)
        
    
    def create_ground_plance(self):
        # configure the ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
        plane_params.distance = 0
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        
        self.gym.add_ground(self.sim, plane_params)
        
    
    def create_viewer(self):
        if not self.args.headless:
            # create viewer using the default camera properties
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                raise ValueError('*** Failed to create viewer')
            
        # position the camera
        cam_pos = gymapi.Vec3(2.0, -1.0, 1.0)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

            
    def load_assets(self):
        # Create Robinon2S asset
        robinion2s_asset_root = "../assets"
        robinion2s_asset_path = "urdf/robinion2s.urdf"
        
        robinion2s_asset_options = gymapi.AssetOptions()
        robinion2s_asset_options.fix_base_link = False
        
        # Load all Robinon2S asset
        self.robinion2s_asset = self.gym.load_asset(self.sim, robinion2s_asset_root, robinion2s_asset_path, robinion2s_asset_options)
        
        self.robinion2s_pose = gymapi.Transform()
        self.robinion2s_pose.p.y = 0.0245
        self.robinion2s_pose.p.x = 0.02
        self.robinion2s_pose.p.z = 0.612
        self.robinion2s_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        
        #Create Board asset
        board_asset_root = "../assets"
        board_asset_path = "urdf/board.urdf"

        board_asset_options = gymapi.AssetOptions()
        board_asset_options.fix_base_link = False

        # Load all Board asset
        self.board_asset = self.gym.load_asset(self.sim, board_asset_root, board_asset_path, board_asset_options)

        self.board_pose = gymapi.Transform()
        self.board_pose.p.z = 0.064
        self.board_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        
    
    def create_envs(self, num_envs, spacing, num_per_row):
        self.create_viewer()
        self.load_assets()
        
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        
        envs = []
        robinion2s_handles = []
        board_handles = []
        
        for i in range(num_envs):
            # Create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            envs.append(env)
            
            # Create Robinion2S
            robinion2s_handle = self.gym.create_actor(env, self.robinion2s_asset, self.robinion2s_pose, "robinion2s", 0, 0)    
            
            props = self.gym.get_actor_dof_properties(env, robinion2s_handle)
            props["driveMode"] = gymapi.DOF_MODE_POS
            props["stiffness"] = 80.0
            props["damping"] = 2.0
            self.gym.set_actor_dof_properties(env, robinion2s_handle, props)
            self.num_dofs = self.gym.get_actor_dof_count(env, robinion2s_handle)
        
            # Create Balance board
            board_handle = self.gym.create_actor(env, self.board_asset, self.board_pose, "board", 0, 0)
            
            robinion2s_handles.append(robinion2s_handle)
            board_handles.append(board_handle)
        
    
    def allocate_tensors(self):
        
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
        
        self.board_rigid_body_dict = self.gym.get_asset_rigid_body_dict(self.board_asset)
        self.robinion2s_rigid_body_dict = self.gym.get_asset_rigid_body_dict(self.robinion2s_asset)
        
        self.reset_root_tensor = self.root_tensor.clone()
        self.reset_positions = self.reset_root_tensor[:, 0:3]
        self.reset_root_tensor[:, 7:] = 0.0

        self.dof_target_tensor = self.dof_pos.clone()
    
    
    def generate_indexes(self):
        self.all_env_indexes = torch.arange(0, self.num_envs, dtype=torch.long, device=self.device)        
        self.agent_indexes = torch.arange(0, self.num_envs * 2, 2, dtype=torch.long, device=self.device)
        self.board_indexes = self.agent_indexes + 1
        
    
    def reset(self, reset_env_ids : torch.Tensor):
        self.pid_roll.reset_pid(reset_env_ids)
        self.pid_pitch.reset_pid(reset_env_ids)
        self.filter_roll.reset_lowpass_filter(reset_env_ids)
        self.filter_pitch.reset_lowpass_filter(reset_env_ids)
        
        self.leg_z_pos_left[reset_env_ids] = 0.0
        self.leg_z_pos_right[reset_env_ids] = 0.0
        
        self.progress_buf[reset_env_ids] = 0
        
        # Convert to int32 for tensor API calls.
        reset_env_ids_int32 = reset_env_ids.to(torch.int32)
        reset_agent_inds_int32 = self.agent_indexes[reset_env_ids].to(torch.int32)
        reset_board_inds_int32 = self.board_indexes[reset_env_ids].to(torch.int32)
        reset_actor_ids_int32 = torch.cat([reset_agent_inds_int32, reset_board_inds_int32])
        
        self.dof_pos[reset_env_ids, :] = self.torch_initial_pos
        self.dof_vel[reset_env_ids, :] = 0.0
        
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_states),
                                    gymtorch.unwrap_tensor(reset_actor_ids_int32), len(reset_actor_ids_int32))
        
        self.dof_target_tensor[reset_env_ids, :] = self.torch_initial_pos
        self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_target_tensor),
                                    gymtorch.unwrap_tensor(reset_actor_ids_int32), len(reset_actor_ids_int32))
        
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.reset_root_tensor),
                                    gymtorch.unwrap_tensor(reset_actor_ids_int32), len(reset_actor_ids_int32))
        
        
    def check_reset_condition(self, agent_roll : torch.Tensor, agent_pitch : torch.Tensor):        
        reset_cond_mask = (torch.abs(agent_roll) > MAX_IMU_ANGLE) | (torch.abs(agent_pitch) > MAX_IMU_ANGLE)        
        reset_indexes = torch.argwhere(reset_cond_mask)        
        return reset_indexes
    
    
    def calcualte_leg_inverse_kinematics(self, pose : torch.Tensor, env_ids : torch.Tensor):
        x = pose[:,0]
        y = pose[:,1]
        z = pose[:,2]
        theta = pose[:,3]
        
        height = z + self.sit_value[env_ids]

        y = torch.where(torch.abs(y)>(2*LEG_LENGTH)-height, 
                        torch.where(y>0, (2*LEG_LENGTH)-height, -((2*LEG_LENGTH)-height)),
                        y)
        
        hip_ankle_dist = torch.sqrt((x * x) + ((2 * LEG_LENGTH) - height) * ((2 * LEG_LENGTH) - height))
        
        hip_ankle_dist = torch.where(hip_ankle_dist>(2*LEG_LENGTH), 2*LEG_LENGTH, hip_ankle_dist)
        
        hip_angle_0 = torch.atan(x / ((2 * LEG_LENGTH) - height))
        hip_angle_1 = torch.acos((hip_ankle_dist / (2 * LEG_LENGTH)))
    
        hip_yaw = theta
        hip_pitch = hip_angle_0 + hip_angle_1
        hip_roll = torch.asin(y / ((2 * LEG_LENGTH) - height))
        
        knee_pitch = 2 * hip_angle_1
        
        ankle_pitch = hip_angle_1 - hip_angle_0
        ankle_roll = torch.asin(y / ((2 * LEG_LENGTH) - height))
        
        return hip_yaw, -hip_roll, -hip_pitch, knee_pitch, -ankle_pitch, ankle_roll
    
    
    def calculate_joint_position(self, pid_roll : torch.Tensor, pid_pitch : torch.Tensor, 
                                weights : torch.Tensor, env_ids : torch.Tensor):
        # leg_gain_weight_roll = weights[:, 0]
        # leg_gain_weight_pitch = weights[:, 1]
        # arm_gain_weight_roll_left = weights[:, 2]
        # arm_gain_weight_pitch_left = weights[:, 3]
        # arm_gain_weight_roll_right = weights[:, 4]
        # arm_gain_weight_pitch_right = weights[:, 5]
        
        leg_gain_weight_roll = 0.41661036014556885
        leg_gain_weight_pitch = 0.2125472128391266
        arm_gain_weight_roll_left = -0.4425053596496582
        arm_gain_weight_pitch_left = -0.7637600302696228
        arm_gain_weight_roll_right = 0.48136627674102783
        arm_gain_weight_pitch_right = -0.41864466667175293
        
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
        
        leg_y_pos_left = leg_y_pos_left + ((HIP_LENGTH-(HIP_LENGTH * torch.cos(pid_roll)))/2)
        leg_y_pos_right = leg_y_pos_right + ((HIP_LENGTH-(HIP_LENGTH * torch.cos(pid_roll)))/2)

        leg_z_pos_left = torch.where(pid_roll <= 0, 
                                     torch.abs(torch.sin(pid_roll) * HIP_LENGTH),
                                     0.0)
        leg_z_pos_right = torch.where(pid_roll > 0, 
                                      torch.sin(pid_roll) * HIP_LENGTH,
                                      0.0)
        
        leg_left = torch.cat((leg_x_pos_left.unsqueeze(1),
            -leg_y_pos_left.unsqueeze(1), leg_z_pos_left.unsqueeze(1), self.theta_hip_yaw[env_ids]),1)
        leg_right = torch.cat((leg_x_pos_right.unsqueeze(1),
            -leg_y_pos_right.unsqueeze(1), leg_z_pos_right.unsqueeze(1), self.theta_hip_yaw[env_ids]),1)
        
        leg_values_left = self.calcualte_leg_inverse_kinematics(leg_left, env_ids)
        leg_values_right = self.calcualte_leg_inverse_kinematics(leg_right, env_ids)
        
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
                                       roll : torch.Tensor, pitch : torch.Tensor, 
                                       pd_gain_roll : torch.Tensor, pd_gain_pitch : torch.Tensor, 
                                       weights : torch.Tensor,
                                       env_ids : torch.Tensor):
        
        # kp_roll = pd_gain_roll[:, 0]
        # kd_roll = pd_gain_roll[:, 1]
        # kp_pitch = pd_gain_pitch[:, 0]
        # kd_pitch = pd_gain_pitch[:, 1]
        
        kp_roll = 8.371222496032715
        kd_roll = 2.5177385807037354
        kp_pitch = 2.145641803741455
        kd_pitch = 4.6283183097839355
        
        self.pid_roll.tune_pd_gains(kp_roll, kd_roll, env_ids)
        self.pid_pitch.tune_pd_gains(kp_pitch, kd_pitch, env_ids)
        
        output_roll = -self.pid_roll.update_pd_control(roll, env_ids)
        output_pitch = -self.pid_pitch.update_pd_control(pitch, env_ids)
        
        filtered_pid_roll = self.filter_roll.update_lowpass_filter(output_roll, self.alpha, env_ids)
        filtered_pid_pitch = self.filter_pitch.update_lowpass_filter(output_pitch, self.alpha, env_ids)
        
        left_leg, right_leg, left_arm, right_arm = self.calculate_joint_position(filtered_pid_roll, filtered_pid_pitch, weights, env_ids)
        
        return left_leg, right_leg, left_arm, right_arm
    
    def save_best_combination(self, reset_env_ids):
        reset_progress_buf = self.progress_buf[reset_env_ids]
        best_progress_i = torch.argmax(reset_progress_buf)
        best_progress_i = reset_env_ids[best_progress_i]
        best_progress = self.progress_buf[best_progress_i].item()
        # self.best_progress = best_progress
        if best_progress > self.best_progress:
            self.best_progress = best_progress
            # TODO: Save best combination
            combination = [self.pd_gain_roll[best_progress_i, 0].item(), 
                           self.pd_gain_roll[best_progress_i, 1].item(),
                           self.pd_gain_pitch[best_progress_i, 0].item(),
                           self.pd_gain_pitch[best_progress_i, 1].item(),
                           self.weights[best_progress_i, 0].item(),
                           self.weights[best_progress_i, 1].item(),
                           self.weights[best_progress_i, 2].item(),
                           self.weights[best_progress_i, 3].item(),
                           self.weights[best_progress_i, 4].item(),
                           self.weights[best_progress_i, 5].item(),
                           self.alpha[best_progress_i].item(),
                           self.sit_value[best_progress_i].item()]
            print(f"New best with {self.best_progress} steps | {self.best_progress * self.dt} seconds.")
            print("===================================================================================")
            print("Parameters: ")
            print(combination)
            print("===================================================================================")
            print("Roll PD gains: ".ljust(16), combination[0:2])
            print("Pitch PD gains: ".ljust(16), combination[2:4])
            print("Weights: ".ljust(16), combination[4:10])
            print("lpf alpha: ".ljust(16), combination[10])
            print("Init Sit value: ".ljust(16), combination[11])
            print("===================================================================================")

    
    def run(self):
        while not self.gym.query_viewer_has_closed(self.viewer):
            # update the viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
        
            # Wait for dt to elapse in real time.
            # This synchronizes the physics simulation with the rendering rate.
            self.gym.sync_frame_time(self.sim)
        
            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            
            # Refresh tensors.
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            
            #### Pre-physics step ###
            roll, pitch, yaw = get_euler_xyz(self.root_orientations[self.agent_indexes])
            
            roll = torch.where(roll > np.pi, roll-(2*np.pi), roll) 
            pitch = torch.where(pitch > np.pi, pitch-(2*np.pi), pitch)

            self.change_action_envs = (self.progress_buf % self.check_pid) == 0
            update_pid_envs = torch.argwhere(self.change_action_envs).squeeze()
            if update_pid_envs.shape == torch.Size([]):
                update_pid_envs = update_pid_envs.unsqueeze(0)
            
            if len(update_pid_envs) > 0:
                left_leg, right_leg, left_arm, right_arm = self.control_balance_board_with_pid(-roll[update_pid_envs],
                        -pitch[update_pid_envs],
                        self.pd_gain_roll[update_pid_envs],
                        self.pd_gain_pitch[update_pid_envs],
                        self.weights[update_pid_envs],
                        update_pid_envs)

                left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee_pitch, left_ankle_pitch, left_ankle_roll = left_leg
                right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee_pitch, right_ankle_pitch, right_ankle_roll = right_leg

                left_arm_roll, left_arm_pitch = left_arm
                right_arm_roll, right_arm_pitch = right_arm
                
                self.dof_target_tensor[update_pid_envs, 10] = left_hip_yaw
                self.dof_target_tensor[update_pid_envs, 11] = left_hip_roll
                self.dof_target_tensor[update_pid_envs, 12] = left_hip_pitch
                self.dof_target_tensor[update_pid_envs, 13] = left_knee_pitch
                self.dof_target_tensor[update_pid_envs, 14] = left_ankle_pitch
                self.dof_target_tensor[update_pid_envs, 15] = left_ankle_roll
                        
                self.dof_target_tensor[update_pid_envs, 16] = right_hip_yaw
                self.dof_target_tensor[update_pid_envs, 17] = right_hip_roll
                self.dof_target_tensor[update_pid_envs, 18] = right_hip_pitch
                self.dof_target_tensor[update_pid_envs, 19] = right_knee_pitch
                self.dof_target_tensor[update_pid_envs, 20] = right_ankle_pitch
                self.dof_target_tensor[update_pid_envs, 21] = right_ankle_roll
                
                self.dof_target_tensor[update_pid_envs, 2] = left_arm_pitch
                self.dof_target_tensor[update_pid_envs, 6] = right_arm_pitch
                self.dof_target_tensor[update_pid_envs, 3] = left_arm_roll
                self.dof_target_tensor[update_pid_envs, 7] = right_arm_roll
                
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_target_tensor))
            
            # (N, 1)
            self.progress_buf += 1
            
            reset_env_ids = self.check_reset_condition(roll, pitch)
            
            if len(reset_env_ids) > 0:
                self.save_best_combination(reset_env_ids)
                self.reset(reset_env_ids)
                self.resample_parameters(reset_env_ids)
                
            self.tick += 1
    
        
if __name__ == "__main__":
    ctrl = BalanceBoard()
    
    ctrl.run()
