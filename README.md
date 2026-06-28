# BalanceBoard_IsaacGym

Isaac Gym-based reinforcement learning environments for humanoid balance control on a balance board.

This repository provides custom IsaacGymEnvs tasks for training and evaluating a humanoid robot balance controller using multiple action representations, including PD-based control, inverse-kinematics-based control, and direct joint-space control.

## Demo Video

[![Balance Board Demo](https://img.youtube.com/vi/-xCmdSmCYeY/0.jpg)](https://youtu.be/-xCmdSmCYeY)

Direct link: [Watch the demo on YouTube](https://youtu.be/-xCmdSmCYeY)

## Overview

The goal of this project is to train a humanoid robot to maintain balance on an unstable balance board in simulation. The environment is implemented using NVIDIA Isaac Gym and IsaacGymEnvs.

The repository includes:

- Humanoid and balance board URDF assets
- Custom IsaacGymEnvs task files
- PPO training configurations
- PID and low-pass filter utilities
- A baseline PD control

## Main Features

- Isaac Gym simulation environment for humanoid balance control
- Balance board task using a humanoid robot model
- Three task/control variants:
  - `BalanceBoardPID`
  - `BalanceBoardIK`
  - `BalanceBoardJoint`
- PPO training configuration for each task
- GPU-based parallel simulation through Isaac Gym
- Baseline PD controller for comparison

## Task Variants

### 1. BalanceBoardPID

`BalanceBoardPID` uses a policy to tune PID-related control parameters and motion weights. The controller converts balance errors into joint targets through PD control, low-pass filtering, and inverse kinematics.

### 2. BalanceBoardIK

`BalanceBoardIK` uses a policy that outputs foot-position, foot-orientation, arm-motion, and filtering parameters. These commands are converted into lower-body joint commands through inverse kinematics.

### 3. BalanceBoardJoint

`BalanceBoardJoint` uses a policy that directly outputs joint-space commands for the humanoid robot. This provides a more direct action representation compared with the PID and IK variants.
