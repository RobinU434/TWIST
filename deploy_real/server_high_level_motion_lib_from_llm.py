#!/usr/bin/env python
"""
High-Level Motion Library Server
---------------------------------
This server streams pre-recorded motion data from a pickle file to control a humanoid robot.
It loads motion sequences, processes them into "mimic observations", and publishes them via Redis
for consumption by robot controllers.

Key Features:
- Loads motion data from .pkl files using MotionLib
- Converts motion frames to mimic observations (robot state representation)
- Publishes at fixed control rate (20Hz default) via Redis
- Optional MuJoCo visualization of the motion playback
- Safe interpolation back to default pose on exit
"""
import argparse
import time
import redis
import json
import numpy as np

# import isaacgym
import torch
from rich import print
import os
import mujoco
from mujoco.viewer import launch_passive

# ---------------------------------------------------------------------
# Example imports: adapt to your actual file structure
# ---------------------------------------------------------------------
# from pose.utils.motion_lib_pkl import MotionLib
from pose.utils.motion_lib_pkl_wo_isaacgym import MotionLib
from data_utils.rot_utils import euler_from_quaternion, quat_rotate_inverse, quat_rotate_inverse_torch

from data_utils.params import DEFAULT_MIMIC_OBS, DEFAULT_ACTION_HAND


# ---------------------------------------------------------------------
# A small helper to replicate "mimic obs" logic from your code
# ---------------------------------------------------------------------
def build_mimic_obs(motion_lib: MotionLib, t_step: int, control_dt: float, tar_obs_steps, robot_type: str = "g1"):
    """
    Build the mimic observation at a given time-step from the motion library.

    Args:
        motion_lib: MotionLib object containing the motion data
        t_step: Current time step in the motion sequence
        control_dt: Control timestep (typically 0.02s = 50Hz)
        tar_obs_steps: Target observation steps (future frame offsets)
        robot_type: Type of robot ("g1" supported)

    Returns:
        Tuple containing:
        - mimic_obs: Flattened observation vector for robot control
        - root_pos: Root position (x, y, z)
        - root_rot: Root orientation quaternion
        - dof_pos: Joint positions
        - root_vel: Root linear velocity
        - root_ang_vel: Root angular velocity
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build time array for the target observation steps
    motion_times = torch.tensor([t_step * control_dt], device=device).unsqueeze(-1)
    obs_motion_times = tar_obs_steps * control_dt + motion_times
    obs_motion_times = obs_motion_times.flatten()

    # Assume single motion sequence in the .pkl file (motion_id = 0)
    motion_ids = torch.zeros(len(tar_obs_steps), dtype=torch.int, device=device)

    # Retrieve motion frame data from the library
    # root_pos.shape: (1, 3)
    # root_rot.shape: (1, 4)
    # root_vel.shape: (1, 3)
    # root_ang_vel.shape: (1, 3)
    # dof_pos.shape: (1, num_joints)
    root_pos, root_rot, root_vel, root_ang_vel, dof_pos, _, _ = motion_lib.calc_motion_frame(
        motion_ids, obs_motion_times
    )

    #########################################################################################
    # HERE I want to exchange the variables root_pos, root_rot, root_vel, root_ang_vel, dof_pos with the data from data_pos, data_dec_out
    # data_pos are the 3D positions of the joints
    # data_dec_out is the output from the vae, it has features like ric positions, root positions, root rotations, root velocities, root angular velocities
    # Somehow I can not directly use data_dec_out to replace the variables, because the output visualization looks weird but I do not know why
    data_pos = np.load("../assets/test_motion_from_llm/a_person_is_walking_pos.npy")
    data_dec_out = np.load("../assets/test_motion_from_llm/a_person_is_walking_dec_out.npy")

    # The dataset of TWIST is 30Hz
    # The dataset from the LLM is 20Hz
    data_idx = (obs_motion_times * 20).int().cpu().numpy()[0]  # Assuming 20Hz data rate

    breakpoint()

    # root_pos, root_rot, dof_pos are used by MuJoCo viewer
    # The RL policy uses only mimic_obs_buf
    #########################################################################################

    # Convert root orientation from quaternion to Euler angles (roll, pitch, yaw)
    roll, pitch, yaw = euler_from_quaternion(root_rot)
    roll = roll.reshape(1, -1, 1)
    pitch = pitch.reshape(1, -1, 1)
    yaw = yaw.reshape(1, -1, 1)

    # Transform velocities from world frame to root-relative frame
    root_vel = quat_rotate_inverse_torch(root_rot, root_vel).reshape(1, -1, 3)
    root_ang_vel = quat_rotate_inverse_torch(root_rot, root_ang_vel).reshape(1, -1, 3)

    # Reshape position arrays
    root_pos = root_pos.reshape(1, -1, 3)
    dof_pos = dof_pos.reshape(1, -1, dof_pos.shape[-1])

    # Special handling for G1 robot: add zero wrist positions
    if robot_type == "g1":
        dof_pos_with_wrist = torch.zeros(25, device=device).reshape(1, 1, 25)
        wrist_ids = [19, 24]  # Wrist joint indices
        other_ids = [f for f in range(25) if f not in wrist_ids]
        dof_pos_with_wrist[..., other_ids] = dof_pos
        dof_pos = dof_pos_with_wrist

    # Concatenate all components into mimic observation buffer
    # Format: [height, roll, pitch, yaw, root_vel (3D), yaw_vel, joint_positions]
    mimic_obs_buf = torch.cat(
        (root_pos[..., 2:3], roll, pitch, yaw, root_vel, root_ang_vel[..., 2:3], dof_pos), dim=-1
    )[
        :, 0:1
    ]  # shape (1, 1, ?)
    mimic_obs_buf = mimic_obs_buf.reshape(1, -1)

    return (
        mimic_obs_buf.detach().cpu().numpy().squeeze(),
        root_pos.detach().cpu().numpy().squeeze(),
        root_rot.detach().cpu().numpy().squeeze(),
        dof_pos.detach().cpu().numpy().squeeze(),
        root_vel.detach().cpu().numpy().squeeze(),
        root_ang_vel.detach().cpu().numpy().squeeze(),
    )


def main(args, xml_file, robot_base):
    """
    Main execution function for the motion server.

    Args:
        args: Command line arguments containing motion_file, robot type, etc.
        xml_file: Path to MuJoCo XML model file for visualization
        robot_base: Name of the robot base body for camera tracking
    """

    # Initialize MuJoCo visualization if requested
    if args.vis:
        sim_model = mujoco.MjModel.from_xml_path(xml_file)
        sim_data = mujoco.MjData(sim_model)
        viewer = launch_passive(model=sim_model, data=sim_data, show_left_ui=False, show_right_ui=False)

        # Print model information for debugging
        print("Degrees of Freedom (DoF) names and their order:")
        for i in range(sim_model.nv):  # 'nv' is the number of DoFs
            dof_name = mujoco.mj_id2name(sim_model, mujoco.mjtObj.mjOBJ_JOINT, sim_model.dof_jntid[i])
            print(f"DoF {i}: {dof_name}")

        # print("Body names and their IDs:")
        # for i in range(self.model.nbody):  # 'nbody' is the number of bodies
        #     body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
        #     print(f"Body ID {i}: {body_name}")

        print("Motor (Actuator) names and their IDs:")
        for i in range(sim_model.nu):  # 'nu' is the number of actuators (motors)
            motor_name = mujoco.mj_id2name(sim_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            print(f"Motor ID {i}: {motor_name}")

    # Step 1: Connect to Redis server for publishing commands
    redis_client = redis.Redis(host="localhost", port=6379, db=0)

    # Step 2: Load motion library from pickle file
    device = "cuda" if torch.cuda.is_available() else "cpu"
    motion_lib = MotionLib(args.motion_file, device=device)

    # Step 3: Parse target observation steps from command line
    tar_obs_steps = [int(x.strip()) for x in args.steps.split(",")]
    tar_obs_steps_tensor = torch.tensor(tar_obs_steps, device=device, dtype=torch.int)

    # Step 4: Calculate motion duration and loop parameters
    control_dt = 0.02  # 50Hz control rate
    motion_id = torch.tensor([0], device=device, dtype=torch.long)
    motion_length = motion_lib.get_motion_length(motion_id)
    num_steps = int(motion_length / control_dt)

    print(f"[Motion Server] Streaming for {num_steps} steps at dt={control_dt:.3f} seconds...")

    # Initialize last observation for safe interpolation on exit
    last_mimic_obs = DEFAULT_MIMIC_OBS[args.robot]

    # Optional velocity visualization (debugging flags)
    vis_root_vel = False
    vis_root_ang_vel = False
    if vis_root_vel:
        root_vel_list = []
    if vis_root_ang_vel:
        root_ang_vel_list = []

    # Main motion streaming loop
    try:
        for t_step in range(num_steps):
            t0 = time.time()

            # Generate mimic observation for current timestep
            # mimic_obs.shape: (33,)
            # root_pos.shape: (3,)
            # root_rot.shape: (4,)
            # dof_pos.shape: (25,)
            # root_vel.shape: (3,)
            # root_ang_vel.shape: (3,)

            mimic_obs, root_pos, root_rot, dof_pos, root_vel, root_ang_vel = build_mimic_obs(
                motion_lib=motion_lib,
                t_step=t_step,
                control_dt=control_dt,
                tar_obs_steps=tar_obs_steps_tensor,
                robot_type=args.robot,
            )

            # Optional: collect velocity data for visualization
            if vis_root_vel:
                root_vel_list.append(root_vel)
            if vis_root_ang_vel:
                root_ang_vel_list.append(root_ang_vel)

            # Publish mimic observation and hand action to Redis
            mimic_obs_list = mimic_obs.tolist() if mimic_obs.ndim == 1 else mimic_obs.flatten().tolist()
            redis_client.set(f"action_mimic_{args.robot}", json.dumps(mimic_obs_list))
            redis_client.set(f"action_hand_{args.robot}", json.dumps(DEFAULT_ACTION_HAND[args.robot].tolist()))
            last_mimic_obs = mimic_obs

            # Progress feedback
            print(f"Step {t_step:4d} => mimic_obs shape = {mimic_obs.shape} published...", end="\r")

            # Update MuJoCo visualization if enabled
            if args.vis:
                sim_data.qpos[:3] = root_pos
                # Convert quaternion format: [x,y,z,w] -> [w,x,y,z]
                root_rot = root_rot[[3, 0, 1, 2]]
                sim_data.qpos[3:7] = root_rot
                sim_data.qpos[7:] = dof_pos
                mujoco.mj_forward(sim_model, sim_data)

                # Camera follows robot base
                robot_base_pos = sim_data.xpos[sim_model.body(robot_base).id]
                viewer.cam.lookat = robot_base_pos
                viewer.cam.distance = 2.0
                viewer.sync()

            # Maintain real-time control rate
            elapsed = time.time() - t0
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)

    except KeyboardInterrupt:
        print("[Motion Server] Keyboard interrupt. Interpolating to default mimic_obs...")
        # Smoothly interpolate back to default standing pose over 2 seconds
        time_back_to_default = 2.0
        for i in range(int(time_back_to_default / control_dt)):
            interp_mimic_obs = last_mimic_obs + (DEFAULT_MIMIC_OBS[args.robot] - last_mimic_obs) * (
                i / (time_back_to_default / control_dt)
            )
            redis_client.set(f"action_mimic_{args.robot}", json.dumps(interp_mimic_obs.tolist()))
            redis_client.set(f"action_hand_{args.robot}", json.dumps(DEFAULT_ACTION_HAND[args.robot].tolist()))
            time.sleep(control_dt)
        redis_client.set(f"action_mimic_{args.robot}", json.dumps(DEFAULT_MIMIC_OBS[args.robot].tolist()))
        redis_client.set(f"action_hand_{args.robot}", json.dumps(DEFAULT_ACTION_HAND[args.robot].tolist()))
        last_mimic_obs = DEFAULT_MIMIC_OBS[args.robot]
        exit()
    finally:
        print("[Motion Server] Exiting...Interpolating to default mimic_obs...")
        # Always ensure safe return to default pose on exit
        time_back_to_default = 2.0
        for i in range(int(time_back_to_default / control_dt)):
            interp_mimic_obs = last_mimic_obs + (DEFAULT_MIMIC_OBS[args.robot] - last_mimic_obs) * (
                i / (time_back_to_default / control_dt)
            )
            redis_client.set(f"action_mimic_{args.robot}", json.dumps(interp_mimic_obs.tolist()))
            redis_client.set(f"action_hand_{args.robot}", json.dumps(DEFAULT_ACTION_HAND[args.robot].tolist()))
            time.sleep(control_dt)
        redis_client.set(f"action_mimic_{args.robot}", json.dumps(DEFAULT_MIMIC_OBS[args.robot].tolist()))
        redis_client.set(f"action_hand_{args.robot}", json.dumps(DEFAULT_ACTION_HAND[args.robot].tolist()))
        last_mimic_obs = DEFAULT_MIMIC_OBS[args.robot]
        exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motion_file",
        help="Path to your *.pkl motion file for MotionLib",
        default="/Users/jbeisswenger/Desktop/PhD/Humanoid_VLA/humanoid-vla/TWIST/twist_motion_dataset/home/yanjieze/projects/g1_wbc/humanoid-motion-imitation/track_dataset/twist_motion_dataset/biomotionlab_ntroje/rub103_0005_normal_walk1.pkl",
    )
    parser.add_argument("--robot", type=str, default="g1", choices=["g1"])
    parser.add_argument(
        "--steps", type=str, default="1", help="Comma-separated steps for future frames (tar_obs_steps)"
    )
    parser.add_argument("--vis", action="store_true", help="Visualize the motion")
    args = parser.parse_args()

    # args.vis = True

    print("Robot type: ", args.robot)
    print("Motion file: ", args.motion_file)
    print("Steps: ", args.steps)

    HERE = os.path.dirname(os.path.abspath(__file__))

    if args.robot == "g1":
        xml_file = f"{HERE}/../assets/g1/g1_mocap_with_wrist_roll.xml"
        robot_base = "pelvis"
    else:
        raise ValueError(f"robot type {args.robot} not supported")

    main(args, xml_file, robot_base)