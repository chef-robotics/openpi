#!/usr/bin/env python3
"""
Trossen Arm <-> OpenPI Policy Server Bridge (Bimanual Version)

Bridge between a bimanual widowx and the OpenPI policy server.
Handles:
1. Collecting observations from the arm (joint positions, images)
2. Sending observations to the policy server via WebSocket
3. Receiving action predictions
4. Executing actions on the arm

Usage:
    python main.py --mode autonomous --task_prompt "grab and handover red cube"

    Test mode (no movement):
    python main.py --mode test --task_prompt "grab and handover red cube"
"""

import argparse
import logging
import time
import sys
import os
import numpy as np
from openpi_client import websocket_client_policy
import cv2
from collections import defaultdict
from scipy.interpolate import PchipInterpolator

from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.robots import (  # noqa: F401
    make_robot_from_config,
)
from lerobot_robot_trossen.config_bi_widowxai_follower import BiWidowXAIFollowerRobotConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _extract_bowl_roi_from_cam_high_bgr(
    cam_high_bgr_hwc: np.ndarray,
    *,
    center_x: int = 330,
    center_y: int = 392,
    crop_size: int = 112,
    out_size: int = 224,
) -> np.ndarray:
    """Extract bowl ROI from cam_high (BGR, HWC) and return ROI (BGR, HWC) resized to out_size."""
    if cam_high_bgr_hwc.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape {cam_high_bgr_hwc.shape}")
    h, w = cam_high_bgr_hwc.shape[:2]
    half = crop_size // 2
    x1 = max(0, center_x - half)
    y1 = max(0, center_y - half)
    x2 = min(w, x1 + crop_size)
    y2 = min(h, y1 + crop_size)
    # Adjust if crop goes out of bounds
    if x2 - x1 < crop_size:
        x1 = max(0, x2 - crop_size)
    if y2 - y1 < crop_size:
        y1 = max(0, y2 - crop_size)
    cropped = cam_high_bgr_hwc[y1:y2, x1:x2]
    return cv2.resize(cropped, (out_size, out_size), interpolation=cv2.INTER_LINEAR)



class TrossenOpenPIBridge:
    """Bridge between a Trossen AI Stationary Kit and OpenPI policy server."""

    def __init__(
        self,
        policy_server_host: str = "localhost",
        policy_server_port: int = 8000,
        control_frequency: int = 30,
        test_mode: str = "autonomous",  # "autonomous" or "test"
        max_steps: int = 1000
    ):
        self.control_frequency = control_frequency
        self.max_steps = max_steps
        self.dt = 1.0 / control_frequency
        self.test_mode = test_mode

        logger.info(f"Connecting to policy server at {policy_server_host}:{policy_server_port}")
        self.policy_client = websocket_client_policy.WebsocketClientPolicy(
            host=policy_server_host,
            port=policy_server_port
        )

        robot_config = BiWidowXAIFollowerRobotConfig(
            id="bimanual_follower",
            left_arm_ip_address="192.168.1.5",
            right_arm_ip_address="192.168.1.4",
            min_time_to_move_multiplier=4.0,
            loop_rate=30,
            cameras={
                "cam_high": RealSenseCameraConfig(
                    serial_number_or_name="230322270292",
                    width=640, height=480, fps=30, use_depth=False
                ),
                "cam_low": RealSenseCameraConfig(
                    serial_number_or_name="230322271134",
                    width=640, height=480, fps=30, use_depth=False
                ),
                "cam_right_wrist": RealSenseCameraConfig(
                    serial_number_or_name="230422272861",
                    width=640, height=480, fps=30, use_depth=False
                ),
                "cam_left_wrist": RealSenseCameraConfig(
                    serial_number_or_name="230322270548",
                    width=640, height=480, fps=30, use_depth=False
                ),
            }
        )
        self.robot = make_robot_from_config(robot_config)
        self.robot.connect()

        self.current_action_chunk = None
        self.action_chunk_idx = 0
        self.action_chunk_size = 50  # Number of actions per chunk from the policy (Defined by the policy server in this case 50)
        self.episode_step = 0
        self.is_running = False
        self.rate_of_inference = 50  # Number of control steps per policy inference (matches README and Pi-0 paper)

        self.temporal_ensemble_coefficient = None  # Temporal ensembling weight (can be set to None for no ensembling)

        # FIFO Buffer for actions
        self.action_buffer = defaultdict(list)
        self.action_buffer_size = self.max_steps + self.action_chunk_size  # Buffer size to hold actions for the entire episode

        self.action_dim = len(self.robot._joint_ft)  # 7 joints per arm * 2 arms

    def execute_action(self, action: np.ndarray):
        """Execute action on the arm."""
        full_action = action.copy()

        if self.test_mode == "test":
            logger.info(f"TEST MODE: Would execute action: {full_action}")
            return
        elif self.test_mode == "autonomous":
            joint_features = list(self.robot._joint_ft.keys())
            action_dict = {k: full_action[i] for i, k in enumerate(joint_features)}
            self.robot.send_action(action_dict)
        else:
            logger.error(f"Unknown mode: {self.test_mode}. No action executed.")

    def move_to_start_position(self, goal_position: np.ndarray, duration: float = 5.0):
        """The first position queried from the policy depends on the training data.
        Assuming the first position is a "stage" position will result in a large jump if the arm is not already there.
        To avoid this, we smoothly move the arm to a first action/position before sending the rest of the actions.
        We use PCHIP interpolation for smooth trajectory generation and give it enough time to reach the position to prevent
        jumps and triggering safety stops (velocity limits)."""

        joint_pos_keys = [k for k in self.robot.get_observation().keys() if k.endswith('.pos')]
        current_pose = np.array([self.robot.get_observation()[k] for k in joint_pos_keys])
        # Example stage_pose for bimanual WidowX arms.
        # Each value corresponds to a joint position (in radians) for the 14 joints:
        # [left_joint_0, left_joint_1, left_joint_2, left_joint_3, left_joint_4, left_joint_5, left_left_carriage_joint,
        #  right_joint_0, right_joint_1, right_joint_2, right_joint_3, right_joint_4, right_joint_5, right_left_carriage_joint]
        # The values below represent a "stage" pose, e.g. arms up and open, ready for task start.
        # stage_pose = np.array([0, np.pi/3, np.pi/6, np.pi/5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        waypoints = np.array([current_pose, goal_position])
        timepoints = np.array([0, duration])  # Use the provided duration
        interpolator_position = PchipInterpolator(timepoints, waypoints, axis=0)

        start_time = time.time()
        end_time = start_time + timepoints[-1]

        while time.time() < end_time:
            loop_start_time = time.time()
            current_time = loop_start_time - start_time
            positions = interpolator_position(current_time)
            self.execute_action(positions)

    def run_episode(
        self,
        task_prompt: str = "look down",
        center_crop: bool = False,
        use_bowl_roi: bool = False,
        bowl_roi_center_x: int = 330,
        bowl_roi_center_y: int = 392,
        bowl_roi_crop_size: int = 112,
    ):
        """Run a single episode of policy execution."""
        logger.info(f"Starting episode with prompt: '{task_prompt}'")
        self.episode_step = 0
        self.action_chunk_idx = 0
        self.current_action_chunk = None
        self.is_running = True
        is_first_step = True

        while self.is_running and self.episode_step < self.max_steps:
            start_loop_time = time.perf_counter()

            # Request new action chunk after consuming the previous one
            if self.current_action_chunk is None or self.action_chunk_idx >= self.rate_of_inference:
                observation_dict = self.robot.get_observation()

                # Extract joint positions from observation
                joint_pos_keys = [k for k in observation_dict.keys() if k.endswith('.pos')]
                joint_positions = np.array([observation_dict[k] for k in joint_pos_keys])

                # Transform and resize images from all cameras
                cameras = list(self.robot._cameras_ft.keys())
                cam_high_bgr_hwc_raw = None
                for cam in cameras:
                    image_hwc = observation_dict[cam] # shape: (H, W, C), BGR
                    if cam == "cam_high":
                        # Keep a copy of the raw cam_high frame for ROI extraction (ROI coords are in original pixels).
                        cam_high_bgr_hwc_raw = image_hwc
                    if center_crop:
                        h, w, _ = image_hwc.shape
                        crop_size = min(h, w)
                        # Compute top, left corner coordinates
                        top = (h - crop_size) // 2
                        left = (w - crop_size) // 2
                        # Center crop
                        image_hwc = image_hwc[top:top + crop_size, left:left + crop_size]
                    #convert BGR to RGB
                    image_resized = cv2.resize(image_hwc, (224, 224))
                    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
                    image_chw = np.transpose(image_rgb, (2, 0, 1))
                    observation_dict[cam] = image_chw

                # Optionally add bowl ROI camera for ROI-trained policies.
                # NOTE: Standard Aloha policies will error if you send unexpected cameras, so keep this gated.
                if use_bowl_roi:
                    if cam_high_bgr_hwc_raw is None:
                        raise RuntimeError("cam_high image not found; cannot compute bowl ROI")
                    roi_bgr = _extract_bowl_roi_from_cam_high_bgr(
                        cam_high_bgr_hwc_raw,
                        center_x=bowl_roi_center_x,
                        center_y=bowl_roi_center_y,
                        crop_size=bowl_roi_crop_size,
                        out_size=224,
                    )
                    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
                    roi_chw = np.transpose(roi_rgb, (2, 0, 1))
                    observation_dict["cam_high_bowl_roi"] = roi_chw

                # Create observation for policy to follow the ALOHA format
                observation = {
                    "state": joint_positions,
                    "images": {cam: observation_dict[cam] for cam in cameras},
                    "prompt": task_prompt
                }
                if use_bowl_roi:
                    observation["images"]["cam_high_bowl_roi"] = observation_dict["cam_high_bowl_roi"]
            
                logger.info(f"Step {self.episode_step}: Requesting new action chunk")
                response = self.policy_client.infer(observation)
                self.current_action_chunk = response["actions"]

                for k in range(self.action_chunk_size):
                    future_t = self.episode_step + k
                    if future_t < self.action_buffer_size:
                        self.action_buffer[future_t].append(self.current_action_chunk[k])

                self.action_chunk_idx = 0
                logger.info(f"Received action chunk: {self.current_action_chunk.shape}")

            # Select action using temporal ensembling if enabled
            if self.temporal_ensemble_coefficient is not None:
                if len(self.action_buffer[self.episode_step]) == 0:
                    a_t = np.zeros(self.action_dim)
                else:
                    candidates = np.array(self.action_buffer[self.episode_step])  # shape: (N, 14)
                    weights = self._get_weights(len(candidates))  # shape: (N,)
                    a_t = np.average(candidates, axis=0, weights=weights)  # shape: (14,)
            else:
                a_t = self.current_action_chunk[self.action_chunk_idx]
            # Execute the current action
            if is_first_step:
                logger.info("Moving to start position to avoid large jumps...")
                self.move_to_start_position(a_t, duration=5.0)
                is_first_step = False
            else:
                self.execute_action(a_t)

            self.action_chunk_idx += 1
            self.episode_step += 1


            dt_s = time.perf_counter() - start_loop_time
            busy_wait_time = self.dt - dt_s

            # Busy wait to maintain control frequency
            if busy_wait_time > 0:
                time.sleep(busy_wait_time)
            loop_s = time.perf_counter() - start_loop_time
            logger.info(f"time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        self.is_running = False
        logger.info(f"Episode completed after {self.episode_step} steps")

    def _get_weights(self, num_preds: int) -> np.ndarray:
        weights = np.exp(-self.temporal_ensemble_coefficient * np.arange(num_preds))
        return weights / weights.sum()


    def autonomous_mode(self, task_prompt: str = "look down"):
        """Run in autonomous mode where the arm executes policy predictions."""
        logger.info("Starting autonomous mode")
        self.run_episode(task_prompt=task_prompt, center_crop=True)

    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up...")
        self.robot.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trossen AI Stationary Kit <-> OpenPI Policy Server Bridge")
    parser.add_argument("--policy_host", default="localhost", help="Policy server host")
    parser.add_argument("--policy_port", type=int, default=8000, help="Policy server port")
    parser.add_argument("--control_freq", type=int, default=30, help="Control frequency in Hz")
    parser.add_argument("--mode", choices=["autonomous", "test"],  default="autonomous",
                        help="Operation mode: autonomous (execute) or test (no movement)")
    parser.add_argument("--task_prompt", default="move the arm to the left", help="Task description for the policy")
    parser.add_argument("--max_steps", type=int, default=3600, help="Maximum steps per episode")
    args = parser.parse_args()

    bridge = TrossenOpenPIBridge(
        policy_server_host=args.policy_host,
        policy_server_port=args.policy_port,
        control_frequency=args.control_freq,
        test_mode=args.mode,
        max_steps=args.max_steps
    )

    try:
        bridge.autonomous_mode(task_prompt=args.task_prompt)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected. Cleaning up...")
    except Exception as e:
        logger.error(f"Error in autonomous mode: {e}")
    finally:
        bridge.cleanup()

