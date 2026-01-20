#!/usr/bin/env python3
"""
Trossen Arm <-> OpenPI Policy Server Bridge with Action Smoothing

Simple approach to reduce discontinuities between action chunks using
exponential moving average (EMA) smoothing.

The smoothing formula:
    smoothed_action = alpha * new_action + (1 - alpha) * previous_action

Where alpha controls responsiveness vs smoothness:
    - Lower alpha (0.2-0.4): More smoothing, slower response
    - Higher alpha (0.6-0.8): Less smoothing, faster response

Usage:
    python main_smooth.py --mode autonomous --task_prompt "grab red cube" --smooth_alpha 0.5
"""

import argparse
import logging
import time
import numpy as np
from openpi_client import websocket_client_policy
import cv2
from scipy.interpolate import PchipInterpolator

from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.robots import make_robot_from_config
from lerobot_robot_trossen.config_bi_widowxai_follower import BiWidowXAIFollowerRobotConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrossenOpenPIBridgeSmooth:
    """Bridge with EMA action smoothing for smoother chunk transitions."""

    def __init__(
        self,
        policy_server_host: str = "localhost",
        policy_server_port: int = 8000,
        control_frequency: int = 30,
        test_mode: str = "autonomous",
        max_steps: int = 1000,
        smooth_alpha: float = 0.5,
    ):
        self.control_frequency = control_frequency
        self.max_steps = max_steps
        self.dt = 1.0 / control_frequency
        self.test_mode = test_mode
        self.smooth_alpha = smooth_alpha

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
        self.action_chunk_size = 50
        self.episode_step = 0
        self.is_running = False
        self.rate_of_inference = 50
        self.action_dim = len(self.robot._joint_ft)
        
        # EMA smoothing state
        self.previous_action = None
        
        logger.info(f"Action smoothing enabled: alpha={smooth_alpha}")

    def smooth_action(self, action: np.ndarray) -> np.ndarray:
        """Apply EMA smoothing to reduce discontinuities."""
        if self.previous_action is None:
            self.previous_action = action.copy()
            return action
        
        smoothed = self.smooth_alpha * action + (1 - self.smooth_alpha) * self.previous_action
        self.previous_action = smoothed.copy()
        return smoothed

    def execute_action(self, action: np.ndarray):
        """Execute action on the arm."""
        if self.test_mode == "test":
            logger.info(f"TEST MODE: Would execute action: {action}")
            return
        elif self.test_mode == "autonomous":
            joint_features = list(self.robot._joint_ft.keys())
            action_dict = {k: action[i] for i, k in enumerate(joint_features)}
            self.robot.send_action(action_dict)

    def move_to_start_position(self, goal_position: np.ndarray, duration: float = 5.0):
        """Smoothly move to the first action position."""
        joint_pos_keys = [k for k in self.robot.get_observation().keys() if k.endswith('.pos')]
        current_pose = np.array([self.robot.get_observation()[k] for k in joint_pos_keys])
        
        waypoints = np.array([current_pose, goal_position])
        timepoints = np.array([0, duration])
        interpolator = PchipInterpolator(timepoints, waypoints, axis=0)

        start_time = time.time()
        end_time = start_time + duration

        while time.time() < end_time:
            current_time = time.time() - start_time
            positions = interpolator(current_time)
            self.execute_action(positions)
        
        # Initialize smoothing state with goal position
        self.previous_action = goal_position.copy()

    def _get_observation_for_policy(self, task_prompt: str, center_crop: bool = False) -> dict:
        """Prepare observation dictionary for policy inference."""
        observation_dict = self.robot.get_observation()

        joint_pos_keys = [k for k in observation_dict.keys() if k.endswith('.pos')]
        joint_positions = np.array([observation_dict[k] for k in joint_pos_keys])

        cameras = list(self.robot._cameras_ft.keys())
        for cam in cameras:
            image_hwc = observation_dict[cam]
            if center_crop:
                h, w, _ = image_hwc.shape
                crop_size = min(h, w)
                top = (h - crop_size) // 2
                left = (w - crop_size) // 2
                image_hwc = image_hwc[top:top + crop_size, left:left + crop_size]
            image_resized = cv2.resize(image_hwc, (224, 224))
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            image_chw = np.transpose(image_rgb, (2, 0, 1))
            observation_dict[cam] = image_chw

        return {
            "state": joint_positions,
            "images": {cam: observation_dict[cam] for cam in cameras},
            "prompt": task_prompt
        }

    def run_episode(self, task_prompt: str = "look down", center_crop: bool = False):
        """Run a single episode with action smoothing."""
        logger.info(f"Starting episode with prompt: '{task_prompt}'")
        self.episode_step = 0
        self.action_chunk_idx = 0
        self.current_action_chunk = None
        self.is_running = True
        self.previous_action = None
        is_first_step = True

        while self.is_running and self.episode_step < self.max_steps:
            start_loop_time = time.perf_counter()

            # Request new action chunk when needed
            if self.current_action_chunk is None or self.action_chunk_idx >= self.rate_of_inference:
                observation = self._get_observation_for_policy(task_prompt, center_crop=center_crop)
                logger.info(f"Step {self.episode_step}: Requesting new action chunk")
                response = self.policy_client.infer(observation)
                self.current_action_chunk = response["actions"]
                self.action_chunk_idx = 0
                logger.info(f"Received action chunk: {self.current_action_chunk.shape}")

            # Get raw action from chunk
            raw_action = self.current_action_chunk[self.action_chunk_idx]
            
            # Apply EMA smoothing
            smoothed_action = self.smooth_action(raw_action)

            if is_first_step:
                logger.info("Moving to start position...")
                self.move_to_start_position(raw_action, duration=5.0)
                is_first_step = False
            else:
                self.execute_action(smoothed_action)

            self.action_chunk_idx += 1
            self.episode_step += 1

            # Maintain control frequency
            dt_s = time.perf_counter() - start_loop_time
            if self.dt - dt_s > 0:
                time.sleep(self.dt - dt_s)
            loop_s = time.perf_counter() - start_loop_time
            logger.info(f"time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        self.is_running = False
        logger.info(f"Episode completed after {self.episode_step} steps")

    def autonomous_mode(self, task_prompt: str = "look down"):
        """Run in autonomous mode."""
        logger.info("Starting autonomous mode with action smoothing")
        self.run_episode(task_prompt=task_prompt, center_crop=True)

    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up...")
        self.robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trossen Bridge with Action Smoothing")
    parser.add_argument("--policy_host", default="localhost", help="Policy server host")
    parser.add_argument("--policy_port", type=int, default=8000, help="Policy server port")
    parser.add_argument("--control_freq", type=int, default=30, help="Control frequency in Hz")
    parser.add_argument("--mode", choices=["autonomous", "test"], default="autonomous",
                        help="Operation mode")
    parser.add_argument("--task_prompt", default="move the arm to the left",
                        help="Task description for the policy")
    parser.add_argument("--max_steps", type=int, default=3600, help="Maximum steps per episode")
    parser.add_argument("--smooth_alpha", type=float, default=0.5,
                        help="EMA smoothing factor (0.2-0.8). Lower=smoother, higher=responsive")
    
    args = parser.parse_args()

    bridge = TrossenOpenPIBridgeSmooth(
        policy_server_host=args.policy_host,
        policy_server_port=args.policy_port,
        control_frequency=args.control_freq,
        test_mode=args.mode,
        max_steps=args.max_steps,
        smooth_alpha=args.smooth_alpha,
    )

    try:
        bridge.autonomous_mode(task_prompt=args.task_prompt)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected. Cleaning up...")
    except Exception as e:
        logger.error(f"Error in autonomous mode: {e}")
    finally:
        bridge.cleanup()
