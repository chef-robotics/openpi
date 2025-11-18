#!/usr/bin/env python3
"""
Trossen Arm <-> OpenPI Policy Server Bridge (Bimanual Version with Skill Selection)

Bridge between a bimanual widowx and the OpenPI policy server with keyboard-based skill switching.
Handles:
1. Collecting observations from the arm (joint positions, images)
2. Sending observations to the policy server via WebSocket
3. Receiving action predictions
4. Executing actions on the arm
5. Real-time skill switching via keyboard input (0-9)

Features:
- Loads skill instructions from a JSONL tasks file
- During execution, press 0-9 to switch to corresponding skill
- Current instruction is logged when requesting new action chunks
- Skill changes take effect on the next policy inference

Usage:
    python main_skills.py --mode autonomous --tasks_file path/to/tasks.jsonl

    Test mode (no movement):
    python main_skills.py --mode test --tasks_file path/to/tasks.jsonl

    Example:
    python main_skills.py --mode autonomous --tasks_file sandi/datasets/sandi/lettuce-sandwich-skills/meta/tasks.jsonl

Tasks file format (JSONL):
    {"task_index": 0, "task": "Pick up one slice of bread."}
    {"task_index": 1, "task": "Put one slice of bread at the blue cross mark on the table."}
    ...

During execution, press 0-9 to switch between the loaded skills.
"""

import argparse
import logging
import time
import sys
import os
import numpy as np
from openpi_client import websocket_client_policy
import cv2
from lerobot.robots import make_robot_from_config
from lerobot.robots.bi_widowxai_follower.config_bi_widowxai_follower import BiWidowXAIFollowerConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from collections import defaultdict
from scipy.interpolate import PchipInterpolator
import select
import tty
import termios
import json


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_skill_instructions(tasks_file: str) -> list:
    """Load skill instructions from a JSONL file.
    
    Args:
        tasks_file: Path to the tasks.jsonl file
        
    Returns:
        List of task strings ordered by task_index
    """
    if not os.path.exists(tasks_file):
        raise FileNotFoundError(f"Tasks file not found: {tasks_file}")
    
    tasks = []
    with open(tasks_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                task_data = json.loads(line)
                tasks.append((task_data['task_index'], task_data['task']))
    
    # Sort by task_index and extract just the task strings
    tasks.sort(key=lambda x: x[0])
    skill_instructions = [task[1] for task in tasks]
    
    logger.info(f"Loaded {len(skill_instructions)} skills from {tasks_file}")
    return skill_instructions


class TrossenOpenPIBridge:
    """Bridge between a Trossen AI Stationary Kit and OpenPI policy server."""

    def __init__(
        self,
        skill_instructions: list,
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
        self.skill_instructions = skill_instructions

        logger.info(f"Connecting to policy server at {policy_server_host}:{policy_server_port}")
        self.policy_client = websocket_client_policy.WebsocketClientPolicy(
            host=policy_server_host,
            port=policy_server_port
        )
        # The camera names should match the ones expected by the policy
        # e.g. "cam_high", "cam_low", "cam_right_wrist", "cam_left_wrist"
        bi_widowx_ai_config = BiWidowXAIFollowerConfig(
            left_arm_ip_address="192.168.1.5",
            right_arm_ip_address="192.168.1.4",
            min_time_to_move_multiplier=4.0,
            id="bimanual_follower",
            cameras={
                "cam_high": RealSenseCameraConfig(
                    serial_number_or_name="218622270304",
                    width=640, height=480, fps=30, use_depth=False
                ),
                "cam_low": RealSenseCameraConfig(
                    serial_number_or_name="130322272628",
                    width=640, height=480, fps=30, use_depth=False
                ),
                "cam_right_wrist": RealSenseCameraConfig(
                    serial_number_or_name="128422271347",
                    width=640, height=480, fps=30, use_depth=False
                ),
                "cam_left_wrist": RealSenseCameraConfig(
                    serial_number_or_name="218622274938",
                    width=640, height=480, fps=30, use_depth=False
                ),
            }
        )
        self.robot = make_robot_from_config(bi_widowx_ai_config)
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

        # Keyboard input tracking for skill selection
        self.current_skill_index = 0  # Default to first skill
        self.old_terminal_settings = None
        
        # Display available skills
        logger.info("=" * 60)
        logger.info("AVAILABLE SKILLS:")
        for idx, skill in enumerate(self.skill_instructions):
            logger.info(f"  [{idx}] {skill}")
        logger.info("=" * 60)
        logger.info(f"Current skill: [{self.current_skill_index}] {self.skill_instructions[self.current_skill_index]}")
        logger.info(f"Press 0-{len(self.skill_instructions)-1} during execution to switch skills")

    def setup_keyboard_input(self):
        """Setup terminal for non-blocking keyboard input."""
        self.old_terminal_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
    
    def restore_keyboard_input(self):
        """Restore terminal settings."""
        if self.old_terminal_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_terminal_settings)
    
    def check_keyboard_input(self):
        """Check for keyboard input without blocking. Returns skill index if valid input, None otherwise."""
        # Check if there's input available
        if select.select([sys.stdin], [], [], 0)[0]:
            char = sys.stdin.read(1)
            if char.isdigit():
                skill_index = int(char)
                if 0 <= skill_index < len(self.skill_instructions):
                    return skill_index
                else:
                    logger.warning(f"Invalid skill index: {skill_index}. Must be 0-{len(self.skill_instructions)-1}")
        return None

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

    def run_episode(self, task_prompt: str = "look down"):
        """Run a single episode of policy execution."""
        logger.info(f"Starting episode with initial prompt: '{task_prompt}'")
        self.episode_step = 0
        self.action_chunk_idx = 0
        self.current_action_chunk = None
        self.is_running = True
        is_first_step = True

        # Setup keyboard input for skill selection
        self.setup_keyboard_input()

        try:
            while self.is_running and self.episode_step < self.max_steps:
                start_loop_time = time.perf_counter()

                # Check for keyboard input to switch skills
                new_skill_index = self.check_keyboard_input()
                if new_skill_index is not None:
                    self.current_skill_index = new_skill_index
                    logger.info("=" * 60)
                    logger.info(f"SKILL SWITCHED: [{self.current_skill_index}] {self.skill_instructions[self.current_skill_index]}")
                    logger.info("=" * 60)

                # Request new action chunk after consuming the previous one
                if self.current_action_chunk is None or self.action_chunk_idx >= self.rate_of_inference:
                    # Get current skill instruction
                    current_instruction = self.skill_instructions[self.current_skill_index]
                    
                    observation_dict = self.robot.get_observation()

                    # Extract joint positions from observation
                    joint_pos_keys = [k for k in observation_dict.keys() if k.endswith('.pos')]
                    joint_positions = np.array([observation_dict[k] for k in joint_pos_keys])

                    # Transform and resize images from all cameras
                    cameras = list(self.robot._cameras_ft.keys())
                    for cam in cameras:
                        image_hwc = observation_dict[cam]
                        #convert BGR to RGB
                        image_resized = cv2.resize(image_hwc, (224, 224))
                        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
                        image_chw = np.transpose(image_rgb, (2, 0, 1))
                        observation_dict[cam] = image_chw

                    # Create observation for policy to follow the ALOHA format
                    observation = {
                        "state": joint_positions,
                        "images": {cam: observation_dict[cam] for cam in cameras},
                        "prompt": current_instruction
                    }
                
                    logger.info(f"Step {self.episode_step}: Requesting new action chunk with instruction: [{self.current_skill_index}] '{current_instruction}'")
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

        finally:
            # Restore terminal settings
            self.restore_keyboard_input()
            self.is_running = False
            logger.info(f"Episode completed after {self.episode_step} steps")

    def _get_weights(self, num_preds: int) -> np.ndarray:
        weights = np.exp(-self.temporal_ensemble_coefficient * np.arange(num_preds))
        return weights / weights.sum()


    def autonomous_mode(self, task_prompt: str = "look down"):
        """Run in autonomous mode where the arm executes policy predictions."""
        logger.info("Starting autonomous mode")
        self.run_episode(task_prompt=task_prompt)

    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up...")
        self.restore_keyboard_input()
        self.robot.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trossen AI Stationary Kit <-> OpenPI Policy Server Bridge with Skill Selection")
    parser.add_argument("--policy_host", default="localhost", help="Policy server host")
    parser.add_argument("--policy_port", type=int, default=8000, help="Policy server port")
    parser.add_argument("--control_freq", type=int, default=30, help="Control frequency in Hz")
    parser.add_argument("--mode", choices=["autonomous", "test"],  default="autonomous",
                        help="Operation mode: autonomous (execute) or test (no movement)")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument("--initial_skill", type=int, default=0, help="Initial skill index")
    parser.add_argument("--tasks_file", type=str, required=True, 
                        help="Path to tasks.jsonl file containing skill instructions")
    args = parser.parse_args()

    # Load skill instructions from file
    try:
        skill_instructions = load_skill_instructions(args.tasks_file)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Error parsing tasks file: {e}")
        sys.exit(1)

    bridge = TrossenOpenPIBridge(
        skill_instructions=skill_instructions,
        policy_server_host=args.policy_host,
        policy_server_port=args.policy_port,
        control_frequency=args.control_freq,
        test_mode=args.mode,
        max_steps=args.max_steps
    )

    # Set initial skill if provided
    if 0 <= args.initial_skill < len(skill_instructions):
        bridge.current_skill_index = args.initial_skill
        logger.info(f"Initial skill set to: [{args.initial_skill}] {skill_instructions[args.initial_skill]}")
    else:
        logger.warning(f"Invalid initial skill index {args.initial_skill}. Using default skill 0.")

    try:
        # Start with the selected skill instruction
        initial_instruction = skill_instructions[bridge.current_skill_index]
        bridge.autonomous_mode(task_prompt=initial_instruction)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected. Cleaning up...")
    except Exception as e:
        logger.error(f"Error in autonomous mode: {e}")
    finally:
        bridge.cleanup()

