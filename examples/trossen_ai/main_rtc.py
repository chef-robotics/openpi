#!/usr/bin/env python3
"""
Trossen Arm <-> OpenPI Policy Server Bridge (Bimanual Version)

Bridge between a bimanual widowx and the OpenPI policy server.
Handles:
1. Collecting observations from the arm (joint positions, images)
2. Sending observations to the policy server via WebSocket
3. Receiving action predictions
4. Executing actions on the arm

Supports Real-Time Chunking (RTC) for smoother action transitions.
Reference: https://github.com/Physical-Intelligence/real-time-chunking-kinetix

Usage:
    python main.py --mode autonomous --task_prompt "grab and handover red cube"

    Test mode (no movement):
    python main.py --mode test --task_prompt "grab and handover red cube"
    
    With RTC enabled (smoother motion):
    python main.py --mode autonomous --task_prompt "grab and handover red cube" --use_rtc
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
import threading
import queue

from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.robots import (  # noqa: F401
    make_robot_from_config,
)
from lerobot_robot_trossen.config_bi_widowxai_follower import BiWidowXAIFollowerRobotConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RTCActionScheduler:
    """
    Real-Time Chunking (RTC) Action Scheduler
    
    Implements inference-time RTC to reduce discontinuities between action chunks.
    Key ideas from the paper "Real-Time Execution of Action Chunking Flow Policies":
    
    1. Overlap execution: Start inferring next chunk before current one finishes
    2. Action blending: Smoothly blend overlapping regions between chunks
    3. Prefix conditioning: New chunks are aware of committed actions
    
    This client-side implementation provides action blending without server modifications.
    For full RTC with inpainting, server-side changes would be needed.
    """
    
    def __init__(
        self,
        action_chunk_size: int = 50,
        execution_horizon: int = 50,  # s: how many steps to execute before re-querying
        blend_horizon: int = 10,  # Number of steps to blend between chunks
        blend_type: str = "linear",  # "linear", "cosine", or "exponential"
    ):
        self.action_chunk_size = action_chunk_size  # H: full chunk size
        self.execution_horizon = execution_horizon  # s: steps before re-query
        self.blend_horizon = blend_horizon
        self.blend_type = blend_type
        
        # Buffer to store committed actions and pending chunks
        self.committed_actions = []  # Actions that have been executed
        self.current_chunk = None  # Current action chunk being executed
        self.next_chunk = None  # Pre-fetched next chunk
        self.chunk_start_idx = 0  # Global step index where current chunk starts
        self.current_step = 0  # Global step counter
        
        # For async inference
        self.inference_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
    def _get_blend_weights(self, num_steps: int) -> np.ndarray:
        """Generate blending weights for smooth transitions."""
        t = np.linspace(0, 1, num_steps)
        
        if self.blend_type == "linear":
            return t
        elif self.blend_type == "cosine":
            # Cosine blending for smoother transitions
            return 0.5 * (1 - np.cos(np.pi * t))
        elif self.blend_type == "exponential":
            # Exponential blending
            return 1 - np.exp(-3 * t)
        else:
            return t
    
    def set_new_chunk(self, actions: np.ndarray, is_first: bool = False):
        """
        Set a new action chunk with proper blending setup.
        
        Args:
            actions: New action chunk of shape (H, action_dim)
            is_first: Whether this is the first chunk (no blending needed)
        """
        if is_first or self.current_chunk is None:
            self.current_chunk = actions.copy()
            self.chunk_start_idx = self.current_step
            self.next_chunk = None
        else:
            # Store as next chunk for blending
            self.next_chunk = actions.copy()
    
    def get_action(self) -> tuple[np.ndarray, bool]:
        """
        Get the next action to execute with RTC blending.
        
        Returns:
            action: The blended action to execute
            needs_new_chunk: Whether a new chunk should be requested
        """
        if self.current_chunk is None:
            raise RuntimeError("No action chunk set. Call set_new_chunk first.")
        
        # Index within current chunk
        local_idx = self.current_step - self.chunk_start_idx
        
        # Check if we need to transition to next chunk
        if local_idx >= self.execution_horizon and self.next_chunk is not None:
            # Transition: current becomes the blended version
            self.current_chunk = self.next_chunk
            self.chunk_start_idx = self.current_step
            self.next_chunk = None
            local_idx = 0
        
        # Determine if we need blending
        action = self.current_chunk[local_idx].copy()
        
        # If we have a next chunk and we're in the overlap region, blend
        if self.next_chunk is not None:
            overlap_start = self.execution_horizon - self.blend_horizon
            if local_idx >= overlap_start:
                # We're in the blending region
                blend_idx = local_idx - overlap_start
                
                # The next chunk's corresponding index
                # New chunk overlaps: next_chunk[0:blend_horizon] should blend with
                # current_chunk[execution_horizon-blend_horizon:execution_horizon]
                next_local_idx = blend_idx
                
                if next_local_idx < len(self.next_chunk):
                    blend_weights = self._get_blend_weights(self.blend_horizon)
                    w = blend_weights[blend_idx]
                    
                    action = (1 - w) * self.current_chunk[local_idx] + w * self.next_chunk[next_local_idx]
        
        self.current_step += 1
        
        # Determine if we need a new chunk
        # Request new chunk when we're blend_horizon steps before the end of execution_horizon
        steps_until_switch = self.execution_horizon - local_idx
        needs_new_chunk = (steps_until_switch <= self.blend_horizon) and (self.next_chunk is None)
        
        return action, needs_new_chunk
    
    def get_committed_actions(self, num_actions: int) -> np.ndarray | None:
        """
        Get the most recent committed actions for prefix conditioning.
        Used for server-side RTC if supported.
        
        Args:
            num_actions: Number of recent actions to return
            
        Returns:
            Array of shape (num_actions, action_dim) or None if not enough history
        """
        if len(self.committed_actions) < num_actions:
            return None
        return np.array(self.committed_actions[-num_actions:])
    
    def commit_action(self, action: np.ndarray):
        """Record an action as executed/committed."""
        self.committed_actions.append(action.copy())
        
    def reset(self):
        """Reset the scheduler for a new episode."""
        self.committed_actions = []
        self.current_chunk = None
        self.next_chunk = None
        self.chunk_start_idx = 0
        self.current_step = 0
class TrossenOpenPIBridge:
    """Bridge between a Trossen AI Stationary Kit and OpenPI policy server."""

    def __init__(
        self,
        policy_server_host: str = "localhost",
        policy_server_port: int = 8000,
        control_frequency: int = 30,
        test_mode: str = "autonomous",  # "autonomous" or "test"
        max_steps: int = 1000,
        use_rtc: bool = False,
        rtc_blend_horizon: int = 10,
        rtc_blend_type: str = "cosine",
        rtc_execution_horizon: int = None,  # If None, uses rate_of_inference
    ):
        self.control_frequency = control_frequency
        self.max_steps = max_steps
        self.dt = 1.0 / control_frequency
        self.test_mode = test_mode
        self.use_rtc = use_rtc

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
        # Number of actions per chunk from the policy
        # Pi0: 50 (flow-based), Pi0.5/Pi0FAST: 32 (autoregressive)
        # Will be auto-detected from first response if not specified
        self.action_chunk_size = 50
        self.episode_step = 0
        self.is_running = False
        # Number of control steps per policy inference
        # For Pi0: typically 50, for Pi0.5: typically 32
        self.rate_of_inference = 50

        self.temporal_ensemble_coefficient = None  # Temporal ensembling weight (can be set to None for no ensembling)

        # FIFO Buffer for actions
        self.action_buffer = defaultdict(list)
        self.action_buffer_size = self.max_steps + self.action_chunk_size  # Buffer size to hold actions for the entire episode

        self.action_dim = len(self.robot._joint_ft)  # 7 joints per arm * 2 arms
        
        # RTC (Real-Time Chunking) for smoother action transitions
        if self.use_rtc:
            execution_horizon = rtc_execution_horizon if rtc_execution_horizon else self.rate_of_inference
            self.rtc_scheduler = RTCActionScheduler(
                action_chunk_size=self.action_chunk_size,
                execution_horizon=execution_horizon,
                blend_horizon=rtc_blend_horizon,
                blend_type=rtc_blend_type,
            )
            logger.info(f"RTC enabled: blend_horizon={rtc_blend_horizon}, blend_type={rtc_blend_type}")
        else:
            self.rtc_scheduler = None

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

    def _get_observation_for_policy(self, task_prompt: str, center_crop: bool = False) -> dict:
        """Prepare observation dictionary for policy inference."""
        observation_dict = self.robot.get_observation()

        # Extract joint positions from observation
        joint_pos_keys = [k for k in observation_dict.keys() if k.endswith('.pos')]
        joint_positions = np.array([observation_dict[k] for k in joint_pos_keys])

        # Transform and resize images from all cameras
        cameras = list(self.robot._cameras_ft.keys())
        for cam in cameras:
            image_hwc = observation_dict[cam] # shape: (H, W, C), BGR
            if center_crop:
                h, w, _ = image_hwc.shape
                crop_size = min(h, w)
                # Compute top, left corner coordinates
                top = (h - crop_size) // 2
                left = (w - crop_size) // 2
                # Center crop
                image_hwc = image_hwc[top:top + crop_size, left:left + crop_size]
            # convert BGR to RGB
            image_resized = cv2.resize(image_hwc, (224, 224))
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            image_chw = np.transpose(image_rgb, (2, 0, 1))
            observation_dict[cam] = image_chw

        # Create observation for policy to follow the ALOHA format
        observation = {
            "state": joint_positions,
            "images": {cam: observation_dict[cam] for cam in cameras},
            "prompt": task_prompt
        }
        return observation

    def run_episode(self, task_prompt: str = "look down", center_crop=False):
        """Run a single episode of policy execution."""
        logger.info(f"Starting episode with prompt: '{task_prompt}'")
        self.episode_step = 0
        self.action_chunk_idx = 0
        self.current_action_chunk = None
        self.is_running = True
        is_first_step = True
        
        # Reset RTC scheduler if enabled
        if self.rtc_scheduler is not None:
            self.rtc_scheduler.reset()
            self._run_episode_rtc(task_prompt, is_first_step, center_crop)
        else:
            self._run_episode_standard(task_prompt, is_first_step, center_crop)

    def _run_episode_standard(self, task_prompt: str, is_first_step: bool, center_crop: bool = False):
        """Standard episode execution without RTC."""
        while self.is_running and self.episode_step < self.max_steps:
            start_loop_time = time.perf_counter()

            # Request new action chunk after consuming the previous one
            if self.current_action_chunk is None or self.action_chunk_idx >= self.rate_of_inference:
                observation = self._get_observation_for_policy(task_prompt, center_crop=center_crop)
            
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

    def _run_episode_rtc(self, task_prompt: str, is_first_step: bool, center_crop: bool = False):
        """
        Episode execution with Real-Time Chunking (RTC) for smoother transitions.
        
        RTC works by:
        1. Pre-fetching the next action chunk before the current one finishes
        2. Blending overlapping regions between consecutive chunks
        3. Optionally using server-side inpainting for prefix conditioning (Pi0 only)
        
        This eliminates discontinuities at chunk boundaries.
        
        Note: Client-side blending works for both Pi0 (flow-based) and Pi0.5/Pi0FAST 
        (autoregressive). Server-side inpainting only works for Pi0.
        """
        logger.info("Running with RTC (Real-Time Chunking) enabled")
        
        # Check if server supports RTC with native inpainting (Pi0 only, not Pi0.5)
        server_supports_rtc = False
        if hasattr(self.policy_client, 'metadata'):
            server_supports_rtc = self.policy_client.metadata.get('rtc_enabled', False)
            if server_supports_rtc:
                logger.info("Server supports native RTC inpainting (Pi0 flow-based)")
        
        # Get initial chunk
        observation = self._get_observation_for_policy(task_prompt, center_crop=center_crop)
        logger.info(f"Step {self.episode_step}: Requesting initial action chunk")
        response = self.policy_client.infer(observation)
        
        # Auto-detect action chunk size from response (Pi0: 50, Pi0.5: 32)
        actual_chunk_size = response["actions"].shape[0]
        if actual_chunk_size != self.rtc_scheduler.action_chunk_size:
            logger.info(f"Auto-detected action chunk size: {actual_chunk_size} (was {self.rtc_scheduler.action_chunk_size})")
            self.rtc_scheduler.action_chunk_size = actual_chunk_size
            # Also adjust execution horizon if it exceeds chunk size
            if self.rtc_scheduler.execution_horizon > actual_chunk_size:
                self.rtc_scheduler.execution_horizon = actual_chunk_size
                logger.info(f"Adjusted execution horizon to {actual_chunk_size}")
        
        self.rtc_scheduler.set_new_chunk(response["actions"], is_first=True)
        logger.info(f"Received initial action chunk: {response['actions'].shape}")
        
        # Track inference timing for adaptive scheduling
        inference_times = []
        
        while self.is_running and self.episode_step < self.max_steps:
            start_loop_time = time.perf_counter()
            
            # Get blended action from RTC scheduler
            a_t, needs_new_chunk = self.rtc_scheduler.get_action()
            
            # Start inference for next chunk if needed
            if needs_new_chunk:
                observation = self._get_observation_for_policy(task_prompt, center_crop=center_crop)
                logger.info(f"Step {self.episode_step}: Pre-fetching next action chunk (RTC)")
                
                # If server supports RTC, send action prefix for inpainting
                if server_supports_rtc:
                    # Get committed actions to use as prefix
                    # The prefix should be the overlap region that the new chunk should match
                    prefix = self.rtc_scheduler.get_committed_actions(self.rtc_scheduler.blend_horizon)
                    if prefix is not None:
                        observation['action_prefix'] = prefix
                        observation['inpainting_strength'] = 1.0
                        logger.info(f"Sending action prefix of length {len(prefix)} for server-side inpainting")
                
                inference_start_time = time.perf_counter()
                response = self.policy_client.infer(observation)
                inference_time = time.perf_counter() - inference_start_time
                inference_times.append(inference_time)
                
                # Log RTC info if available
                if 'rtc_info' in response:
                    rtc_info = response['rtc_info']
                    if rtc_info.get('native_inpainting'):
                        logger.info(f"Server used native inpainting (prefix_len={rtc_info['prefix_len']})")
                
                self.rtc_scheduler.set_new_chunk(response["actions"])
                logger.info(f"Received action chunk (inference: {inference_time*1000:.1f}ms)")
            
            # Execute the current action
            if is_first_step:
                logger.info("Moving to start position to avoid large jumps...")
                self.move_to_start_position(a_t, duration=5.0)
                is_first_step = False
            else:
                self.execute_action(a_t)
            
            # Commit the action for potential future prefix conditioning
            self.rtc_scheduler.commit_action(a_t)
            
            self.episode_step += 1

            dt_s = time.perf_counter() - start_loop_time
            busy_wait_time = self.dt - dt_s

            # Busy wait to maintain control frequency
            if busy_wait_time > 0:
                time.sleep(busy_wait_time)
            loop_s = time.perf_counter() - start_loop_time
            logger.info(f"time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        self.is_running = False
        
        # Report inference timing statistics
        if inference_times:
            avg_inference = np.mean(inference_times) * 1000
            logger.info(f"Episode completed. Avg inference time: {avg_inference:.1f}ms")
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
    
    # RTC (Real-Time Chunking) arguments for smoother action transitions
    parser.add_argument("--use_rtc", action="store_true",
                        help="Enable Real-Time Chunking for smoother action transitions")
    parser.add_argument("--rtc_blend_horizon", type=int, default=10,
                        help="Number of steps to blend between action chunks (default: 10)")
    parser.add_argument("--rtc_blend_type", choices=["linear", "cosine", "exponential"], default="cosine",
                        help="Blending function type: linear, cosine, or exponential (default: cosine)")
    parser.add_argument("--rtc_execution_horizon", type=int, default=None,
                        help="Steps to execute before re-querying (default: same as rate_of_inference=50)")
    
    args = parser.parse_args()

    bridge = TrossenOpenPIBridge(
        policy_server_host=args.policy_host,
        policy_server_port=args.policy_port,
        control_frequency=args.control_freq,
        test_mode=args.mode,
        max_steps=args.max_steps,
        use_rtc=args.use_rtc,
        rtc_blend_horizon=args.rtc_blend_horizon,
        rtc_blend_type=args.rtc_blend_type,
        rtc_execution_horizon=args.rtc_execution_horizon,
    )

    try:
        bridge.autonomous_mode(task_prompt=args.task_prompt)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected. Cleaning up...")
    except Exception as e:
        logger.error(f"Error in autonomous mode: {e}")
    finally:
        bridge.cleanup()

