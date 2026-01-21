#!/usr/bin/env python3
"""
Trossen Arm <-> OpenPI Policy Server Bridge (PCHIP Interpolation Version)

This version uses PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) 
interpolation for action chunk boundaries to achieve:
1. C1 continuity (smooth velocity transitions)
2. Shape-preserving interpolation (no overshoots)
3. Better boundary gap compensation than linear/cosine blending

Usage:
    python main_interp.py --mode autonomous --task_prompt "grab and handover red cube"
    
    With different interpolation modes:
    python main_interp.py --mode autonomous --task_prompt "task" --interp_mode pchip
    python main_interp.py --mode autonomous --task_prompt "task" --interp_mode cubic
"""

import argparse
import logging
import time
import numpy as np
from openpi_client import websocket_client_policy
import cv2
from collections import defaultdict
from scipy.interpolate import PchipInterpolator, CubicSpline, interp1d

from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.robots import make_robot_from_config
from lerobot_robot_trossen.config_bi_widowxai_follower import BiWidowXAIFollowerRobotConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class InterpActionScheduler:
    """
    PCHIP-based Action Scheduler for smooth action chunk transitions.
    
    Uses Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) to interpolate
    between action chunks, providing:
    - C1 continuity (continuous first derivative)
    - Shape-preserving interpolation (monotonic, no overshoots)
    - Smooth velocity transitions at chunk boundaries
    
    The key idea is to treat overlapping action predictions as data points
    and fit a smooth interpolating curve through them, rather than simple blending.
    """
    
    def __init__(
        self,
        action_chunk_size: int = 50,
        execution_horizon: int = 50,
        overlap_horizon: int = 15,  # Number of overlapping steps to use for interpolation
        interp_mode: str = "pchip",  # "pchip", "cubic", or "linear"
        use_velocity_matching: bool = True,  # Match velocities at boundaries
    ):
        self.action_chunk_size = action_chunk_size
        self.execution_horizon = execution_horizon
        self.overlap_horizon = overlap_horizon
        self.interp_mode = interp_mode
        self.use_velocity_matching = use_velocity_matching
        
        # Action chunk storage
        self.chunks = []  # List of (start_step, actions) tuples
        self.current_step = 0
        self.action_dim = None
        
        # Interpolator cache
        self._interpolator = None
        self._interp_valid_range = (0, 0)
        
        # History for velocity estimation
        self.executed_actions = []
        
    def set_new_chunk(self, actions: np.ndarray, is_first: bool = False):
        """
        Add a new action chunk to the scheduler.
        
        Args:
            actions: Action chunk of shape (H, action_dim)
            is_first: Whether this is the first chunk
        """
        if self.action_dim is None:
            self.action_dim = actions.shape[1]
            
        chunk_start = self.current_step
        if not is_first and len(self.chunks) > 0:
            # New chunks start from current step
            pass
        
        self.chunks.append((chunk_start, actions.copy()))
        
        # Invalidate interpolator cache
        self._interpolator = None
        
        # Keep only recent chunks to limit memory
        self._prune_old_chunks()
        
        # Rebuild interpolator
        self._build_interpolator()
        
        if is_first:
            logger.info(f"Initial chunk set at step {chunk_start}")
        else:
            logger.info(f"New chunk added at step {chunk_start}, total chunks: {len(self.chunks)}")
    
    def _prune_old_chunks(self):
        """Remove chunks that are no longer relevant."""
        # Keep chunks that might still be used for interpolation
        min_step = self.current_step - self.overlap_horizon
        self.chunks = [(start, actions) for start, actions in self.chunks 
                       if start + len(actions) > min_step]
    
    def _build_interpolator(self):
        """
        Build the interpolator from all available action chunks.
        
        For overlapping regions, we create weighted data points that
        smoothly transition from the old chunk to the new chunk.
        """
        if len(self.chunks) == 0:
            return
        
        # Determine the valid interpolation range
        min_step = min(start for start, _ in self.chunks)
        max_step = max(start + len(actions) for start, actions in self.chunks)
        
        # Build time points and action values
        # For each time step, collect predictions from all applicable chunks
        time_points = []
        action_points = []
        
        for t in range(min_step, max_step):
            # Collect all predictions for this timestep
            predictions = []
            weights = []
            
            for chunk_start, chunk_actions in self.chunks:
                chunk_idx = t - chunk_start
                if 0 <= chunk_idx < len(chunk_actions):
                    predictions.append(chunk_actions[chunk_idx])
                    
                    # Weight based on how "fresh" this prediction is
                    # More recent chunks get higher weight
                    freshness = 1.0 / (1.0 + (t - chunk_start) / self.action_chunk_size)
                    weights.append(freshness)
            
            if len(predictions) > 0:
                predictions = np.array(predictions)
                weights = np.array(weights)
                weights /= weights.sum()
                
                # Weighted average of predictions
                weighted_action = np.sum(predictions * weights[:, np.newaxis], axis=0)
                
                time_points.append(t)
                action_points.append(weighted_action)
        
        if len(time_points) < 2:
            return
        
        time_points = np.array(time_points)
        action_points = np.array(action_points)
        
        # Create interpolator based on mode
        if self.interp_mode == "pchip":
            # PCHIP: Shape-preserving, C1 continuous
            self._interpolator = PchipInterpolator(time_points, action_points, axis=0)
        elif self.interp_mode == "cubic":
            # Cubic spline: C2 continuous, may have overshoots
            self._interpolator = CubicSpline(time_points, action_points, axis=0, bc_type='natural')
        elif self.interp_mode == "linear":
            # Linear: Simple but may have velocity discontinuities
            self._interpolator = interp1d(time_points, action_points, axis=0, 
                                          kind='linear', fill_value='extrapolate')
        else:
            raise ValueError(f"Unknown interpolation mode: {self.interp_mode}")
        
        self._interp_valid_range = (time_points[0], time_points[-1])
    
    def get_action(self) -> tuple[np.ndarray, bool]:
        """
        Get the interpolated action for the current timestep.
        
        Returns:
            action: The interpolated action
            needs_new_chunk: Whether a new chunk should be requested
        """
        if self._interpolator is None or len(self.chunks) == 0:
            raise RuntimeError("No action chunk set. Call set_new_chunk first.")
        
        # Get interpolated action
        t = float(self.current_step)
        
        # Clamp to valid range
        t_clamped = np.clip(t, self._interp_valid_range[0], self._interp_valid_range[1])
        action = self._interpolator(t_clamped)
        
        # Store executed action for velocity estimation
        self.executed_actions.append(action.copy())
        if len(self.executed_actions) > 100:
            self.executed_actions.pop(0)
        
        self.current_step += 1
        
        # Determine if we need a new chunk
        # Request when we're overlap_horizon steps away from running out of data
        latest_chunk_start, latest_chunk = self.chunks[-1]
        steps_remaining = (latest_chunk_start + len(latest_chunk)) - self.current_step
        needs_new_chunk = steps_remaining <= self.overlap_horizon
        
        return action, needs_new_chunk
    
    def get_current_velocity(self) -> np.ndarray | None:
        """
        Estimate current velocity from recent executed actions.
        Can be used for velocity matching at chunk boundaries.
        """
        if len(self.executed_actions) < 2:
            return None
        
        # Simple finite difference
        return self.executed_actions[-1] - self.executed_actions[-2]
    
    def reset(self):
        """Reset the scheduler for a new episode."""
        self.chunks = []
        self.current_step = 0
        self._interpolator = None
        self._interp_valid_range = (0, 0)
        self.executed_actions = []


class InterpActionSchedulerV2:
    """
    Version 2: Direct PCHIP interpolation at chunk boundaries.
    
    Instead of pre-computing a weighted average, this version directly
    uses PCHIP to interpolate through the boundary region between chunks.
    
    For two consecutive chunks A and B with overlap:
    - Uses the tail of chunk A as the first waypoints
    - Uses the head of chunk B as the later waypoints
    - PCHIP interpolates smoothly through all points
    """
    
    def __init__(
        self,
        action_chunk_size: int = 50,
        execution_horizon: int = 50,
        transition_steps: int = 10,  # Steps over which to transition
        num_anchor_points: int = 5,  # Points from each chunk to use as anchors
        interp_mode: str = "pchip",
    ):
        self.action_chunk_size = action_chunk_size
        self.execution_horizon = execution_horizon
        self.transition_steps = transition_steps
        self.num_anchor_points = num_anchor_points
        self.interp_mode = interp_mode
        
        self.current_chunk = None
        self.next_chunk = None
        self.chunk_start_idx = 0
        self.current_step = 0
        self.action_dim = None
        
        # Transition interpolator (created when next_chunk is set)
        self._transition_interp = None
        self._transition_start = None
        self._transition_end = None
        
    def set_new_chunk(self, actions: np.ndarray, is_first: bool = False):
        """Set a new action chunk."""
        if self.action_dim is None:
            self.action_dim = actions.shape[1]
            
        if is_first or self.current_chunk is None:
            self.current_chunk = actions.copy()
            self.chunk_start_idx = self.current_step
            self.next_chunk = None
            self._transition_interp = None
        else:
            self.next_chunk = actions.copy()
            self._build_transition_interpolator()
    
    def _build_transition_interpolator(self):
        """
        Build PCHIP interpolator for the transition region.
        
        Uses anchor points from both chunks to create a smooth transition.
        """
        if self.current_chunk is None or self.next_chunk is None:
            return
        
        local_idx = self.current_step - self.chunk_start_idx
        
        # Transition starts when we enter the overlap region
        transition_start = self.chunk_start_idx + self.execution_horizon - self.transition_steps
        transition_end = transition_start + self.transition_steps + self.num_anchor_points
        
        # Get anchor points from current chunk (tail portion)
        current_anchor_start = self.execution_horizon - self.transition_steps - self.num_anchor_points
        current_anchor_end = self.execution_horizon
        current_anchor_start = max(0, current_anchor_start)
        
        # Get anchor points from next chunk (head portion)
        next_anchor_start = 0
        next_anchor_end = min(self.num_anchor_points + self.transition_steps, len(self.next_chunk))
        
        # Build time points and values
        time_points = []
        action_points = []
        
        # Add current chunk anchors
        for i in range(current_anchor_start, min(current_anchor_end, len(self.current_chunk))):
            t = self.chunk_start_idx + i
            time_points.append(t)
            action_points.append(self.current_chunk[i])
        
        # Add next chunk anchors (time-shifted)
        next_chunk_start_time = self.chunk_start_idx + self.execution_horizon
        for i in range(next_anchor_start, next_anchor_end):
            t = next_chunk_start_time + i
            if t > time_points[-1]:  # Avoid duplicate time points
                time_points.append(t)
                action_points.append(self.next_chunk[i])
        
        if len(time_points) < 2:
            return
        
        time_points = np.array(time_points)
        action_points = np.array(action_points)
        
        # Create PCHIP interpolator
        if self.interp_mode == "pchip":
            self._transition_interp = PchipInterpolator(time_points, action_points, axis=0)
        elif self.interp_mode == "cubic":
            self._transition_interp = CubicSpline(time_points, action_points, axis=0, bc_type='natural')
        else:
            self._transition_interp = interp1d(time_points, action_points, axis=0, 
                                               kind='linear', fill_value='extrapolate')
        
        self._transition_start = time_points[0]
        self._transition_end = time_points[-1]
        
        logger.debug(f"Built transition interpolator: t=[{self._transition_start}, {self._transition_end}]")
    
    def get_action(self) -> tuple[np.ndarray, bool]:
        """Get the next action with PCHIP interpolation at boundaries."""
        if self.current_chunk is None:
            raise RuntimeError("No action chunk set. Call set_new_chunk first.")
        
        local_idx = self.current_step - self.chunk_start_idx
        
        # Check if we need to transition to next chunk
        if local_idx >= self.execution_horizon and self.next_chunk is not None:
            self.current_chunk = self.next_chunk
            self.chunk_start_idx = self.current_step
            self.next_chunk = None
            self._transition_interp = None
            local_idx = 0
        
        # Determine if we're in transition region
        in_transition = (
            self._transition_interp is not None and 
            self._transition_start <= self.current_step <= self._transition_end
        )
        
        if in_transition:
            # Use PCHIP interpolation
            action = self._transition_interp(float(self.current_step))
        else:
            # Use raw chunk action
            if local_idx < len(self.current_chunk):
                action = self.current_chunk[local_idx].copy()
            else:
                # Extrapolate if needed (shouldn't happen normally)
                action = self.current_chunk[-1].copy()
        
        self.current_step += 1
        
        # Determine if we need a new chunk
        steps_until_switch = self.execution_horizon - local_idx
        needs_new_chunk = (steps_until_switch <= self.transition_steps) and (self.next_chunk is None)
        
        return action, needs_new_chunk
    
    def reset(self):
        """Reset for a new episode."""
        self.current_chunk = None
        self.next_chunk = None
        self.chunk_start_idx = 0
        self.current_step = 0
        self._transition_interp = None
        self._transition_start = None
        self._transition_end = None


class TrossenOpenPIBridgeInterp:
    """Bridge with PCHIP interpolation for smooth action transitions."""

    def __init__(
        self,
        policy_server_host: str = "localhost",
        policy_server_port: int = 8000,
        control_frequency: int = 30,
        test_mode: str = "autonomous",
        max_steps: int = 1000,
        interp_mode: str = "pchip",
        transition_steps: int = 10,
        num_anchor_points: int = 5,
        scheduler_version: int = 2,  # 1 or 2
    ):
        self.control_frequency = control_frequency
        self.max_steps = max_steps
        self.dt = 1.0 / control_frequency
        self.test_mode = test_mode
        self.interp_mode = interp_mode

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

        self.action_chunk_size = 50
        self.episode_step = 0
        self.is_running = False
        self.rate_of_inference = 50

        self.action_dim = len(self.robot._joint_ft)
        
        # Create interpolation scheduler
        if scheduler_version == 1:
            self.interp_scheduler = InterpActionScheduler(
                action_chunk_size=self.action_chunk_size,
                execution_horizon=self.rate_of_inference,
                overlap_horizon=transition_steps + num_anchor_points,
                interp_mode=interp_mode,
            )
        else:
            self.interp_scheduler = InterpActionSchedulerV2(
                action_chunk_size=self.action_chunk_size,
                execution_horizon=self.rate_of_inference,
                transition_steps=transition_steps,
                num_anchor_points=num_anchor_points,
                interp_mode=interp_mode,
            )
        
        logger.info(f"PCHIP interpolation enabled: mode={interp_mode}, "
                   f"transition_steps={transition_steps}, anchor_points={num_anchor_points}")

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
        """Smoothly move to start position using PCHIP interpolation."""
        joint_pos_keys = [k for k in self.robot.get_observation().keys() if k.endswith('.pos')]
        current_pose = np.array([self.robot.get_observation()[k] for k in joint_pos_keys])
        
        waypoints = np.array([current_pose, goal_position])
        timepoints = np.array([0, duration])
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

        observation = {
            "state": joint_positions,
            "images": {cam: observation_dict[cam] for cam in cameras},
            "prompt": task_prompt
        }
        return observation

    def run_episode(self, task_prompt: str = "look down", center_crop=False):
        """Run episode with PCHIP interpolation for action chunks."""
        logger.info(f"Starting episode with prompt: '{task_prompt}'")
        logger.info(f"Using {self.interp_mode.upper()} interpolation for chunk boundaries")
        
        self.episode_step = 0
        self.is_running = True
        is_first_step = True
        
        self.interp_scheduler.reset()
        
        # Get initial chunk
        observation = self._get_observation_for_policy(task_prompt, center_crop=center_crop)
        logger.info(f"Step {self.episode_step}: Requesting initial action chunk")
        response = self.policy_client.infer(observation)
        
        # Auto-detect action chunk size
        actual_chunk_size = response["actions"].shape[0]
        if actual_chunk_size != self.action_chunk_size:
            logger.info(f"Auto-detected action chunk size: {actual_chunk_size}")
            self.action_chunk_size = actual_chunk_size
            if hasattr(self.interp_scheduler, 'action_chunk_size'):
                self.interp_scheduler.action_chunk_size = actual_chunk_size
            if hasattr(self.interp_scheduler, 'execution_horizon'):
                if self.interp_scheduler.execution_horizon > actual_chunk_size:
                    self.interp_scheduler.execution_horizon = actual_chunk_size
        
        self.interp_scheduler.set_new_chunk(response["actions"], is_first=True)
        logger.info(f"Received initial action chunk: {response['actions'].shape}")
        
        inference_times = []
        
        while self.is_running and self.episode_step < self.max_steps:
            start_loop_time = time.perf_counter()
            
            # Get interpolated action
            a_t, needs_new_chunk = self.interp_scheduler.get_action()
            
            # Pre-fetch next chunk if needed
            if needs_new_chunk:
                observation = self._get_observation_for_policy(task_prompt, center_crop=center_crop)
                logger.info(f"Step {self.episode_step}: Pre-fetching next action chunk")
                
                inference_start_time = time.perf_counter()
                response = self.policy_client.infer(observation)
                inference_time = time.perf_counter() - inference_start_time
                inference_times.append(inference_time)
                
                self.interp_scheduler.set_new_chunk(response["actions"])
                logger.info(f"Received action chunk (inference: {inference_time*1000:.1f}ms)")
            
            # Execute action
            if is_first_step:
                logger.info("Moving to start position to avoid large jumps...")
                self.move_to_start_position(a_t, duration=5.0)
                is_first_step = False
            else:
                self.execute_action(a_t)
            
            self.episode_step += 1

            dt_s = time.perf_counter() - start_loop_time
            busy_wait_time = self.dt - dt_s

            if busy_wait_time > 0:
                time.sleep(busy_wait_time)
            loop_s = time.perf_counter() - start_loop_time
            logger.info(f"time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        self.is_running = False
        
        if inference_times:
            avg_inference = np.mean(inference_times) * 1000
            logger.info(f"Episode completed. Avg inference time: {avg_inference:.1f}ms")
        logger.info(f"Episode completed after {self.episode_step} steps")

    def autonomous_mode(self, task_prompt: str = "look down"):
        """Run in autonomous mode with PCHIP interpolation."""
        logger.info("Starting autonomous mode with PCHIP interpolation")
        self.run_episode(task_prompt=task_prompt, center_crop=True)

    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up...")
        self.robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trossen AI <-> OpenPI Bridge with PCHIP Interpolation"
    )
    parser.add_argument("--policy_host", default="localhost", help="Policy server host")
    parser.add_argument("--policy_port", type=int, default=8000, help="Policy server port")
    parser.add_argument("--control_freq", type=int, default=30, help="Control frequency in Hz")
    parser.add_argument("--mode", choices=["autonomous", "test"], default="autonomous",
                        help="Operation mode: autonomous (execute) or test (no movement)")
    parser.add_argument("--task_prompt", default="move the arm to the left", 
                        help="Task description for the policy")
    parser.add_argument("--max_steps", type=int, default=3600, help="Maximum steps per episode")
    
    # Interpolation arguments
    parser.add_argument("--interp_mode", choices=["pchip", "cubic", "linear"], default="pchip",
                        help="Interpolation mode: pchip (shape-preserving), cubic (C2), linear")
    parser.add_argument("--transition_steps", type=int, default=10,
                        help="Number of steps for the transition region (default: 10)")
    parser.add_argument("--num_anchor_points", type=int, default=5,
                        help="Number of anchor points from each chunk for interpolation (default: 5)")
    parser.add_argument("--scheduler_version", type=int, choices=[1, 2], default=2,
                        help="Scheduler version: 1 (weighted average) or 2 (direct PCHIP at boundaries)")
    
    args = parser.parse_args()

    bridge = TrossenOpenPIBridgeInterp(
        policy_server_host=args.policy_host,
        policy_server_port=args.policy_port,
        control_frequency=args.control_freq,
        test_mode=args.mode,
        max_steps=args.max_steps,
        interp_mode=args.interp_mode,
        transition_steps=args.transition_steps,
        num_anchor_points=args.num_anchor_points,
        scheduler_version=args.scheduler_version,
    )

    try:
        bridge.autonomous_mode(task_prompt=args.task_prompt)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected. Cleaning up...")
    except Exception as e:
        logger.error(f"Error in autonomous mode: {e}")
    finally:
        bridge.cleanup()
