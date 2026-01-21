#!/usr/bin/env python3
"""
Trossen Arm <-> OpenPI Policy Server Bridge (Receding Horizon Version)

Implements Receding Horizon Control with Temporal Ensembling for smooth action transitions.

Key concepts:
1. Receding Horizon: Execute k < H steps, then re-query (creates overlap)
2. Temporal Ensembling: Average predictions from multiple overlapping chunks
3. Exponential weighting: Recent predictions weighted higher than older ones

Reference papers:
- Diffusion Policy (Chi et al., 2023): https://arxiv.org/abs/2303.04137
- ACT (Zhao et al., 2023): https://arxiv.org/abs/2304.13705

Usage:
    # Basic receding horizon (execute 10 of 50 steps)
    python main_receding_horizon.py --mode autonomous --task_prompt "your task" --execution_horizon 10
    
    # With temporal ensembling
    python main_receding_horizon.py --mode autonomous --task_prompt "your task" --execution_horizon 10 --use_temporal_ensemble
    
    # Adjust ensemble decay (lower = more smoothing)
    python main_receding_horizon.py --mode autonomous --task_prompt "your task" --execution_horizon 10 --use_temporal_ensemble --ensemble_decay 0.01
    
    
    The best result was 30ctrl_50act_ema0.7_scoop with jerkiness 0.0187. To compare fairly:

  # Setup H: Match your EMA baseline frequency
  python main_receding_horizon.py \
      --control_freq 30 \
      --action_chunk_size 50 \
      --execution_horizon 50 \
      --use_temporal_ensemble \
      --ensemble_decay 0.01 \
      --use_ema_smoothing \
      --ema_alpha 0.7 \
      --task_prompt "your task"
      
    tests;  
    --execution_horizon 10 --use_temporal_ensemble --ensemble_decay 0.01
    --execution_horizon 15 --use_temporal_ensemble --ensemble_decay 0.02
    --execution_horizon 10 --use_temporal_ensemble --use_ema_smoothing --ema_alpha 0.7
  
"""

import argparse
import logging
import time
import numpy as np
from openpi_client import websocket_client_policy
import cv2
from collections import deque
from scipy.interpolate import PchipInterpolator

from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.robots import make_robot_from_config
from lerobot_robot_trossen.config_bi_widowxai_follower import BiWidowXAIFollowerRobotConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RecedingHorizonScheduler:
    """
    Receding Horizon Action Scheduler.
    
    Classic MPC-style approach:
    - Predict H steps into the future
    - Execute only k steps (k < H)
    - Re-query policy and repeat
    
    This creates natural overlap between consecutive predictions,
    which can be exploited for temporal ensembling.
    """
    
    def __init__(
        self,
        action_chunk_size: int = 50,      # H: prediction horizon
        execution_horizon: int = 10,       # k: steps to execute before re-query
    ):
        self.action_chunk_size = action_chunk_size
        self.execution_horizon = execution_horizon
        
        self.current_chunk = None
        self.chunk_start_idx = 0
        self.current_step = 0
        self.action_dim = None
        
    def set_new_chunk(self, actions: np.ndarray, is_first: bool = False):
        """Set a new action chunk."""
        if self.action_dim is None:
            self.action_dim = actions.shape[1]
            
        self.current_chunk = actions.copy()
        self.chunk_start_idx = self.current_step
        
        if is_first:
            logger.info(f"Initial chunk set at step {self.chunk_start_idx}, shape={actions.shape}")
        else:
            logger.debug(f"New chunk at step {self.chunk_start_idx}")
    
    def get_action(self) -> tuple[np.ndarray, bool]:
        """
        Get the next action using receding horizon.
        
        Returns:
            action: The action to execute
            needs_new_chunk: Whether a new chunk should be requested
        """
        if self.current_chunk is None:
            raise RuntimeError("No action chunk set. Call set_new_chunk first.")
        
        local_idx = self.current_step - self.chunk_start_idx
        
        # Get action from current chunk
        if local_idx < len(self.current_chunk):
            action = self.current_chunk[local_idx].copy()
        else:
            # Shouldn't happen with proper scheduling, but fallback
            action = self.current_chunk[-1].copy()
            logger.warning(f"Exceeded chunk bounds: local_idx={local_idx}, chunk_size={len(self.current_chunk)}")
        
        self.current_step += 1
        
        # Request new chunk when we've executed k steps
        needs_new_chunk = (local_idx + 1) >= self.execution_horizon
        
        return action, needs_new_chunk
    
    def reset(self):
        """Reset for a new episode."""
        self.current_chunk = None
        self.chunk_start_idx = 0
        self.current_step = 0


class TemporalEnsembleScheduler:
    """
    Temporal Ensemble Action Scheduler (from ACT/Diffusion Policy).
    
    Maintains multiple overlapping action chunks and computes weighted
    average of all predictions for each timestep.
    
    Key insight: Each timestep t may have predictions from multiple chunks:
    - Chunk queried at t-k predicts a[t] at index k
    - Chunk queried at t-2k predicts a[t] at index 2k
    - etc.
    
    We weight these predictions exponentially, favoring more recent chunks.
    
    Formula:
        a_t = Σ_i w_i * chunk_i[t - start_i] / Σ_i w_i
        where w_i = exp(-decay * (t - start_i))
    """
    
    def __init__(
        self,
        action_chunk_size: int = 50,      # H: prediction horizon
        execution_horizon: int = 10,       # k: steps to execute before re-query
        ensemble_decay: float = 0.01,      # Exponential decay rate (lower = more smoothing)
        max_chunks: int = 10,              # Maximum chunks to keep in ensemble
    ):
        self.action_chunk_size = action_chunk_size
        self.execution_horizon = execution_horizon
        self.ensemble_decay = ensemble_decay
        self.max_chunks = max_chunks
        
        # Store chunks as (start_step, actions) tuples
        self.chunks: deque = deque(maxlen=max_chunks)
        self.current_step = 0
        self.action_dim = None
        
    def set_new_chunk(self, actions: np.ndarray, is_first: bool = False):
        """Add a new action chunk to the ensemble."""
        if self.action_dim is None:
            self.action_dim = actions.shape[1]
            
        chunk_start = self.current_step
        self.chunks.append((chunk_start, actions.copy()))
        
        if is_first:
            logger.info(f"Initial chunk added at step {chunk_start}, ensemble size: 1")
        else:
            logger.debug(f"Chunk added at step {chunk_start}, ensemble size: {len(self.chunks)}")
    
    def _compute_ensemble_action(self) -> np.ndarray:
        """
        Compute the temporally ensembled action for current timestep.
        
        Averages all applicable predictions weighted by recency.
        """
        t = self.current_step
        
        weighted_sum = np.zeros(self.action_dim)
        weight_total = 0.0
        
        for chunk_start, chunk_actions in self.chunks:
            local_idx = t - chunk_start
            
            # Check if this chunk has a prediction for timestep t
            if 0 <= local_idx < len(chunk_actions):
                # Weight by recency: more recent predictions get higher weight
                # w = exp(-decay * local_idx)
                # local_idx = 0 means this was just predicted, highest weight
                weight = np.exp(-self.ensemble_decay * local_idx)
                
                weighted_sum += weight * chunk_actions[local_idx]
                weight_total += weight
        
        if weight_total > 0:
            return weighted_sum / weight_total
        else:
            # Fallback: return last known action
            if len(self.chunks) > 0:
                _, last_chunk = self.chunks[-1]
                return last_chunk[0].copy()
            else:
                raise RuntimeError("No chunks available for ensemble")
    
    def get_action(self) -> tuple[np.ndarray, bool]:
        """
        Get the temporally ensembled action.
        
        Returns:
            action: The ensembled action
            needs_new_chunk: Whether a new chunk should be requested
        """
        if len(self.chunks) == 0:
            raise RuntimeError("No action chunks set. Call set_new_chunk first.")
        
        # Compute ensembled action
        action = self._compute_ensemble_action()
        
        # Determine if we need a new chunk
        last_chunk_start, _ = self.chunks[-1]
        steps_since_last_query = self.current_step - last_chunk_start
        needs_new_chunk = steps_since_last_query >= self.execution_horizon
        
        self.current_step += 1
        
        # Prune old chunks that are no longer useful
        self._prune_old_chunks()
        
        return action, needs_new_chunk
    
    def _prune_old_chunks(self):
        """Remove chunks that no longer contribute to ensemble."""
        while len(self.chunks) > 0:
            chunk_start, chunk_actions = self.chunks[0]
            chunk_end = chunk_start + len(chunk_actions)
            
            # Remove if chunk is entirely in the past
            if chunk_end <= self.current_step:
                self.chunks.popleft()
            else:
                break
    
    def get_ensemble_stats(self) -> dict:
        """Get statistics about the current ensemble state."""
        t = self.current_step
        num_contributing = 0
        
        for chunk_start, chunk_actions in self.chunks:
            local_idx = t - chunk_start
            if 0 <= local_idx < len(chunk_actions):
                num_contributing += 1
        
        return {
            "total_chunks": len(self.chunks),
            "contributing_chunks": num_contributing,
            "current_step": t,
        }
    
    def reset(self):
        """Reset for a new episode."""
        self.chunks.clear()
        self.current_step = 0


class HybridEnsembleScheduler:
    """
    Hybrid scheduler combining temporal ensemble with EMA smoothing.
    
    This provides two levels of smoothing:
    1. Temporal ensemble: Averages multiple chunk predictions
    2. EMA: Smooths the output relative to last executed action
    
    This can provide better smoothing than either approach alone.
    """
    
    def __init__(
        self,
        action_chunk_size: int = 50,
        execution_horizon: int = 10,
        ensemble_decay: float = 0.01,
        ema_alpha: float = 0.8,  # Higher alpha = less additional smoothing
        max_chunks: int = 10,
    ):
        self.ensemble_scheduler = TemporalEnsembleScheduler(
            action_chunk_size=action_chunk_size,
            execution_horizon=execution_horizon,
            ensemble_decay=ensemble_decay,
            max_chunks=max_chunks,
        )
        self.ema_alpha = ema_alpha
        self.last_executed_action = None
        self.action_dim = None
        
    def set_new_chunk(self, actions: np.ndarray, is_first: bool = False):
        """Add a new action chunk."""
        if self.action_dim is None:
            self.action_dim = actions.shape[1]
        self.ensemble_scheduler.set_new_chunk(actions, is_first)
    
    def get_action(self) -> tuple[np.ndarray, bool]:
        """Get action with ensemble + EMA smoothing."""
        ensemble_action, needs_new_chunk = self.ensemble_scheduler.get_action()
        
        if self.last_executed_action is None:
            smoothed_action = ensemble_action
        else:
            smoothed_action = (
                self.ema_alpha * ensemble_action + 
                (1 - self.ema_alpha) * self.last_executed_action
            )
        
        self.last_executed_action = smoothed_action.copy()
        return smoothed_action, needs_new_chunk
    
    def reset(self):
        """Reset for a new episode."""
        self.ensemble_scheduler.reset()
        self.last_executed_action = None


class TrossenOpenPIBridgeRecedingHorizon:
    """Bridge with Receding Horizon Control and optional Temporal Ensembling."""

    def __init__(
        self,
        policy_server_host: str = "localhost",
        policy_server_port: int = 8000,
        control_frequency: int = 30,
        test_mode: str = "autonomous",
        max_steps: int = 1000,
        action_chunk_size: int = 50,
        execution_horizon: int = 10,
        use_temporal_ensemble: bool = True,
        ensemble_decay: float = 0.01,
        use_ema_smoothing: bool = False,
        ema_alpha: float = 0.8,
    ):
        self.control_frequency = control_frequency
        self.max_steps = max_steps
        self.dt = 1.0 / control_frequency
        self.test_mode = test_mode
        self.action_chunk_size = action_chunk_size
        self.execution_horizon = execution_horizon
        self.use_temporal_ensemble = use_temporal_ensemble

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

        self.episode_step = 0
        self.is_running = False

        self.action_dim = len(self.robot._joint_ft)
        
        # Create appropriate scheduler
        if use_temporal_ensemble:
            if use_ema_smoothing:
                self.scheduler = HybridEnsembleScheduler(
                    action_chunk_size=action_chunk_size,
                    execution_horizon=execution_horizon,
                    ensemble_decay=ensemble_decay,
                    ema_alpha=ema_alpha,
                )
                logger.info(f"Using Hybrid (Temporal Ensemble + EMA) scheduler")
                logger.info(f"  - Execution horizon: {execution_horizon}/{action_chunk_size}")
                logger.info(f"  - Ensemble decay: {ensemble_decay}")
                logger.info(f"  - EMA alpha: {ema_alpha}")
            else:
                self.scheduler = TemporalEnsembleScheduler(
                    action_chunk_size=action_chunk_size,
                    execution_horizon=execution_horizon,
                    ensemble_decay=ensemble_decay,
                )
                logger.info(f"Using Temporal Ensemble scheduler")
                logger.info(f"  - Execution horizon: {execution_horizon}/{action_chunk_size}")
                logger.info(f"  - Ensemble decay: {ensemble_decay}")
                logger.info(f"  - Max overlap: {action_chunk_size // execution_horizon} chunks")
        else:
            self.scheduler = RecedingHorizonScheduler(
                action_chunk_size=action_chunk_size,
                execution_horizon=execution_horizon,
            )
            logger.info(f"Using basic Receding Horizon scheduler")
            logger.info(f"  - Execution horizon: {execution_horizon}/{action_chunk_size}")

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
        """Run episode with receding horizon control."""
        logger.info(f"Starting episode with prompt: '{task_prompt}'")
        logger.info(f"Receding Horizon: execute {self.execution_horizon} of {self.action_chunk_size} steps")
        
        self.episode_step = 0
        self.is_running = True
        is_first_step = True
        
        self.scheduler.reset()
        
        # Get initial chunk
        observation = self._get_observation_for_policy(task_prompt, center_crop=center_crop)
        logger.info(f"Step {self.episode_step}: Requesting initial action chunk")
        response = self.policy_client.infer(observation)
        
        # Auto-detect action chunk size
        actual_chunk_size = response["actions"].shape[0]
        if actual_chunk_size != self.action_chunk_size:
            logger.info(f"Auto-detected action chunk size: {actual_chunk_size}")
            self.action_chunk_size = actual_chunk_size
            if hasattr(self.scheduler, 'action_chunk_size'):
                self.scheduler.action_chunk_size = actual_chunk_size
            if hasattr(self.scheduler, 'ensemble_scheduler'):
                self.scheduler.ensemble_scheduler.action_chunk_size = actual_chunk_size
        
        self.scheduler.set_new_chunk(response["actions"], is_first=True)
        logger.info(f"Received initial action chunk: {response['actions'].shape}")
        
        inference_times = []
        inference_count = 0
        
        while self.is_running and self.episode_step < self.max_steps:
            start_loop_time = time.perf_counter()
            
            # Get action (ensembled or raw depending on scheduler)
            a_t, needs_new_chunk = self.scheduler.get_action()
            
            # Request new chunk when needed
            if needs_new_chunk:
                observation = self._get_observation_for_policy(task_prompt, center_crop=center_crop)
                
                inference_start_time = time.perf_counter()
                response = self.policy_client.infer(observation)
                inference_time = time.perf_counter() - inference_start_time
                inference_times.append(inference_time)
                inference_count += 1
                
                self.scheduler.set_new_chunk(response["actions"])
                
                # Log ensemble stats if available
                if hasattr(self.scheduler, 'get_ensemble_stats'):
                    stats = self.scheduler.get_ensemble_stats()
                    logger.info(f"Step {self.episode_step}: New chunk #{inference_count} "
                               f"(inference: {inference_time*1000:.1f}ms, "
                               f"ensemble: {stats['contributing_chunks']} chunks)")
                elif hasattr(self.scheduler, 'ensemble_scheduler'):
                    stats = self.scheduler.ensemble_scheduler.get_ensemble_stats()
                    logger.info(f"Step {self.episode_step}: New chunk #{inference_count} "
                               f"(inference: {inference_time*1000:.1f}ms, "
                               f"ensemble: {stats['contributing_chunks']} chunks)")
                else:
                    logger.info(f"Step {self.episode_step}: New chunk #{inference_count} "
                               f"(inference: {inference_time*1000:.1f}ms)")
            
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
            
            # Log less frequently to reduce noise
            if self.episode_step % 30 == 0:
                logger.info(f"Step {self.episode_step}: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        self.is_running = False
        
        if inference_times:
            avg_inference = np.mean(inference_times) * 1000
            logger.info(f"\nEpisode Summary:")
            logger.info(f"  Total steps: {self.episode_step}")
            logger.info(f"  Total inferences: {inference_count}")
            logger.info(f"  Avg inference time: {avg_inference:.1f}ms")
            logger.info(f"  Inference frequency: every {self.execution_horizon} steps")

    def autonomous_mode(self, task_prompt: str = "look down"):
        """Run in autonomous mode with receding horizon."""
        logger.info("Starting autonomous mode with Receding Horizon Control")
        self.run_episode(task_prompt=task_prompt, center_crop=True)

    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up...")
        self.robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trossen AI <-> OpenPI Bridge with Receding Horizon Control"
    )
    parser.add_argument("--policy_host", default="localhost", help="Policy server host")
    parser.add_argument("--policy_port", type=int, default=8000, help="Policy server port")
    parser.add_argument("--control_freq", type=int, default=30, help="Control frequency in Hz")
    parser.add_argument("--mode", choices=["autonomous", "test"], default="autonomous",
                        help="Operation mode: autonomous (execute) or test (no movement)")
    parser.add_argument("--task_prompt", default="move the arm to the left", 
                        help="Task description for the policy")
    parser.add_argument("--max_steps", type=int, default=3600, help="Maximum steps per episode")
    
    # Receding horizon arguments
    parser.add_argument("--action_chunk_size", type=int, default=50,
                        help="Full action chunk size H (default: 50)")
    parser.add_argument("--execution_horizon", type=int, default=10,
                        help="Steps to execute before re-query k (default: 10). "
                             "Lower = more overlap = smoother but more inference")
    
    # Temporal ensemble arguments
    parser.add_argument("--use_temporal_ensemble", action="store_true",
                        help="Enable temporal ensembling (recommended)")
    parser.add_argument("--ensemble_decay", type=float, default=0.01,
                        help="Exponential decay for ensemble weights (default: 0.01). "
                             "Lower = more smoothing, higher = favor recent predictions")
    
    # Additional smoothing
    parser.add_argument("--use_ema_smoothing", action="store_true",
                        help="Add EMA smoothing on top of ensemble")
    parser.add_argument("--ema_alpha", type=float, default=0.8,
                        help="EMA alpha when using hybrid mode (default: 0.8)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.execution_horizon > args.action_chunk_size:
        logger.warning(f"execution_horizon ({args.execution_horizon}) > action_chunk_size ({args.action_chunk_size})")
        logger.warning("Setting execution_horizon = action_chunk_size")
        args.execution_horizon = args.action_chunk_size

    bridge = TrossenOpenPIBridgeRecedingHorizon(
        policy_server_host=args.policy_host,
        policy_server_port=args.policy_port,
        control_frequency=args.control_freq,
        test_mode=args.mode,
        max_steps=args.max_steps,
        action_chunk_size=args.action_chunk_size,
        execution_horizon=args.execution_horizon,
        use_temporal_ensemble=args.use_temporal_ensemble,
        ensemble_decay=args.ensemble_decay,
        use_ema_smoothing=args.use_ema_smoothing,
        ema_alpha=args.ema_alpha,
    )

    try:
        bridge.autonomous_mode(task_prompt=args.task_prompt)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected. Cleaning up...")
    except Exception as e:
        logger.error(f"Error in autonomous mode: {e}")
        import traceback
        traceback.print_exc()
    finally:
        bridge.cleanup()
