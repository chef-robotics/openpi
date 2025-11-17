"""Custom LeRobotDataset subclass with robust video frame loading.

This module provides a more robust version of LeRobotDataset that handles
edge cases in video frame loading, particularly for episodes where timestamp
rounding could cause frame index out-of-bounds errors.
"""

import logging
from pathlib import Path

import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.video_utils import decode_video_frames


logger = logging.getLogger(__name__)


def decode_video_frames_safe(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str | None = None,
) -> torch.Tensor:
    """Safely decode video frames with better error handling and edge case management.
    
    This function wraps the standard decode_video_frames but adds:
    1. Better error messages
    2. Clamping of frame indices to valid range
    3. Logging of problematic episodes
    
    Args:
        video_path: Path to the video file
        timestamps: List of timestamps to extract frames
        tolerance_s: Allowed deviation in seconds for frame retrieval
        backend: Backend to use for decoding
    
    Returns:
        Decoded frames as a torch.Tensor
    """
    try:
        # Try the standard decoding first
        return decode_video_frames(video_path, timestamps, tolerance_s, backend)
    except (RuntimeError, ValueError) as e:
        error_msg = str(e)
        
        # Check if this is a frame index error
        if "frame index" in error_msg.lower() or "out of bounds" in error_msg.lower():
            logger.warning(
                f"Frame index out of bounds for video {video_path}. "
                f"Timestamps: {timestamps}. "
                f"This might be due to timestamp/video duration mismatch. "
                f"Attempting to load with clamped frame indices."
            )
            
            # Get video metadata to find valid frame range
            try:
                import torchcodec
                from torchcodec.decoders import VideoDecoder
                
                decoder = VideoDecoder(str(video_path))
                metadata = decoder.metadata
                num_frames = metadata.num_frames
                average_fps = metadata.average_fps
                
                # Calculate frame indices with clamping
                frame_indices = []
                for ts in timestamps:
                    frame_idx = round(ts * average_fps)
                    # Clamp to valid range
                    frame_idx = max(0, min(frame_idx, num_frames - 1))
                    frame_indices.append(frame_idx)
                
                logger.info(
                    f"Clamped frame indices: {frame_indices} "
                    f"(video has {num_frames} frames, fps={average_fps:.2f})"
                )
                
                # Retrieve frames using clamped indices
                frames_batch = decoder.get_frames_at(indices=frame_indices)
                frames = torch.stack([frame for frame in frames_batch.data])
                
                return frames
                
            except Exception as e2:
                logger.error(f"Failed to recover from frame index error: {e2}")
                raise RuntimeError(
                    f"Could not load frames from {video_path} at timestamps {timestamps}. "
                    f"Original error: {error_msg}. Recovery attempt failed: {str(e2)}"
                ) from e
        else:
            # Re-raise if it's not a frame index error
            raise


class RobustLeRobotDataset(LeRobotDataset):
    """LeRobotDataset with more robust video frame loading.
    
    This subclass overrides the _query_videos method to use a more robust
    video decoding function that handles edge cases better, particularly:
    - Episodes where timestamps round to frame indices beyond video length
    - Videos with slight duration/frame count mismatches
    
    Usage:
        Simply replace LeRobotDataset with RobustLeRobotDataset in your code:
        
        # Before:
        dataset = lerobot_dataset.LeRobotDataset(...)
        
        # After:
        dataset = RobustLeRobotDataset(...)
    """
    
    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        """Query video frames with robust error handling.
        
        Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault. This probably happens because a memory reference to the video loader is created in
        the main process and a subprocess fails to access it.
        
        This method is identical to the parent class except it uses decode_video_frames_safe
        instead of decode_video_frames.
        """
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            
            # Use the safe decoding function instead of the standard one
            frames = decode_video_frames_safe(
                video_path, 
                query_ts, 
                self.tolerance_s, 
                self.video_backend
            )
            item[vid_key] = frames.squeeze(0)

        return item

