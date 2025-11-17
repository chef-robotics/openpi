# Fixing LeRobot Issues in OpenPI

This document explains two approaches to fix issues in the LeRobot library that OpenPI depends on.

## The Problem

The LeRobot dataset loading code has a potential issue with video frame loading. In `lerobot/common/datasets/video_utils.py`, line 205:

```python
frame_indices = [round(ts * average_fps) for ts in timestamps]
```

This can cause "frame index out of bounds" errors when:
- Timestamps round to frame indices >= video frame count
- Video duration/frame count mismatches occur
- Edge cases in episode boundaries

## Solution: Custom Dataset Subclass (✅ Implemented)

**This is the recommended approach** and is already implemented in OpenPI.

### How it Works

1. **Create a custom subclass**: `RobustLeRobotDataset` in `openpi/training/robust_lerobot_dataset.py`
2. **Override only the problematic method**: `_query_videos()`
3. **Use the custom class**: Replace `LeRobotDataset` with `RobustLeRobotDataset` in `data_loader.py`

### Benefits

✅ **No fork needed**: We don't need to maintain a fork of the entire lerobot library  
✅ **Minimal changes**: Only override the specific method that needs fixing  
✅ **Easy to maintain**: Updates to lerobot are automatically inherited  
✅ **Easy to remove**: If lerobot fixes the issue upstream, just switch back to `LeRobotDataset`  
✅ **Clear code**: The fix is isolated and well-documented  

### Implementation

The fix is already integrated:
- `robust_lerobot_dataset.py` contains `RobustLeRobotDataset` class
- `data_loader.py` imports and uses `RobustLeRobotDataset` instead of `LeRobotDataset`
- The robust version includes:
  - Better error handling
  - Frame index clamping to valid ranges
  - Detailed logging of problematic episodes
  - Graceful recovery from frame index errors

### What Gets Fixed

The `decode_video_frames_safe()` function:
1. **Tries standard decoding first**: Most episodes work fine
2. **Catches frame index errors**: Detects out-of-bounds frame access
3. **Calculates valid frame range**: Gets actual video frame count
4. **Clamps indices**: `frame_idx = max(0, min(frame_idx, num_frames - 1))`
5. **Logs warnings**: Helps identify problematic episodes
6. **Returns valid frames**: Continues training without crashing

## Alternative: Fork LeRobot (❌ Not Recommended)

This approach would involve:

1. **Fork lerobot** into `openpi/third_party/lerobot/`
2. **Modify the source**: Edit `video_utils.py` directly
3. **Remove dependency**: Update `pyproject.toml` to not use `pip install lerobot`
4. **Maintain fork**: Keep up with upstream changes manually

### Why We Didn't Choose This

❌ **High maintenance burden**: Need to manually sync with upstream  
❌ **Larger surface area**: Harder to track what we changed  
❌ **Deployment complexity**: Have to package the entire forked library  
❌ **Testing overhead**: Need to test all of lerobot, not just our change  

### When to Use This Approach

Consider forking if:
- You need to make extensive changes across many files
- The upstream project is no longer maintained
- You need to diverge significantly from upstream
- The changes are fundamental architectural changes

## Testing

To verify the fix works:

```bash
# Run training on the problematic dataset
python scripts/train.py --config your_config

# Or test specific episodes
python scripts/debug_split_dataset.ipynb
```

The notebook will show:
- Which episodes have length 158 (or other problematic lengths)
- Whether they load successfully
- Any warnings about frame index clamping

## Reverting to Standard LeRobot

If you want to use the standard LeRobot dataset (e.g., if they fix the issue upstream):

```python
# In data_loader.py, change:
dataset = RobustLeRobotDataset(...)

# Back to:
dataset = lerobot_dataset.LeRobotDataset(...)
```

And remove the import of `RobustLeRobotDataset`.

## Summary

We chose the **custom subclass approach** because it:
- Solves the problem with minimal code changes
- Doesn't require maintaining a fork
- Is easy to update or remove
- Keeps the fix isolated and well-documented
- Allows us to inherit upstream improvements automatically

The fix is production-ready and already integrated into OpenPI's training pipeline.

