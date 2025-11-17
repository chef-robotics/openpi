#!/usr/bin/env python3
"""
Test script to verify gripper transformations work correctly.

This script demonstrates and tests the forward and reverse gripper transformations
used in the normalization pipeline.
"""

import numpy as np


def forward_transform(physical_value, min_val=0.0, max_val=0.045):
    """Transform from physical space [0, 0.045] to policy space [1, 0]."""
    return 1.0 - (physical_value - min_val) / (max_val - min_val)


def reverse_transform(policy_value, min_val=0.0, max_val=0.045):
    """Transform from policy space [1, 0] to physical space [0, 0.045]."""
    return min_val + (1.0 - policy_value) * (max_val - min_val)


def test_transformations():
    """Test that forward and reverse transformations are correct."""
    print("=" * 60)
    print("Testing Gripper Transformations")
    print("=" * 60)
    
    # Test edge cases
    test_cases = [
        ("Closed gripper", 0.0, 1.0),
        ("Open gripper", 0.045, 0.0),
        ("Half open", 0.0225, 0.5),
        ("Quarter open", 0.01125, 0.75),
    ]
    
    print("\nForward Transform (Physical → Policy):")
    print("-" * 60)
    all_correct = True
    for name, physical, expected_policy in test_cases:
        policy = forward_transform(physical)
        is_correct = np.isclose(policy, expected_policy, atol=1e-6)
        status = "✓" if is_correct else "✗"
        print(f"{status} {name:20s}: {physical:7.5f} → {policy:7.5f} (expected {expected_policy:7.5f})")
        all_correct = all_correct and is_correct
    
    print("\nReverse Transform (Policy → Physical):")
    print("-" * 60)
    for name, expected_physical, policy in test_cases:
        physical = reverse_transform(policy)
        is_correct = np.isclose(physical, expected_physical, atol=1e-6)
        status = "✓" if is_correct else "✗"
        print(f"{status} {name:20s}: {policy:7.5f} → {physical:7.5f} (expected {expected_physical:7.5f})")
        all_correct = all_correct and is_correct
    
    print("\nRound-trip Test (Physical → Policy → Physical):")
    print("-" * 60)
    physical_values = np.linspace(0.0, 0.045, 10)
    for physical_in in physical_values:
        policy = forward_transform(physical_in)
        physical_out = reverse_transform(policy)
        is_correct = np.isclose(physical_in, physical_out, atol=1e-6)
        status = "✓" if is_correct else "✗"
        print(f"{status} {physical_in:7.5f} → {policy:7.5f} → {physical_out:7.5f}")
        all_correct = all_correct and is_correct
    
    print("\nRound-trip Test (Policy → Physical → Policy):")
    print("-" * 60)
    policy_values = np.linspace(0.0, 1.0, 10)
    for policy_in in policy_values:
        physical = reverse_transform(policy_in)
        policy_out = forward_transform(physical)
        is_correct = np.isclose(policy_in, policy_out, atol=1e-6)
        status = "✓" if is_correct else "✗"
        print(f"{status} {policy_in:7.5f} → {physical:7.5f} → {policy_out:7.5f}")
        all_correct = all_correct and is_correct
    
    print("\n" + "=" * 60)
    if all_correct:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
    print("=" * 60)
    
    return all_correct


def test_array_transforms():
    """Test transformations on arrays (like full action vectors)."""
    print("\n" + "=" * 60)
    print("Testing Array Transformations (Bimanual Robot)")
    print("=" * 60)
    
    # Simulate a bimanual robot with 14 joints
    # Indices 6 and 13 are gripper joints
    gripper_indices = [6, 13]
    min_val, max_val = 0.0, 0.045
    
    # Create a sample action in physical space
    physical_action = np.array([
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6,  # left arm joints 0-5
        0.020,                          # left gripper (closed halfway) - index 6
        0.7, 0.8, 0.9, 1.0, 1.1, 1.2,  # right arm joints 7-12
        0.040,                          # right gripper (mostly open) - index 13
    ])
    
    print("\nOriginal Physical Action:")
    print(f"  Full action: {physical_action}")
    print(f"  Left gripper  (idx {gripper_indices[0]}): {physical_action[gripper_indices[0]]:.5f} m")
    print(f"  Right gripper (idx {gripper_indices[1]}): {physical_action[gripper_indices[1]]:.5f} m")
    
    # Forward transform (as done in compute_norm_stats.py)
    policy_action = physical_action.copy()
    for idx in gripper_indices:
        policy_action[idx] = forward_transform(policy_action[idx], min_val, max_val)
    
    print("\nTransformed to Policy Space:")
    print(f"  Full action: {policy_action}")
    print(f"  Left gripper  (idx {gripper_indices[0]}): {policy_action[gripper_indices[0]]:.5f}")
    print(f"  Right gripper (idx {gripper_indices[1]}): {policy_action[gripper_indices[1]]:.5f}")
    
    # Reverse transform (as done in main.py)
    reconstructed_action = policy_action.copy()
    for idx in gripper_indices:
        reconstructed_action[idx] = reverse_transform(reconstructed_action[idx], min_val, max_val)
    
    print("\nReconstructed Physical Action:")
    print(f"  Full action: {reconstructed_action}")
    print(f"  Left gripper  (idx {gripper_indices[0]}): {reconstructed_action[gripper_indices[0]]:.5f} m")
    print(f"  Right gripper (idx {gripper_indices[1]}): {reconstructed_action[gripper_indices[1]]:.5f} m")
    
    # Verify
    is_correct = np.allclose(physical_action, reconstructed_action, atol=1e-6)
    print("\n" + "=" * 60)
    if is_correct:
        print("✓ Array transformation test passed!")
    else:
        print("✗ Array transformation test failed!")
        print(f"  Max error: {np.max(np.abs(physical_action - reconstructed_action))}")
    print("=" * 60)
    
    return is_correct


if __name__ == "__main__":
    test1_passed = test_transformations()
    test2_passed = test_array_transforms()
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Basic transformations: {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"Array transformations: {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    print("=" * 60)
    
    exit(0 if (test1_passed and test2_passed) else 1)

