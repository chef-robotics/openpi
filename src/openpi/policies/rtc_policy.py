"""
RTC-enabled Policy wrapper for Real-Time Chunking support.

This module provides a policy wrapper that supports action prefix conditioning
for inference-time RTC. It can work with both standard Pi0 models (with client-side
blending) and Pi0RTC models (with server-side inpainting).

Reference: https://github.com/Physical-Intelligence/real-time-chunking-kinetix
"""

import logging
import time
from collections.abc import Sequence
from typing import Any, Optional, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

logger = logging.getLogger(__name__)

BasePolicy: TypeAlias = _base_policy.BasePolicy


class RTCPolicy(BasePolicy):
    """
    Policy wrapper with Real-Time Chunking (RTC) support.
    
    This policy accepts an optional 'action_prefix' in the observation dict
    and uses it for prefix conditioning during action generation.
    
    For models that support native inpainting (Pi0RTC), the prefix is passed
    directly to the model. For standard models, the prefix is used for
    validation and the client should handle blending.
    """
    
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        enable_inpainting: bool = True,
        default_inpainting_strength: float = 1.0,
    ):
        """
        Initialize RTC-enabled policy.
        
        Args:
            model: The underlying model (Pi0 or Pi0RTC)
            rng: Random key
            transforms: Input transforms
            output_transforms: Output transforms
            sample_kwargs: Additional kwargs for sampling
            metadata: Policy metadata
            enable_inpainting: Whether to enable server-side inpainting
            default_inpainting_strength: Default strength for prefix conditioning
        """
        self._model = model
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._enable_inpainting = enable_inpainting
        self._default_inpainting_strength = default_inpainting_strength
        
        # Check if model supports native inpainting
        self._supports_native_inpainting = hasattr(model, 'sample_actions_with_prefix')
        
        # Add RTC info to metadata
        self._metadata['rtc_enabled'] = True
        self._metadata['rtc_native_inpainting'] = self._supports_native_inpainting

    @override
    def infer(self, obs: dict) -> dict:
        """
        Infer actions with optional RTC prefix conditioning.
        
        The observation dict can include:
        - 'action_prefix': np.ndarray of shape (prefix_len, action_dim) - committed actions
        - 'inpainting_strength': float (0-1) - how strongly to enforce prefix
        
        Returns:
            Dict with 'actions' and timing info
        """
        # Extract RTC-specific fields before transformation
        action_prefix = obs.pop('action_prefix', None)
        inpainting_strength = obs.pop('inpainting_strength', self._default_inpainting_strength)
        
        # Make a copy since transformations may modify the inputs in place
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        
        # Make a batch and convert to jax.Array
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        
        # Prepare action prefix if provided
        prefix_array = None
        if action_prefix is not None and self._enable_inpainting:
            prefix_array = jnp.asarray(action_prefix)[np.newaxis, ...]  # Add batch dim
        
        start_time = time.monotonic()
        self._rng, sample_rng = jax.random.split(self._rng)
        
        # Build sample kwargs
        sample_kwargs = dict(self._sample_kwargs)
        if prefix_array is not None and self._supports_native_inpainting:
            sample_kwargs['action_prefix'] = prefix_array
            sample_kwargs['inpainting_strength'] = inpainting_strength
        
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(
                sample_rng, 
                _model.Observation.from_dict(inputs),
                **sample_kwargs
            ),
        }
        
        # Unbatch and convert to np.ndarray
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        model_time = time.monotonic() - start_time
        
        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        
        # Add RTC metadata
        outputs["rtc_info"] = {
            "prefix_len": action_prefix.shape[0] if action_prefix is not None else 0,
            "inpainting_strength": inpainting_strength if action_prefix is not None else 0.0,
            "native_inpainting": self._supports_native_inpainting and prefix_array is not None,
        }
        
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


def create_rtc_policy(
    model: _model.BaseModel,
    transforms: Sequence[_transforms.DataTransformFn] = (),
    output_transforms: Sequence[_transforms.DataTransformFn] = (),
    **kwargs
) -> RTCPolicy:
    """
    Factory function to create an RTC-enabled policy.
    
    Args:
        model: Pi0 or Pi0RTC model
        transforms: Input transforms
        output_transforms: Output transforms
        **kwargs: Additional kwargs passed to RTCPolicy
        
    Returns:
        RTCPolicy instance
    """
    return RTCPolicy(
        model,
        transforms=transforms,
        output_transforms=output_transforms,
        **kwargs
    )
