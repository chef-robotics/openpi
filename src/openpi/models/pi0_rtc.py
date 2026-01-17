"""
Pi0 model extension with Real-Time Chunking (RTC) support.

This module extends the base Pi0 model to support inference-time RTC through
action prefix conditioning (inpainting). This allows for smoother transitions
between action chunks by conditioning new predictions on previously committed actions.

Reference: https://github.com/Physical-Intelligence/real-time-chunking-kinetix
Paper: "Real-Time Execution of Action Chunking Flow Policies"
"""

import dataclasses
import logging
from typing import Optional

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


@dataclasses.dataclass(frozen=True)
class Pi0RTCConfig(_pi0.Pi0Config):
    """Config for Pi0 with RTC support."""
    
    # RTC-specific parameters
    default_prefix_len: int = 0  # Default number of prefix actions to condition on
    inpainting_strength: float = 1.0  # Strength of prefix conditioning (0-1)
    
    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0RTC":
        return Pi0RTC(self, rngs=nnx.Rngs(rng))


class Pi0RTC(_pi0.Pi0):
    """
    Pi0 model with Real-Time Chunking (RTC) support.
    
    Extends the base Pi0 model to support action prefix conditioning during
    inference, enabling smoother transitions between action chunks.
    """
    
    def __init__(self, config: Pi0RTCConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self._rtc_config = config
    
    @at.typecheck
    def sample_actions_with_prefix(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        action_prefix: Optional[at.Float[at.Array, "b prefix_len action_dim"]] = None,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        inpainting_strength: float = 1.0,
    ) -> _model.Actions:
        """
        Sample actions with optional prefix conditioning (inpainting).
        
        This implements inference-time RTC by fixing the prefix of the action
        sequence to previously committed actions and generating only the suffix.
        
        Args:
            rng: Random key for sampling
            observation: Current observation
            action_prefix: Optional prefix actions to condition on, shape (batch, prefix_len, action_dim)
                          These are the committed actions from the previous chunk.
            num_steps: Number of flow integration steps
            inpainting_strength: How strongly to enforce the prefix (0 = ignore prefix, 1 = hard fix)
        
        Returns:
            Action sequence of shape (batch, action_horizon, action_dim)
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        
        # If we have a prefix, determine the prefix length
        prefix_len = 0
        if action_prefix is not None:
            prefix_len = action_prefix.shape[1]
            # Ensure prefix doesn't exceed action horizon
            prefix_len = min(prefix_len, self.action_horizon)
        
        # First fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        
        def step(carry):
            x_t, time = carry
            
            # Apply inpainting: replace prefix portion with interpolated committed actions
            if action_prefix is not None and prefix_len > 0:
                # Linear interpolation for flow matching: x_t = t * noise + (1-t) * x_0
                # At time t, the prefix should be: t * noise[:, :prefix_len] + (1-t) * action_prefix
                time_expanded = time[..., None, None]
                
                # Get the noise for the prefix region
                prefix_noise = noise[:, :prefix_len, :]
                
                # Compute what the prefix should be at this timestep
                target_prefix = time_expanded * prefix_noise + (1 - time_expanded) * action_prefix[:, :prefix_len, :]
                
                # Soft inpainting: blend between model prediction and target
                # With strength=1, we fully replace; with strength=0, we don't modify
                mask = jnp.ones((batch_size, self.action_horizon, 1))
                mask = mask.at[:, :prefix_len, :].set(1 - inpainting_strength)
                
                # Apply soft mask to keep prefix close to target
                x_t_inpainted = mask * x_t + (1 - mask) * target_prefix
                
                # For the suffix portion, keep x_t unchanged
                x_t = jnp.concatenate([x_t_inpainted[:, :prefix_len, :], x_t[:, prefix_len:, :]], axis=1)
            
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            
            suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_for_suffix = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask_for_suffix, suffix_attn_mask], axis=-1)
            
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache
            )
            
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
            
            x_next = x_t + dt * v_t
            
            # Post-step inpainting to ensure prefix stays on track
            if action_prefix is not None and prefix_len > 0:
                next_time = time + dt
                next_time_expanded = next_time[..., None, None]
                
                # Compute target for next timestep
                target_prefix_next = next_time_expanded * prefix_noise + (1 - next_time_expanded) * action_prefix[:, :prefix_len, :]
                
                # Replace prefix with target (hard inpainting after velocity update)
                x_next = jnp.concatenate([
                    inpainting_strength * target_prefix_next + (1 - inpainting_strength) * x_next[:, :prefix_len, :],
                    x_next[:, prefix_len:, :]
                ], axis=1)
            
            return x_next, time + dt
        
        def cond(carry):
            x_t, time = carry
            return time >= -dt / 2
        
        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        
        # Final inpainting: ensure prefix matches exactly
        if action_prefix is not None and prefix_len > 0:
            x_0 = jnp.concatenate([action_prefix[:, :prefix_len, :], x_0[:, prefix_len:, :]], axis=1)
        
        return x_0
    
    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        action_prefix: Optional[at.Float[at.Array, "b prefix_len action_dim"]] = None,
        inpainting_strength: float = 1.0,
    ) -> _model.Actions:
        """
        Sample actions with optional RTC prefix conditioning.
        
        This overrides the base sample_actions to support action prefix.
        If no prefix is provided, falls back to standard sampling.
        """
        if action_prefix is not None:
            return self.sample_actions_with_prefix(
                rng, observation, action_prefix,
                num_steps=num_steps,
                inpainting_strength=inpainting_strength,
            )
        else:
            # Fall back to parent implementation
            return super().sample_actions(rng, observation, num_steps=num_steps)
