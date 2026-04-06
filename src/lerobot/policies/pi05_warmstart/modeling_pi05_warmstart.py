#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PI05 policy with warm-start flow matching.

Only the flow-matching training pass and inference initialisation are
changed relative to the original PI05.  All VLM / action-expert weights,
attention masking, and image preprocessing are inherited unchanged.

Key changes (marked with ── WARM-START ──):
  • PI05WarmstartPytorch.forward(): uses B_t (shifted prev chunk) as the
    source distribution instead of Gaussian noise.
  • PI05WarmstartPytorch.sample_actions(): initialises from B_t + σ·ε and
    runs only n_warmstart_steps instead of num_inference_steps.
  • PI05WarmstartPolicy.__init__(): replaces self.model with the warmstart
    variant; tracks self._prev_chunk across select_action() calls.
  • PI05WarmstartPolicy.forward(): splits extended action tensor into
    prev_actions and current actions; detects cold-start from is_pad mask.
  • PI05WarmstartPolicy.select_action(): stores the predicted chunk as the
    next warm-start initialisation; resets on self.reset().
"""

import builtins
import logging
from collections import deque
from pathlib import Path
from typing import Unpack

import torch
import torch.cuda.nvtx as nvtx
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.modeling_pi05 import (
    PI05Policy,
    PI05Pytorch,
    ActionSelectKwargs,
    pad_vector,
)
from lerobot.policies.pi05_warmstart.configuration_pi05_warmstart import PI05WarmstartConfig
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _compute_shifted_prior(prev_actions: Tensor, execution_horizon: int) -> Tensor:
    """Compute B_t = shift(prev_chunk, E).

    shift(A, E) = [a_E, a_{E+1}, ..., a_{H-1}, a_{H-1}, ..., a_{H-1}]
    Shifts the chunk left by E steps and pads the tail by repeating
    the last action E times.

    Args:
        prev_actions: shape (B, H, d_action)
        execution_horizon: E, number of steps to shift

    Returns:
        B_t: shape (B, H, d_action)
    """
    E = execution_horizon
    # Shifted body: actions from index E onwards
    shifted = prev_actions[:, E:, :]          # (B, H-E, d_action)
    # Tail: last action repeated E times
    last = prev_actions[:, -1:, :].expand(-1, E, -1)  # (B, E, d_action)
    return torch.cat([shifted, last], dim=1)  # (B, H, d_action)


# ─────────────────────────────────────────────────────────────────────────────
# Core model: overrides only forward() and sample_actions()
# ─────────────────────────────────────────────────────────────────────────────

class PI05WarmstartPytorch(PI05Pytorch):
    """PI05 core model with warm-start flow matching.

    All helper methods (embed_prefix, embed_suffix, denoise_step, …) are
    inherited unchanged from PI05Pytorch.  Only the two top-level entry
    points that implement the flow-matching objective are overridden.
    """

    def forward(  # type: ignore[override]
        self,
        images,
        img_masks,
        tokens,
        masks,
        actions: Tensor,
        noise: Tensor | None = None,
        time: Tensor | None = None,
        prev_actions: Tensor | None = None,
        source: Tensor | None = None,
    ) -> Tensor:
        """Training forward pass with warm-start flow matching.

        Args:
            images, img_masks, tokens, masks: observation inputs (unchanged)
            actions: current action chunk (B, chunk_size, max_action_dim)
            noise: optional pre-sampled noise (unused when source is given,
                   kept for API compatibility; used only in all-cold fallback)
            time: optional pre-sampled time (beta-distributed, unchanged)
            prev_actions: previous action chunk (B, chunk_size, max_action_dim)
                or None for cold-start.  Ignored when ``source`` is provided.
            source: pre-computed source distribution (B, chunk_size, max_action_dim)
                already shifted for warm-start rows and Gaussian noise for cold-start
                rows.  When provided, bypasses internal ``_compute_shifted_prior``
                and handles mixed batches correctly: cold-start rows use Gaussian
                noise matching inference, warm-start rows use the shifted prior B_t.

        Returns:
            Per-element MSE losses of shape (B, chunk_size, max_action_dim).
        """
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]  # (B, 1, 1) for broadcasting

        # ── WARM-START: training interpolation ───────────────────────────────
        if source is not None:
            # Unified path: source was pre-computed by the policy forward.
            # Warm rows: source = B_t (shifted prev chunk).
            # Cold rows: source = Gaussian noise (matches cold-start inference).
            # Same formula applies to all samples in the batch.
            sigma = self.config.warmstart_sigma
            eps = torch.randn_like(actions)
            x_t = time_expanded * source + (1 - time_expanded) * actions + sigma * eps
            u_t = source - actions

        elif prev_actions is not None and self.config.use_warmstart:
            # All-warm batch: compute shift internally (backward-compat path).
            E = self.config.execution_horizon
            sigma = self.config.warmstart_sigma

            B_t = _compute_shifted_prior(prev_actions, E)  # (B, H, d_action)

            # Interpolation: x_t = t*B_t + (1-t)*A* + σ*ε
            # (Code convention: t=1 → source; t=0 → target/clean)
            eps = torch.randn_like(actions)
            x_t = time_expanded * B_t + (1 - time_expanded) * actions + sigma * eps

            # Velocity target: u_t = B_t - A*
            # (points from clean action toward the warm-start prior;
            #  during inference we integrate backwards, i.e., dt < 0)
            u_t = B_t - actions

        else:
            # ── COLD-START: standard flow matching (fallback) ─────────────
            if noise is None:
                noise = self.sample_noise(actions.shape, actions.device)
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions
        # ── END WARM-START ────────────────────────────────────────────────

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, tokens, masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0]
            .self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.chunk_size:]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(  # type: ignore[override]
        self,
        images,
        img_masks,
        tokens,
        masks,
        noise: Tensor | None = None,
        num_steps: int | None = None,
        prev_chunk: Tensor | None = None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Inference with warm-start initialisation.

        Args:
            images, img_masks, tokens, masks: observation inputs (unchanged).
            noise: ignored when warm-starting; used for cold-start if supplied.
            num_steps: overrides step count; if None, auto-selected.
            prev_chunk: previous action chunk (B, chunk_size, max_action_dim)
                or None for cold-start (first step of an episode).

        Returns:
            actions: (B, chunk_size, max_action_dim) denoised action chunk.
        """
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
        import copy

        bsize = tokens.shape[0]
        device = tokens.device

        # ── WARM-START: initialise from shifted prior ─────────────────────
        if prev_chunk is not None and self.config.use_warmstart:
            E = self.config.execution_horizon
            sigma = self.config.warmstart_sigma

            B_t = _compute_shifted_prior(prev_chunk, E)  # (B, H, d_action)
            x_t = B_t + sigma * torch.randn_like(B_t)
            n_steps = self.config.n_warmstart_steps

        else:
            # ── COLD-START: pure Gaussian noise, full denoising ───────────
            if noise is None:
                actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
                noise = self.sample_noise(actions_shape, device)
            x_t = noise
            n_steps = self.config.num_inference_steps
        # ── END WARM-START ────────────────────────────────────────────────

        if num_steps is not None:
            n_steps = num_steps  # explicit override

        nvtx.range_push("vlm_prefix")
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, tokens, masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        nvtx.range_pop()  # vlm_prefix

        dt = -1.0 / n_steps

        nvtx.range_push("action_expert_all")
        for step in range(n_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=input_x_t,
                    timestep=current_timestep,
                )

            nvtx.range_push(f"denoise_step_{step:02d}")
            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)
            nvtx.range_pop()  # denoise_step

            x_t = x_t + dt * v_t

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)
        nvtx.range_pop()  # action_expert_all

        return x_t


# ─────────────────────────────────────────────────────────────────────────────
# Policy wrapper
# ─────────────────────────────────────────────────────────────────────────────

class PI05WarmstartPolicy(PI05Policy):
    """PI05 policy with warm-start flow matching.

    Inherits all weight-loading, preprocessing, and RTC logic from PI05Policy.
    Only the training loss and inference initialisation differ.
    """

    config_class = PI05WarmstartConfig
    name = "pi05_warmstart"

    def __init__(self, config: PI05WarmstartConfig, **kwargs):
        # Call PI05Policy.__init__ which sets up rtc_processor and self.model
        super().__init__(config, **kwargs)

        # Replace PI05Pytorch with the warm-start variant.
        # Both share the same architecture, so pretrained weights load correctly.
        self.model = PI05WarmstartPytorch(config, rtc_processor=self.rtc_processor)

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)

        # Warm-start state: the last predicted action chunk
        self._prev_chunk: Tensor | None = None

    # ── class method: load pretrained (delegates to PI05Policy logic) ─────

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        **kwargs,
    ) -> T:
        """Load pretrained weights into the warm-start policy.

        Accepts any pi05 or pi05_warmstart checkpoint.  If loading a plain
        pi05 checkpoint, pass ``config=PI05WarmstartConfig(...)`` explicitly.
        """
        return super().from_pretrained(pretrained_name_or_path, config=config, **kwargs)

    # ── reset ─────────────────────────────────────────────────────────────

    def reset(self):
        """Reset action queue AND warm-start state (call at episode start)."""
        super().reset()
        self._prev_chunk = None

    # ── action selection (inference) ──────────────────────────────────────

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action; refills queue with warm-started chunk."""
        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action; use predict_action_chunk"
        )

        self.eval()

        if len(self._action_queue) == 0:
            # Pass the stored prev_chunk (None → cold-start on first call)
            full_chunk = self.predict_action_chunk(batch, prev_chunk=self._prev_chunk)
            actions = full_chunk[:, : self.config.n_action_steps]

            # Store the FULL chunk (chunk_size steps) as the next warm-start prior.
            # Must be stored before slicing to n_action_steps, otherwise
            # _compute_shifted_prior receives (B, n_action_steps, d) instead of
            # (B, chunk_size, d) and produces a wrong-shaped source tensor.
            # predict_action_chunk returns original_action_dim; re-pad to max_action_dim.
            self._prev_chunk = pad_vector(
                full_chunk, self.config.max_action_dim
            ).detach()

            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Tensor],
        prev_chunk: Tensor | None = None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Predict an action chunk with optional warm-start.

        Args:
            batch: observation batch.
            prev_chunk: (B, chunk_size, max_action_dim) previous padded chunk.
                If None, cold-start is used.
        """
        self.eval()

        images, img_masks = self._preprocess_images(batch)
        tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions = self.model.sample_actions(
            images, img_masks, tokens, masks,
            prev_chunk=prev_chunk,
            **kwargs,
        )

        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]
        return actions

    # ── training forward ──────────────────────────────────────────────────

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        """Training loss with warm-start flow matching.

        The batch "action" key has shape (B, 2*chunk_size, action_dim) because
        action_delta_indices is extended to [-chunk_size..chunk_size-1].
        This method splits it into prev_actions and current_actions, then builds
        a unified source tensor for the flow-matching interpolation:

          - Warm-start rows: source = B_t = shift(prev_actions, E)
          - Cold-start rows: source = Gaussian noise ~ N(0, I)

        Using Gaussian noise for cold-start rows (instead of zeros) matches the
        inference cold-start path exactly, eliminating the train/inference mismatch.
        """
        images, img_masks = self._preprocess_images(batch)
        tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        # ── split extended action tensor ───────────────────────────────────
        raw_actions = batch[ACTION]  # (B, 2*chunk_size, action_dim)
        chunk_size = self.config.chunk_size

        if raw_actions.shape[1] == 2 * chunk_size:
            # Warm-start training: dataset provided both halves
            prev_raw = raw_actions[:, :chunk_size, :]       # (B, H, d)
            curr_raw = raw_actions[:, chunk_size:, :]       # (B, H, d)

            prev_actions = pad_vector(prev_raw, self.config.max_action_dim)
            actions = pad_vector(curr_raw, self.config.max_action_dim)

            # ── build unified source tensor ────────────────────────────────
            # "action_is_pad" shape: (B, 2*chunk_size)
            # The first chunk_size entries correspond to prev_actions.
            # If any prev entry is padded the sample is at an episode boundary
            # (cold-start): its source should be Gaussian noise to match inference.
            is_pad = batch.get("action_is_pad", None)
            if is_pad is not None:
                prev_is_pad = is_pad[:, :chunk_size]           # (B, chunk_size)
                cold_start_mask = prev_is_pad.any(dim=1)       # (B,) bool

                # Gaussian noise for cold-start rows (matches inference cold-start)
                cold_noise = torch.randn_like(actions)

                if cold_start_mask.all():
                    # Every sample is cold-start: source is pure Gaussian noise
                    source = cold_noise
                elif cold_start_mask.any():
                    # Mixed batch: warm rows get B_t, cold rows get Gaussian noise
                    B_t = _compute_shifted_prior(prev_actions, self.config.execution_horizon)
                    source = B_t.clone()
                    source[cold_start_mask] = cold_noise[cold_start_mask]
                    logging.debug(
                        "Mixed warm/cold batch: %d/%d cold-start samples",
                        cold_start_mask.sum().item(),
                        cold_start_mask.shape[0],
                    )
                else:
                    # All warm-start: source is the shifted prior B_t
                    source = _compute_shifted_prior(prev_actions, self.config.execution_horizon)
            else:
                # No padding info: assume all warm-start (could be a bug → log warning)
                logging.warning(
                    "action_is_pad not found in batch; assuming all warm-start. "
                    "This is unexpected — check dataset loading."
                )
                source = _compute_shifted_prior(prev_actions, self.config.execution_horizon)

        elif raw_actions.shape[1] == chunk_size:
            # Dataset returned only current chunk (e.g., baseline run or wrong config)
            # → always cold-start: source is None, model uses Gaussian noise internally
            actions = pad_vector(raw_actions, self.config.max_action_dim)
            source = None
            logging.warning(
                "action tensor has chunk_size=%d steps, expected %d. "
                "Falling back to cold-start. Check action_delta_indices.",
                chunk_size, 2 * chunk_size,
            )
        else:
            raise ValueError(
                f"Unexpected action shape: {raw_actions.shape}. "
                f"Expected (B, {chunk_size}, d) or (B, {2 * chunk_size}, d)."
            )

        # ── compute warm-start flow-matching loss ──────────────────────────
        losses = self.model.forward(
            images, img_masks, tokens, masks, actions,
            source=source,
        )

        # Truncate to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]

        per_dim = losses.mean(dim=[0, 1]).detach().cpu().numpy()  # (action_dim,)
        # Expand list into individual scalars so WandB can plot each dimension
        loss_dict = {f"loss_dim_{i}": float(v) for i, v in enumerate(per_dim)}

        if reduction == "none":
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            loss = losses.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict
