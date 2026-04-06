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

"""Configuration for PI05 with warm-start flow matching.

This extends the standard PI05Config with warm-start parameters.
The only architectural change is in the flow-matching interpolation:
instead of pure Gaussian noise as the source distribution, we use
a shifted version of the previous action chunk (B_t), with small
Gaussian noise sigma added for training robustness.

Shift operator:
    shift(A, E) = [a_E, a_{E+1}, ..., a_{H-1}, a_{H-1}, ..., a_{H-1}]
    (shift left by E, pad tail by repeating last action E times)

Training interpolation (replaces standard x_t = t*ε + (1-t)*A*):
    x_t = t * B_t + (1-t) * A* + sigma * ε

Velocity target (replaces standard u_t = ε - A*):
    u_t = B_t - A*

Inference (replaces x_0 = ε ~ N(0,I), n_steps denoising):
    x_0 = B_t + sigma * ε,  denoised for n_warmstart_steps

Cold-start fallback (first chunk of each episode, no prev chunk):
    Falls back to standard flow matching (Gaussian source, full steps).

Dataset note:
    action_delta_indices is extended to [-chunk_size..chunk_size-1],
    so the batch "action" key has shape (B, 2*chunk_size, action_dim).
    The policy splits this into prev_actions (first half) and current
    actions (second half).  The "action_is_pad" mask from the dataset
    flags which prev_actions are before the episode boundary (cold-start).
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

DEFAULT_IMAGE_SIZE = 224


@PreTrainedConfig.register_subclass("pi05_warmstart")
@dataclass
class PI05WarmstartConfig(PreTrainedConfig):
    # ── model backbone (identical to PI05Config) ──────────────────────────
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"  # Options: "bfloat16", "float32"

    n_obs_steps: int = 1
    chunk_size: int = 50  # Number of action steps to predict (action_horizon)
    n_action_steps: int = 10  # Number of action steps to execute

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters (identical to PI05Config)
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    image_resolution: tuple[int, int] = (
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )

    empty_cameras: int = 0
    tokenizer_max_length: int = 200

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    # Training settings
    gradient_checkpointing: bool = False
    compile_model: bool = False
    compile_mode: str = "max-autotune"
    device: str | None = None

    # Finetuning settings
    freeze_vision_encoder: bool = False
    train_expert_only: bool = False  # Freeze VLM, train only action expert + projections

    # Optimizer settings
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # Scheduler settings
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    # ── warm-start parameters (new) ───────────────────────────────────────
    execution_horizon: int = 10
    """E: number of actions actually executed before re-planning.
    The shift operator slides the previous chunk left by E steps and
    pads the tail with the last action repeated E times.
    Typical values: 1 (re-plan every step), 10 (re-plan every 10 steps).
    Must satisfy: 1 <= execution_horizon <= chunk_size."""

    warmstart_sigma: float = 0.02
    """sigma: standard deviation of the small Gaussian noise added to the
    interpolated trajectory during training and to B_t at inference time.
    Keeps the model robust to imperfect warm starts.
    Typical range: 0.01 - 0.05."""

    n_warmstart_steps: int = 2 # 1
    """K: number of denoising steps when warm-starting (prev chunk available).
    Much smaller than num_inference_steps (10) because B_t is already close
    to the target.  Typical values: 1, 2, 3, 5."""

    use_warmstart: bool = True
    """Toggle for ablation.  Set to False to use cold-start (standard PI05)
    even when a previous chunk is available."""

    def __post_init__(self):
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than "
                f"chunk_size ({self.chunk_size})"
            )
        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")
        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")
        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")
        if not (1 <= self.execution_horizon <= self.chunk_size):
            raise ValueError(
                f"execution_horizon ({self.execution_horizon}) must be in "
                f"[1, chunk_size={self.chunk_size}]"
            )

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = OBS_IMAGES + f".empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),
            )
            self.input_features[key] = empty_camera

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )
            self.input_features[OBS_STATE] = state_feature

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
            )
            self.output_features[ACTION] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        """Fetch both previous chunk and current chunk.

        Returns indices [-chunk_size, ..., -1, 0, 1, ..., chunk_size-1].
        The dataset will provide both in the "action" key as a tensor of
        shape (2*chunk_size, action_dim).  "action_is_pad" flags which
        entries are clamped (before episode start → cold-start).
        """
        return list(range(-self.chunk_size, self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
