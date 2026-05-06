#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Configuration for plug-and-play long-term memory controllers."""

from dataclasses import dataclass
from pathlib import Path


LONG_MEM_PROVIDERS = ("mock", "openai")
DEFAULT_OPENAI_LONG_MEM_MODEL = "gpt-5.4-mini"


@dataclass
class LongMemConfig:
    """Configuration for the high-level long-term memory runtime.

    This is intentionally not a PreTrainedConfig: long memory is an inference-time
    controller/wrapper around a policy, not a trainable LeRobot policy.
    """

    enabled: bool = False
    provider: str = "openai"
    model: str = DEFAULT_OPENAI_LONG_MEM_MODEL

    # High-level segment timing
    decision_interval_chunks: int = 1
    verify_only_at_chunk_boundary: bool = True
    call_llm_only_at_safe_boundary: bool = True

    # Conservative memory update / retry behavior
    confidence_threshold: float = 0.75
    max_retries_per_subtask: int = 3

    # Image input settings
    image_keys: list[str] | None = None
    max_image_size: int = 512
    jpeg_quality: int = 80

    # Logging
    log_dir: Path = Path("outputs/memory_rollouts")
    save_memory_trace: bool = False

    def __post_init__(self) -> None:
        if self.provider not in LONG_MEM_PROVIDERS:
            raise ValueError(
                f"Unsupported long memory provider '{self.provider}'. "
                f"Expected one of {LONG_MEM_PROVIDERS}."
            )
        if self.decision_interval_chunks < 1:
            raise ValueError("decision_interval_chunks must be >= 1")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if self.max_retries_per_subtask < 0:
            raise ValueError("max_retries_per_subtask must be >= 0")
        if self.max_image_size < 1:
            raise ValueError("max_image_size must be >= 1")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
