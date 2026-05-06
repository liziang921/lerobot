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
"""Factory helpers for long-memory controllers and policy wrappers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import nn

from lerobot.policies.long_mem.configuration_long_mem import LongMemConfig
from lerobot.policies.long_mem.controller import LLMHighLevelController
from lerobot.policies.long_mem.controllers.mock import MockLongMemController
from lerobot.policies.long_mem.controllers.openai import OpenAILongMemController
from lerobot.policies.long_mem.wrapped_policy import MemoryWrappedPolicy


def make_long_mem_controller(config: LongMemConfig) -> LLMHighLevelController:
    """Build the configured high-level long-memory controller."""

    if config.provider == "mock":
        return MockLongMemController()
    if config.provider == "openai":
        return OpenAILongMemController.from_config(config)
    raise ValueError(f"Unsupported long-memory provider: {config.provider!r}")


def wrap_policy_with_long_mem(
    *,
    base_policy: nn.Module,
    config: LongMemConfig,
    policy_preprocessor: Callable[[dict[str, Any]], dict[str, Any]],
    episode_id: str = "long_mem_episode",
    env_id: str | None = None,
    batch_id: int | None = None,
) -> MemoryWrappedPolicy:
    """Wrap a base LeRobot policy with the configured long-memory controller.

    The wrapper owns the policy preprocessor so it can rewrite the raw ``task``
    field before language tokenization.
    """

    controller = make_long_mem_controller(config)
    return MemoryWrappedPolicy(
        base_policy=base_policy,
        controller=controller,
        config=config,
        policy_preprocessor=policy_preprocessor,
        episode_id=episode_id,
        env_id=env_id,
        batch_id=batch_id,
    )
