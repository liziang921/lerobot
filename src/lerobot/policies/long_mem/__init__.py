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
"""Long-term memory utilities for high-level VLA subtask management."""

from lerobot.policies.long_mem.configuration_long_mem import DEFAULT_OPENAI_LONG_MEM_MODEL, LongMemConfig
from lerobot.policies.long_mem.controller import LLMHighLevelController, make_continue_fallback_decision
from lerobot.policies.long_mem.controllers.mock import MockLongMemController
from lerobot.policies.long_mem.controllers.openai import OpenAILongMemController
from lerobot.policies.long_mem.factory import make_long_mem_controller, wrap_policy_with_long_mem
from lerobot.policies.long_mem.image_utils import (
    EncodedObservationImage,
    extract_and_encode_observation_images,
)
from lerobot.policies.long_mem.logger import LongMemDecisionLogger, build_long_mem_decision_log_record
from lerobot.policies.long_mem.prompt import (
    LongMemPrompt,
    build_long_mem_prompt,
    memory_decision_json_schema,
)
from lerobot.policies.long_mem.schemas import MemoryDecision, MemoryState
from lerobot.policies.long_mem.wrapped_policy import MemoryWrappedPolicy

__all__ = [
    "EncodedObservationImage",
    "DEFAULT_OPENAI_LONG_MEM_MODEL",
    "LLMHighLevelController",
    "LongMemConfig",
    "LongMemDecisionLogger",
    "LongMemPrompt",
    "MemoryDecision",
    "MemoryState",
    "MemoryWrappedPolicy",
    "MockLongMemController",
    "OpenAILongMemController",
    "build_long_mem_decision_log_record",
    "build_long_mem_prompt",
    "extract_and_encode_observation_images",
    "make_long_mem_controller",
    "make_continue_fallback_decision",
    "memory_decision_json_schema",
    "wrap_policy_with_long_mem",
]
