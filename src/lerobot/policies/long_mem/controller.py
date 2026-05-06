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
"""Provider-agnostic controller interface for long-term memory decisions."""

from __future__ import annotations

from typing import Any, Protocol

from lerobot.policies.long_mem.schemas import MemoryDecision, MemoryState


class LLMHighLevelController(Protocol):
    """Interface implemented by mock, OpenAI, and future multimodal LLM controllers."""

    provider: str

    def decide(
        self,
        state: MemoryState,
        start_observation: dict[str, Any] | None,
        end_observation: dict[str, Any] | None,
    ) -> MemoryDecision:
        """Return one high-level decision for the current segment boundary."""
        ...


def make_continue_fallback_decision(
    state: MemoryState,
    *,
    warning: str,
    evidence: str = "Controller decision failed or was unavailable.",
) -> MemoryDecision:
    """Build a conservative fallback decision that does not update semantic memory."""

    next_subtask = state.current_subtask
    decision = "continue" if next_subtask else "next"
    if next_subtask is None:
        next_subtask = state.goal

    return MemoryDecision(
        subtask_status="uncertain",
        confidence=0.0,
        evidence=evidence,
        should_update_semantic_memory=False,
        semantic_memory=state.semantic_memory,
        attempt_memory=state.attempt_memory,
        decision=decision,
        next_subtask=next_subtask,
        task_complete=False,
        warnings=[warning],
    )
