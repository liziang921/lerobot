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
"""Mock long-memory controller for local integration work without API calls."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from lerobot.policies.long_mem.controller import make_continue_fallback_decision
from lerobot.policies.long_mem.schemas import MemoryDecision, MemoryState


class MockLongMemController:
    """Deterministic long-memory controller for tests and dry runs.

    If scripted decisions are provided, each call returns the next scripted decision.
    Otherwise, the first call starts with a concrete default subtask and later calls
    conservatively continue the current subtask without updating semantic memory.
    """

    provider = "mock"

    def __init__(
        self,
        *,
        first_subtask: str | None = None,
        scripted_decisions: Iterable[MemoryDecision | dict[str, Any]] | None = None,
        script_path: str | Path | None = None,
    ) -> None:
        if scripted_decisions is not None and script_path is not None:
            raise ValueError("Provide scripted_decisions or script_path, not both")

        self.first_subtask = first_subtask
        self._scripted_decisions: deque[MemoryDecision] = deque()
        self.num_calls = 0

        if script_path is not None:
            scripted_decisions = self._load_script(script_path)
        if scripted_decisions is not None:
            self._scripted_decisions = deque(self._coerce_decision(item) for item in scripted_decisions)

    def decide(
        self,
        state: MemoryState,
        start_observation: dict[str, Any] | None,
        end_observation: dict[str, Any] | None,
    ) -> MemoryDecision:
        del start_observation, end_observation

        self.num_calls += 1
        if self._scripted_decisions:
            return self._scripted_decisions.popleft()

        if state.current_subtask is None:
            return self._first_subtask_decision(state)

        return self._continue_decision(state)

    def _first_subtask_decision(self, state: MemoryState) -> MemoryDecision:
        next_subtask = self.first_subtask or _default_first_subtask(state.goal)
        return MemoryDecision(
            subtask_status="uncertain",
            confidence=1.0,
            evidence="Mock controller selected the first subtask without visual verification.",
            should_update_semantic_memory=False,
            semantic_memory=state.semantic_memory,
            attempt_memory=state.attempt_memory,
            decision="next",
            next_subtask=next_subtask,
            task_complete=False,
            warnings=["mock_controller"],
        )

    def _continue_decision(self, state: MemoryState) -> MemoryDecision:
        return MemoryDecision(
            subtask_status="incomplete",
            confidence=1.0,
            evidence="Mock controller is configured to keep executing the current subtask.",
            should_update_semantic_memory=False,
            semantic_memory=state.semantic_memory,
            attempt_memory=state.attempt_memory,
            decision="continue",
            next_subtask=state.current_subtask,
            task_complete=False,
            warnings=["mock_controller"],
        )

    @staticmethod
    def _coerce_decision(value: MemoryDecision | dict[str, Any]) -> MemoryDecision:
        if isinstance(value, MemoryDecision):
            return value
        if isinstance(value, dict):
            return MemoryDecision(**value)
        raise TypeError(f"Scripted decision must be MemoryDecision or dict, got {type(value)}")

    @staticmethod
    def _load_script(script_path: str | Path) -> list[dict[str, Any]]:
        path = Path(script_path)
        try:
            with path.open() as f:
                data = json.load(f)
        except OSError as exc:
            raise OSError(f"Failed to load mock long-memory script from {path}") from exc

        if isinstance(data, dict):
            data = data.get("decisions")
        if not isinstance(data, list):
            raise ValueError("Mock long-memory script must be a list or an object with a 'decisions' list")
        if not all(isinstance(item, dict) for item in data):
            raise ValueError("Every scripted mock decision must be a JSON object")
        return data

    def fallback(self, state: MemoryState, *, warning: str) -> MemoryDecision:
        """Return the shared conservative fallback decision."""

        return make_continue_fallback_decision(state, warning=warning)


def _default_first_subtask(goal: str) -> str:
    stripped_goal = goal.strip()
    if stripped_goal:
        return stripped_goal
    return "Inspect the scene and prepare for the task."
