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
"""Schemas shared by long-term memory controllers and runtimes."""

from dataclasses import dataclass, field
from typing import Any


SUBTASK_STATUSES = (
    "complete",
    "incomplete",
    "failed",
    "uncertain",
)

MEMORY_DECISIONS = (
    "continue",
    "retry",
    "next",
    "task_complete",
)


@dataclass
class MemoryState:
    """Python-owned state passed into each stateless high-level LLM call."""

    episode_id: str
    goal: str
    semantic_memory: str = ""
    attempt_memory: str = ""
    current_subtask: str | None = None
    previous_subtask: str | None = None
    subtask_index: int = 0
    high_level_step: int = 0
    retry_count_for_current_subtask: int = 0
    completed_events: list[str] = field(default_factory=list)
    failed_attempts: list[str] = field(default_factory=list)
    decision_history: list[dict[str, Any]] = field(default_factory=list)
    env_id: str | None = None
    batch_id: int | None = None
    task_complete: bool = False
    task_failed: bool = False
    terminal_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise ValueError("episode_id must be a non-empty string")
        if not isinstance(self.goal, str):
            raise ValueError("goal must be a string")
        if self.subtask_index < 0:
            raise ValueError("subtask_index must be >= 0")
        if self.high_level_step < 0:
            raise ValueError("high_level_step must be >= 0")
        if self.retry_count_for_current_subtask < 0:
            raise ValueError("retry_count_for_current_subtask must be >= 0")
        if self.batch_id is not None and self.batch_id < 0:
            raise ValueError("batch_id must be >= 0 when provided")
        if self.task_complete and self.task_failed:
            raise ValueError("task_complete and task_failed cannot both be True")
        if self.terminal_reason is not None and not isinstance(self.terminal_reason, str):
            raise ValueError("terminal_reason must be a string or None")


@dataclass
class MemoryDecision:
    """Validated output from a high-level memory controller.

    subtask_status describes only the current subtask. decision="task_complete"
    with task_complete=True means the whole original goal is complete.
    """

    subtask_status: str
    confidence: float
    evidence: str
    should_update_semantic_memory: bool
    semantic_memory: str
    attempt_memory: str
    decision: str
    next_subtask: str | None
    task_complete: bool
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.subtask_status not in SUBTASK_STATUSES:
            raise ValueError(
                f"Invalid subtask_status '{self.subtask_status}'. Expected one of {SUBTASK_STATUSES}."
            )
        if self.decision not in MEMORY_DECISIONS:
            raise ValueError(f"Invalid decision '{self.decision}'. Expected one of {MEMORY_DECISIONS}.")
        if not isinstance(self.confidence, int | float):
            raise ValueError("confidence must be numeric")
        self.confidence = float(self.confidence)
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if self.task_complete and self.decision != "task_complete":
            raise ValueError("task_complete=True requires decision='task_complete'")
        if self.decision == "task_complete" and not self.task_complete:
            raise ValueError("decision='task_complete' requires task_complete=True")
        if self.decision == "task_complete" and self.next_subtask is not None:
            raise ValueError("next_subtask must be None when decision='task_complete'")
        if self.decision != "task_complete" and (
            self.next_subtask is None or not self.next_subtask.strip()
        ):
            raise ValueError("next_subtask must be a non-empty string unless decision='task_complete'")
        if self.subtask_status != "complete" and self.should_update_semantic_memory:
            raise ValueError("Only subtask_status='complete' may update semantic memory")
        if self.next_subtask is not None and not isinstance(self.next_subtask, str):
            raise ValueError("next_subtask must be a string or None")
        if not isinstance(self.warnings, list) or not all(isinstance(w, str) for w in self.warnings):
            raise ValueError("warnings must be a list of strings")
