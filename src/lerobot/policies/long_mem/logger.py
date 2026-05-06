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
"""JSONL decision logging for long-memory rollouts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lerobot.policies.long_mem.schemas import MemoryDecision, MemoryState


class LongMemDecisionLogger:
    """Append high-level memory decisions to a JSONL file."""

    def __init__(
        self,
        log_dir: str | Path,
        *,
        filename: str = "long_mem_decisions.jsonl",
    ) -> None:
        self.log_dir = Path(log_dir)
        self.path = self.log_dir / filename

    def log_decision(
        self,
        *,
        state_before: MemoryState,
        decision: MemoryDecision,
        state_after: MemoryState | None = None,
        provider: str | None = None,
        model: str | None = None,
        saved_image_paths: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Append one high-level decision record and return the JSONL path."""

        record = build_long_mem_decision_log_record(
            state_before=state_before,
            decision=decision,
            state_after=state_after,
            provider=provider,
            model=model,
            saved_image_paths=saved_image_paths,
            extra=extra,
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(record, sort_keys=True, default=_json_default))
            f.write("\n")
        return self.path


def build_long_mem_decision_log_record(
    *,
    state_before: MemoryState,
    decision: MemoryDecision,
    state_after: MemoryState | None = None,
    provider: str | None = None,
    model: str | None = None,
    saved_image_paths: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable record for one high-level memory decision.

    The wrapper should pass state_after after applying conservative update rules.
    If state_after is omitted, the *_after fields reflect the controller's proposed
    memory strings from MemoryDecision.
    """

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "episode_id": state_before.episode_id,
        "env_id": state_before.env_id,
        "batch_id": state_before.batch_id,
        "high_level_step": state_before.high_level_step,
        "goal": state_before.goal,
        "previous_subtask": state_before.previous_subtask,
        "current_subtask": state_before.current_subtask,
        "semantic_memory_before": state_before.semantic_memory,
        "attempt_memory_before": state_before.attempt_memory,
        "decision": asdict(decision),
        "semantic_memory_after": (
            state_after.semantic_memory if state_after is not None else decision.semantic_memory
        ),
        "attempt_memory_after": (
            state_after.attempt_memory if state_after is not None else decision.attempt_memory
        ),
        "next_subtask": decision.next_subtask,
        "confidence": decision.confidence,
        "saved_image_paths": _jsonable(saved_image_paths or {}),
        "provider": provider,
        "model": model,
        "warnings": list(decision.warnings),
        "state_after": asdict(state_after) if state_after is not None else None,
        "extra": _jsonable(extra or {}),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
