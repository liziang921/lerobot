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

import pytest

from lerobot.policies.long_mem import LongMemConfig, MemoryDecision, MemoryState


def make_decision(**overrides) -> MemoryDecision:
    values = {
        "subtask_status": "complete",
        "confidence": 0.9,
        "evidence": "The object is visibly on the target surface.",
        "should_update_semantic_memory": True,
        "semantic_memory": "The object has been placed on the target surface.",
        "attempt_memory": "",
        "decision": "next",
        "next_subtask": "Move to the next visible object.",
        "task_complete": False,
        "warnings": [],
    }
    values.update(overrides)
    return MemoryDecision(**values)


def test_long_mem_config_defaults_are_valid():
    cfg = LongMemConfig()

    assert not cfg.enabled
    assert cfg.provider == "mock"
    assert cfg.decision_interval_chunks == 1
    assert cfg.confidence_threshold == 0.75


@pytest.mark.parametrize(
    "kwargs",
    [
        {"provider": "bad-provider"},
        {"decision_interval_chunks": 0},
        {"confidence_threshold": -0.1},
        {"confidence_threshold": 1.1},
        {"max_retries_per_subtask": -1},
        {"max_image_size": 0},
        {"jpeg_quality": 0},
        {"jpeg_quality": 101},
    ],
)
def test_long_mem_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        LongMemConfig(**kwargs)


def test_memory_state_defaults_are_valid_and_independent():
    first = MemoryState(episode_id="episode-1", goal="Put the mug on the shelf.")
    second = MemoryState(episode_id="episode-2", goal="Open the drawer.")

    first.completed_events.append("The mug was grasped.")

    assert first.semantic_memory == ""
    assert first.batch_id is None
    assert second.completed_events == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"episode_id": ""},
        {"subtask_index": -1},
        {"high_level_step": -1},
        {"retry_count_for_current_subtask": -1},
        {"batch_id": -1},
    ],
)
def test_memory_state_rejects_invalid_values(kwargs):
    values = {"episode_id": "episode-1", "goal": "Put the mug on the shelf."}
    values.update(kwargs)

    with pytest.raises(ValueError):
        MemoryState(**values)


def test_memory_decision_accepts_valid_values():
    decision = make_decision()

    assert decision.subtask_status == "complete"
    assert decision.confidence == 0.9
    assert decision.should_update_semantic_memory


@pytest.mark.parametrize(
    "kwargs",
    [
        {"subtask_status": "done"},
        {"decision": "advance"},
        {"confidence": -0.1},
        {"confidence": 1.1},
        {"confidence": "high"},
        {"task_complete": True, "decision": "next"},
        {"task_complete": False, "decision": "task_complete"},
        {"next_subtask": 3},
        {"warnings": "not-a-list"},
        {"warnings": [3]},
    ],
)
def test_memory_decision_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        make_decision(**kwargs)


@pytest.mark.parametrize("subtask_status", ["incomplete", "failed", "uncertain"])
def test_memory_decision_rejects_semantic_update_for_unsafe_statuses(subtask_status):
    with pytest.raises(ValueError):
        make_decision(subtask_status=subtask_status, should_update_semantic_memory=True)


def test_memory_decision_accepts_task_complete_without_next_subtask():
    decision = make_decision(
        subtask_status="complete",
        decision="task_complete",
        task_complete=True,
        next_subtask=None,
    )

    assert decision.task_complete
    assert decision.next_subtask is None
