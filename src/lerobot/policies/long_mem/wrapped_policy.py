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
"""Generic policy wrapper for long-memory high-level subtask control."""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from lerobot.policies.long_mem.configuration_long_mem import LongMemConfig
from lerobot.policies.long_mem.controller import LLMHighLevelController
from lerobot.policies.long_mem.logger import LongMemDecisionLogger
from lerobot.policies.long_mem.schemas import MemoryDecision, MemoryState
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


logger = logging.getLogger(__name__)

TASK_KEY = "task"


class MemoryWrappedPolicy(nn.Module):
    """Inference-time long-memory wrapper around a low-level LeRobot policy.

    v1 supports batch size 1 and synchronous high-level decisions at configured
    action-chunk boundaries. The wrapper expects to see the raw task field before
    language tokenization; if a policy_preprocessor is provided, it is called
    after the wrapper rewrites task to the current subtask.

    Both select_action and predict_action_chunk are supported. This matters for
    VLA policies such as pi0.5: normal eval calls select_action, while async
    inference servers may call predict_action_chunk directly.

    Metadata:
        episode_id identifies one rollout/trajectory attempt, not a permanent
            task class. The eval-loop integration can auto-fill this later.
        env_id is optional environment/task metadata.
        batch_id is reserved for future vectorized eval. Since v1 supports batch
            size 1 only, it is usually None or 0.

    Timing:
        action_chunk_steps is the number of low-level actions actually executed
            per VLA chunk/replan. This may be smaller than the predicted action
            horizon. For example, a policy may predict horizon 50 but execute
            only 10 actions before replanning; in that case action_chunk_steps
            should be 10.
    """

    def __init__(
        self,
        *,
        base_policy: nn.Module,
        controller: LLMHighLevelController,
        config: LongMemConfig | None = None,
        policy_preprocessor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        goal: str | None = None,
        episode_id: str = "long_mem_episode",
        env_id: str | None = None,
        batch_id: int | None = None,
        action_chunk_steps: int | None = None,
        reset_base_policy_on_subtask_change: bool = True,
        model: str | None = None,
    ) -> None:
        super().__init__()
        self.base_policy = base_policy
        self.controller = controller
        self.long_mem_config = config or LongMemConfig()
        self.provider = _get_controller_provider(controller)
        _validate_provider_matches_config(self.long_mem_config, self.provider)
        self.policy_preprocessor = policy_preprocessor
        self.initial_goal = goal
        self.episode_id = episode_id
        self.env_id = env_id
        self.batch_id = batch_id
        inferred_action_chunk_steps = _infer_action_chunk_steps(base_policy)

        if action_chunk_steps is None and inferred_action_chunk_steps is None:
            raise ValueError(
                "Could not infer action_chunk_steps from base_policy.config. "
                "Please pass action_chunk_steps explicitly so long-memory decisions occur at chunk boundaries."
            )

        self.action_chunk_steps = (
            action_chunk_steps if action_chunk_steps is not None else inferred_action_chunk_steps
        )
        self.reset_base_policy_on_subtask_change = reset_base_policy_on_subtask_change
        self.model = model or getattr(controller, "model", self.long_mem_config.model)
        self.decision_logger = (
            LongMemDecisionLogger(self.long_mem_config.log_dir)
            if self.long_mem_config.save_memory_trace
            else None
        )

        if self.action_chunk_steps < 1:
            raise ValueError("action_chunk_steps must be >= 1")

        self.memory_state: MemoryState | None = None
        self.segment_start_observation: dict[str, Any] | None = None
        self.actions_since_decision = 0
        self.chunks_since_decision = 0
        self.task_complete = False
        self.task_failed = False
        self.terminal_reason: str | None = None
        self._last_action: Tensor | None = None

    @property
    def config(self) -> Any:
        """Expose the wrapped policy config for code that expects policy.config."""

        return getattr(self.base_policy, "config", None)

    @property
    def segment_length_actions(self) -> int:
        return self.long_mem_config.decision_interval_chunks * self.action_chunk_steps

    @property
    def is_terminal(self) -> bool:
        return self.task_complete or self.task_failed

    def reset(
        self,
        *,
        goal: str | None = None,
        episode_id: str | None = None,
        env_id: str | None = None,
        batch_id: int | None = None,
    ) -> None:
        """Reset wrapper and base-policy state at an episode boundary."""

        if hasattr(self.base_policy, "reset"):
            self.base_policy.reset()

        self.initial_goal = goal if goal is not None else self.initial_goal
        self.episode_id = episode_id if episode_id is not None else self.episode_id
        self.env_id = env_id if env_id is not None else self.env_id
        self.batch_id = batch_id if batch_id is not None else self.batch_id
        self.memory_state = None
        self.segment_start_observation = None
        self.actions_since_decision = 0
        self.chunks_since_decision = 0
        self.task_complete = False
        self.task_failed = False
        self.terminal_reason = None
        self._last_action = None

    def get_optim_params(self) -> Any:
        if not hasattr(self.base_policy, "get_optim_params"):
            raise AttributeError("Wrapped base_policy does not define get_optim_params")
        return self.base_policy.get_optim_params()

    def use_original_modules(self) -> Any:
        if hasattr(self.base_policy, "use_original_modules"):
            return self.base_policy.use_original_modules()
        return None

    def forward(self, batch: dict[str, Tensor], *args, **kwargs) -> Any:
        return self.base_policy(batch, *args, **kwargs)

    @torch.no_grad()
    def predict_action_chunk(self, observation: dict[str, Any], *args, **kwargs) -> Tensor:
        """Return one action chunk while managing long-memory decisions at chunk boundaries."""

        if not hasattr(self.base_policy, "predict_action_chunk"):
            raise AttributeError("Wrapped base_policy does not define predict_action_chunk")

        self._warn_if_terminal_action_requested()
        self._ensure_batch_size_one(observation)
        self._ensure_pre_tokenization_observation(observation)
        self._ensure_memory_state(observation)

        if self.segment_start_observation is None:
            self.segment_start_observation = _snapshot_observation(observation)

        if not self.is_terminal and self.memory_state.current_subtask is None:
            self._run_high_level_decision(
                start_observation=self.segment_start_observation,
                end_observation=None,
            )
            self.segment_start_observation = _snapshot_observation(observation)
            self.actions_since_decision = 0
            self.chunks_since_decision = 0
        elif not self.is_terminal and self.chunks_since_decision >= self.long_mem_config.decision_interval_chunks:
            segment_end_observation = _snapshot_observation(observation)
            self._run_high_level_decision(
                start_observation=self.segment_start_observation,
                end_observation=segment_end_observation,
            )
            self.segment_start_observation = segment_end_observation
            self.actions_since_decision = 0
            self.chunks_since_decision = 0

        policy_observation = self._prepare_policy_observation(observation)
        action_chunk = self.base_policy.predict_action_chunk(policy_observation, *args, **kwargs)
        self.chunks_since_decision += 1
        self.actions_since_decision += self.action_chunk_steps
        return action_chunk

    @torch.no_grad()
    def select_action(self, observation: dict[str, Any], **kwargs) -> Tensor:
        """Return one low-level action while managing high-level memory decisions."""

        self._warn_if_terminal_action_requested()
        self._ensure_batch_size_one(observation)
        self._ensure_pre_tokenization_observation(observation)
        self._ensure_memory_state(observation)

        if self.segment_start_observation is None:
            self.segment_start_observation = _snapshot_observation(observation)

        if not self.is_terminal and self.memory_state.current_subtask is None:
            self._run_high_level_decision(
                start_observation=self.segment_start_observation,
                end_observation=None,
            )
            self.segment_start_observation = _snapshot_observation(observation)
            self.actions_since_decision = 0
            self.chunks_since_decision = 0
        elif not self.is_terminal and self.actions_since_decision >= self.segment_length_actions:
            segment_end_observation = _snapshot_observation(observation)
            self._run_high_level_decision(
                start_observation=self.segment_start_observation,
                end_observation=segment_end_observation,
            )
            self.segment_start_observation = segment_end_observation
            self.actions_since_decision = 0
            self.chunks_since_decision = 0

        policy_observation = self._prepare_policy_observation(observation)
        action = self.base_policy.select_action(policy_observation, **kwargs)
        self._last_action = action
        self.actions_since_decision += 1
        self.chunks_since_decision = self.actions_since_decision // self.action_chunk_steps
        return action

    def _ensure_memory_state(self, observation: dict[str, Any]) -> None:
        if self.memory_state is not None:
            return

        goal = self.initial_goal or _extract_goal_from_observation(observation)
        self.memory_state = MemoryState(
            episode_id=self.episode_id,
            goal=goal,
            env_id=self.env_id,
            batch_id=self.batch_id,
        )

    def _run_high_level_decision(
        self,
        *,
        start_observation: dict[str, Any] | None,
        end_observation: dict[str, Any] | None,
    ) -> MemoryDecision:
        if self.memory_state is None:
            raise RuntimeError("Memory state must be initialized before high-level decisions")

        state_before = copy.deepcopy(self.memory_state)
        previous_subtask = self.memory_state.current_subtask
        decision = self.controller.decide(
            state=self.memory_state,
            start_observation=start_observation,
            end_observation=end_observation,
        )
        self._apply_decision(decision)
        subtask_changed = self.memory_state.current_subtask != previous_subtask
        if subtask_changed and self.reset_base_policy_on_subtask_change:
            self._flush_base_policy()

        if self.decision_logger is not None:
            self.decision_logger.log_decision(
                state_before=state_before,
                decision=decision,
                state_after=copy.deepcopy(self.memory_state),
                provider=self.provider,
                model=self.model,
            )

        return decision

    def _apply_decision(self, decision: MemoryDecision) -> None:
        if self.memory_state is None:
            raise RuntimeError("Memory state must be initialized before applying decisions")

        state = self.memory_state
        previous_subtask = state.current_subtask
        semantic_before = state.semantic_memory

        state.decision_history.append(asdict(decision))
        if decision.subtask_status == "complete" and decision.confidence >= self.long_mem_config.confidence_threshold:
            if decision.should_update_semantic_memory:
                state.semantic_memory = decision.semantic_memory
                if decision.semantic_memory:
                    state.completed_events.append(decision.semantic_memory)
            state.attempt_memory = decision.attempt_memory
            state.retry_count_for_current_subtask = 0
        else:
            state.semantic_memory = semantic_before
            state.attempt_memory = decision.attempt_memory
            if previous_subtask is not None and decision.subtask_status in (
                "incomplete",
                "failed",
                "uncertain",
            ):
                state.failed_attempts.append(decision.evidence)
            if decision.decision == "retry":
                state.retry_count_for_current_subtask += 1
                if state.retry_count_for_current_subtask > self.long_mem_config.max_retries_per_subtask:
                    logger.warning(
                        "Long-memory retry limit exceeded for subtask %r: %d > %d",
                        previous_subtask,
                        state.retry_count_for_current_subtask,
                        self.long_mem_config.max_retries_per_subtask,
                    )
                    state.task_failed = True
                    state.terminal_reason = "max_retries_exceeded"
                    self.task_failed = True
                    self.terminal_reason = "max_retries_exceeded"
            elif decision.decision == "continue":
                pass
            else:
                state.retry_count_for_current_subtask = 0

        if decision.decision == "task_complete":
            state.previous_subtask = previous_subtask
            state.current_subtask = None
            state.task_complete = True
            state.task_failed = False
            state.terminal_reason = "task_complete"
            self.task_complete = True
            self.task_failed = False
            self.terminal_reason = "task_complete"
        else:
            if decision.next_subtask is None:
                logger.warning(
                    "MemoryDecision decision=%r returned next_subtask=None. "
                    "Falling back to previous_subtask=%r or original goal.",
                    decision.decision,
                    previous_subtask,
                )
            next_subtask = decision.next_subtask or previous_subtask or state.goal
            if next_subtask != previous_subtask:
                state.previous_subtask = previous_subtask
                if previous_subtask is not None:
                    state.subtask_index += 1
            state.current_subtask = next_subtask
            state.task_complete = False
            self.task_complete = False
            if not state.task_failed:
                state.terminal_reason = None
                self.terminal_reason = None

        state.high_level_step += 1

    def _prepare_policy_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self.memory_state is None:
            raise RuntimeError("Memory state must be initialized before preparing observations")

        subtask = self.memory_state.current_subtask or self.memory_state.previous_subtask or self.memory_state.goal
        policy_observation = _with_task(observation, subtask)
        if self.policy_preprocessor is not None:
            policy_observation = self.policy_preprocessor(policy_observation)
        return policy_observation

    def _flush_base_policy(self) -> None:
        if hasattr(self.base_policy, "reset"):
            self.base_policy.reset()

    def _warn_if_terminal_action_requested(self) -> None:
        if self.is_terminal:
            logger.warning(
                "MemoryWrappedPolicy action requested after terminal state; "
                "outer rollout loop should stop. task_complete=%s task_failed=%s reason=%s",
                self.task_complete,
                self.task_failed,
                self.terminal_reason,
            )

    def _ensure_batch_size_one(self, observation: dict[str, Any]) -> None:
        batch_size = _infer_batch_size(observation)
        if batch_size != 1:
            raise NotImplementedError(
                f"MemoryWrappedPolicy v1 supports batch_size=1 only, got batch_size={batch_size}"
            )

    def _ensure_pre_tokenization_observation(self, observation: dict[str, Any]) -> None:
        if _has_tokenized_language(observation):
            raise ValueError(
                "MemoryWrappedPolicy must receive observations before language tokenization. "
                "For pi0.5, pass the raw observation with 'task' to the wrapper and provide "
                "policy_preprocessor to MemoryWrappedPolicy, instead of calling the policy preprocessor "
                "before the wrapper."
            )
        if self.memory_state is None and self.initial_goal is None and TASK_KEY not in observation:
            raise ValueError(
                "MemoryWrappedPolicy needs a raw 'task' field on the first observation or an explicit "
                "goal=... at reset/construction time."
            )


def _infer_action_chunk_steps(base_policy: nn.Module) -> int | None:
    config = getattr(base_policy, "config", None)
    for name in ("n_action_steps", "action_chunk_steps", "chunk_size"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    return None


def _get_controller_provider(controller: LLMHighLevelController) -> str:
    provider = getattr(controller, "provider", None)
    if not isinstance(provider, str) or not provider:
        raise ValueError(
            "Long-memory controller must define a non-empty string 'provider' attribute, "
            "for example provider='mock' or provider='openai'."
        )
    return provider


def _validate_provider_matches_config(config: LongMemConfig, controller_provider: str) -> None:
    if config.provider != controller_provider:
        raise ValueError(
            f"LongMemConfig.provider={config.provider!r} does not match "
            f"controller.provider={controller_provider!r}. Use a matching controller/config pair."
        )


def _extract_goal_from_observation(observation: dict[str, Any]) -> str:
    task = observation.get(TASK_KEY, "")
    if isinstance(task, str):
        return task
    if isinstance(task, (list, tuple)):
        if len(task) == 0:
            return ""
        if not isinstance(task[0], str):
            raise TypeError("observation['task'] entries must be strings")
        return task[0]
    raise TypeError("observation['task'] must be a string or a batch-size-one list/tuple of strings")


def _with_task(observation: dict[str, Any], task: str) -> dict[str, Any]:
    updated = dict(observation)
    old_task = observation.get(TASK_KEY)
    if isinstance(old_task, str):
        updated[TASK_KEY] = task
    elif isinstance(old_task, (list, tuple)):
        if len(old_task) != 1:
            raise NotImplementedError(
                f"MemoryWrappedPolicy v1 supports batch_size=1 only, got task batch size {len(old_task)}"
            )
        updated[TASK_KEY] = [task]
    elif old_task is None:
        updated[TASK_KEY] = [task]
    else:
        raise TypeError("observation['task'] must be a string or list/tuple of strings")
    return updated


def _snapshot_observation(observation: dict[str, Any]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for key, value in observation.items():
        if key == TASK_KEY and isinstance(value, list):
            snapshot[key] = list(value)
        elif key == TASK_KEY and isinstance(value, tuple):
            snapshot[key] = tuple(value)
        else:
            snapshot[key] = value
    return snapshot


def _has_tokenized_language(observation: dict[str, Any]) -> bool:
    return OBS_LANGUAGE_TOKENS in observation or OBS_LANGUAGE_ATTENTION_MASK in observation


def _infer_batch_size(observation: dict[str, Any]) -> int:
    if TASK_KEY in observation:
        if isinstance(observation[TASK_KEY], str):
            return 1
        if isinstance(observation[TASK_KEY], (list, tuple)):
            return len(observation[TASK_KEY])
    for value in observation.values():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.shape[0])
        if isinstance(value, np.ndarray) and value.ndim > 0:
            return int(value.shape[0])
    return 1
