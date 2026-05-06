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
"""Prompt construction for long-term memory high-level decisions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from lerobot.policies.long_mem.schemas import MEMORY_DECISIONS, SUBTASK_STATUSES, MemoryState


SYSTEM_PROMPT = """You are a high-level long-term memory manager for a robot VLA policy.

Your job is to inspect segment-boundary observations and decide what the low-level robot policy
should do next. A segment starts at start_obs, executes current_subtask for one or more complete
action chunks, and ends at end_obs. Only judge subtask success from segment-boundary observations.
Do not judge success from mid-action observations.

You must make exactly one high-level decision:
1. Judge whether the current subtask is complete, incomplete, failed, or uncertain.
2. Update semantic_memory and attempt_memory conservatively.
3. Choose whether to continue, retry, move to the next subtask, or mark the whole task complete.

Subtask vs full-task completion:
- subtask_status describes only the current_subtask.
- subtask_status="complete" means the current subtask is complete; it does not automatically mean the original goal is complete.
- decision="task_complete" and task_complete=true mean the whole original goal is complete.
- Use task_complete=true only when the original goal is satisfied, considering the original goal, semantic_memory, current_subtask, and start/end observations.
- If the current subtask is complete but the original goal still has remaining work, use subtask_status="complete", decision="next", task_complete=false, and provide the next_subtask.
- If the current subtask completes the final remaining part of the original goal, use subtask_status="complete", decision="task_complete", task_complete=true, and next_subtask=null.

First-call rule:
- If current_subtask is null, no subtask has been executed yet.
- In this case, do not verify completion and do not update semantic_memory.
- Choose the first short concrete subtask from the original goal.
- Use subtask_status="incomplete", should_update_semantic_memory=false, decision="next".

Semantic memory rules:
- semantic_memory stores only completed, high-confidence task progress.
- Only update semantic_memory when subtask_status="complete" and the observations provide clear evidence of successful world-state progress.
- If subtask_status is "incomplete", "failed", or "uncertain", should_update_semantic_memory must be false and semantic_memory should remain unchanged.
- Failed, repeated, ambiguous, or uncertain attempts belong in attempt_memory, not semantic_memory.
- If a completed subtask makes previous attempt_memory irrelevant, clear or compress attempt_memory.

Uncertainty and recovery rules:
- If observations are occluded, ambiguous, or do not show the target area, use subtask_status="uncertain".
- When uncertain, do not claim progress and do not update semantic_memory.
- If helpful, choose decision="retry" and output a safe, concrete visibility-improving or recovery instruction.
- Examples: "Move the gripper slightly upward and away from the plate.", "Move the arm back to reveal the drawer."
- Avoid vague instructions like "inspect the scene" or "get a better view."

Status-decision consistency rules:
- If subtask_status="incomplete", usually use decision="continue".
- If subtask_status="failed", usually use decision="retry".
- If subtask_status="uncertain", use decision="continue" when more execution may finish the same attempt, or decision="retry" when a concrete recovery/visibility action is safer.
- If subtask_status="complete", use decision="next" or decision="task_complete".
- Avoid subtask_status="failed" with decision="continue" unless there is a specific reason to keep executing the exact same instruction.

Instruction rules:
- next_subtask must be one short, concrete, robot-actionable instruction for the low-level VLA.
- Use demonstration-style language, such as "Pick up the blue plate from the sink."
- Avoid abstract planning text, explanations, or multi-step instructions inside next_subtask.
- For decision="continue", repeat the current subtask or provide a very similar concrete instruction.
- For decision="retry", provide a modified concrete instruction that may fix the failure or improve visibility.
- For decision="next", provide the next concrete subtask.
- For decision="task_complete", set task_complete=true and next_subtask=null.

Output rules:
- Do not include chain-of-thought.
- Output only valid JSON matching the provided schema.
"""


@dataclass(frozen=True)
class LongMemPrompt:
    """Provider-agnostic prompt bundle for a long-memory controller."""

    system: str
    user: str
    response_schema: dict[str, Any]


def memory_decision_json_schema() -> dict[str, Any]:
    """Return a JSON schema for MemoryDecision-compatible controller output."""

    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "subtask_status",
            "confidence",
            "evidence",
            "should_update_semantic_memory",
            "semantic_memory",
            "attempt_memory",
            "decision",
            "next_subtask",
            "task_complete",
            "warnings",
        ],
        "properties": {
            "subtask_status": {
                "type": "string",
                "enum": list(SUBTASK_STATUSES),
                "description": (
                    "Judgment of the current_subtask after comparing start_obs and end_obs. "
                    "This is not the same as full original-goal completion."
                ),
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Confidence in the subtask_status and high-level decision.",
            },
            "evidence": {
                "type": "string",
                "description": "Short visual evidence summary. Do not include chain-of-thought.",
            },
            "should_update_semantic_memory": {
                "type": "boolean",
                "description": "True only when completed progress should be committed to semantic_memory.",
            },
            "semantic_memory": {
                "type": "string",
                "description": "Updated compressed completed-progress memory, or the prior value if unchanged.",
            },
            "attempt_memory": {
                "type": "string",
                "description": "Temporary notes about failed, incomplete, repeated, or uncertain attempts.",
            },
            "decision": {
                "type": "string",
                "enum": list(MEMORY_DECISIONS),
                "description": (
                    "High-level control decision for the wrapper. Recommended mapping: "
                    "incomplete->continue, failed->retry, uncertain->continue or retry, "
                    "complete->next or task_complete."
                ),
            },
            "next_subtask": {
                "type": ["string", "null"],
                "description": "Short concrete VLA instruction, or null if the whole task is complete.",
            },
            "task_complete": {
                "type": "boolean",
                "description": (
                    "Whether the whole original long-horizon goal is complete, not merely whether "
                    "the current subtask is complete."
                ),
            },
            "warnings": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Short warnings about uncertainty, occlusion, missing views, or safety concerns.",
            },
        },
    }


def build_state_payload(state: MemoryState, max_decision_history: int = 5) -> dict[str, Any]:
    """Build the compact state payload included in the user prompt."""

    payload = asdict(state)
    if max_decision_history < 0:
        raise ValueError("max_decision_history must be >= 0")
    if max_decision_history == 0:
        payload["decision_history"] = []
    else:
        payload["decision_history"] = payload["decision_history"][-max_decision_history:]
    return payload


def build_long_mem_prompt(
    state: MemoryState,
    *,
    start_image_keys: Sequence[str] | None = None,
    end_image_keys: Sequence[str] | None = None,
    max_decision_history: int = 5,
) -> LongMemPrompt:
    """Build a provider-agnostic prompt for one segment-boundary memory decision.

    Images are not embedded here. The provider-specific controller should attach image bytes
    separately and use the key lists in this prompt to label them as start_obs or end_obs.
    """

    payload = {
        "memory_state": build_state_payload(state, max_decision_history=max_decision_history),
        "segment_observations": {
            "start_obs": {
                "description": "Observation before executing current_subtask for this high-level segment.",
                "image_keys": list(start_image_keys or []),
            },
            "end_obs": {
                "description": "Observation after executing complete action chunk(s) for this segment.",
                "image_keys": list(end_image_keys or []),
            },
        },
        "decision_vocabulary": {
            "subtask_status": {
                "complete": "The current subtask succeeded.",
                "incomplete": "The current subtask is not finished yet, with no clear failure.",
                "failed": "The current attempt visibly failed.",
                "uncertain": "The observations are insufficient for a confident judgment.",
            },
            "decision": {
                "continue": "Keep executing the same current_subtask; this is not a failed retry.",
                "retry": "Retry or recover with a modified concrete instruction after failure or unsafe uncertainty.",
                "next": "Move to a new subtask after completed progress. If current_subtask is null, choose the first subtask.",
                "task_complete": "The whole original goal is complete after completed progress.",
            },
            "recommended_status_decision_pairs": {
                "incomplete": ["continue"],
                "failed": ["retry"],
                "uncertain": ["continue", "retry"],
                "complete": ["next", "task_complete"],
            },
        },
        "response_schema": memory_decision_json_schema(),
    }

    user = (
        "Use the attached start_obs and end_obs images, if present, together with this JSON context. "
        "Return exactly one JSON object matching response_schema.\n\n"
        f"{json.dumps(payload, indent=2)}"
    )
    return LongMemPrompt(system=SYSTEM_PROMPT, user=user, response_schema=memory_decision_json_schema())
