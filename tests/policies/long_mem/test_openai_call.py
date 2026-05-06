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
"""Opt-in OpenAI smoke test for the long-memory controller.

Run directly to exercise the real OpenAI API:

    python tests/policies/long_mem/test_openai_call.py --save-memory-trace

Or run with pytest explicitly:

    RUN_OPENAI_LONG_MEM_SMOKE=1 pytest tests/policies/long_mem/test_openai_call.py -s
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import NamedTuple

import numpy as np

from lerobot.policies.long_mem import LongMemDecisionLogger, MemoryDecision, MemoryState
from lerobot.policies.long_mem.controllers.openai import (
    DEFAULT_OPENAI_LONG_MEM_MODEL,
    OpenAILongMemController,
)
from lerobot.policies.long_mem.schemas import MEMORY_DECISIONS, SUBTASK_STATUSES


IMAGE_KEY = "observation.images.front"


class SmokeTestDecisions(NamedTuple):
    first_call: MemoryDecision
    verification: MemoryDecision


def make_dummy_scene(*, cup_in_box: bool) -> np.ndarray:
    """Create a simple RGB scene for image-encoding and multimodal API smoke tests."""

    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ImportError("Pillow is required for the OpenAI long-memory smoke test") from exc

    image = Image.new("RGB", (512, 512), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    draw.rectangle((0, 340, 512, 512), fill=(230, 230, 230))
    draw.text((24, 24), "Robot scene", fill=(0, 0, 0))

    box = (300, 210, 455, 360)
    draw.rectangle(box, outline=(25, 90, 200), width=8)
    draw.text((315, 180), "box", fill=(25, 90, 200))

    if cup_in_box:
        cup = (350, 255, 405, 330)
        label = "end: cup in box"
    else:
        cup = (105, 260, 160, 335)
        label = "start: cup on table"

    draw.rectangle(cup, fill=(220, 30, 45), outline=(120, 0, 10), width=4)
    draw.text((24, 64), label, fill=(0, 0, 0))
    return np.asarray(image, dtype=np.uint8)


def run_openai_smoke_test(
    *,
    model: str = DEFAULT_OPENAI_LONG_MEM_MODEL,
    save_memory_trace: bool = False,
    log_dir: str | Path = "outputs/memory_rollouts",
) -> SmokeTestDecisions:
    start_obs = {IMAGE_KEY: make_dummy_scene(cup_in_box=False)}
    end_obs = {IMAGE_KEY: make_dummy_scene(cup_in_box=True)}
    first_call_state = MemoryState(
        episode_id="openai_smoke_test_first_call",
        goal="Put the red cup into the box.",
        semantic_memory="",
        attempt_memory="",
        current_subtask=None,
    )
    verification_state = MemoryState(
        episode_id="openai_smoke_test_verify",
        goal="Put the red cup into the box.",
        semantic_memory="",
        attempt_memory="",
        current_subtask="Pick up the red cup and place it into the box.",
    )
    controller = OpenAILongMemController(
        model=model,
        image_keys=[IMAGE_KEY],
        max_image_size=512,
        jpeg_quality=80,
    )

    first_call_decision = controller.decide(
        state=first_call_state,
        start_observation=start_obs,
        end_observation=None,
    )
    _assert_real_openai_response(controller)
    _assert_valid_decision(first_call_decision)
    assert first_call_decision.decision == "next"
    assert first_call_decision.next_subtask is not None
    assert not first_call_decision.task_complete
    assert not first_call_decision.should_update_semantic_memory
    _print_decision_report(
        title="Case 1: first-call planning",
        controller=controller,
        decision=first_call_decision,
    )
    if save_memory_trace:
        _save_trace(
            log_dir=log_dir,
            state=first_call_state,
            decision=first_call_decision,
            controller=controller,
        )

    verification_decision = controller.decide(
        state=verification_state,
        start_observation=start_obs,
        end_observation=end_obs,
    )
    _assert_real_openai_response(controller)
    _assert_valid_decision(verification_decision)
    _print_decision_report(
        title="Case 2: segment verification",
        controller=controller,
        decision=verification_decision,
    )
    if save_memory_trace:
        _save_trace(
            log_dir=log_dir,
            state=verification_state,
            decision=verification_decision,
            controller=controller,
        )

    return SmokeTestDecisions(first_call=first_call_decision, verification=verification_decision)


def _assert_valid_decision(decision: MemoryDecision) -> None:
    assert isinstance(decision, MemoryDecision)
    assert decision.decision in MEMORY_DECISIONS
    assert decision.subtask_status in SUBTASK_STATUSES
    assert decision.task_complete or decision.next_subtask is not None


def _assert_real_openai_response(controller: OpenAILongMemController) -> None:
    assert controller.last_raw_output is not None, (
        "Expected a real OpenAI raw output, but got fallback/no output."
    )


def _print_decision_report(
    *,
    title: str,
    controller: OpenAILongMemController,
    decision: MemoryDecision,
) -> None:
    print(f"\n=== {title} ===")
    print("OpenAI long-memory smoke test decision:")
    print(decision)
    print("\nRaw model JSON output:")
    print(controller.last_raw_output)
    print("\nPrompt/image metadata:")
    print(f"model={controller.model}")
    print(f"start_image_keys={[image.key for image in controller.last_start_images]}")
    print(f"end_image_keys={[image.key for image in controller.last_end_images]}")


def _save_trace(
    *,
    log_dir: str | Path,
    state: MemoryState,
    decision: MemoryDecision,
    controller: OpenAILongMemController,
) -> None:
    log_path = LongMemDecisionLogger(log_dir).log_decision(
        state_before=state,
        decision=decision,
        provider="openai",
        model=controller.model,
        extra={
            "start_image_keys": [image.key for image in controller.last_start_images],
            "end_image_keys": [image.key for image in controller.last_end_images],
        },
    )
    print(f"\nAppended memory decision log: {log_path}")


def test_openai_long_mem_smoke() -> None:
    if os.getenv("RUN_OPENAI_LONG_MEM_SMOKE") != "1":
        import pytest

        pytest.skip("Set RUN_OPENAI_LONG_MEM_SMOKE=1 to run the real OpenAI smoke test")
    run_openai_smoke_test(save_memory_trace=os.getenv("SAVE_MEMORY_TRACE") == "1")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the OpenAI long-memory smoke test.")
    parser.add_argument("--model", default=DEFAULT_OPENAI_LONG_MEM_MODEL)
    parser.add_argument("--save-memory-trace", action="store_true")
    parser.add_argument("--log-dir", default="outputs/memory_rollouts")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    if not os.getenv("OPENAI_API_KEY"):
        logging.warning("OPENAI_API_KEY is not set; the controller should return a conservative fallback")

    run_openai_smoke_test(
        model=args.model,
        save_memory_trace=args.save_memory_trace,
        log_dir=args.log_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
