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
"""OpenAI-backed long-memory controller."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Sequence

try:
    import openai
except ImportError:
    openai = None

from lerobot.policies.long_mem.configuration_long_mem import (
    DEFAULT_OPENAI_LONG_MEM_MODEL,
    LongMemConfig,
)
from lerobot.policies.long_mem.controller import make_continue_fallback_decision
from lerobot.policies.long_mem.image_utils import (
    EncodedObservationImage,
    extract_and_encode_observation_images,
)
from lerobot.policies.long_mem.prompt import LongMemPrompt, build_long_mem_prompt
from lerobot.policies.long_mem.schemas import MemoryDecision, MemoryState


logger = logging.getLogger(__name__)


class OpenAILongMemController:
    """Multimodal OpenAI controller for segment-boundary long-memory decisions.

    The controller is stateless from the model's point of view: every call sends the
    Python-owned MemoryState plus the current segment boundary images. If the SDK,
    credentials, model name, request, or response validation fails, decide() returns
    a conservative fallback that does not update semantic memory.
    """

    provider = "openai"

    def __init__(
        self,
        *,
        model: str = "",
        api_key: str | None = None,
        image_keys: Sequence[str] | None = None,
        max_image_size: int = 512,
        jpeg_quality: int = 80,
        timeout_s: float = 30.0,
        max_retries: int = 2,
        retry_backoff_s: float = 1.0,
        max_output_tokens: int = 1024,
        store: bool = False,
    ) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_LONG_MEM_MODEL)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.image_keys = list(image_keys) if image_keys is not None else None
        self.max_image_size = max_image_size
        self.jpeg_quality = jpeg_quality
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self.max_output_tokens = max_output_tokens
        self.store = store

        self.last_prompt: LongMemPrompt | None = None
        self.last_raw_output: str | None = None
        self.last_response: Any | None = None
        self.last_start_images: list[EncodedObservationImage] = []
        self.last_end_images: list[EncodedObservationImage] = []

        self._client: Any | None = None
        self._init_error: str | None = self._validate_config()
        if self._init_error is None:
            try:
                self._client = openai.OpenAI(
                    api_key=self.api_key,
                    timeout=self.timeout_s,
                    max_retries=0,
                )
            except Exception as exc:
                error_message = _safe_error_message(exc, api_key=self.api_key)
                self._init_error = (
                    f"OpenAI client initialization failed: {error_message}"
                )

    @classmethod
    def from_config(
        cls,
        config: LongMemConfig,
        *,
        api_key: str | None = None,
        timeout_s: float = 30.0,
        max_retries: int = 2,
        retry_backoff_s: float = 1.0,
        max_output_tokens: int = 1024,
        store: bool = False,
    ) -> OpenAILongMemController:
        """Create an OpenAI controller from the shared long-memory config."""

        return cls(
            model=config.model,
            api_key=api_key,
            image_keys=config.image_keys,
            max_image_size=config.max_image_size,
            jpeg_quality=config.jpeg_quality,
            timeout_s=timeout_s,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
            max_output_tokens=max_output_tokens,
            store=store,
        )

    def decide(
        self,
        state: MemoryState,
        start_observation: dict[str, Any] | None,
        end_observation: dict[str, Any] | None,
    ) -> MemoryDecision:
        if self._init_error is not None:
            logger.warning("OpenAI long-memory controller is unavailable: %s", self._init_error)
            return make_continue_fallback_decision(
                state,
                warning=self._init_error,
                evidence="OpenAI controller was not initialized.",
            )

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._decide_once(state, start_observation, end_observation)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "OpenAI long-memory decision attempt %d/%d failed: %s",
                    attempt + 1,
                    self.max_retries + 1,
                    _safe_error_message(exc, api_key=self.api_key),
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_s * (2**attempt))

        logger.warning(
            "OpenAI long-memory controller failed after %d attempts; returning conservative fallback: %s",
            self.max_retries + 1,
            _safe_error_message(last_error, api_key=self.api_key),
        )
        return make_continue_fallback_decision(
            state,
            warning=f"openai_controller_error: {_safe_error_message(last_error, api_key=self.api_key)}",
            evidence="OpenAI controller failed, so the wrapper should preserve semantic memory.",
        )

    def _validate_config(self) -> str | None:
        if openai is None:
            return "openai package is not installed"
        if not self.api_key:
            return "OPENAI_API_KEY is not set"
        if not self.model:
            return "OpenAI model is not set; pass LongMemConfig.model or set OPENAI_MODEL"
        if self.max_image_size < 1:
            return "max_image_size must be >= 1"
        if not 1 <= self.jpeg_quality <= 100:
            return "jpeg_quality must be between 1 and 100"
        if self.timeout_s <= 0:
            return "timeout_s must be > 0"
        if self.max_retries < 0:
            return "max_retries must be >= 0"
        if self.retry_backoff_s < 0:
            return "retry_backoff_s must be >= 0"
        if self.max_output_tokens < 1:
            return "max_output_tokens must be >= 1"
        return None

    def _decide_once(
        self,
        state: MemoryState,
        start_observation: dict[str, Any] | None,
        end_observation: dict[str, Any] | None,
    ) -> MemoryDecision:
        self.last_start_images = extract_and_encode_observation_images(
            start_observation,
            image_keys=self.image_keys,
            max_image_size=self.max_image_size,
            jpeg_quality=self.jpeg_quality,
        )
        self.last_end_images = extract_and_encode_observation_images(
            end_observation,
            image_keys=self.image_keys,
            max_image_size=self.max_image_size,
            jpeg_quality=self.jpeg_quality,
        )
        self.last_prompt = build_long_mem_prompt(
            state,
            start_image_keys=[image.key for image in self.last_start_images],
            end_image_keys=[image.key for image in self.last_end_images],
        )

        response = self._client.responses.create(
            model=self.model,
            instructions=self.last_prompt.system,
            input=[
                {
                    "role": "user",
                    "content": _build_multimodal_content(
                        self.last_prompt,
                        self.last_start_images,
                        self.last_end_images,
                    ),
                }
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "long_mem_decision",
                    "schema": self.last_prompt.response_schema,
                    "strict": True,
                }
            },
            max_output_tokens=self.max_output_tokens,
            store=self.store,
        )
        self.last_response = response
        self.last_raw_output = _extract_response_text(response)
        if not self.last_raw_output:
            refusal = _extract_response_refusal(response)
            if refusal:
                raise ValueError(f"OpenAI response refusal: {refusal}")
            raise ValueError("OpenAI response did not include output text")

        try:
            decision_payload = json.loads(self.last_raw_output)
        except json.JSONDecodeError as exc:
            raise ValueError("OpenAI response was not valid JSON") from exc

        return MemoryDecision(**decision_payload)


def _build_multimodal_content(
    prompt: LongMemPrompt,
    start_images: Sequence[EncodedObservationImage],
    end_images: Sequence[EncodedObservationImage],
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt.user}]
    content.extend(_image_content("start_obs", start_images))
    content.extend(_image_content("end_obs", end_images))
    return content


def _image_content(segment_name: str, images: Sequence[EncodedObservationImage]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for image in images:
        content.append(
            {
                "type": "input_text",
                "text": f"{segment_name} image '{image.key}' ({image.width}x{image.height}).",
            }
        )
        content.append(
            {
                "type": "input_image",
                "image_url": image.data_url,
            }
        )
    return content


def _extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = _as_iterable(getattr(response, "output", None))
    text_parts: list[str] = []
    for item in output:
        for content_item in _as_iterable(_get_value(item, "content")):
            text = _get_value(content_item, "text")
            if isinstance(text, str):
                text_parts.append(text)
    return "\n".join(text_parts).strip()


def _extract_response_refusal(response: Any) -> str | None:
    output = _as_iterable(getattr(response, "output", None))
    for item in output:
        for content_item in _as_iterable(_get_value(item, "content")):
            refusal = _get_value(content_item, "refusal")
            if isinstance(refusal, str) and refusal.strip():
                return refusal.strip()
    return None


def _get_value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _as_iterable(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _safe_error_message(exc: Exception | None, *, api_key: str | None = None) -> str:
    if exc is None:
        return "unknown error"
    message = str(exc)
    for secret in (api_key, os.getenv("OPENAI_API_KEY")):
        if secret:
            message = message.replace(secret, "[redacted]")
    return message
