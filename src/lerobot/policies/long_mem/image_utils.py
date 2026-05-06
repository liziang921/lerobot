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
"""Image extraction and JPEG encoding helpers for long-memory prompts."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from lerobot.utils.constants import OBS_IMAGE, OBS_IMAGES


@dataclass(frozen=True)
class EncodedObservationImage:
    """An observation image encoded for a multimodal LLM request."""

    key: str
    image_bytes: bytes
    width: int
    height: int
    mime_type: str = "image/jpeg"

    @property
    def data_url(self) -> str:
        encoded = base64.b64encode(self.image_bytes).decode("ascii")
        return f"data:{self.mime_type};base64,{encoded}"


def is_observation_image_key(key: str) -> bool:
    """Return whether a LeRobot observation key looks like an image feature."""

    return key == OBS_IMAGE or key.startswith(f"{OBS_IMAGES}.")


def get_observation_image_keys(
    observation: Mapping[str, Any],
    image_keys: Sequence[str] | None = None,
) -> list[str]:
    """Return configured image keys or auto-detected LeRobot image keys."""

    if image_keys is not None:
        return list(image_keys)
    return [key for key in observation if is_observation_image_key(key)]


def extract_and_encode_observation_images(
    observation: Mapping[str, Any] | None,
    *,
    image_keys: Sequence[str] | None = None,
    batch_index: int = 0,
    max_image_size: int = 512,
    jpeg_quality: int = 80,
) -> list[EncodedObservationImage]:
    """Extract image features from one LeRobot observation and encode them as JPEG bytes.

    v1 callers should pass batch size 1 observations. The batch_index argument is kept so
    future batched memory can select a specific vector-env slot without changing this API.
    """

    if observation is None:
        return []
    if batch_index < 0:
        raise ValueError("batch_index must be >= 0")
    if max_image_size < 1:
        raise ValueError("max_image_size must be >= 1")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")

    encoded_images: list[EncodedObservationImage] = []
    for key in get_observation_image_keys(observation, image_keys=image_keys):
        if key not in observation:
            raise KeyError(f"Configured image key '{key}' is not present in the observation")
        image = observation[key]
        image_np = observation_image_to_numpy(image, batch_index=batch_index)
        image_bytes, width, height = encode_numpy_image_to_jpeg(
            image_np,
            max_image_size=max_image_size,
            jpeg_quality=jpeg_quality,
        )
        encoded_images.append(
            EncodedObservationImage(
                key=key,
                image_bytes=image_bytes,
                width=width,
                height=height,
            )
        )
    return encoded_images


def observation_image_to_numpy(image: Any, *, batch_index: int = 0) -> np.ndarray:
    """Convert a LeRobot image tensor/array into channel-last uint8 RGB."""

    if isinstance(image, torch.Tensor):
        image_np = image.detach().to("cpu").numpy()
    elif isinstance(image, np.ndarray):
        image_np = image
    else:
        image_np = np.asarray(image)

    if image_np.ndim == 4:
        if batch_index >= image_np.shape[0]:
            raise IndexError(
                f"batch_index {batch_index} is out of bounds for image batch size {image_np.shape[0]}"
            )
        image_np = image_np[batch_index]

    if image_np.ndim == 2:
        image_np = np.repeat(image_np[..., None], 3, axis=-1)

    if image_np.ndim != 3:
        raise ValueError(f"Expected image with 2, 3, or 4 dimensions, got shape {image_np.shape}")

    # Convert channel-first CHW to channel-last HWC when the first dimension is a color channel.
    if image_np.shape[0] in (1, 3, 4) and image_np.shape[-1] not in (1, 3, 4):
        image_np = np.moveaxis(image_np, 0, -1)

    if image_np.shape[-1] == 1:
        image_np = np.repeat(image_np, 3, axis=-1)
    elif image_np.shape[-1] == 4:
        image_np = image_np[..., :3]
    elif image_np.shape[-1] != 3:
        raise ValueError(f"Expected image channel dimension to be 1, 3, or 4, got shape {image_np.shape}")

    if np.issubdtype(image_np.dtype, np.floating):
        max_value = float(np.nanmax(image_np)) if image_np.size else 0.0
        if max_value <= 1.0:
            image_np = image_np * 255.0
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    elif image_np.dtype != np.uint8:
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(image_np)


def encode_numpy_image_to_jpeg(
    image: np.ndarray,
    *,
    max_image_size: int = 512,
    jpeg_quality: int = 80,
) -> tuple[bytes, int, int]:
    """Resize and encode a channel-last uint8 RGB image as JPEG bytes."""

    if max_image_size < 1:
        raise ValueError("max_image_size must be >= 1")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected channel-last RGB image with shape (H, W, 3), got {image.shape}")

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required to encode long-memory observation images") from exc

    height, width = image.shape[:2]
    scale = min(1.0, max_image_size / max(height, width))
    pil_image = Image.fromarray(image)
    if scale < 1.0:
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        pil_image = pil_image.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)

    from io import BytesIO

    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG", quality=int(jpeg_quality), optimize=True)
    width, height = pil_image.size
    return buffer.getvalue(), width, height
