# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""IFBench (AllenAI) verifier.

Vendored from https://github.com/allenai/IFBench so verl can score the harder
IFBench instruction set offline.
"""

import logging
from collections.abc import Sequence
from typing import Any

from . import evaluation_lib

InputExample = evaluation_lib.InputExample

logger = logging.getLogger(__name__)
JsonDict = dict[str, Any]
KwargsDict = dict[str, "str | int | float | None"]


def _normalize_instruction_ids(raw_ids: Sequence[Any]) -> list[str]:
    normalized: list[str] = []
    for entry in raw_ids or []:
        if entry is None:
            continue
        text = str(entry).strip()
        if not text:
            continue
        normalized.append(text)
    return normalized


def _coerce_scalar(value: Any) -> Any:
    """Restore integral kwargs that parquet stored as float.

    Nullable integer kwargs (e.g. ``small_n``) are promoted to ``float64`` when
    serialized into a shared parquet struct, so ``5`` comes back as ``5.0``. The
    IFBench checkers use these values for indexing / ``range()`` and reject
    floats, so convert whole-number floats back to ``int`` (leaving genuine
    fractional values untouched).
    """
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _coerce_kwargs_list(raw_kwargs: Any, num_instructions: int) -> list[KwargsDict]:
    if isinstance(raw_kwargs, list):
        processed: list[KwargsDict] = []
        for entry in raw_kwargs:
            processed.append(dict(entry) if isinstance(entry, dict) else {})
    elif isinstance(raw_kwargs, dict):
        processed = [dict(raw_kwargs) for _ in range(num_instructions)]
    else:
        processed = [{} for _ in range(num_instructions)]

    if len(processed) < num_instructions:
        tail = processed[-1] if processed else {}
        processed.extend([dict(tail) for _ in range(num_instructions - len(processed))])
    elif len(processed) > num_instructions:
        processed = processed[:num_instructions]

    sanitized: list[KwargsDict] = []
    for entry in processed:
        sanitized.append({k: _coerce_scalar(v) for k, v in entry.items() if v is not None})
    return sanitized


def _build_input_example(metadata: JsonDict) -> "InputExample | None":
    instruction_ids = _normalize_instruction_ids(metadata.get("instruction_id_list"))
    if not instruction_ids:
        logger.debug("Missing instruction identifiers in metadata: %s", metadata)
        return None

    prompt_text = metadata.get("prompt_text")
    prompt_text = "" if prompt_text is None else str(prompt_text)

    kwargs_list = _coerce_kwargs_list(metadata.get("kwargs"), len(instruction_ids))

    return InputExample(
        key=int(metadata.get("record_id") or 0),
        instruction_id_list=instruction_ids,
        prompt=prompt_text,
        kwargs=kwargs_list,
    )


def compute_ifbench_reward(response: str, metadata: "JsonDict | None" = None) -> float:
    """Score a model response using the official IFBench (strict) rules."""

    if metadata is None:
        logger.debug("No metadata provided for IFBench scoring.")
        return 0.0
    if response is None:
        return 0.0

    inp = _build_input_example(metadata)
    if inp is None:
        return 0.0

    prompt_to_response = {inp.prompt: str(response or "")}
    output = evaluation_lib.test_instruction_following_strict(inp, prompt_to_response)
    return 1.0 if output.follow_all_instructions else 0.0
