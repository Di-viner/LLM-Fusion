# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Instruction-following verifier for verl.

Wraps two rule-based backends: ``ifevalg`` (Google IFEval +
extended constraints) and ``ifbench`` (AllenAI IFBench). The per-sample
verification spec (``rm_type`` / ``instruction_id_list`` / ``kwargs`` /
``prompt_text``) is carried in ``ground_truth`` as a JSON string produced by the
data-preprocessing step, so it flows through verl's standard
``compute_score(data_source, solution_str, ground_truth, extra_info)`` path.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _strip_think(response: str) -> str:
    """Keep the text after the final think block, matching rollout post-processing.

    Some pipelines hard-fail (reward 0) when ``</think>`` is missing because their
    rollout always emits a reasoning trace. verl is used with many chat templates, so we
    only strip when the tag is present and otherwise score the full response.
    """
    if response is None:
        return ""
    if "</think>" in response:
        response = response.split("</think>")[-1]
    if response.endswith("<|im_end|>"):
        response = response[: response.rfind("<|im_end|>")]
    return response.strip()


def _parse_spec(ground_truth: Any, extra_info: dict | None) -> dict:
    """Recover the IF verification spec from ground_truth (JSON str/dict)."""
    spec: dict = {}
    if isinstance(ground_truth, str):
        try:
            spec = json.loads(ground_truth)
        except (json.JSONDecodeError, TypeError):
            spec = {}
    elif isinstance(ground_truth, dict):
        spec = dict(ground_truth)

    # Allow extra_info to supply / override fields (useful for custom pipelines).
    if isinstance(extra_info, dict):
        for key in ("rm_type", "instruction_id_list", "kwargs", "prompt_text"):
            if not spec.get(key) and extra_info.get(key) is not None:
                spec[key] = extra_info[key]
    return spec


def compute_score(
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
) -> float:
    """Verify an instruction-following response.

    Returns a plain float score, matching the other verifiers (math / science /
    sandbox code). Returning a dict here would make the reward manager emit a
    different set of ``reward_extra_info`` keys for IF samples than for the
    others (and include a non-numeric ``rm_type``), which breaks batch assembly
    (``KeyError`` in agent_loop ``_postprocess``) and validation metric
    aggregation when IF is mixed with other data sources in one run.
    """
    spec = _parse_spec(ground_truth, extra_info)
    rm_type = str(spec.get("rm_type") or "ifevalg").strip().lower()
    response = _strip_think(solution_str)

    metadata = {
        "instruction_id_list": spec.get("instruction_id_list") or [],
        "kwargs": spec.get("kwargs"),
        "prompt_text": spec.get("prompt_text", ""),
        "record_id": spec.get("record_id", 0),
    }

    try:
        if rm_type == "ifbench":
            from .ifbench import compute_ifbench_reward

            score = compute_ifbench_reward(response, metadata=metadata)
        else:  # default: ifevalg
            from .ifevalg import compute_ifevalg_reward

            score = compute_ifevalg_reward(response, metadata=metadata)
    except Exception as e:  # noqa: BLE001 - never crash the training loop on a bad sample
        logger.warning("IF reward (%s) computation failed: %s", rm_type, e)
        score = 0.0

    return float(score)
