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
"""WorkBench (``agent`` domain) reward for verl.

The reward is *outcome/state* based, mirroring nemo-gym's ``workplace_assistant``
verifier: replay the tool calls the policy actually executed and the
ground-truth tool calls on two freshly-seeded WorkBench environments and check
that all workplace databases end in the same state (see
``verl.utils.workbench.verify``).

Predicted tool calls are captured during the multi-turn rollout by
``verl.tools.workbench_tool`` into ``agent_data.extra_fields['workbench_actions']``,
which the agent loop forwards to the reward manager as ``tool_extra_fields`` and
is merged into ``extra_info`` here. As a fallback (e.g. offline scoring without
the tool agent loop) the calls are parsed from the response text.

``ground_truth`` carries the JSON-encoded list of ``{"name", "arguments"}``
ground-truth tool calls, so scoring flows through verl's standard
``compute_score(data_source, solution_str, ground_truth, extra_info)`` path.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from verl.utils.workbench.verify import compute_score as _state_compute_score

logger = logging.getLogger(__name__)

# Hermes/Qwen tool-call blocks emitted in the assistant text, used only as a
# fallback when the rollout did not record structured actions.
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _normalize_actions(actions: Any) -> list[dict[str, Any]]:
    """Coerce a captured/parsed action list into ``[{name, arguments}]`` dicts."""
    if actions is None:
        return []
    if isinstance(actions, str):
        try:
            actions = json.loads(actions)
        except (json.JSONDecodeError, TypeError):
            return []
    normalized: list[dict[str, Any]] = []
    for a in actions:
        if not isinstance(a, dict):
            continue
        name = a.get("name")
        if not name:
            continue
        args = a.get("arguments", a.get("parameters", {}))
        normalized.append({"name": name, "arguments": args})
    return normalized


def _parse_actions_from_text(solution_str: str) -> list[dict[str, Any]]:
    """Best-effort extraction of tool calls from Hermes-style response text."""
    actions: list[dict[str, Any]] = []
    for match in _TOOL_CALL_RE.findall(solution_str or ""):
        try:
            call = json.loads(match)
        except (json.JSONDecodeError, TypeError):
            continue
        name = call.get("name")
        if not name:
            continue
        actions.append({"name": name, "arguments": call.get("arguments", {})})
    return actions


def _parse_ground_truth(ground_truth: Any) -> list[dict[str, Any]]:
    """Recover the ground-truth tool-call list from ``reward_model.ground_truth``."""
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except (json.JSONDecodeError, TypeError):
            return []
    return _normalize_actions(ground_truth)


def compute_score(
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
) -> float:
    """Score one WorkBench trajectory (1.0 if end-state matches ground truth)."""
    gt_actions = _parse_ground_truth(ground_truth)

    predicted_actions: list[dict[str, Any]] = []
    if isinstance(extra_info, dict) and extra_info.get("workbench_actions") is not None:
        predicted_actions = _normalize_actions(extra_info.get("workbench_actions"))
    else:
        # Fallback: recover the calls from the response text.
        predicted_actions = _parse_actions_from_text(solution_str)

    try:
        return float(_state_compute_score(predicted_actions, gt_actions))
    except Exception as e:  # noqa: BLE001 - never crash the training loop on a bad sample
        logger.warning("WorkBench reward computation failed: %s", e)
        return 0.0
