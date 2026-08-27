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
"""Reward wrapper for BFCL v3 multi-turn (state-based) evaluation.

The BFCL multi-turn spec is JSON-encoded into ``reward_model.ground_truth``::

    {
        "test_id": "multi_turn_base_0",
        "involved_classes": ["GorillaFileSystem", "TwitterAPI"],
        "initial_config": {...},
        "ground_truth": [ [call, ...], ... ],   # per user-turn call strings
        "long_context": false
    }

The model's executed calls (per turn, per step) come from the rollout and are
passed through ``extra_info["bfcl_multiturn_calls"]`` with shape
``list[turn][step][call_str]``. Scoring replays both sides on fresh vendored
backends and compares final per-turn state + execution responses
(:func:`verl.utils.bfcl_multiturn.checker.multi_turn_checker`).
"""

import json
import logging

from verl.utils.bfcl_multiturn import multi_turn_checker

logger = logging.getLogger(__name__)


def _parse_spec(ground_truth):
    if isinstance(ground_truth, str):
        return json.loads(ground_truth)
    return ground_truth


def _normalize_model_decoded(calls, num_turns):
    """Coerce recorded model calls into ``list[turn][step][str]``.

    Accepts either the already-nested per-turn/per-step structure or a flat
    per-turn list of call strings (treated as a single step per turn).
    """
    if calls is None:
        return [[[]] for _ in range(num_turns)]
    decoded = []
    for turn in calls:
        if not turn:
            decoded.append([[]])
        elif isinstance(turn[0], list):
            # already list[step][call]
            decoded.append([[str(c) for c in step] for step in turn])
        else:
            # list[call] -> single step
            decoded.append([[str(c) for c in turn]])
    return decoded


def compute_score(solution_str: str, ground_truth, extra_info: dict | None = None) -> float:
    """Return 1.0 if the model's multi-turn trajectory matches BFCL ground truth."""
    try:
        spec = _parse_spec(ground_truth)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("BFCL multi-turn: failed to parse ground truth: %s", e)
        return 0.0

    gt_turns = spec.get("ground_truth", [])
    involved_classes = spec.get("involved_classes", [])
    initial_config = spec.get("initial_config", {})
    long_context = bool(spec.get("long_context", False))

    model_calls = None
    if isinstance(extra_info, dict):
        model_calls = extra_info.get("bfcl_multiturn_calls")

    model_decoded = _normalize_model_decoded(model_calls, len(gt_turns))

    try:
        result = multi_turn_checker(
            model_decoded, gt_turns, initial_config, involved_classes, long_context
        )
    except Exception as e:
        logger.warning("BFCL multi-turn: checker raised: %s", e)
        return 0.0

    return 1.0 if result.get("valid") else 0.0
