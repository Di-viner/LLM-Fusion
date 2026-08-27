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
"""Reward wrapper for BFCL v3 (prompting mode) evaluation.

``compute_score`` decodes the model's textual function-call output and scores it
against the BFCL ground-truth spec using the vendored AST / relevance checkers
in :mod:`verl.utils.reward_score.bfcl`.

The ground truth carried in the dataset is a JSON object of the form::

    {
        "category": "simple",            # BFCL test category
        "language": "python",            # python | java | javascript
        "functions": [ {func doc}, ... ], # raw (un-preprocessed) function docs
        "possible_answer": [ ... ] | null # None for relevance/irrelevance
    }
"""

import json
import logging
import re

from .bfcl import ast_checker, decode_ast, is_empty_output
from .bfcl.ast_parse import is_function_calling_format_output

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _is_relevance_category(category: str) -> bool:
    return "relevance" in category or "irrelevance" in category


def _clean_solution(solution_str: str) -> str:
    """Strip a leading reasoning block and common wrappers before decoding.

    BFCL prompting mode instructs the model to return *only* the function calls.
    Reasoning models (e.g. Qwen3) may still emit a ``<think>...</think>`` block;
    we drop it so the trailing answer can be parsed.
    """
    if solution_str is None:
        return ""
    text = _THINK_RE.sub("", solution_str)
    # If the model left an explicit reasoning open tag without a close tag,
    # keep only the content after the last one.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    return text.strip()


def _parse_ground_truth(ground_truth):
    if isinstance(ground_truth, str):
        return json.loads(ground_truth)
    return ground_truth


def compute_score(solution_str: str, ground_truth, extra_info: dict | None = None) -> float:
    """Return 1.0 if the decoded function call(s) satisfy the BFCL check, else 0.0."""
    try:
        spec = _parse_ground_truth(ground_truth)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("BFCL: failed to parse ground truth: %s", e)
        return 0.0

    category = spec.get("category", "")
    language = spec.get("language", "python")
    functions = spec.get("functions", [])
    possible_answer = spec.get("possible_answer")

    cleaned = _clean_solution(solution_str)

    # ---- Relevance / irrelevance: judged purely on whether a valid call exists.
    if _is_relevance_category(category):
        contain_func_call = False
        try:
            decoded = decode_ast(cleaned)
            contain_func_call = True
            if is_empty_output(decoded):
                contain_func_call = False
        except Exception:
            contain_func_call = False

        if "irrelevance" in category:
            success = not contain_func_call
        else:  # (live_)relevance
            success = contain_func_call
        return 1.0 if success else 0.0

    # ---- AST categories: decode, validate format, run the checker.
    try:
        decoded = decode_ast(cleaned)
    except Exception:
        return 0.0

    if not is_function_calling_format_output(decoded):
        return 0.0

    if possible_answer is None:
        return 0.0

    try:
        result = ast_checker(functions, decoded, possible_answer, language, category)
    except Exception as e:
        logger.warning("BFCL: checker raised for category %s: %s", category, e)
        return 0.0

    return 1.0 if result.get("valid") else 0.0
