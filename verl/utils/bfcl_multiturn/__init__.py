# Copyright 2024 The Gorilla Team and the Berkeley Function Calling Leaderboard authors
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from https://github.com/ShishirPatil/gorilla
# (berkeley-function-call-leaderboard/bfcl_eval), licensed under Apache-2.0.
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
"""Vendored BFCL v3 multi-turn stateful backends + checker.

Ports the multi-turn evaluation core from Gorilla ``bfcl_eval`` so verl can run
and score BFCL v3 ``multi_turn`` tasks (the paradigm closest to the WorkBench
agentic training). The 8 stateful backends (GorillaFileSystem, MathAPI,
MessageAPI, TwitterAPI, TicketAPI, TradingBot, TravelAPI, VehicleControlAPI)
live under ``func_source_code/`` and their prompt schemas under
``multi_turn_func_doc/``.
"""

import functools
import json
import os

from .checker import multi_turn_checker, multi_turn_irrelevance_checker
from .executor import (
    CLASS_MAPPING,
    FUNC_DOC_FILE_MAPPING,
    execute_calls,
    make_instances,
)

_FUNC_DOC_DIR = os.path.join(os.path.dirname(__file__), "multi_turn_func_doc")


@functools.lru_cache(maxsize=None)
def _load_func_doc(class_name: str) -> tuple:
    # The func-doc files are JSONL (one function schema per line).
    path = os.path.join(_FUNC_DOC_DIR, FUNC_DOC_FILE_MAPPING[class_name])
    with open(path) as f:
        return tuple(json.loads(line) for line in f if line.strip())


def get_involved_function_docs(involved_classes) -> list:
    """Return the concatenated function schemas for the involved backend classes."""
    docs = []
    for class_name in involved_classes:
        if class_name not in FUNC_DOC_FILE_MAPPING:
            raise ValueError(f"Unsupported BFCL multi-turn class: {class_name}")
        docs.extend([json.loads(json.dumps(d)) for d in _load_func_doc(class_name)])
    return docs


__all__ = [
    "CLASS_MAPPING",
    "FUNC_DOC_FILE_MAPPING",
    "make_instances",
    "execute_calls",
    "multi_turn_checker",
    "multi_turn_irrelevance_checker",
    "get_involved_function_docs",
]
