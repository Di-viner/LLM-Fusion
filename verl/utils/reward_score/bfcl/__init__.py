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
"""Vendored, self-contained subset of the Berkeley Function Calling Leaderboard
(BFCL) v3 evaluation logic.

This package ports the AST-based (single-turn) and relevance/irrelevance
checkers from the Gorilla ``bfcl_eval`` package so that verl can score BFCL v3
rollouts offline without pulling in the full (heavy) upstream dependency tree
(tree-sitter, model handlers, network REST executors, ...).

Scope:
  * Python AST categories: simple / multiple / parallel / parallel_multiple and
    their ``live_*`` variants.
  * Relevance: irrelevance / live_irrelevance / live_relevance.

Out of scope (deliberately not vendored):
  * java / javascript  (require tree-sitter parsers)
  * exec_* / rest / sql (require real code / REST execution)
  * multi_turn_*        (require the stateful multi-turn backend)

The upstream reference is https://github.com/ShishirPatil/gorilla
(``berkeley-function-call-leaderboard/bfcl_eval``).
"""

from .ast_parse import decode_ast, is_empty_output
from .checker import ast_checker
from .constants import DEFAULT_SYSTEM_PROMPT, build_bfcl_system_prompt

__all__ = [
    "ast_checker",
    "decode_ast",
    "is_empty_output",
    "DEFAULT_SYSTEM_PROMPT",
    "build_bfcl_system_prompt",
]
