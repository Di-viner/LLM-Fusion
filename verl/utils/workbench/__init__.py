# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from https://github.com/NVIDIA-NeMo/Gym
# (resources_servers/workplace_assistant), licensed under Apache-2.0.
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
"""Vendored WorkBench (nemo-gym ``workplace_assistant``) agent environment.

This package bundles everything needed to train and score the ``agent``
domain inside verl without depending on an external nemo-gym checkout:

- ``workplace_assistant_tools/`` + ``csv_data/``: the stateful workplace
  toolkits and their seed data (verbatim from nemo-gym).
- ``verify``: state-based outcome reward (replay predicted vs ground-truth
  tool calls, compare final database states).
- ``schemas``: OpenAI tool schemas exposed to the policy during rollout.

The training-time multi-turn rollout tools live in
``verl.tools.workbench_tool`` and the reward entry point in
``verl.utils.reward_score.workbench_verify``.
"""

from verl.utils.workbench.verify import compute_score, get_tools, is_correct
from verl.utils.workbench.schemas import (
    WORKBENCH_TOOL_NAMES,
    get_workbench_tool_schemas,
    get_workbench_tool_schemas_dict,
)

__all__ = [
    "compute_score",
    "get_tools",
    "is_correct",
    "WORKBENCH_TOOL_NAMES",
    "get_workbench_tool_schemas",
    "get_workbench_tool_schemas_dict",
]
