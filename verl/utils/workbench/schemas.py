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
"""WorkBench tool schemas.

The vendored toolkit files declare their schemas in the OpenAI *Responses* flat
layout (``{"type": "function", "name": ..., "parameters": ...}``). verl's tool
stack expects the *Chat Completions* nested layout
(``{"type": "function", "function": {"name": ..., "parameters": ...}}``) via
:class:`verl.tools.schemas.OpenAIFunctionToolSchema`. This module is the single
place that owns the conversion so the toolkit files can stay byte-for-byte
identical to upstream.
"""

from __future__ import annotations

import copy
from typing import Any

from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.utils.workbench.workplace_assistant_tools.analytics import analytics_tool_schemas
from verl.utils.workbench.workplace_assistant_tools.calendar import calendar_tool_schemas
from verl.utils.workbench.workplace_assistant_tools.company_directory import company_directory_tool_schemas
from verl.utils.workbench.workplace_assistant_tools.customer_relationship_manager import (
    customer_relationship_manager_tool_schemas,
)
from verl.utils.workbench.workplace_assistant_tools.email import email_tool_schemas
from verl.utils.workbench.workplace_assistant_tools.project_management import project_management_tool_schemas

# Raw flat schemas, in the order tools are registered by ``verify.get_tools``.
RAW_TOOL_SCHEMAS: list[dict[str, Any]] = [
    *company_directory_tool_schemas,
    *email_tool_schemas,
    *calendar_tool_schemas,
    *analytics_tool_schemas,
    *project_management_tool_schemas,
    *customer_relationship_manager_tool_schemas,
]


def _to_nested(flat: dict[str, Any]) -> dict[str, Any]:
    """Convert a flat OpenAI-Responses schema to the nested Chat-Completions layout."""
    flat = copy.deepcopy(flat)
    parameters = flat.get("parameters") or {"type": "object", "properties": {}}
    # verl's pydantic schema does not model ``additionalProperties``; drop it.
    parameters.pop("additionalProperties", None)
    function = {
        "name": flat["name"],
        "description": flat.get("description", ""),
        "parameters": parameters,
    }
    if "strict" in flat:
        function["strict"] = bool(flat["strict"])
    return {"type": "function", "function": function}


def get_workbench_tool_schemas_dict() -> list[dict[str, Any]]:
    """All WorkBench tool schemas as nested OpenAI Chat-Completions dicts."""
    return [_to_nested(s) for s in RAW_TOOL_SCHEMAS]


def get_workbench_tool_schemas() -> list[OpenAIFunctionToolSchema]:
    """All WorkBench tool schemas as validated ``OpenAIFunctionToolSchema`` objects."""
    return [OpenAIFunctionToolSchema.model_validate(s) for s in get_workbench_tool_schemas_dict()]


def get_schema_by_name(name: str) -> OpenAIFunctionToolSchema:
    """Return the validated schema for a single tool by function name."""
    for schema in get_workbench_tool_schemas_dict():
        if schema["function"]["name"] == name:
            return OpenAIFunctionToolSchema.model_validate(schema)
    raise KeyError(f"Unknown WorkBench tool: {name}")


# Ordered list of all tool (function) names exposed by WorkBench.
WORKBENCH_TOOL_NAMES: list[str] = [s["name"] for s in RAW_TOOL_SCHEMAS]
