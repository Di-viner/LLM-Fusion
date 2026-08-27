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
"""BFCL v3 multi-turn tool schemas for verl's (Hermes) tool agent loop.

The vendored BFCL func-docs (``multi_turn_func_doc/*.json``, one JSON schema per
line) use BFCL's flat ``{"name", "description", "parameters": {"type": "dict",
...}}`` layout with Python-ish primitive types (``dict`` / ``float``). verl's
tool stack expects the OpenAI *Chat-Completions* nested layout
(``{"type": "function", "function": {...}}``) validated by
:class:`verl.tools.schemas.OpenAIFunctionToolSchema`.

This module is the single place that converts BFCL func-docs to that layout so
each BFCL backend method can be registered as an individual verl tool (mirroring
:mod:`verl.utils.workbench.schemas`). Across the 8 backend classes the 128
method names are globally unique, so tools can be keyed by the bare method name
(which is also what :func:`verl.utils.bfcl_multiturn.executor.execute_calls`
expects before it prepends the owning class name).
"""

from __future__ import annotations

import functools
import json
import os
from typing import Any

from verl.tools.schemas import OpenAIFunctionToolSchema

from .executor import FUNC_DOC_FILE_MAPPING

_FUNC_DOC_DIR = os.path.join(os.path.dirname(__file__), "multi_turn_func_doc")

# BFCL primitive type -> JSON-schema type.
_TYPE_MAP = {"dict": "object", "float": "number"}


def _convert_type(t: Any) -> Any:
    if isinstance(t, list):
        return [_TYPE_MAP.get(x, x) for x in t]
    return _TYPE_MAP.get(t, t)


def _convert_property(pdef: dict[str, Any]) -> dict[str, Any]:
    """Keep only the fields verl's ``OpenAIFunctionPropertySchema`` models.

    Nested ``items`` / ``properties`` / ``default`` are dropped (the pydantic
    schema does not model them; WorkBench schemas are handled the same way).
    """
    prop: dict[str, Any] = {"type": _convert_type(pdef.get("type", "string"))}
    if pdef.get("description"):
        prop["description"] = pdef["description"]
    if pdef.get("enum"):
        prop["enum"] = pdef["enum"]
    return prop


def _to_nested(doc: dict[str, Any]) -> dict[str, Any]:
    """Convert one BFCL flat func-doc to the nested Chat-Completions layout."""
    params = doc.get("parameters") or {}
    properties = {name: _convert_property(pdef) for name, pdef in (params.get("properties") or {}).items()}
    function = {
        "name": doc["name"],
        "description": doc.get("description", ""),
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(params.get("required", [])),
        },
    }
    return {"type": "function", "function": function}


@functools.lru_cache(maxsize=None)
def _load_func_doc(class_name: str) -> tuple:
    path = os.path.join(_FUNC_DOC_DIR, FUNC_DOC_FILE_MAPPING[class_name])
    with open(path) as f:
        return tuple(json.loads(line) for line in f if line.strip())


@functools.lru_cache(maxsize=1)
def _build_indexes() -> tuple[dict[str, dict], dict[str, str], dict[str, list[str]]]:
    """Return (name -> nested schema dict, name -> class, class -> [names])."""
    name_to_schema: dict[str, dict] = {}
    name_to_class: dict[str, str] = {}
    class_to_names: dict[str, list[str]] = {}
    for class_name in FUNC_DOC_FILE_MAPPING:
        names: list[str] = []
        for doc in _load_func_doc(class_name):
            name = doc["name"]
            if name in name_to_schema:
                raise ValueError(
                    f"Duplicate BFCL method name '{name}' "
                    f"({name_to_class[name]} and {class_name}); bare-name keying is unsafe."
                )
            name_to_schema[name] = _to_nested(doc)
            name_to_class[name] = class_name
            names.append(name)
        class_to_names[class_name] = names
    return name_to_schema, name_to_class, class_to_names


def get_bfcl_tool_schemas_dict() -> list[dict[str, Any]]:
    """All BFCL tool schemas as nested OpenAI Chat-Completions dicts."""
    name_to_schema, _, _ = _build_indexes()
    return list(name_to_schema.values())


def get_bfcl_schema_by_name(name: str) -> OpenAIFunctionToolSchema:
    """Return the validated schema for a single BFCL method by name."""
    name_to_schema, _, _ = _build_indexes()
    if name not in name_to_schema:
        raise KeyError(f"Unknown BFCL tool: {name}")
    return OpenAIFunctionToolSchema.model_validate(name_to_schema[name])


def get_tool_names_for_classes(involved_classes) -> list[str]:
    """Method names exposed by the given involved backend classes (in order)."""
    _, _, class_to_names = _build_indexes()
    names: list[str] = []
    for class_name in involved_classes:
        if class_name not in class_to_names:
            raise ValueError(f"Unsupported BFCL multi-turn class: {class_name}")
        names.extend(class_to_names[class_name])
    return names


# Ordered list of every BFCL method (tool) name across all backend classes.
def _all_names() -> list[str]:
    name_to_schema, _, _ = _build_indexes()
    return list(name_to_schema.keys())


BFCL_TOOL_NAMES: list[str] = _all_names()
