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
"""BFCL v3 multi-turn tools for verl's (Hermes) tool agent loop.

Mirrors :mod:`verl.tools.workbench_tool`. BFCL multi-turn exposes stateful
backend classes (e.g. ``GorillaFileSystem``, ``TwitterAPI``) whose public
methods mutate a per-sample environment seeded from ``initial_config``. Unlike
WorkBench (single fixed toolset + fixed initial state), BFCL is *per sample*:

- The involved classes / initial config differ per test case; they travel in the
  parquet under ``extra_info.tools_kwargs['__bfcl_seed__']`` and are read here to
  lazily instantiate the backends on first use.
- The advertised toolset is filtered per sample via ``extra_info.tool_selection``
  (handled by ``ToolAgentLoop``), so each rollout only sees the methods of its
  involved classes.

As with WorkBench, ``ToolAgentLoop`` creates/releases a tool on every call, so
trajectory state is stashed on the per-trajectory ``AgentData``:

- ``agent_data._bfcl_instances``: the live backend instances (plain attribute,
  never serialized).
- ``agent_data.extra_fields['bfcl_multiturn_calls']``: the executed call strings
  in the shape ``[[call, ...]]`` (single user turn) that the reward manager
  forwards as ``extra_info`` to
  :func:`verl.utils.reward_score.bfcl_multiturn_verify`, which replays them on
  fresh backends for a state-based reward.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.bfcl_multiturn.executor import execute_calls, make_instances
from verl.utils.bfcl_multiturn.tool_schemas import get_bfcl_schema_by_name
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Attribute name used to cache the live BFCL backend instances on AgentData.
_INSTANCES_ATTR = "_bfcl_instances"
# Key under the per-sample ``tools_kwargs`` holding the trajectory seed.
SEED_KEY = "__bfcl_seed__"
# extra_fields key holding the executed calls as list[turn] of call strings.
CALLS_KEY = "bfcl_multiturn_calls"


def _format_call(name: str, arguments: dict[str, Any]) -> str:
    """Reconstruct a Python call string ``name(k=repr(v), ...)``.

    The executor prepends the owning class (``Class.name(...)``) and ``eval``s it
    against the backend instances, so arguments must render as Python literals.
    """
    parts = [f"{k}={v!r}" for k, v in arguments.items()]
    return f"{name}({', '.join(parts)})"


def _get_or_create_instances(agent_data: Any) -> Optional[dict[str, Any]]:
    """Return the trajectory-scoped BFCL backends, creating them on first use."""
    instances = getattr(agent_data, _INSTANCES_ATTR, None)
    if instances is not None:
        return instances
    seed = (getattr(agent_data, "tools_kwargs", None) or {}).get(SEED_KEY)
    if not seed:
        return None
    # The seed is stored as a JSON string (``{"spec": "<json>"}``) so parquet
    # keeps a uniform schema; fall back to a plain dict for direct callers.
    spec = seed.get("spec") if isinstance(seed, dict) else None
    spec = json.loads(spec) if isinstance(spec, str) else seed
    instances = make_instances(
        spec.get("involved_classes", []),
        spec.get("initial_config", {}),
        bool(spec.get("long_context", False)),
    )
    setattr(agent_data, _INSTANCES_ATTR, instances)
    return instances


def _record_call(agent_data: Any, call_str: str) -> None:
    """Append an executed call to this trajectory's single-turn call log."""
    calls = agent_data.extra_fields.setdefault(CALLS_KEY, [[]])
    calls[0].append(call_str)


class BfclTool(BaseTool):
    """One BFCL backend method, sharing a per-trajectory environment via ``AgentData``.

    A single class backs every BFCL method; the tool config supplies a different
    ``function_name`` per entry, whose schema is resolved from the vendored BFCL
    func-docs.
    """

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        if tool_schema is None:
            function_name = config.get("function_name")
            if not function_name:
                raise ValueError("BfclTool requires either a `tool_schema` or a `function_name` in its config.")
            tool_schema = get_bfcl_schema_by_name(function_name)
        super().__init__(config, tool_schema)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        # No per-tool state: backends live on AgentData. Just mint an id.
        return instance_id or str(uuid4()), ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        agent_data = kwargs.get("agent_data")
        if agent_data is None:
            return ToolResponse(text="BFCL environment unavailable."), 0.0, {}

        if not isinstance(parameters, dict):
            parameters = {}
        # Drop null arguments so optional params keep their defaults.
        arguments = {k: v for k, v in parameters.items() if v is not None}

        # Record the attempted call *before* execution: the reward replays the
        # model's action list on a fresh environment regardless of runtime errors.
        call_str = _format_call(self.name, arguments)
        _record_call(agent_data, call_str)

        instances = _get_or_create_instances(agent_data)
        if instances is None:
            return ToolResponse(text="Error: BFCL backend seed is missing for this sample."), 0.0, {}

        try:
            results = execute_calls([call_str], instances)
            text = results[0] if results else ""
        except Exception as e:  # noqa: BLE001 - surface the error so the model can self-correct
            text = f"Error executing tool '{self.name}': {e}"
        return ToolResponse(text=str(text)), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        # Trajectory reward is computed by the reward manager from the recorded
        # call log (see bfcl_multiturn_verify), not per-tool.
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        # Backend lifetime is tied to AgentData, nothing to release per call.
        return None
