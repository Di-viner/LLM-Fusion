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
"""Multi-turn WorkBench tools for verl's tool agent loop.

WorkBench exposes ~27 workplace functions (email / calendar / analytics /
project-management / CRM / company-directory) that all mutate a *single* shared
in-memory database. verl instantiates one :class:`WorkbenchTool` per function,
but they must operate on the same environment within a rollout trajectory.

Because ``ToolAgentLoop`` creates/releases a tool on every individual call, we
cannot keep per-trajectory state on the tool instance. Instead we stash the
live environment (and the ordered list of executed calls) on the per-trajectory
``AgentData`` object that the loop passes into ``execute``:

- ``agent_data._workbench_env``: the live tool environment. It is a plain
  attribute (not part of ``extra_fields``) so it is never serialized.
- ``agent_data.extra_fields["workbench_actions"]``: the ordered ``[{name,
  arguments}]`` the policy actually invoked. ``extra_fields`` flows to the
  reward manager as ``tool_extra_fields`` -> ``extra_info``, where
  ``verl.utils.reward_score.workbench_verify`` replays it against the
  ground-truth trajectory for a state-based reward.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.workbench.schemas import get_schema_by_name
from verl.utils.workbench.verify import get_tools

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Attribute name used to cache the live WorkBench environment on AgentData.
_ENV_ATTR = "_workbench_env"
# extra_fields key holding the ordered list of executed tool calls.
ACTIONS_KEY = "workbench_actions"


def _get_or_create_env(agent_data: Any) -> dict[str, Any]:
    """Return the trajectory-scoped WorkBench environment, creating it on first use."""
    env = getattr(agent_data, _ENV_ATTR, None)
    if env is None:
        env = get_tools()
        setattr(agent_data, _ENV_ATTR, env)
    return env


def _record_action(agent_data: Any, name: str, arguments: dict[str, Any]) -> None:
    """Append an executed call to the trajectory's serializable action log."""
    actions = agent_data.extra_fields.setdefault(ACTIONS_KEY, [])
    actions.append({"name": name, "arguments": arguments})


class WorkbenchTool(BaseTool):
    """One WorkBench function, sharing a per-trajectory environment via ``AgentData``.

    The tool's identity comes entirely from its schema's function name, so a
    single class backs every WorkBench function; the tool config simply supplies
    a different ``tool_schema`` per entry.
    """

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        # Allow the schema to be resolved by name from the vendored WorkBench
        # schemas, so tool configs can omit the (large) parameter definitions.
        if tool_schema is None:
            function_name = config.get("function_name")
            if not function_name:
                raise ValueError(
                    "WorkbenchTool requires either a `tool_schema` or a `function_name` in its config."
                )
            tool_schema = get_schema_by_name(function_name)
        super().__init__(config, tool_schema)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        # No per-tool state: the environment lives on AgentData. Just mint an id.
        return instance_id or str(uuid4()), ToolResponse()

    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        agent_data = kwargs.get("agent_data")
        if agent_data is None:
            # Without AgentData there is no shared env / action log to update.
            return ToolResponse(text="WorkBench environment unavailable."), 0.0, {}

        if not isinstance(parameters, dict):
            parameters = {}
        # Drop null arguments (nemo-gym behavior) so optional params keep defaults.
        arguments = {k: v for k, v in parameters.items() if v is not None}

        env = _get_or_create_env(agent_data)

        # Record the attempted call *before* execution: the reward replays the
        # model's action list on a fresh environment regardless of runtime errors.
        _record_action(agent_data, self.name, arguments)

        fn = env["functions"].get(self.name)
        if fn is None:
            return ToolResponse(text=f"Error: unknown WorkBench tool '{self.name}'."), 0.0, {}

        try:
            result = fn(**arguments)
        except Exception as e:  # noqa: BLE001 - surface the error so the model can self-correct
            return ToolResponse(text=f"Error executing tool '{self.name}': {e}"), 0.0, {}

        if isinstance(result, (dict, list)):
            text = json.dumps(result, ensure_ascii=False)
        else:
            text = str(result)
        return ToolResponse(text=text), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        # Trajectory reward is computed by the reward manager from the recorded
        # action log (see workbench_verify), not per-tool.
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        # Environment lifetime is tied to AgentData, nothing to release per call.
        return None
