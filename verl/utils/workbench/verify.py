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
"""State-based WorkBench verifier.

Ported from the NVIDIA nemo-gym ``workplace_assistant`` resource server
(``utils.py``). The reward is *outcome* based: replay the model's
executed tool calls and the ground-truth tool calls on two freshly seeded
environments, then check that every workplace database ends in the same state.

The heavy toolkit code (and its CSV seed data) is vendored under this package so
both the training-time rollout tools (``verl.tools.workbench_tool``) and the
reward path (``verl.utils.reward_score.workbench_verify``) share one source of
truth and stay independent of the external nemo-gym checkout.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from verl.utils.workbench.workplace_assistant_tools.analytics import AnalyticsTool
from verl.utils.workbench.workplace_assistant_tools.calendar import CalendarTool
from verl.utils.workbench.workplace_assistant_tools.company_directory import CompanyDirectoryTool
from verl.utils.workbench.workplace_assistant_tools.customer_relationship_manager import (
    CustomerRelationshipManagerTool,
)
from verl.utils.workbench.workplace_assistant_tools.email import EmailTool
from verl.utils.workbench.workplace_assistant_tools.project_management import ProjectManagementTool

logger = logging.getLogger(__name__)

# Toolkits enabled for every WorkBench session (mirrors nemo-gym ``seed_session``).
DEFAULT_TOOLKITS = (
    "email",
    "calendar",
    "analytics",
    "project_management",
    "customer_relationship_manager",
)


def get_tools(toolkits: tuple[str, ...] | list[str] = DEFAULT_TOOLKITS) -> dict[str, Any]:
    """Build a fresh WorkBench tool environment.

    Returns a dict with ``containers`` (stateful tool objects keyed by toolkit),
    ``functions`` (callable name -> bound method) and ``schemas`` (OpenAI-style
    tool schemas). The company directory is always available.
    """
    tool_env: dict[str, Any] = {"containers": {}, "functions": {}, "schemas": []}

    company_directory = CompanyDirectoryTool()
    tool_env["containers"]["company_directory"] = company_directory
    tool_env["functions"]["company_directory_find_email_address"] = company_directory.find_email_address

    if "email" in toolkits:
        email = EmailTool()
        tool_env["containers"]["email"] = email
        tool_env["functions"]["email_get_email_information_by_id"] = email.get_email_information_by_id
        tool_env["functions"]["email_search_emails"] = email.search_emails
        tool_env["functions"]["email_send_email"] = email.send_email
        tool_env["functions"]["email_delete_email"] = email.delete_email
        tool_env["functions"]["email_forward_email"] = email.forward_email
        tool_env["functions"]["email_reply_email"] = email.reply_email

    if "calendar" in toolkits:
        calendar = CalendarTool()
        tool_env["containers"]["calendar"] = calendar
        tool_env["functions"]["calendar_get_event_information_by_id"] = calendar.get_event_information_by_id
        tool_env["functions"]["calendar_search_events"] = calendar.search_events
        tool_env["functions"]["calendar_create_event"] = calendar.create_event
        tool_env["functions"]["calendar_delete_event"] = calendar.delete_event
        tool_env["functions"]["calendar_update_event"] = calendar.update_event

    if "analytics" in toolkits:
        analytics = AnalyticsTool()
        tool_env["containers"]["analytics"] = analytics
        tool_env["functions"]["analytics_engaged_users_count"] = analytics.engaged_users_count
        tool_env["functions"]["analytics_get_visitor_information_by_id"] = analytics.get_visitor_information_by_id
        tool_env["functions"]["analytics_create_plot"] = analytics.create_plot
        tool_env["functions"]["analytics_traffic_source_count"] = analytics.traffic_source_count
        tool_env["functions"]["analytics_total_visits_count"] = analytics.total_visits_count
        tool_env["functions"]["analytics_get_average_session_duration"] = analytics.get_average_session_duration

    if "project_management" in toolkits:
        project_management = ProjectManagementTool()
        tool_env["containers"]["project_management"] = project_management
        tool_env["functions"]["project_management_get_task_information_by_id"] = (
            project_management.get_task_information_by_id
        )
        tool_env["functions"]["project_management_search_tasks"] = project_management.search_tasks
        tool_env["functions"]["project_management_create_task"] = project_management.create_task
        tool_env["functions"]["project_management_delete_task"] = project_management.delete_task
        tool_env["functions"]["project_management_update_task"] = project_management.update_task

    if "customer_relationship_manager" in toolkits:
        crm = CustomerRelationshipManagerTool()
        tool_env["containers"]["customer_relationship_manager"] = crm
        tool_env["functions"]["customer_relationship_manager_search_customers"] = crm.search_customers
        tool_env["functions"]["customer_relationship_manager_update_customer"] = crm.update_customer
        tool_env["functions"]["customer_relationship_manager_add_customer"] = crm.add_customer
        tool_env["functions"]["customer_relationship_manager_delete_customer"] = crm.delete_customer

    return tool_env


def _coerce_arguments(arguments: Any) -> dict[str, Any]:
    """Normalize a tool call's ``arguments`` into a kwargs dict."""
    if arguments is None:
        return {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            return {}
    if not isinstance(arguments, dict):
        return {}
    return arguments


def execute_actions_and_reset_state(actions: list[dict[str, Any]]) -> dict[str, Any]:
    """Replay ``actions`` on a fresh tool environment and return it.

    Errors from individual tool calls are swallowed (matching nemo-gym): a
    malformed prediction simply fails to mutate state and will therefore differ
    from the ground-truth state.
    """
    tool_env = get_tools(DEFAULT_TOOLKITS)
    for action in actions or []:
        try:
            name = action["name"]
            arguments = _coerce_arguments(action.get("arguments"))
            tool_env["functions"][name](**arguments)
        except Exception as e:  # noqa: BLE001 - a bad predicted call must not crash scoring
            logger.debug("WorkBench replay skipped action %s: %s", action, e)
            continue
    return tool_env


def _lowercase_state(df):
    """Case-insensitive comparison for most string columns (mirrors nemo-gym)."""
    fields_not_to_convert = ["status", "list_name", "board"]
    for col in df.columns:
        if col not in fields_not_to_convert:
            df[col] = df[col].str.lower()
    return df


def is_correct(predicted_actions: list[dict[str, Any]], ground_truth_actions: list[dict[str, Any]]) -> bool:
    """Return True iff predicted actions reproduce the ground-truth end state."""
    predict_env = execute_actions_and_reset_state(predicted_actions)
    ground_truth_env = execute_actions_and_reset_state(ground_truth_actions)

    checks = [
        ("calendar", "_calendar_events"),
        ("email", "_emails"),
        ("analytics", "_plots_data"),
        ("project_management", "_project_tasks"),
        ("customer_relationship_manager", "_crm_data"),
    ]
    for container_name, state_attr in checks:
        predicted_state = _lowercase_state(getattr(predict_env["containers"][container_name], state_attr))
        ground_truth_state = _lowercase_state(getattr(ground_truth_env["containers"][container_name], state_attr))
        if not predicted_state.equals(ground_truth_state):
            return False
    return True


def compute_score(predicted_actions: list[dict[str, Any]], ground_truth_actions: list[dict[str, Any]]) -> float:
    """Binary WorkBench reward: 1.0 when the end states match, else 0.0."""
    try:
        return 1.0 if is_correct(predicted_actions, ground_truth_actions) else 0.0
    except Exception as e:  # noqa: BLE001 - never crash the training loop on a bad sample
        logger.warning("WorkBench verify failed: %s", e)
        return 0.0
