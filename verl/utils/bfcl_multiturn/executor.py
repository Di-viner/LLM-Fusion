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
"""Instantiate and execute BFCL multi-turn stateful backends.

Adapted from ``bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils``. The
upstream code caches instances in ``globals()`` keyed by model/test id to keep
state across turns; here we instead keep an explicit per-trajectory ``instances``
dict so the same logic works without global state (safe for concurrent scoring
and for reuse inside a verl tool).
"""

import copy
import inspect
import json
import re

from .func_source_code.gorilla_file_system import GorillaFileSystem
from .func_source_code.math_api import MathAPI
from .func_source_code.message_api import MessageAPI
from .func_source_code.posting_api import TwitterAPI
from .func_source_code.ticket_api import TicketAPI
from .func_source_code.trading_bot import TradingBot
from .func_source_code.travel_booking import TravelAPI
from .func_source_code.vehicle_control import VehicleControlAPI

# BFCL class name -> backend class object.
CLASS_MAPPING = {
    "GorillaFileSystem": GorillaFileSystem,
    "MathAPI": MathAPI,
    "MessageAPI": MessageAPI,
    "TwitterAPI": TwitterAPI,
    "TicketAPI": TicketAPI,
    "TradingBot": TradingBot,
    "TravelAPI": TravelAPI,
    "VehicleControlAPI": VehicleControlAPI,
}

# BFCL class name -> the JSON func-doc file name under ``multi_turn_func_doc/``.
FUNC_DOC_FILE_MAPPING = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
}

# Stateless classes take no initial configuration / scenario seeding.
STATELESS_CLASSES = ["MathAPI"]

# Disallow obviously dangerous calls (matches upstream guard).
_FORBIDDEN_CALLS = {"kill", "exit", "quit", "remove", "unlink", "popen", "Popen", "run"}


def make_instances(involved_classes, initial_config, long_context=False):
    """Instantiate and seed the involved backend classes for one trajectory."""
    instances = {}
    for class_name in involved_classes:
        if class_name not in CLASS_MAPPING:
            raise ValueError(f"Unsupported BFCL multi-turn class: {class_name}")
        instance = CLASS_MAPPING[class_name]()
        if class_name not in STATELESS_CLASSES:
            class_initial_config = (initial_config or {}).get(class_name, {})
            instance._load_scenario(copy.deepcopy(class_initial_config), long_context=long_context)
        instances[class_name] = instance
    return instances


def build_method_mapping(instances):
    """Map every public method name to its owning class name."""
    mapping = {}
    for class_name, instance in instances.items():
        for method_name, _ in inspect.getmembers(instance, predicate=inspect.ismethod):
            if method_name.startswith("_"):
                continue
            mapping[method_name] = class_name
    return mapping


def _process_method_calls(function_call_string: str, method_mapping: dict) -> str:
    """Prepend the owning class name to each bare method name in the call string."""

    def replace_function(match):
        func_name = match.group(1)
        if func_name in method_mapping:
            return f"{method_mapping[func_name]}.{func_name}"
        return func_name

    pattern = r"\b([a-zA-Z_]\w*)\s*(?=\()"
    return re.sub(pattern, replace_function, function_call_string)


def _stringify(result):
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        try:
            return json.dumps(result)
        except Exception:
            return str(result)
    return str(result)


def execute_calls(func_call_list, instances):
    """Execute a list of call strings against ``instances`` (mutated in place).

    Returns the list of stringified execution results, matching upstream
    ``execute_multi_turn_func_call`` semantics.
    """
    method_mapping = build_method_mapping(instances)
    namespace = dict(instances)  # class_name -> instance, used as eval locals
    execution_results = []
    for func_call in func_call_list:
        processed = _process_method_calls(func_call, method_mapping)
        try:
            head = processed.split("(")[0] if "(" in processed else processed
            if "." in head:
                head = head.split(".")[1]
            if head in _FORBIDDEN_CALLS:
                raise Exception(f"Function call {head} is not allowed.")
            result = eval(processed, {"__builtins__": __builtins__}, namespace)
            execution_results.append(_stringify(result))
        except Exception as e:
            execution_results.append(f"Error during execution: {str(e)}")
    return execution_results


def is_empty_execute_response(input_list: list) -> bool:
    if len(input_list) == 0:
        return True
    if len(input_list) == 1 and len(input_list[0]) == 0:
        return True
    return False
