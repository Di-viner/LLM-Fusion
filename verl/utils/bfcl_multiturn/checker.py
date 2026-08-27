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
"""BFCL multi-turn checker (state + response based).

Ported from ``bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker`` with
the execution backend swapped for :mod:`verl.utils.bfcl_multiturn.executor`
(explicit per-trajectory instances instead of ``globals()`` caching).

``multi_turn_checker`` returns ``{"valid": bool, ...}``. A turn passes when:
  * the model produced a non-empty response whenever the ground truth did, and
  * every involved backend instance ends the turn in the same public state as
    the ground-truth instance, and
  * the cumulative model execution responses contain (unordered) all of the
    ground-truth execution responses for that turn.
"""

from .executor import execute_calls, is_empty_execute_response, make_instances


def multi_turn_checker(
    model_decoded: list,  # list[turn] of list[step] of list[str call]
    ground_truth: list,  # list[turn] of list[str call]
    initial_config: dict,
    involved_classes: list,
    long_context: bool = False,
) -> dict:
    model_instances = make_instances(involved_classes, initial_config, long_context)
    ground_truth_instances = make_instances(involved_classes, initial_config, long_context)

    all_turn_model_execution_results = []

    for turn_index, single_turn_ground_truth_list in enumerate(ground_truth):
        single_turn_model_response_list = (
            model_decoded[turn_index] if turn_index < len(model_decoded) else []
        )

        # Execute the model's calls (possibly multiple steps within the turn).
        single_turn_model_execution_results = []
        for single_step_model_response in single_turn_model_response_list:
            step_results = execute_calls(single_step_model_response, model_instances)
            single_turn_model_execution_results.extend(step_results)

        # Execute the ground-truth calls for this turn.
        single_turn_ground_truth_execution_results = execute_calls(
            single_turn_ground_truth_list, ground_truth_instances
        )

        all_turn_model_execution_results.extend(single_turn_model_execution_results)

        # If GT expects calls this turn, the model must produce a non-empty response.
        if len(single_turn_ground_truth_list) > 0:
            if not single_turn_model_response_list or is_empty_execute_response(
                single_turn_model_response_list
            ):
                return {
                    "valid": False,
                    "error_message": f"Model response list is empty for turn {turn_index}",
                    "error_type": "multi_turn:empty_turn_model_response",
                }

        # GT empty for this turn -> nothing to compare (irrelevance handled elsewhere).
        if not single_turn_ground_truth_list:
            continue

        state_check_result = state_checker(model_instances, ground_truth_instances)
        if not state_check_result["valid"]:
            return state_check_result

        response_check_result = response_checker(
            all_turn_model_execution_results,
            single_turn_ground_truth_execution_results,
            turn_index,
        )
        if not response_check_result["valid"]:
            return response_check_result

    return {"valid": True}


def multi_turn_irrelevance_checker(model_decoded: list, ground_truth: list) -> dict:
    """When GT is empty for a turn, the model must not emit valid calls."""
    for turn_index, single_turn_ground_truth_list in enumerate(ground_truth):
        single_turn_model_response_list = (
            model_decoded[turn_index] if turn_index < len(model_decoded) else []
        )
        if len(single_turn_ground_truth_list) == 0:
            if is_empty_execute_response(single_turn_model_response_list):
                continue
            return {
                "valid": False,
                "error_message": f"Model outputs valid function calls when it should not for turn {turn_index}.",
                "error_type": "multi_turn:irrelevance_error:decoder_success",
            }
    return {"valid": True}


#### Sub-checkers ####


def state_checker(model_instances: dict, ground_truth_instances: dict) -> dict:
    for class_name, ground_truth_instance in ground_truth_instances.items():
        model_instance = model_instances[class_name]
        valid, differences = _compare_instances(model_instance, ground_truth_instance)
        if not valid:
            return {
                "valid": False,
                "error_message": f"Model instance for {class_name} does not match the state with ground truth instance.",
                "error_type": "multi_turn:instance_state_mismatch",
                "details": {"differences": differences},
            }
    return {"valid": True}


def response_checker(model_response_list: list, ground_truth_response_list: list, turn_index: int) -> dict:
    is_subsequence, missing_items = _is_subsequence_unordered(
        ground_truth_response_list, model_response_list
    )
    if not is_subsequence:
        return {
            "valid": False,
            "error_message": f"Model response execution results so far does not contain all the ground truth response execution results for turn {turn_index}.",
            "error_type": "multi_turn:execution_response_mismatch",
            "details": {"missing_items": missing_items},
        }
    return {"valid": True}


#### Helpers ####


def _compare_instances(model_object, ground_truth_object) -> tuple[bool, dict]:
    assert type(model_object) == type(ground_truth_object), "Objects are not of the same type."
    differences = {}
    valid = True
    for attr_name in vars(ground_truth_object):
        if attr_name.startswith("_"):
            continue
        model_attr = getattr(model_object, attr_name)
        ground_truth_attr = getattr(ground_truth_object, attr_name)
        if model_attr != ground_truth_attr:
            valid = False
            differences[attr_name] = {"model": model_attr, "ground_truth": ground_truth_attr}
    return valid, differences


def _is_subsequence_unordered(list1, list2) -> tuple[bool, list]:
    list2_copy = list2[:]
    missing_elements = []
    for item in list1:
        try:
            list2_copy.remove(item)
        except ValueError:
            missing_elements.append(item)
    return len(missing_elements) == 0, missing_elements
