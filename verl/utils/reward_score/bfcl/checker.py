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
"""BFCL AST checker (single-turn).

Ported from ``bfcl_eval.eval_checker.ast_eval.ast_checker`` with two
simplifications that do not change the scoring semantics for the supported
categories:

  * ``language`` is a plain string ("python" / "java" / "javascript") instead of
    the upstream ``Language`` enum.
  * The model-specific ``convert_func_name`` (dot/underscore rewriting for
    OpenAI/Mistral/Google function-calling APIs) is dropped, since prompting
    mode keeps the original function names.
"""

import re

from .constants import (
    JAVA_TYPE_CONVERSION,
    JS_TYPE_CONVERSION,
    NESTED_CONVERSION_TYPE_LIST,
    PYTHON_NESTED_TYPE_CHECK_LIST,
    PYTHON_TYPE_MAPPING,
)
from .type_converters import java_type_converter, js_type_converter


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def ast_checker(func_description, model_output, possible_answer, language: str, test_category: str):
    if "parallel" in test_category:
        return parallel_function_checker_no_order(func_description, model_output, possible_answer, language)
    elif "multiple" in test_category:
        return multiple_function_checker(func_description, model_output, possible_answer, language)
    else:
        if len(model_output) != 1:
            return {
                "valid": False,
                "error": ["Wrong number of functions."],
                "error_type": "simple_function_checker:wrong_count",
            }
        return simple_function_checker(func_description[0], model_output[0], possible_answer[0], language)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_description(func_descriptions, name):
    if isinstance(func_descriptions, list):
        for func_description in func_descriptions:
            if func_description["name"] == name:
                return func_description
        return None
    return func_descriptions


def get_possible_answer_type(possible_answer: list):
    for answer in possible_answer:
        if answer != "":  # Optional parameter
            return type(answer)
    return None


def type_checker(param, value, possible_answer, expected_type_description, expected_type_converted, nested_type_converted):
    # Only supports nested type checking one level deep (matches upstream).
    result = {"valid": True, "error": [], "is_variable": False, "error_type": "type_error:simple"}

    is_variable = False
    possible_answer_type = get_possible_answer_type(possible_answer)
    if possible_answer_type is not None:
        if possible_answer_type != expected_type_converted:
            is_variable = True

    if type(value) == expected_type_converted:
        if nested_type_converted is None:
            result["is_variable"] = is_variable
            return result
        else:
            for possible_answer_item in possible_answer:
                flag = True
                if type(possible_answer_item) == list:
                    for value_item in value:
                        checker_result = type_checker(
                            param, value_item, possible_answer_item, str(nested_type_converted), nested_type_converted, None
                        )
                        if not checker_result["valid"]:
                            flag = False
                            break
                if flag:
                    return {"valid": True, "error": [], "is_variable": is_variable}

            result["valid"] = False
            result["error"] = [
                f"Nested type checking failed for parameter {repr(param)}. Expected outer type "
                f"{expected_type_description} with inner type {str(nested_type_converted)}. "
                f"Parameter value: {repr(value)}."
            ]
            result["error_type"] = "type_error:nested"

    possible_answer_type = get_possible_answer_type(possible_answer)
    if possible_answer_type is not None:
        if type(value) == possible_answer_type:
            result["is_variable"] = True
            return result

    result["valid"] = False
    result["error"].append(
        f"Incorrect type for parameter {repr(param)}. Expected type {expected_type_description}, "
        f"got {type(value).__name__}. Parameter value: {repr(value)}."
    )
    result["error_type"] = "type_error:simple"
    return result


def standardize_string(input_string: str):
    regex_string = r"[ \,\.\/\-\_\*\^]"
    return re.sub(regex_string, "", input_string).lower().replace("'", '"')


def string_checker(param, model_output, possible_answer):
    standardize_possible_answer = []
    standardize_model_output = standardize_string(model_output)
    for i in range(len(possible_answer)):
        if type(possible_answer[i]) == str:
            standardize_possible_answer.append(standardize_string(possible_answer[i]))

    if standardize_model_output not in standardize_possible_answer:
        return {
            "valid": False,
            "error": [
                f"Invalid value for parameter {repr(param)}: {repr(model_output)}. "
                f"Expected one of {possible_answer}. Case insensitive."
            ],
            "error_type": "value_error:string",
        }
    return {"valid": True, "error": []}


def list_checker(param, model_output, possible_answer):
    standardize_model_output = list(model_output)
    for i in range(len(standardize_model_output)):
        if type(standardize_model_output[i]) == str:
            standardize_model_output[i] = standardize_string(model_output[i])

    standardize_possible_answer = []
    for i in range(len(possible_answer)):
        standardize_possible_answer.append([])
        for j in range(len(possible_answer[i])):
            if type(possible_answer[i][j]) == str:
                standardize_possible_answer[i].append(standardize_string(possible_answer[i][j]))
            else:
                standardize_possible_answer[i].append(possible_answer[i][j])

    if standardize_model_output not in standardize_possible_answer:
        return {
            "valid": False,
            "error": [
                f"Invalid value for parameter {repr(param)}: {repr(model_output)}. Expected one of {possible_answer}."
            ],
            "error_type": "value_error:list/tuple",
        }
    return {"valid": True, "error": []}


def dict_checker(param, model_output, possible_answers):
    result = {"valid": False, "error": [], "error_type": "dict_checker:unclear"}
    for i in range(len(possible_answers)):
        if possible_answers[i] == "":
            continue

        result = {"valid": False, "error": [], "error_type": "dict_checker:unclear"}
        flag = True
        possible_answer = possible_answers[i]

        for key, value in model_output.items():
            if key not in possible_answer:
                result["valid"] = False
                result["error"].append(f"Unexpected dict key parameter: '{key}'.")
                result["error_type"] = "value_error:dict_key"
                flag = False
                break

            standardize_value = value
            if type(value) == str:
                standardize_value = standardize_string(value)

            standardize_possible_answer = []
            for j in range(len(possible_answer[key])):
                if type(possible_answer[key][j]) == str:
                    standardize_possible_answer.append(standardize_string(possible_answer[key][j]))
                else:
                    standardize_possible_answer.append(possible_answer[key][j])

            if standardize_value not in standardize_possible_answer:
                result["valid"] = False
                result["error"].append(
                    f"Invalid value for parameter {repr(key)}: {repr(value)}. "
                    f"Expected one of {standardize_possible_answer}."
                )
                result["error_type"] = "value_error:dict_value"
                flag = False
                break

        for key, value in possible_answer.items():
            if key not in model_output and "" not in value:
                result["valid"] = False
                result["error"].append(f"Missing dict key parameter: '{key}'.")
                result["error_type"] = "value_error:dict_key"
                flag = False
                break

        if flag:
            return {"valid": True, "error": []}

    return result


def list_dict_checker(param, model_output, possible_answers):
    result = {"valid": False, "error": [], "error_type": "list_dict_checker:unclear"}
    for answer_index in range(len(possible_answers)):
        flag = True
        if len(model_output) != len(possible_answers[answer_index]):
            result["valid"] = False
            result["error"] = ["Wrong number of dictionaries in the list."]
            result["error_type"] = "value_error:list_dict_count"
            flag = False
            continue

        for dict_index in range(len(model_output)):
            result = dict_checker(param, model_output[dict_index], [possible_answers[answer_index][dict_index]])
            if not result["valid"]:
                flag = False
                break
        if flag:
            return {"valid": True, "error": []}

    return result


def simple_function_checker(func_description, model_output, possible_answer, language: str):
    possible_answer = list(possible_answer.values())[0]
    func_name = func_description["name"]
    param_details = func_description["parameters"]["properties"]
    required_params = func_description["parameters"]["required"]

    result = {"valid": True, "error": [], "error_type": "simple_function_checker:unclear"}

    if func_name not in model_output:
        result["valid"] = False
        result["error"].append(f"Function name {repr(func_name)} not found in model output.")
        result["error_type"] = "simple_function_checker:wrong_func_name"
        return result

    model_params = model_output[func_name]

    for param in required_params:
        if param not in model_params:
            result["valid"] = False
            result["error"].append(f"Missing required parameter: {repr(param)}.")
            result["error_type"] = "simple_function_checker:missing_required"
            return result

    for param, value in model_params.items():
        if param not in param_details or param not in possible_answer:
            result["valid"] = False
            result["error"].append(f"Unexpected parameter: {repr(param)}.")
            result["error_type"] = "simple_function_checker:unexpected_param"
            return result

        full_param_details = param_details[param]
        expected_type_description = full_param_details["type"]
        is_variable = False
        nested_type_converted = None

        if language == "java":
            expected_type_converted = JAVA_TYPE_CONVERSION[expected_type_description]
            if expected_type_description in JAVA_TYPE_CONVERSION:
                if type(value) != str:
                    result["valid"] = False
                    result["error"].append(
                        f"Incorrect type for parameter {repr(param)}. Expected type String, "
                        f"got {type(value).__name__}. Parameter value: {repr(value)}."
                    )
                    result["error_type"] = "type_error:java"
                    return result
                if expected_type_description in NESTED_CONVERSION_TYPE_LIST:
                    nested_type = param_details[param]["items"]["type"]
                    nested_type_converted = JAVA_TYPE_CONVERSION[nested_type]
                    value = java_type_converter(value, expected_type_description, nested_type)
                else:
                    value = java_type_converter(value, expected_type_description)

        elif language == "javascript":
            expected_type_converted = JS_TYPE_CONVERSION[expected_type_description]
            if expected_type_description in JS_TYPE_CONVERSION:
                if type(value) != str:
                    result["valid"] = False
                    result["error"].append(
                        f"Incorrect type for parameter {repr(param)}. Expected type String, "
                        f"got {type(value).__name__}. Parameter value: {repr(value)}."
                    )
                    result["error_type"] = "type_error:js"
                    return result
                if expected_type_description in NESTED_CONVERSION_TYPE_LIST:
                    nested_type = param_details[param]["items"]["type"]
                    nested_type_converted = JS_TYPE_CONVERSION[nested_type]
                    value = js_type_converter(value, expected_type_description, nested_type)
                else:
                    value = js_type_converter(value, expected_type_description)

        elif language == "python":
            expected_type_converted = PYTHON_TYPE_MAPPING[expected_type_description]
            if expected_type_description in PYTHON_NESTED_TYPE_CHECK_LIST:
                nested_type = param_details[param]["items"]["type"]
                nested_type_converted = PYTHON_TYPE_MAPPING[nested_type]
        else:
            raise ValueError(f"Unsupported language: {language}")

        # tuple in possible answers becomes list after json round-trip
        if expected_type_description == "tuple" and type(value) == tuple:
            value = list(value)

        # allow python auto conversion from int to float
        if language == "python" and expected_type_description == "float" and type(value) == int:
            value = float(value)

        type_check_result = type_checker(
            param, value, possible_answer[param], expected_type_description, expected_type_converted, nested_type_converted
        )
        is_variable = type_check_result["is_variable"]
        if not type_check_result["valid"]:
            return type_check_result

        if not is_variable:
            if expected_type_converted == dict:
                result = dict_checker(param, value, possible_answer[param])
                if not result["valid"]:
                    return result
                continue
            elif expected_type_converted == list and nested_type_converted == dict:
                result = list_dict_checker(param, value, possible_answer[param])
                if not result["valid"]:
                    return result
                continue
            elif expected_type_converted == str:
                result = string_checker(param, value, possible_answer[param])
                if not result["valid"]:
                    return result
                continue
            elif expected_type_converted == list:
                result = list_checker(param, value, possible_answer[param])
                if not result["valid"]:
                    return result
                continue

        if value not in possible_answer[param]:
            result["valid"] = False
            result["error"].append(
                f"Invalid value for parameter {repr(param)}: {repr(value)}. Expected one of {possible_answer[param]}."
            )
            result["error_type"] = "value_error:others"
            return result

    for param in possible_answer:
        if param not in model_params and "" not in possible_answer[param]:
            result["valid"] = False
            result["error"].append(
                f"Optional parameter {repr(param)} not provided and not marked as optional."
            )
            result["error_type"] = "simple_function_checker:missing_optional"
            return result

    return result


def parallel_function_checker_no_order(func_descriptions, model_output, possible_answers, language: str):
    if len(model_output) != len(possible_answers):
        return {
            "valid": False,
            "error": ["Wrong number of functions."],
            "error_type": "parallel_function_checker_no_order:wrong_count",
        }

    matched_indices = []
    for i in range(len(possible_answers)):
        func_name_expected = list(possible_answers[i].keys())[0]
        func_description = find_description(func_descriptions, func_name_expected)

        all_errors = []
        result = {"valid": False}
        for index in range(len(model_output)):
            if index in matched_indices:
                continue
            result = simple_function_checker(func_description, model_output[index], possible_answers[i], language)
            if result["valid"]:
                matched_indices.append(index)
                break
            else:
                all_errors.append(
                    {
                        f"Model Result Index {index}": {
                            "sub_error": result["error"],
                            "sub_error_type": result["error_type"],
                            "model_output_item": model_output[index],
                            "possible_answer_item": possible_answers[i],
                        }
                    }
                )

        if not result["valid"]:
            considered_indices = [idx for idx in range(len(model_output)) if idx not in matched_indices]
            all_errors.insert(
                0,
                f"Could not find a matching function among index {considered_indices} of model output "
                f"for index {i} of possible answers.",
            )
            return {
                "valid": False,
                "error": all_errors,
                "error_type": "parallel_function_checker_no_order:cannot_find_match",
            }

    return {"valid": True, "error": []}


def multiple_function_checker(func_descriptions, model_output, possible_answers, language: str):
    if len(model_output) != len(possible_answers):
        return {
            "valid": False,
            "error": ["Wrong number of functions."],
            "error_type": "multiple_function_checker:wrong_count",
        }

    func_name_expected = list(possible_answers[0].keys())[0]
    func_description = find_description(func_descriptions, func_name_expected)
    return simple_function_checker(func_description, model_output[0], possible_answers[0], language)
