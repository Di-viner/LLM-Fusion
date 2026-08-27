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
"""Java / JavaScript value type converters.

Ported verbatim (aside from the import path) from
``bfcl_eval.eval_checker.ast_eval.type_convertor``. These are only exercised
when the AST checker is given already-decoded Java/JavaScript arguments; the
vendored decoder itself only handles Python, so in practice these are unused
unless a caller supplies decoded Java/JS output directly.
"""

import re
from typing import Dict, List, Union

from .constants import JAVA_TYPE_CONVERSION, JS_TYPE_CONVERSION


# ---------------------------------------------------------------------------
# Java
# ---------------------------------------------------------------------------
def java_type_converter(value, expected_type, nested_type=None):
    if expected_type not in JAVA_TYPE_CONVERSION:
        raise ValueError(f"Unsupported type: {expected_type}")
    if expected_type in ("byte", "short", "integer"):
        if not re.match(r"^-?\d+$", value):
            return str(value)
        return int(value)
    elif expected_type == "float":
        if not re.match(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?[fF]$", value):
            return str(value)
        return float(re.sub(r"[fF]$", "", value))
    elif expected_type == "double":
        if not re.match(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?$", value):
            return str(value)
        return float(value)
    elif expected_type == "long":
        if not re.match(r"^-?\d+[lL]$", value):
            return str(value)
        return int(re.sub(r"[lL]$", "", value))
    elif expected_type == "boolean":
        if value not in ["true", "false"]:
            return str(value)
        return value == "true"
    elif expected_type == "char":
        if not re.match(r"^\'.$\'", value):
            return str(value)
        return value
    elif expected_type in ("Array", "ArrayList"):
        return _parse_java_collection(value, expected_type, nested_type)
    elif expected_type == "HashMap":
        return _parse_java_collection(value, expected_type, nested_type)
    elif expected_type in ("Set", "Hashtable", "Queue", "Stack"):
        raise NotImplementedError(f"{expected_type} conversion is not implemented")
    elif expected_type in ("String", "any"):
        return str(value)
    else:
        raise ValueError(f"Unsupported type: {expected_type}")


def _parse_java_collection(input_str: str, type_str: str, nested_type=None) -> Union[List, Dict]:
    if type_str == "ArrayList":
        return _parse_arraylist(input_str, nested_type)
    elif type_str == "Array":
        return _parse_array(input_str, nested_type)
    elif type_str == "HashMap":
        return _parse_hashmap(input_str)
    else:
        raise ValueError(f"Unsupported type: {type_str}")


def _parse_arraylist(input_str: str, nested_type=None) -> List:
    match_as_list = re.search(r"new\s+ArrayList<\w*>\(Arrays\.asList\((.+?)\)\)", input_str)
    if match_as_list:
        elements = []
        for element_str in match_as_list.group(1).split(","):
            element_str = element_str.strip()
            if nested_type in ("char", "String"):
                element = element_str[1:-1]
            else:
                element = java_type_converter(element_str, nested_type) if nested_type else _parse_java_value(element_str)
            elements.append(element)
        return elements

    match_add = re.search(r"new\s+ArrayList<\w*>\(\)\s*\{\{\s*(.+?)\s*\}\}", input_str, re.DOTALL)
    if match_add:
        elements = []
        for match in re.findall(r"add\((.+?)\)", match_add.group(1)):
            value_str = match.strip()
            if nested_type in ("char", "String"):
                value = value_str[1:-1]
            else:
                value = java_type_converter(value_str, nested_type) if nested_type else _parse_java_value(value_str)
            elements.append(value)
        return elements

    if re.search(r"new\s+ArrayList<\w*>\(\)", input_str):
        return []
    return input_str


def _parse_array(input_str: str, nested_type=None) -> List:
    match = re.search(r"new\s+\w+\[\]\s*\{(.*?)\}", input_str)
    if match:
        elements_str = match.group(1)
        if nested_type:
            return [java_type_converter(x.strip(), nested_type) for x in elements_str.split(",") if x.strip()]
        return [_parse_java_value(x.strip()) for x in elements_str.split(",") if x.strip()]
    return input_str


def _parse_hashmap(input_str: str) -> Dict:
    elements: Dict = {}
    match = re.search(r"new\s+HashMap<.*?>\s*\(\)\s*\{\s*\{?\s*(.*?)\s*\}?\s*\}", input_str, re.DOTALL)
    if match:
        puts_str = match.group(1)
        if puts_str.strip():
            for m in re.findall(r"put\(\"(.*?)\",\s*(.*?)\)", puts_str):
                elements[m[0]] = _parse_java_value(m[1].strip())
        return elements
    if re.search(r"new\s+HashMap<.*?>\s*\(\)", input_str):
        return {}
    return input_str


def _parse_java_value(value_str: str):
    if value_str == "true":
        return True
    elif value_str == "false":
        return False
    elif value_str.startswith('"') and value_str.endswith('"'):
        return value_str[1:-1]
    elif re.match(r"^-?\d+[lL]$", value_str):
        return int(value_str[:-1])
    elif re.match(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?[fF]$", value_str):
        return float(re.sub(r"[fF]$", "", value_str))
    else:
        try:
            return int(value_str)
        except ValueError:
            try:
                return float(value_str)
            except ValueError:
                return value_str


# ---------------------------------------------------------------------------
# JavaScript
# ---------------------------------------------------------------------------
def js_type_converter(value, expected_type, nested_type=None):
    if expected_type not in JS_TYPE_CONVERSION:
        raise ValueError(f"Unsupported type: {expected_type}")

    if expected_type == "String":
        if not (value.startswith('"') and value.endswith('"')) and not (
            value.startswith("'") and value.endswith("'")
        ):
            return str(value)
        return value[1:-1]
    elif expected_type == "integer":
        if not re.match(r"^-?\d+$", value):
            return str(value)
        return int(value)
    elif expected_type == "float":
        if not re.match(r"^-?\d+(\.\d+)?$", value):
            return str(value)
        return float(value)
    elif expected_type == "Bigint":
        if not re.match(r"^-?\d+n$", value):
            return str(value)
        return int(value[:-1])
    elif expected_type == "Boolean":
        if value not in ["true", "false"]:
            return str(value)
        return value == "true"
    elif expected_type == "dict":
        return _parse_js_collection(value, "dict", nested_type)
    elif expected_type == "array":
        return _parse_js_collection(value, "array", nested_type)
    elif expected_type == "any":
        return str(value)
    else:
        raise ValueError(f"Unsupported type: {expected_type}")


def _parse_js_collection(code, type_str, nested_type=None):
    code = code.strip()
    if type_str == "array":
        array_2d_pattern = r"\[\s*\[.*?\]\s*(,\s*\[.*?\]\s*)*\]|\bnew\s+Array\(\s*\[.*?\]\s*(,\s*\[.*?\]\s*)*\)"
        array_pattern = r"\[(.*?)\]|\bnew\s+Array\((.*?)\)"
        array_2d_match = re.match(array_2d_pattern, code)
        try:
            if array_2d_match:
                elements_str = array_2d_match.group(0)
                inner_arrays = re.findall(r"\[(.*?)\]", elements_str)
                elements = []
                for idx, inner_array_str in enumerate(inner_arrays):
                    inner_array_str = inner_array_str.strip()
                    if idx == 0 and inner_array_str.startswith("["):
                        inner_array_str = inner_array_str[1:]
                    inner_array_elements = [e.strip() for e in inner_array_str.split(",")]
                    elements.append([_parse_js_value(e) for e in inner_array_elements])
                return elements

            array_match = re.match(array_pattern, code)
            if array_match:
                if array_match.group(1) is not None:
                    elements_str = array_match.group(1).strip()
                    elements = elements_str.split(",") if elements_str else []
                elif array_match.group(2) is not None:
                    elements_str = array_match.group(2).strip()
                    elements = elements_str.split(",") if elements_str else []
                else:
                    elements = []
                if nested_type:
                    elements = [
                        (
                            js_type_converter(e.strip(), nested_type, "String")
                            if (e.strip().startswith("'") or e.strip().startswith('"'))
                            else js_type_converter(e.strip(), nested_type)
                        )
                        for e in elements
                    ]
                else:
                    elements = [_parse_js_value(e.strip()) for e in elements]
                return elements
            return code
        except Exception:
            return code

    elif type_str == "dict":
        if code == "{}":
            return {}
        dict_match = re.match(r"\{(.*?)\}", code)
        if dict_match:
            try:
                content = dict_match.group(1)
                pairs = re.findall(r"([^:]+):\s*(.*?)(?:,\s*(?=[^,]+:)|$)", content)
                dictionary = {}
                for key, value in pairs:
                    key = key.strip().strip("'\"")
                    value = value.strip()
                    if value.startswith("[") and value.endswith("]"):
                        dictionary[key] = _parse_js_collection(value, "array")
                    elif value.startswith("{") and value.endswith("}"):
                        dictionary[key] = _parse_js_collection(value, "dict")
                    else:
                        dictionary[key] = _parse_js_value(value.strip("'\""))
                return dictionary
            except Exception:
                return code
        return code
    else:
        raise ValueError(f"Unsupported type: {type_str}")


def _parse_js_value(value_str: str):
    value_str = value_str.strip()
    if value_str == "true":
        return True
    elif value_str == "false":
        return False
    elif (value_str.startswith('"') and value_str.endswith('"')) or (
        value_str.startswith("'") and value_str.endswith("'")
    ):
        return value_str[1:-1]
    else:
        try:
            return int(value_str)
        except ValueError:
            try:
                return float(value_str)
            except ValueError:
                return value_str
