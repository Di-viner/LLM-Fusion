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
"""Python function-call parser for BFCL prompting-mode outputs.

Ported from ``bfcl_eval.model_handler.utils`` (``ast_parse``,
``resolve_ast_call``, ``resolve_ast_by_type``, ``default_decode_ast_prompting``)
and ``bfcl_eval.utils`` (``is_function_calling_format_output``,
``is_empty_output``). Only the Python language path is supported.
"""

import ast


def resolve_ast_by_type(value):
    if isinstance(value, ast.Constant):
        output = "..." if value.value is Ellipsis else value.value
    elif isinstance(value, ast.UnaryOp):
        output = -value.operand.value
    elif isinstance(value, ast.List):
        output = [resolve_ast_by_type(v) for v in value.elts]
    elif isinstance(value, ast.Dict):
        output = {resolve_ast_by_type(k): resolve_ast_by_type(v) for k, v in zip(value.keys, value.values)}
    elif isinstance(value, ast.NameConstant):  # boolean values
        output = value.value
    elif isinstance(value, ast.BinOp):  # function calls as arguments
        output = eval(ast.unparse(value))
    elif isinstance(value, ast.Name):
        output = value.id
    elif isinstance(value, ast.Call):
        if len(value.keywords) == 0:
            output = ast.unparse(value)
        else:
            output = resolve_ast_call(value)
    elif isinstance(value, ast.Tuple):
        output = tuple(resolve_ast_by_type(v) for v in value.elts)
    elif isinstance(value, ast.Lambda):
        output = eval(ast.unparse(value.body[0].value))
    elif isinstance(value, ast.Ellipsis):
        output = "..."
    elif isinstance(value, ast.Subscript):
        try:
            output = ast.unparse(value.body[0].value)
        except Exception:
            output = ast.unparse(value.value) + "[" + ast.unparse(value.slice) + "]"
    else:
        raise Exception(f"Unsupported AST type: {type(value)}")
    return output


def resolve_ast_call(elem):
    # Handle nested attributes for deeply nested module paths (e.g. a.b.c(...)).
    func_parts = []
    func_part = elem.func
    while isinstance(func_part, ast.Attribute):
        func_parts.append(func_part.attr)
        func_part = func_part.value
    if isinstance(func_part, ast.Name):
        func_parts.append(func_part.id)
    func_name = ".".join(reversed(func_parts))
    args_dict = {}
    for arg in elem.keywords:
        args_dict[arg.arg] = resolve_ast_by_type(arg.value)
    return {func_name: args_dict}


def ast_parse(input_str: str) -> list[dict]:
    """Parse a Python-style function-call string into ``[{name: {param: value}}]``."""
    cleaned_input = input_str.strip().strip("'")
    parsed = ast.parse(cleaned_input, mode="eval")
    extracted = []
    if isinstance(parsed.body, ast.Call):
        extracted.append(resolve_ast_call(parsed.body))
    else:
        for elem in parsed.body.elts:
            assert isinstance(elem, ast.Call)
            extracted.append(resolve_ast_call(elem))
    return extracted


def decode_ast(result: str) -> list[dict]:
    """Decode a prompting-mode model response into the BFCL function-call list.

    Mirrors ``default_decode_ast_prompting``: wraps the response in brackets if
    needed and parses it as a Python list of calls. Raises on invalid syntax.
    """
    result = result.strip("`\n ")
    if not result.startswith("["):
        result = "[" + result
    if not result.endswith("]"):
        result = result + "]"
    return ast_parse(result)


def is_function_calling_format_output(decoded_output) -> bool:
    """Ensure the output is ``[{func: {param: val, ...}}, ...]`` (empty list is OK)."""
    if not isinstance(decoded_output, list):
        return False
    for item in decoded_output:
        if not isinstance(item, dict):
            return False
        if len(item) != 1:
            return False
        if not isinstance(list(item.values())[0], dict):
            return False
    return True


def is_empty_output(decoded_output) -> bool:
    """``[]``, ``[{}]`` or any non-function-calling output is considered empty."""
    if not is_function_calling_format_output(decoded_output):
        return True
    if len(decoded_output) == 0:
        return True
    if len(decoded_output) == 1 and len(decoded_output[0]) == 0:
        return True
    return False
