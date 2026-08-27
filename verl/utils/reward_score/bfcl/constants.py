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
"""Constants and prompt builder for the vendored BFCL v3 checker.

Ported from ``bfcl_eval.constants`` (type mappings) and
``bfcl_eval.constants.default_prompts`` (the classic default system prompt used
for prompting-mode evaluation in BFCL v3).
"""

import json

# ---------------------------------------------------------------------------
# Type conversion tables (bfcl_eval.constants.type_mappings)
# ---------------------------------------------------------------------------

PYTHON_TYPE_MAPPING = {
    "string": str,
    "integer": int,
    "float": float,
    "boolean": bool,
    "array": list,
    "tuple": list,
    "dict": dict,
    "any": str,
}

# Types whose values we recursively check one level deep.
PYTHON_NESTED_TYPE_CHECK_LIST = ["array", "tuple"]

NESTED_CONVERSION_TYPE_LIST = ["Array", "ArrayList", "array"]

JAVA_TYPE_CONVERSION = {
    "byte": int,
    "short": int,
    "integer": int,
    "float": float,
    "double": float,
    "long": int,
    "boolean": bool,
    "char": str,
    "Array": list,
    "ArrayList": list,
    "Set": set,
    "HashMap": dict,
    "Hashtable": dict,
    "Queue": list,
    "Stack": list,
    "String": str,
    "any": str,
}

JS_TYPE_CONVERSION = {
    "String": str,
    "integer": int,
    "float": float,
    "Bigint": int,
    "Boolean": bool,
    "dict": dict,
    "array": list,
    "any": str,
}


# ---------------------------------------------------------------------------
# Default system prompt (BFCL v3 "classic" prompting mode)
#
# This is the ``_DEFAULT_SYSTEM_PROMPT`` reference template shipped in
# ``bfcl_eval.constants.default_prompts``. The model is instructed to emit a
# Python list of function calls, e.g. ``[func_name(a=1, b="x")]``.
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function calls in your response.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.

At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.

Here is a list of functions in JSON format that you can invoke.
{functions}
"""


def _get_language_specific_hint(test_category: str) -> str:
    """Language hint appended to each function description (bfcl_eval.utils)."""
    if "java" in test_category:
        return " Note that the provided function is in Java 8 SDK syntax."
    elif "javascript" in test_category:
        return " Note that the provided function is in JavaScript syntax."
    else:
        return " Note that the provided function is in Python 3 syntax."


def func_doc_language_specific_pre_processing(functions: list[dict], test_category: str) -> list[dict]:
    """Append the language-specific hint to each function description.

    For Python categories this is the only preprocessing that affects the
    prompt. Java/JavaScript parameter-type rewriting is intentionally omitted
    because those categories are out of scope for this vendored checker.
    """
    hint = _get_language_specific_hint(test_category)
    for item in functions:
        if "description" in item:
            item["description"] = item["description"] + hint
    return functions


def build_bfcl_system_prompt(functions: list[dict], test_category: str = "simple") -> str:
    """Build the BFCL prompting-mode system prompt for a set of function docs."""
    processed = func_doc_language_specific_pre_processing(
        json.loads(json.dumps(functions)), test_category
    )
    return DEFAULT_SYSTEM_PROMPT.format(functions=json.dumps(processed))
