# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import json
import logging
import random
import traceback

from .utils import check_correctness

"""
Verify code correctness using the Sandbox Fusion (https://github.com/bytedance/SandboxFusion).
You can either deploy the sandbox_fusion service yourself or use the
FaaS service provided by public cloud, eg: volcengine.com.
"""
logger = logging.getLogger(__name__)

# When continuous=True, the score is computed over (at most) N test cases. When there
# are more than N cases we randomly sample N of them,
# and only execute those N so we don't waste sandbox calls on cases that never
# contribute to the score.
CONTINUOUS_SCORE_NUM_CASES = 10


def compute_score(
    sandbox_fusion_url,
    concurrent_semaphore,
    memory_limit_mb,
    completion,
    test_cases,
    continuous=False,
    timeout=10,
    pass_all=False,
):
    """
    Computes the code score using the remote sandbox API.

    Args:
        sandbox_fusion_url: The URL of the sandbox_fusion service, eg: "https://<your service endpoint>/run_code"

        completion: The completion string containing the code.
        test_cases: JSON string or dictionary containing "inputs" and "outputs".
        continuous: Whether to restrict scoring to (at most) N randomly sampled test cases.
        timeout: Timeout for each test case.
        pass_all: When True, the score is binary (1.0 only if every evaluated test case
            passes, else 0.0). When False, the
            score is the pass rate over the evaluated cases.

    Returns:
        A tuple (score, metadata_list).
        score: Float score (0.0 to 1.0).
        metadata_list: List containing execution metadata for each test case.
    """
    solution = completion
    if "```python" in completion:
        solution = completion.split("```python")[-1].split("```")[0]
    elif "```" in completion:
        # Handle cases like ```\ncode\n```
        parts = completion.split("```")
        if len(parts) >= 2:
            solution = parts[1]
            # Remove potential language specifier like 'python\n'
            if "\n" in solution:
                first_line, rest = solution.split("\n", 1)
                if first_line.strip().isalpha():  # Simple check for language name
                    solution = rest
    else:
        return 0.0, [{"error": "Invalid completion (missing code block)"}]

    try:
        if not isinstance(test_cases, dict):
            try:
                test_cases = json.loads(test_cases)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse test_cases JSON: {e}")
                return 0.0, [{"error": "Invalid test_cases JSON format"}]

        if test_cases is not None and "assert_case" in test_cases and isinstance(test_cases.get("assert_case"), list):
            assert_cases = test_cases.get("assert_case")
            test_cases.setdefault("inputs", ["" for _ in assert_cases])
            test_cases.setdefault("outputs", [None for _ in assert_cases])
        elif not test_cases or "inputs" not in test_cases or "outputs" not in test_cases:
            logger.error("Invalid test_cases structure.")
            return 0.0, [{"error": "Invalid test_cases structure (missing inputs/outputs)"}]

        # For continuous scoring only N cases count, so only execute those N to avoid
        # wasting sandbox calls on cases that never affect the score. When there are
        # more than N cases we randomly sample N of them instead of
        # always taking the first N.
        if continuous:
            list_keys = [key for key in ("inputs", "outputs", "assert_case") if isinstance(test_cases.get(key), list)]
            num_total = len(test_cases["inputs"]) if isinstance(test_cases.get("inputs"), list) else 0
            if num_total > CONTINUOUS_SCORE_NUM_CASES:
                sampled_indices = sorted(random.sample(range(num_total), CONTINUOUS_SCORE_NUM_CASES))
                for key in list_keys:
                    test_cases[key] = [test_cases[key][i] for i in sampled_indices]

        # Check all test cases
        # Note: The return value of check_correctness might need adaptation here
        # Assume check_correctness returns (results_list, metadata_list)
        # results_list contains True, False, or error codes (-1, -2, -3, etc.)
        res_list, metadata_list = check_correctness(
            sandbox_fusion_url=sandbox_fusion_url,
            in_outs=test_cases,
            generation=solution,
            timeout=timeout,
            concurrent_semaphore=concurrent_semaphore,
            memory_limit_mb=memory_limit_mb,
        )

        # Calculate score
        if not res_list:  # If there are no results (e.g., invalid input)
            return 0.0, metadata_list

        if continuous:
            # Only the (at most) N sampled cases were executed (see sampling above), so
            # res_list already contains exactly the cases that count toward the score.
            num_to_consider = len(res_list)
            passed_count = sum(1 for r in res_list if r is True)
            if num_to_consider == 0:
                score = 0.0
            elif pass_all:
                # Binary reward: full credit only if every evaluated case passes.
                score = 1.0 if passed_count == num_to_consider else 0.0
            else:
                score = passed_count / num_to_consider
            final_metadata = metadata_list
        else:
            # Score over all test cases
            passed_count = sum(1 for r in res_list if r is True)
            total_cases = len(res_list)
            if total_cases == 0:
                score = 0.0
            elif pass_all:
                score = 1.0 if passed_count == total_cases else 0.0
            else:
                score = passed_count / total_cases
            final_metadata = metadata_list

    except Exception as e:
        logger.error(f"Error during compute_score: {e}")
        traceback.print_exc()
        score = 0.0
        # Try to return partial metadata if available, otherwise return error info
        final_metadata = metadata_list if "metadata_list" in locals() else [{"error": f"Unhandled exception: {e}"}]

    # Ensure float and list are returned
    return float(score), final_metadata if isinstance(final_metadata, list) else [final_metadata]
