from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig

import logging
logging.getLogger("math_verify").setLevel(logging.ERROR)

_SOLUTION_CLIP_CHARS = 300

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        print(f"remove_boxed error: {s}")
        return None

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]

    return retval

def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0) -> bool:
    # Support options A-Z
    options = tuple(chr(ord('A') + i) for i in range(26))
    verify_func = math_metric(
        gold_extraction_target=(StringExtractionConfig(strings=options),),
        pred_extraction_target=(StringExtractionConfig(strings=options),),
    )
    ret_score = 0.0

    trancated_model_output = last_boxed_only_string(model_output[-_SOLUTION_CLIP_CHARS:])
    trancated_model_output = remove_boxed(trancated_model_output)

    try:
        ret_score, _ = verify_func([ground_truth], [trancated_model_output])
    
    except TimeoutException as e:
        print(f"[MATH_VERIFY TimeoutException] {e}")
        ret_score = timeout_score
    except Exception as e:
        print(f"[MATH_VERIFY Exception] {e}")
        pass

    return ret_score
