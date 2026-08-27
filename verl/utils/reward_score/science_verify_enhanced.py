import logging
import multiprocessing
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from concurrent.futures.process import BrokenProcessPool

try:
    from math_verify.errors import TimeoutException
except ImportError:

    class TimeoutException(Exception):
        pass

    print("To use Math-Verify, please install it first by running `pip install math-verify`.")


logging.getLogger("math_verify").setLevel(logging.ERROR)

_SOLUTION_CLIP_CHARS = 300
_pool = None
_pool_lock = threading.Lock()


def _get_pool():
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn"))
    return _pool


def _reset_pool():
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.shutdown(wait=False, cancel_futures=True)
            _pool = None


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
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

    return None if right_brace_idx is None else string[idx : right_brace_idx + 1]


def _compute_score_in_subprocess(model_output: str, ground_truth: str) -> float:
    from math_verify.metric import math_metric
    from math_verify.parser import StringExtractionConfig

    options = tuple(chr(ord("A") + i) for i in range(26))
    verify_func = math_metric(
        gold_extraction_target=(StringExtractionConfig(strings=options),),
        pred_extraction_target=(StringExtractionConfig(strings=options),),
    )

    truncated_model_output = last_boxed_only_string(model_output[-_SOLUTION_CLIP_CHARS:])
    truncated_model_output = remove_boxed(truncated_model_output)
    score, _ = verify_func([ground_truth], [truncated_model_output])
    return score


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0, timeout: float = 30.0) -> float:
    ret_score = 0.0
    try:
        future = _get_pool().submit(_compute_score_in_subprocess, model_output, ground_truth)
        ret_score = future.result(timeout=timeout)
    except (FuturesTimeoutError, TimeoutException):
        ret_score = timeout_score
    except BrokenProcessPool:
        # A worker crashed (OOM / native crash / SIGKILL). Recreate the pool and retry once.
        _reset_pool()
        try:
            future = _get_pool().submit(_compute_score_in_subprocess, model_output, ground_truth)
            ret_score = future.result(timeout=timeout)
        except (FuturesTimeoutError, TimeoutException):
            ret_score = timeout_score
        except Exception as e:
            print(f"Error in science_verify_enhanced compute_score after pool reset: {e}")
    except Exception as e:
        print(f"Error in science_verify_enhanced compute_score: {e}")
    return ret_score
