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


def _verify_in_subprocess(ground_truth_boxed: str, model_output: str) -> float:
    """Run math_verify in a subprocess where signal.alarm() works."""
    from math_verify.grader import verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse

    gold_targets = (LatexExtractionConfig(),)
    pred_targets = (ExprExtractionConfig(), LatexExtractionConfig())

    clipped_model_output = model_output[-_SOLUTION_CLIP_CHARS:]
    last_boxed = last_boxed_only_string(clipped_model_output)
    extracted_gold = parse(ground_truth_boxed, gold_targets)
    extracted_pred = parse(last_boxed or "", pred_targets)
    if extracted_gold and extracted_pred:
        return max(1.0 if any(verify(g, p) for g in extracted_gold) else 0.0 for p in extracted_pred)
    return 0.0


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0, timeout: float = 30.0) -> float:
    ret_score = 0.0
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        future = _get_pool().submit(_verify_in_subprocess, ground_truth_boxed, model_output)
        ret_score = future.result(timeout=timeout)
    except (FuturesTimeoutError, TimeoutException):
        ret_score = timeout_score
    except BrokenProcessPool:
        # A worker crashed (OOM / native crash / SIGKILL), which permanently breaks the whole
        # pool. Without rebuilding it here, every subsequent call would fail and silently score
        # 0.0. Recreate the pool and retry once.
        _reset_pool()
        try:
            future = _get_pool().submit(_verify_in_subprocess, ground_truth_boxed, model_output)
            ret_score = future.result(timeout=timeout)
        except (FuturesTimeoutError, TimeoutException):
            ret_score = timeout_score
        except Exception as e:
            print(f"Error in math_verify_enhanced compute_score after pool reset: {e}")
    except Exception as e:
        print(f"Error in math_verify_enhanced compute_score: {e}")
    return ret_score
