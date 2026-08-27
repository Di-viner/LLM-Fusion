import re

_SOLUTION_CLIP_CHARS = 300
_VALID_OPTIONS = frozenset(chr(ord("A") + i) for i in range(26))


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


def _extract_boxed_content(boxed_str: str | None) -> str | None:
    if boxed_str is None:
        return None
    if boxed_str.startswith("\\boxed{"):
        return remove_boxed(boxed_str)
    if boxed_str.startswith("\\fbox{"):
        return remove_boxed(boxed_str.replace("\\fbox{", "\\boxed{", 1))
    if boxed_str.startswith("\\boxed "):
        return boxed_str[len("\\boxed ") :].strip()
    return None


def _normalize_option_letter(text: str | None) -> str | None:
    if text is None:
        return None

    text = text.strip()
    if not text:
        return None

    text = text.strip("$")
    for pattern in (
        r"^\\text\{(.+)\}$",
        r"^\\textbf\{(.+)\}$",
        r"^\\mathrm\{(.+)\}$",
    ):
        match = re.fullmatch(pattern, text.strip())
        if match:
            text = match.group(1).strip()
            break

    text = text.strip().strip("().[],;:")

    if len(text) == 1:
        letter = text.upper()
        return letter if letter in _VALID_OPTIONS else None

    match = re.search(r"(?<![A-Z])([A-Z])(?![A-Z])", text.upper())
    if match and match.group(1) in _VALID_OPTIONS:
        return match.group(1)

    return None


def _compute_score(model_output: str, ground_truth: str) -> float:
    boxed = last_boxed_only_string(model_output[-_SOLUTION_CLIP_CHARS:])
    pred = _normalize_option_letter(_extract_boxed_content(boxed))
    gt = _normalize_option_letter(ground_truth)
    if pred is None or gt is None:
        return 0.0
    return 1.0 if pred == gt else 0.0


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0, timeout: float = 30.0) -> float:
    del timeout  # kept for API compatibility with science_verify_enhanced
    try:
        return _compute_score(model_output, ground_truth)
    except Exception as e:
        print(f"Error in science_verify_enhanced_rule compute_score: {e}")
        return timeout_score
