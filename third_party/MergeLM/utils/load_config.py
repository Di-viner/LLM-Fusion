import os

# Local modification: upstream hardcoded the original author's cache paths.
# merge_llms.py only reaches this value when from_pretrained falls back for a
# Hub model id, so local expert directories are unaffected either way.
cache_dir = (
    os.environ.get("HF_HOME")
    or os.environ.get("TRANSFORMERS_CACHE")
    or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
)
