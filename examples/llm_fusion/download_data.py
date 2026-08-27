"""Download the released training and evaluation data from the Hugging Face Hub.

Two dataset repos:

    LLM-Fusion-Train   six per-domain RL training configs, split `train`
    LLM-Fusion-Test    the eight benchmarks in the paper, split `test`

Each config lands as sharded parquet under ``<output-dir>/<target>/<Config>/``,
which is the layout ``paths.sh`` expects.

Examples::

    # everything (~3.4 GiB; the code sets are most of it)
    python examples/llm_fusion/download_data.py

    # only the benchmarks
    python examples/llm_fusion/download_data.py --target test

    # a couple of configs while iterating
    python examples/llm_fusion/download_data.py --target test --configs AIME2025 BFCL_v3
"""

import argparse
import glob
import os
import sys

DEFAULT_OWNER = os.environ.get("LLM_FUSION_OWNER", "Siye01")

# target -> (repo suffix, split name, {config: expected rows})
TARGETS = {
    "train": (
        "LLM-Fusion-Train",
        "train",
        {
            "Math": 38131,
            "Science": 50000,
            "Code": 19169,
            "IF": 16575,
            "Agent": 10229,
            "Mix": 87699,
        },
    ),
    "test": (
        "LLM-Fusion-Test",
        "test",
        {
            "AIME2025": 30,
            "AIME2026": 30,
            "GPQA": 198,
            "LCB_v5": 167,
            "LCB_v6": 175,
            "IFEval": 541,
            "IFBench": 300,
            "BFCL_v3": 200,
        },
    ),
}


def human(num_bytes: int) -> str:
    return f"{num_bytes / 2**20:,.1f} MiB"


def download(repo_id: str, configs: list[str], output_dir: str) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id,
        repo_type="dataset",
        local_dir=output_dir,
        allow_patterns=["README.md"] + [f"{c}/*" for c in configs],
    )


def verify(configs: list[str], expected: dict[str, int], split: str, output_dir: str) -> bool:
    import pyarrow.parquet as pq

    print(f"\n{'config':10s} {'shards':>6s} {'rows':>9s} {'expected':>9s} {'size':>12s}")
    ok = True
    for config in configs:
        shards = sorted(glob.glob(os.path.join(output_dir, config, f"{split}-*.parquet")))
        if not shards:
            print(f"{config:10s} {'--':>6s}  no parquet files found")
            ok = False
            continue
        rows = sum(pq.ParquetFile(s).metadata.num_rows for s in shards)
        size = sum(os.path.getsize(s) for s in shards)
        flag = "" if rows == expected[config] else "  <-- MISMATCH"
        ok &= rows == expected[config]
        print(f"{config:10s} {len(shards):6d} {rows:9,d} {expected[config]:9,d} {human(size):>12s}{flag}")
    return ok


def run_target(target: str, args) -> tuple[bool, str, list[str]]:
    repo_name, split, expected = TARGETS[target]
    repo_id = args.repo_id or f"{DEFAULT_OWNER}/{repo_name}"
    output_dir = os.path.abspath(os.path.join(args.output_dir, target))

    configs = args.configs or list(expected)
    unknown = [c for c in configs if c not in expected]
    if unknown:
        sys.exit(f"unknown config(s) for --target {target}: {', '.join(unknown)}\n"
                 f"available: {', '.join(expected)}")

    print(f"\n=== {target}: {', '.join(configs)} from {repo_id}")
    if not args.skip_download:
        download(repo_id, configs, output_dir)
    return verify(configs, expected, split, output_dir), output_dir, configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", choices=["train", "test", "both"], default="both")
    parser.add_argument("--repo-id", help="override the repo id (single target only)")
    parser.add_argument(
        "--configs",
        nargs="+",
        metavar="CONFIG",
        help="limit to these configs (single target only)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("LLM_FUSION_DATA_DIR", os.path.join("data", "llm_fusion")),
        help="parent of train/ and test/ (default: data/llm_fusion)",
    )
    parser.add_argument("--skip-download", action="store_true", help="only verify what is on disk")
    args = parser.parse_args()

    targets = ["train", "test"] if args.target == "both" else [args.target]
    if len(targets) > 1 and (args.repo_id or args.configs):
        sys.exit("--repo-id and --configs need a single --target")

    ok = True
    roots = {}
    for target in targets:
        target_ok, output_dir, _ = run_target(target, args)
        ok &= target_ok
        roots[target] = output_dir

    if not ok:
        sys.exit("\nRow counts do not match; re-run to resume the download.")

    print("\nPaths for paths.sh:")
    if "train" in roots:
        print(f"  TRAIN_DATA_ROOT={roots['train']}")
    if "test" in roots:
        print(f"  EVAL_DATA_ROOT={roots['test']}")


if __name__ == "__main__":
    main()
