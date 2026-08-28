#!/usr/bin/env bash
# Merge step 1: reassemble each expert's FSDP checkpoint into a fp32 HF model.
#
#   FSDP shards --(this script)--> fp32 HF experts --(run_merge.sh)--> merged model
#
# fp32 and not bf16 on purpose. verl's own `python -m verl.model_merger merge`
# casts to bf16 on export, which is fine for evaluation because vLLM runs bf16
# anyway. It is not fine here: the RL task vector theta_expert - theta_base is
# small, so the fp32->bf16 round-trip perturbs it by an amount comparable to the
# delta itself and changes what the merge produces. Every Merge number in the
# paper came from fp32 experts.
#
# Cost: ~15 GB per expert at 4B, ~30 GB at 8B, so budget ~75 GB / ~150 GB for
# the five experts. They are only needed until run_merge.sh has finished.
#
# Point MATH_ACTOR .. AGENT_ACTOR at the checkpoints you want to fuse, e.g.
# ${OUTPUT_ROOT}/math_expert_4b/global_step_<N>/actor.

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

source "${SCRIPT_DIR}/../paths.sh"
require_paths OUTPUT_ROOT MATH_ACTOR SCIENCE_ACTOR CODE_ACTOR IF_ACTOR AGENT_ACTOR

PYTHON=${PYTHON:-python3}
CONVERT=${CONVERT:-"${PYTHON} ${SCRIPT_DIR}/convert_experts_to_hf.py"}

mkdir -p "${HF_EXPERTS_ROOT}"

convert () {
    local name="$1" actor="$2"
    local out="${HF_EXPERTS_ROOT}/${name}"
    if [ -f "${out}/config.json" ]; then
        echo "[skip] ${name} already converted at ${out}"
        return
    fi
    if [ ! -d "${actor}" ]; then
        echo "ERROR: ${name} actor dir does not exist: ${actor}" >&2
        exit 1
    fi
    echo "[convert-fp32] ${name}: ${actor} -> ${out}"
    ${CONVERT} --local_dir "${actor}" --target_dir "${out}"
}

convert math    "${MATH_ACTOR}"
convert science "${SCIENCE_ACTOR}"
convert code    "${CODE_ACTOR}"
convert if      "${IF_ACTOR}"
convert agent   "${AGENT_ACTOR}"

echo "All five experts exported as fp32 under ${HF_EXPERTS_ROOT}"
ls -la "${HF_EXPERTS_ROOT}"
