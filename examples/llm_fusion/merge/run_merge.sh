#!/usr/bin/env bash
# Merge step 2: fuse the five fp32 experts into one model.
#
# Runs the vendored MergeLM (third_party/MergeLM), which produced every Merge
# result in the paper.
#
# The defaults reproduce the released Merge checkpoint exactly: Task Arithmetic
# at lambda = 0.6 over math, code, science, if, agent, saved as bf16. That is
# the Merge row of the main results table, and the model published as
# Qwen3-4B-Inst-Merge / Qwen3-8B-NT-Merge.
#
# Set RUN_ALL_METHODS=1 for the other four methods compared in the paper's
# merging-methods table: Average, TIES, DARE-TA, SCE.
#
# Prereq: convert_experts_to_hf.sh, so the experts exist as fp32 HF models.
#
# MEMORY. merge_llms.py loads the base and all five experts into CPU RAM in
# fp32 at once: roughly 90 GB at 4B and 180 GB at 8B. It needs no GPU. If the
# host cannot hold that, merge fewer experts via MERGE_MODELS.

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

source "${SCRIPT_DIR}/../paths.sh"
require_paths MODEL_PATH OUTPUT_ROOT

PYTHON=${PYTHON:-python3}

# ---- what to merge -------------------------------------------------------
# MODEL_PATH is the task-vector reference. It MUST be the model the experts were
# GRPO-trained from (Qwen3-4B-Instruct-2507 or the Qwen3-8B non-thinking
# backbone), not a base pretrained checkpoint, or every task vector is wrong.
#
# Order matters only for reading the logs; the merge itself is symmetric. This
# is the order used for the released checkpoints.
MATH_MODEL=${MATH_MODEL:-${HF_EXPERTS_ROOT}/math}
CODE_MODEL=${CODE_MODEL:-${HF_EXPERTS_ROOT}/code}
SCIENCE_MODEL=${SCIENCE_MODEL:-${HF_EXPERTS_ROOT}/science}
IF_MODEL=${IF_MODEL:-${HF_EXPERTS_ROOT}/if}
AGENT_MODEL=${AGENT_MODEL:-${HF_EXPERTS_ROOT}/agent}

MERGE_MODELS=${MERGE_MODELS:-${MATH_MODEL},${CODE_MODEL},${SCIENCE_MODEL},${IF_MODEL},${AGENT_MODEL}}

# ---- merge hyperparameters ----------------------------------------------
# Lambda for TA and DARE-TA.
SCALINGS=${SCALINGS:-"0.6"}
# Scaling coefficient for the methods whose result does not depend on it
# (Average, SCE) and for TIES. 1.0 is MergeLM's default.
SCALING=${SCALING:-1.0}
# DARE drop rate p, used only by DARE-TA. MergeLM's default, and the paper's.
# Survivors are rescaled by 1/(1-p), so the task vector is unchanged in
# expectation and DARE-TA at a given lambda is the same expected step as TA.
WEIGHT_MASK_RATE=${WEIGHT_MASK_RATE:-0.2}
# The merge is always computed in fp32; this is the on-disk dtype of the result.
# bf16 halves the files and is lossless for vLLM eval, which casts to bf16.
SAVE_DTYPE=${SAVE_DTYPE:-bfloat16}
# Average, TIES, DARE-TA, SCE, i.e. everything except plain TA.
RUN_ALL_METHODS=${RUN_ALL_METHODS:-0}

for d in "${MODEL_PATH}" ${MERGE_MODELS//,/ }; do
    if [ ! -f "${d}/config.json" ]; then
        echo "ERROR: ${d} is not an HF model dir (no config.json)." >&2
        echo "Run examples/llm_fusion/merge/convert_experts_to_hf.sh first." >&2
        exit 1
    fi
done

# merge_llms.py writes to ./merged_models and ./save_merge_llm_logs, so the
# output location is chosen by the working directory.
mkdir -p "${MERGE_OUT_ROOT}"
cd "${MERGE_OUT_ROOT}"
export PYTHONPATH="${MERGELM}:${PYTHONPATH:-}"

run_merge () {
    ${PYTHON} "${MERGELM}/merge_llms.py" \
        --pretrained_model_name "${MODEL_PATH}" \
        --models_to_merge "${MERGE_MODELS}" \
        --weight_mask_rate "${WEIGHT_MASK_RATE}" \
        --save_dtype "${SAVE_DTYPE}" \
        "$@"
}

# Mirrors merge_llms.py's save_model_name so a finished merge can be skipped.
# The lambda goes through float() exactly as Python formats it, so SCALINGS="0.60"
# still resolves to the ..._0.6 directory.
merged_dir_name () {
    local method=$1 lam=$2 dare=$3
    case "${method}" in
        average_merging|sce_merging) echo "${method}" ;;
        *)
            local lam_norm
            lam_norm=$(${PYTHON} -c "print(float('${lam}'))")
            if [ "${dare}" = "1" ]; then
                echo "dare_${method}_scaling_coefficient_${lam_norm}"
            else
                echo "${method}_scaling_coefficient_${lam_norm}"
            fi
            ;;
    esac
}

already_merged () {
    [ -f "${MERGE_OUT_ROOT}/merged_models/$1/config.json" ]
}

# ---- Task Arithmetic (the released configuration) -------------------------
for LAM in ${SCALINGS}; do
    name=$(merged_dir_name task_arithmetic "${LAM}" 0)
    if already_merged "${name}"; then
        echo "SKIP ${name} (already merged)"
    else
        run_merge --merging_method_name task_arithmetic --scaling_coefficient "${LAM}"
    fi
done

# ---- the rest of the method comparison (opt-in) --------------------------
if [ "${RUN_ALL_METHODS}" = "1" ]; then
    # Average (model soup): the mean of the task vectors, i.e. TA at 1/N.
    already_merged "$(merged_dir_name average_merging "" 0)" \
        || run_merge --merging_method_name average_merging --scaling_coefficient "${SCALING}"

    # TIES: trim each task vector to its largest-magnitude 20% of entries, elect
    # one sign per parameter, average only the entries that agree with it.
    already_merged "$(merged_dir_name ties_merging "${SCALING}" 0)" \
        || run_merge --merging_method_name ties_merging --scaling_coefficient "${SCALING}"

    # DARE-TA: drop entries at WEIGHT_MASK_RATE, rescale the survivors, then TA.
    # Same lambda as TA above, as in the paper.
    for LAM in ${SCALINGS}; do
        already_merged "$(merged_dir_name task_arithmetic "${LAM}" 1)" \
            || run_merge --merging_method_name task_arithmetic --scaling_coefficient "${LAM}" \
                   --apply_weight_mask True --use_weight_rescale
    done

    # SCE: weight each task vector per parameter matrix by its mean squared
    # magnitude, erase sign-disagreeing entries, normalize by the survivors.
    already_merged "$(merged_dir_name sce_merging "" 0)" \
        || run_merge --merging_method_name sce_merging --scaling_coefficient "${SCALING}"
fi

echo "Merged models (${SAVE_DTYPE}) under ${MERGE_OUT_ROOT}/merged_models:"
ls -la "${MERGE_OUT_ROOT}/merged_models"
echo ""
echo "Evaluate one with:"
echo "  MODEL_PATH=${MERGE_OUT_ROOT}/merged_models/<method> \\"
echo "  EXPERIMENT_NAME=eval_merge bash examples/llm_fusion/eval/eval.sh"
