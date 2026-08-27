#!/usr/bin/env bash
# Evaluate one checkpoint on the eight benchmarks reported in the paper:
#
#   Math      AIME 2025, AIME 2026
#   Science   GPQA
#   Code      LiveCodeBench v5, LiveCodeBench v6
#   IF        IFEval, IFBench
#   Agent     BFCL v3 (multi-turn tool calling)
#
# Runs verl's trainer in val_only mode, which pays Ray and vLLM start-up once and
# then evaluates every set in a single job. Per-set accuracy is logged under
# `val-core/<data_source>/...`; with VAL_N > 1 both mean@n and best@n are
# reported, and the raw generations land in ${VALIDATION_DATA_DIR} as one JSONL
# per benchmark.
#
# MODEL_PATH points at whatever you want scored: the base model, a trained
# checkpoint exported to HuggingFace format, or a merged model.
#
# Usage:
#   bash examples/llm_fusion/eval/eval.sh
#   MODEL_PATH=/path/to/model VAL_N=16 bash examples/llm_fusion/eval/eval.sh

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

source "${SCRIPT_DIR}/../paths.sh"
require_paths MODEL_PATH TRAIN_DATA_ROOT EVAL_DATA_ROOT OUTPUT_ROOT

# A train file is still required by the trainer config in val_only mode and is
# never read. It must hold at least TRAIN_BATCH_SIZE rows: the loader drops the
# last partial batch and then asserts the dataloader is non-empty, so a 30-row
# benchmark would fail. The Agent training set is the smallest one large enough.
TRAIN_FILE=${TRAIN_FILE:-${AGENT_TRAIN}}

# ---- cluster ----
NNODES=${NNODES:-4}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

INFER_BACKEND=${INFER_BACKEND:-vllm}
if [ "${INFER_BACKEND}" = "vllm" ]; then
    export VLLM_USE_V1=1
fi

# ---- sampling ----
VAL_N=${VAL_N:-16}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.6}
VAL_TOP_P=${VAL_TOP_P:-0.95}
# Left unset, verl puts the whole suite in one batch: ~1.4k prompts times VAL_N.
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-128}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-5120}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}

ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.8}

# verl builds the actor worker even in val_only mode, so these still have to be
# set: FSDPActorConfig refuses to instantiate without either use_dynamic_bsz or
# an explicit micro batch size.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
ROLLOUT_N=${ROLLOUT_N:-16}
ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
NORM_ADV_BY_STD_IN_GRPO=${NORM_ADV_BY_STD_IN_GRPO:-FALSE}

MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-16}
MAX_USER_TURNS=${MAX_USER_TURNS:-16}
MAX_PARALLEL_CALLS=${MAX_PARALLEL_CALLS:-1}
MAX_TOOL_RESPONSE_LENGTH=${MAX_TOOL_RESPONSE_LENGTH:-2048}
MULTITURN_FORMAT=${MULTITURN_FORMAT:-hermes}

# ---- bookkeeping ----
PROJECT_NAME=${PROJECT_NAME:-llm_fusion_eval}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-eval_$(basename "${MODEL_PATH}")}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-${OUTPUT_ROOT}/eval/${EXPERIMENT_NAME}}

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.norm_adv_by_std_in_grpo=${NORM_ADV_BY_STD_IN_GRPO}
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILE}"
    data.val_files="${EVAL_SUITE}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.val_batch_size=${VAL_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=False
    data.truncation='error'
    data.return_raw_chat=True
)

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768
    actor_rollout_ref.actor.use_dynamic_bsz=True
)

REF=(
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.max_num_batched_tokens=32768
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.temperature=1
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N}
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P}
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.multi_turn.tool_config_path=${TOOL_CONFIG}
    actor_rollout_ref.rollout.multi_turn.format=${MULTITURN_FORMAT}
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_ASSISTANT_TURNS}
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${MAX_USER_TURNS}
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=${MAX_PARALLEL_CALLS}
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=${MAX_TOOL_RESPONSE_LENGTH}
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.val_only=True
    trainer.val_before_train=True
    # Load the HuggingFace weights at MODEL_PATH directly. Without this verl
    # looks for a training checkpoint to resume from and would score that
    # instead. It also makes validate() report step 0, so the dumps land under
    # ${VALIDATION_DATA_DIR}/0.
    trainer.resume_mode=disable
    trainer.save_freq=-1
    trainer.test_freq=1
    trainer.total_epochs=1
    trainer.logger='["console"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.validation_data_dir=${VALIDATION_DATA_DIR}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
)

REWARD=(
    reward.sandbox_fusion.url=${SANDBOX_FUSION_URL}
    reward.sandbox_fusion.max_concurrent=${SANDBOX_MAX_CONCURRENT}
    reward.sandbox_fusion.memory_limit_mb=${SANDBOX_MEMORY_LIMIT_MB}
)

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${REWARD[@]}" \
    "$@"
