#!/usr/bin/env bash
# Mixed-domain GRPO baseline | Qwen3-4B | FSDP | vLLM
#
# Trains one policy on all five domains at once, using the `Mix` config: the
# blend of Math, Science, Code, IF, and Agent released alongside this code. Each
# row keeps its own `data_source`, so the reward router picks the right verifier
# per sample and validation metrics stay separable by domain.
#
# This is the joint-training point of comparison for model merging and for
# multi-teacher distillation (train_mopd.sh).
#
# Rollout dispatch: unlike the single-domain experts, Mix contains Agent rows
# that need multi-turn tool calling. Those rows carry `agent_name=tool_agent` in
# the data, so they are routed to the tool agent loop regardless of the default
# below, while the four single-turn domains fall back to single_turn_agent.
#
# Multi-node: start the Ray cluster first, then run this on the head node only.

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

source "${SCRIPT_DIR}/../paths.sh"
require_paths MODEL_PATH TRAIN_DATA_ROOT EVAL_DATA_ROOT OUTPUT_ROOT

TRAIN_FILE=${TRAIN_FILE:-${MIX_TRAIN}}

# ---- cluster ----
NNODES=${NNODES:-4}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

INFER_BACKEND=${INFER_BACKEND:-vllm}
if [ "${INFER_BACKEND}" = "vllm" ]; then
    export VLLM_USE_V1=1
fi

# ---- optimisation ----
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-5120}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
ACTOR_LR=${ACTOR_LR:-1e-6}
USE_KL_LOSS=${USE_KL_LOSS:-True}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
NORM_ADV_BY_STD_IN_GRPO=${NORM_ADV_BY_STD_IN_GRPO:-FALSE}

# ---- rollout ----
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.9}
ROLLOUT_N=${ROLLOUT_N:-16}

MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-16}
MAX_USER_TURNS=${MAX_USER_TURNS:-16}
MAX_PARALLEL_CALLS=${MAX_PARALLEL_CALLS:-1}
MAX_TOOL_RESPONSE_LENGTH=${MAX_TOOL_RESPONSE_LENGTH:-2048}
MULTITURN_FORMAT=${MULTITURN_FORMAT:-hermes}

# ---- bookkeeping ----
PROJECT_NAME=${PROJECT_NAME:-llm_fusion}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-mix_rl_qwen3_4b}
SAVE_FREQ=${SAVE_FREQ:-50}
TEST_FREQ=${TEST_FREQ:-25}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}
# The mixed set is five domains' worth of data, so it takes many more steps than
# a single-domain expert.
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1500}

CKPT_DIR=${CKPT_DIR:-${OUTPUT_ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-${OUTPUT_ROOT}/rollout_data/${EXPERIMENT_NAME}}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-${OUTPUT_ROOT}/validation_data/${EXPERIMENT_NAME}}

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.norm_adv_by_std_in_grpo=${NORM_ADV_BY_STD_IN_GRPO}
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILE}"
    data.val_files="${EVAL_SUITE}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.shuffle=True
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
    actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS}
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768
    actor_rollout_ref.actor.use_dynamic_bsz=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.temperature=1
    actor_rollout_ref.rollout.max_num_batched_tokens=32768
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=4
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95
    # Every Mix row carries an explicit agent_name, so this default only applies
    # if you swap in data that does not.
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.multi_turn.tool_config_path=${TOOL_CONFIG}
    actor_rollout_ref.rollout.multi_turn.format=${MULTITURN_FORMAT}
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_ASSISTANT_TURNS}
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${MAX_USER_TURNS}
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=${MAX_PARALLEL_CALLS}
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=${MAX_TOOL_RESPONSE_LENGTH}
)

REF=(
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.val_before_train=True
    trainer.logger='["console", "wandb"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.default_local_dir=${CKPT_DIR}
    trainer.rollout_data_dir=${ROLLOUT_DATA_DIR}
    trainer.validation_data_dir=${VALIDATION_DATA_DIR}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
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
