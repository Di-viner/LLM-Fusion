#!/usr/bin/env bash
# Multi-teacher on-policy distillation (MOPD) | Qwen3-4B | FSDP | vLLM
#
# The distillation counterpart to train_mix_rl.sh: the student samples from the
# same mixed-domain data, but instead of a verifiable reward it is distilled from
# five domain-specialized teachers -- the per-domain experts produced by
# train_expert_math.sh and its siblings, exported to HuggingFace format.
#
# Teachers are routed per sample by `data_source` (distillation.teacher_key):
#
#   data_source            teacher directory
#   LLM-Fusion/Math        ${TEACHER_ROOT}/math
#   LLM-Fusion/Science     ${TEACHER_ROOT}/science
#   LLM-Fusion/Code        ${TEACHER_ROOT}/code
#   LLM-Fusion/IF          ${TEACHER_ROOT}/if
#   LLM-Fusion/Agent       ${TEACHER_ROOT}/agent
#
# Export teachers in bf16: vLLM loads bf16 regardless, so it is lossless for
# teacher scoring and halves both disk and load-time host memory.
#
# GPU layout. The teacher pool is a SEPARATE Ray resource pool from the training
# pool, so its GPUs are NOT part of NNODES x NGPUS_PER_NODE -- they are requested
# through distillation.nnodes x distillation.n_gpus_per_node. The pool size must
# equal sum(num_replicas * tensor_parallel_size) over all teachers. With the
# defaults below:
#
#   training pool:  3 nodes x 8 = 24 GPUs
#   teacher pool:   1 node  x 5 =  5 GPUs  (five 4B teachers, tp1, one replica)
#
# which fits a 4x8 cluster with three GPUs idle on the teacher node.
#
# Multi-node: start the Ray cluster first, then run this on the head node only.

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

source "${SCRIPT_DIR}/../paths.sh"
require_paths MODEL_PATH TRAIN_DATA_ROOT EVAL_DATA_ROOT OUTPUT_ROOT TEACHER_ROOT

TRAIN_FILE=${TRAIN_FILE:-${MIX_TRAIN}}

MATH_TEACHER=${MATH_TEACHER:-${TEACHER_ROOT}/math}
SCIENCE_TEACHER=${SCIENCE_TEACHER:-${TEACHER_ROOT}/science}
CODE_TEACHER=${CODE_TEACHER:-${TEACHER_ROOT}/code}
IF_TEACHER=${IF_TEACHER:-${TEACHER_ROOT}/if}
AGENT_TEACHER=${AGENT_TEACHER:-${TEACHER_ROOT}/agent}

# ---- cluster (training pool only; teachers are counted separately) ----
NNODES=${NNODES:-3}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

INFER_BACKEND=${INFER_BACKEND:-vllm}
if [ "${INFER_BACKEND}" = "vllm" ]; then
    export VLLM_USE_V1=1
fi

# ---- teacher pool ----
TEACHER_TP=${TEACHER_TP:-1}
TEACHER_NUM_REPLICAS=${TEACHER_NUM_REPLICAS:-1}
TEACHER_NNODES=${TEACHER_NNODES:-1}
TEACHER_GPU_MEM_UTIL=${TEACHER_GPU_MEM_UTIL:-0.8}
NUM_TEACHERS=5
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-$(( NUM_TEACHERS * TEACHER_NUM_REPLICAS * TEACHER_TP ))}

# ---- optimisation ----
# 264 = 24 x 11, divisible by the 3 x 8 training pool's data-parallel size.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-264}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-264}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-5120}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
MAX_NUM_TOKENS=${MAX_NUM_TOKENS:-$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1 ))}
ACTOR_LR=${ACTOR_LR:-1e-6}
NORM_ADV_BY_STD_IN_GRPO=${NORM_ADV_BY_STD_IN_GRPO:-FALSE}

# ---- rollout ----
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.9}
ROLLOUT_N=${ROLLOUT_N:-4}

MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-16}
MAX_USER_TURNS=${MAX_USER_TURNS:-16}
MAX_PARALLEL_CALLS=${MAX_PARALLEL_CALLS:-1}
MAX_TOOL_RESPONSE_LENGTH=${MAX_TOOL_RESPONSE_LENGTH:-2048}
MULTITURN_FORMAT=${MULTITURN_FORMAT:-hermes}

# ---- distillation loss ----
DISTILLATION_LOSS_MODE=${DISTILLATION_LOSS_MODE:-k1}
USE_POLICY_GRADIENT=${USE_POLICY_GRADIENT:-True}
USE_TASK_REWARDS=${USE_TASK_REWARDS:-False}

# ---- bookkeeping ----
PROJECT_NAME=${PROJECT_NAME:-llm_fusion}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-mopd_5teacher_qwen3_4b}
SAVE_FREQ=${SAVE_FREQ:-25}
TEST_FREQ=${TEST_FREQ:-25}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-400}

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
    data.reward_fn_key=data_source
)

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.use_kl_loss=False
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
    actor_rollout_ref.rollout.max_model_len=${MAX_NUM_TOKENS}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=4
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95
    # Mix rows all carry an explicit agent_name; this default only applies if
    # you swap in data that does not.
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

teacher_args() {
    local name=$1 key=$2 path=$3
    echo "+distillation.teacher_models.${name}.key=${key}"
    echo "+distillation.teacher_models.${name}.model_path=${path}"
    echo "+distillation.teacher_models.${name}.num_replicas=${TEACHER_NUM_REPLICAS}"
    echo "+distillation.teacher_models.${name}.inference.name=${INFER_BACKEND}"
    echo "+distillation.teacher_models.${name}.inference.tensor_model_parallel_size=${TEACHER_TP}"
    echo "+distillation.teacher_models.${name}.inference.gpu_memory_utilization=${TEACHER_GPU_MEM_UTIL}"
    echo "+distillation.teacher_models.${name}.inference.prompt_length=${MAX_PROMPT_LENGTH}"
    echo "+distillation.teacher_models.${name}.inference.response_length=${MAX_RESPONSE_LENGTH}"
    echo "+distillation.teacher_models.${name}.inference.max_model_len=${MAX_NUM_TOKENS}"
}

DISTILL=(
    distillation.enabled=True
    distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE}
    distillation.nnodes=${TEACHER_NNODES}
    distillation.teacher_key=data_source
    $(teacher_args math "LLM-Fusion/Math" "${MATH_TEACHER}")
    $(teacher_args science "LLM-Fusion/Science" "${SCIENCE_TEACHER}")
    $(teacher_args code "LLM-Fusion/Code" "${CODE_TEACHER}")
    $(teacher_args if "LLM-Fusion/IF" "${IF_TEACHER}")
    $(teacher_args agent "LLM-Fusion/Agent" "${AGENT_TEACHER}")
    distillation.distillation_loss.loss_mode=${DISTILLATION_LOSS_MODE}
    distillation.distillation_loss.use_task_rewards=${USE_TASK_REWARDS}
    distillation.distillation_loss.use_policy_gradient=${USE_POLICY_GRADIENT}
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${REWARD[@]}" \
    "${DISTILL[@]}" \
    "$@"
