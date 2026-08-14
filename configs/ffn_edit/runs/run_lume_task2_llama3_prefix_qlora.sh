#!/usr/bin/env bash
# LUME Task2 plain-prefix QLoRA for Llama3-8B baseline.
# Run from repo root:
#   bash configs/ffn_edit/runs/run_lume_task2_llama3_prefix_qlora.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

export LLAMA_TRAIN_JSON="${LLAMA_TRAIN_JSON:-${REPO_ROOT}/data/lume/api4_prefix_task2/sft_true_prefix_no_instruction_train.json}"
export LLAMA_VAL_JSON="${LLAMA_VAL_JSON:-${REPO_ROOT}/data/lume/api4_prefix_task2/sft_true_prefix_no_instruction_val.json}"
export LLAMA_MODEL_PATH="${LLAMA_MODEL_PATH:-${REPO_ROOT}/models/llama3-8B/baseline}"
export LLAMA_OUTPUT_DIR="${LLAMA_OUTPUT_DIR:-${REPO_ROOT}/checkpoints/ffn_edit/llama3-8b_lora/lume_task2_prefix_plain}"
export LLAMA_FINAL_MODEL_DIR="${LLAMA_FINAL_MODEL_DIR:-${REPO_ROOT}/models/llama3-8B/lume_task2_prefix_plain_qlora}"
export LLAMA_LOG_FILE="${LLAMA_LOG_FILE:-${REPO_ROOT}/logs/ffn_edit/train/run_llama3_lume_task2_prefix_qlora_$(date +%Y%m%d_%H%M%S).log}"
export LLAMA_RESUME_FROM_CHECKPOINT="${LLAMA_RESUME_FROM_CHECKPOINT:-}"

mkdir -p "${LLAMA_OUTPUT_DIR}" "${LLAMA_FINAL_MODEL_DIR}" "$(dirname "${LLAMA_LOG_FILE}")"

echo "[lume_qlora] LOG_FILE=${LLAMA_LOG_FILE}" | tee -a "${LLAMA_LOG_FILE}"
echo "[lume_qlora] MODEL_PATH=${LLAMA_MODEL_PATH}" | tee -a "${LLAMA_LOG_FILE}"
echo "[lume_qlora] TRAIN=${LLAMA_TRAIN_JSON}" | tee -a "${LLAMA_LOG_FILE}"
echo "[lume_qlora] VAL=${LLAMA_VAL_JSON}" | tee -a "${LLAMA_LOG_FILE}"
echo "[lume_qlora] OUTPUT_DIR=${LLAMA_OUTPUT_DIR}" | tee -a "${LLAMA_LOG_FILE}"
echo "[lume_qlora] FINAL_MODEL_DIR=${LLAMA_FINAL_MODEL_DIR}" | tee -a "${LLAMA_LOG_FILE}"

setsid env PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
  PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  bash -c '
set -euo pipefail
trap '\''status=$?; echo "[lume_qlora][inner] exit_status=${status}"'\'' EXIT
echo "[lume_qlora][inner] start train_dir=$1 model=$2"
cd "$1"
python -u ../utils/accelerate_cli.py launch \
  --num_processes 1 \
  --num_machines 1 \
  --main_process_port 12361 \
  run_clm_no_trainer.py \
  --model_name_or_path "$2" \
  --train_file "$3" \
  --validation_file "$4" \
  --config_name "$2" \
  --tokenizer_name "$2" \
  --sft_plain_completion \
  --use_lora \
  --load_in_4bit \
  --bnb_4bit_use_double_quant \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --num_train_epochs 2 \
  --max_train_steps 494 \
  --checkpointing_steps 100 \
  --keep_last_checkpoints 3 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --lr_scheduler_type cosine \
  --num_warmup_steps 50 \
  --max_grad_norm 0.3 \
  --block_size 512 \
  --group_by_length \
  --output_dir "$5" \
  --torch_dtype bfloat16 \
  --low_cpu_mem_usage \
  --gradient_checkpointing \
  --dataloader_num_workers 2
echo "[lume_qlora] training finished; syncing adapter to FINAL_MODEL_DIR=$6"
rsync -a --delete --exclude "step_*" --exclude "epoch_*" --exclude "train.pid" "$5/" "$6/"
echo "[lume_qlora] synced to $6"
' _ "${REPO_ROOT}/srcs/ffn_edit/train" "${LLAMA_MODEL_PATH}" "${LLAMA_TRAIN_JSON}" "${LLAMA_VAL_JSON}" "${LLAMA_OUTPUT_DIR}" "${LLAMA_FINAL_MODEL_DIR}" </dev/null >> "${LLAMA_LOG_FILE}" 2>&1 &

TRAIN_PID=$!
echo "${TRAIN_PID}" > "${LLAMA_OUTPUT_DIR}/train.pid"
echo "[lume_qlora] detached PID=${TRAIN_PID}" | tee -a "${LLAMA_LOG_FILE}"
echo "[lume_qlora] tail -f ${LLAMA_LOG_FILE}" | tee -a "${LLAMA_LOG_FILE}"
