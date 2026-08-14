#!/usr/bin/env bash
# Llama3-8B QLoRA SFT: plain completion (input+output, no instruction) for sft_true_prefix_no_instruction.json
# Single RTX 4090 (24GB) recommended: 4bit + LoRA + gradient checkpointing; effective batch = per_device * grad_accum.
#
# Optional environment variables:
#   LLAMA_MODEL_PATH     base model (default REPO/models/llama3-8B/baseline)
#   LLAMA_TRAIN_JSON     train JSON (default REPO/data/api4_200k/sft_true_prefix_no_instruction.json)
#   LLAMA_VAL_JSON       val JSON (default *_val.json; split by source_id to avoid random train leakage)
#   LLAMA_OUTPUT_DIR     checkpoint dir (step_* + final adapter)
#   LLAMA_FINAL_MODEL_DIR  sync final adapter here after training (default models/llama3-8B/api4_prefix_plain_qlora)
#   LLAMA_RESUME_FROM_CHECKPOINT  resume checkpoint; unset means train from base. Example: LLAMA_RESUME_FROM_CHECKPOINT=/path/to/step_7500
#   LLAMA_LOG_FILE       log file
#
# Step count reference (default num_train_epochs=2, grad_accum=16, per_device=1):
#   train set ~105k (train split); val set via LLAMA_VAL_JSON; steps/epoch ~ ceil(N_train / 16); 2 epochs ~ 2x.
#
# Fails without GPU (e.g. some remote sandboxes); set SKIP_CUDA_CHECK=1 to skip the check (not recommended for QLoRA).
#
# Training is detached via nohup: closing the terminal/SSH session will not SIGHUP it; logs append to LOG_FILE; PID is written to OUTPUT_DIR/train.pid.

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../_common.sh"

TRAIN_DIR="${REPO_ROOT}/src/ffn_edit/train"
DATA_JSON="${LLAMA_TRAIN_JSON:-${REPO_ROOT}/data/api4_200k/sft_true_prefix_no_instruction.json}"
VAL_JSON="${LLAMA_VAL_JSON:-${REPO_ROOT}/data/api4_200k/sft_true_prefix_no_instruction_val.json}"
MODEL_PATH="${LLAMA_MODEL_PATH:-${REPO_ROOT}/models/llama3-8B/baseline}"
OUTPUT_DIR="${LLAMA_OUTPUT_DIR:-${REPO_ROOT}/checkpoints/ffn_edit/llama3-8b_lora/api4_prefix_plain}"
FINAL_MODEL_DIR="${LLAMA_FINAL_MODEL_DIR:-${REPO_ROOT}/models/llama3-8B/api4_prefix_plain_qlora}"
ACCELERATE_CLI_MODULE="ffn_edit.train.accelerate_cli"
RESUME_FROM_CHECKPOINT="${LLAMA_RESUME_FROM_CHECKPOINT:-}"

if [ "${SKIP_CUDA_CHECK:-0}" != "1" ]; then
    if ! python3 -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo "[run_llama3_api4_prefix_qlora] ERROR: torch.cuda.is_available() is False; cannot load 4-bit weights." >&2
        echo "[run_llama3_api4_prefix_qlora] Run this script on a machine with NVIDIA drivers and CUDA (e.g. bash on the 4090 host)." >&2
        exit 1
    fi
fi

mkdir -p "${OUTPUT_DIR}" "${FINAL_MODEL_DIR}"
LOG_DIR="${REPO_ROOT}/logs/ffn_edit/train"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LLAMA_LOG_FILE:-${LOG_DIR}/run_llama3_api4_prefix_qlora_$(date +%Y%m%d_%H%M%S).log}"

banner() {
    printf '%s\n' "$@" | tee -a "${LOG_FILE}"
}

banner "[run_llama3_api4_prefix_qlora] LOG_FILE=${LOG_FILE}"
banner "[run_llama3_api4_prefix_qlora] MODEL_PATH=${MODEL_PATH}"
banner "[run_llama3_api4_prefix_qlora] DATA_JSON=${DATA_JSON}"
banner "[run_llama3_api4_prefix_qlora] VAL_JSON=${VAL_JSON}"
banner "[run_llama3_api4_prefix_qlora] OUTPUT_DIR=${OUTPUT_DIR}"
banner "[run_llama3_api4_prefix_qlora] FINAL_MODEL_DIR=${FINAL_MODEL_DIR}"
if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    banner "[run_llama3_api4_prefix_qlora] RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT}"
else
    banner "[run_llama3_api4_prefix_qlora] RESUME_FROM_CHECKPOINT=(none)"
fi

cmd=(
    python -u -m "${ACCELERATE_CLI_MODULE}" launch
    --num_processes 1
    --num_machines 1
    --main_process_port 12359
    run_clm_no_trainer.py
    --model_name_or_path "${MODEL_PATH}"
    --train_file "${DATA_JSON}"
    --validation_file "${VAL_JSON}"
    --config_name "${MODEL_PATH}"
    --tokenizer_name "${MODEL_PATH}"
    --sft_plain_completion
    --use_lora
    --load_in_4bit
    --bnb_4bit_use_double_quant
    --lora_r 64
    --lora_alpha 128
    --lora_dropout 0.05
    --num_train_epochs 2
    --checkpointing_steps 1500
    --keep_last_checkpoints 3
    --per_device_train_batch_size 1
    --per_device_eval_batch_size 2
    --gradient_accumulation_steps 16
    --learning_rate 2e-4
    --lr_scheduler_type cosine
    --num_warmup_steps 200
    --max_grad_norm 0.3
    --block_size 512
    --group_by_length
    --output_dir "${OUTPUT_DIR}"
    --torch_dtype bfloat16
    --low_cpu_mem_usage
    --gradient_checkpointing
    --dataloader_num_workers 2
)

if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    cmd+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

nohup env PYTHONUNBUFFERED=1 \
    REPO_ROOT="${REPO_ROOT}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    FINAL_MODEL_DIR="${FINAL_MODEL_DIR}" \
    bash -c '
set -euo pipefail
cd "$1"
shift
"$@"
echo "[run_llama3_api4_prefix_qlora] training finished; syncing adapter to FINAL_MODEL_DIR=${FINAL_MODEL_DIR}"
mkdir -p "${FINAL_MODEL_DIR}"
rsync -a --delete \
    --exclude "step_*" \
    --exclude "epoch_*" \
    --exclude "train.pid" \
    "${OUTPUT_DIR}/" "${FINAL_MODEL_DIR}/"
echo "[run_llama3_api4_prefix_qlora] synced to ${FINAL_MODEL_DIR}"
' _ "${TRAIN_DIR}" "${cmd[@]}" >> "${LOG_FILE}" 2>&1 &
TRAIN_PID=$!
echo "${TRAIN_PID}" > "${OUTPUT_DIR}/train.pid"
banner "[run_llama3_api4_prefix_qlora] nohup started detached PID=${TRAIN_PID} (train.pid written under OUTPUT_DIR)"
banner "[run_llama3_api4_prefix_qlora] follow logs: tail -f ${LOG_FILE}"
banner "[run_llama3_api4_prefix_qlora] stop training: kill ${TRAIN_PID}"
