#!/usr/bin/env bash
# LUME Task2 hard-balanced plain-prefix QLoRA for Qwen3-8B-Base.
# Input: data/lume/api4_prefix_task2_hard_balanced train/val JSON.
# Output: QLoRA adapter under checkpoints/ and synced final adapter under models/qwen3-8b-base/.
# Impact: trains adapter weights only; base Qwen3-8B-Base weights are not modified.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

TRAIN_JSON="${QWEN_TRAIN_JSON:-${REPO_ROOT}/data/lume/api4_prefix_task2_hard_balanced/sft_true_prefix_no_instruction_train.json}"
VAL_JSON="${QWEN_VAL_JSON:-${REPO_ROOT}/data/lume/api4_prefix_task2_hard_balanced/sft_true_prefix_no_instruction_val.json}"
MODEL_PATH="${QWEN_MODEL_PATH:-${REPO_ROOT}/models/qwen3-8b-base}"
OUTPUT_DIR="${QWEN_OUTPUT_DIR:-${REPO_ROOT}/checkpoints/ffn_edit/qwen3-8b_lora/lume_task2_hard_balanced_r64}"
FINAL_MODEL_DIR="${QWEN_FINAL_MODEL_DIR:-${REPO_ROOT}/models/qwen3-8b-base/lume_task2_hard_balanced_qlora}"
LOG_FILE="${QWEN_LOG_FILE:-${REPO_ROOT}/logs/ffn_edit/train/run_qwen3_lume_task2_hard_balanced_qlora_$(date +%Y%m%d_%H%M%S).log}"
RESUME_FROM_CHECKPOINT="${QWEN_RESUME_FROM_CHECKPOINT:-}"

mkdir -p "${OUTPUT_DIR}" "${FINAL_MODEL_DIR}" "$(dirname "${LOG_FILE}")"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[START] $(date '+%F %T %Z') qwen3_lume_hard_balanced_qlora"
echo "[INFO] train=${TRAIN_JSON}"
echo "[INFO] val=${VAL_JSON}"
echo "[INFO] model=${MODEL_PATH}"
echo "[INFO] output=${OUTPUT_DIR}"
echo "[INFO] final_adapter=${FINAL_MODEL_DIR}"
echo "[INFO] impact=adapter-only training; base weights unchanged"

python - <<PY
from pathlib import Path
required = ["${TRAIN_JSON}", "${VAL_JSON}", "${MODEL_PATH}"]
missing = [p for p in required if not Path(p).exists()]
if missing:
    raise SystemExit("missing required files: " + ", ".join(missing))
print("[INFO] required files present", flush=True)
PY

resume_args=()
if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
  resume_args+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

cd "${REPO_ROOT}/srcs/ffn_edit/train"
python -u ../utils/accelerate_cli.py launch \
  --num_processes 1 \
  --num_machines 1 \
  --main_process_port 12467 \
  run_clm_no_trainer.py \
  --model_name_or_path "${MODEL_PATH}" \
  --train_file "${TRAIN_JSON}" \
  --validation_file "${VAL_JSON}" \
  --config_name "${MODEL_PATH}" \
  --tokenizer_name "${MODEL_PATH}" \
  --sft_plain_completion \
  --use_lora \
  --load_in_4bit \
  --bnb_4bit_use_double_quant \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_dropout 0.02 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --num_train_epochs 3 \
  --max_train_steps 1000 \
  --checkpointing_steps 250 \
  --keep_last_checkpoints 2 \
  --eval_steps 250 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1.2e-4 \
  --lr_scheduler_type cosine \
  --num_warmup_steps 80 \
  --max_grad_norm 0.25 \
  --block_size 512 \
  --group_by_length \
  --output_dir "${OUTPUT_DIR}" \
  --torch_dtype bfloat16 \
  --low_cpu_mem_usage \
  --gradient_checkpointing \
  --dataloader_num_workers 2 \
  "${resume_args[@]}"

echo "[INFO] training finished; syncing adapter to ${FINAL_MODEL_DIR}"
rsync -a --delete --exclude "step_*" --exclude "epoch_*" --exclude "train.pid" "${OUTPUT_DIR}/" "${FINAL_MODEL_DIR}/"
echo "[DONE] $(date '+%F %T %Z') synced=${FINAL_MODEL_DIR}"
