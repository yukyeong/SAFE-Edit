#!/usr/bin/env bash
# LUME Task2 hard-balanced plain-prefix QLoRA for Ministral 3 8B Base 2512.
# Input: data/lume/api4_prefix_task2_hard_balanced train/val JSON + local Ministral3 base checkpoint.
# Output: QLoRA adapter under checkpoints/ and synced final adapter under models/ministral-3-8b-base/.
# Impact: trains adapter weights only; base Ministral3 weights are not modified.

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../_common.sh"


MINISTRAL_PYTHON="${MINISTRAL_PYTHON:-${REPO_ROOT}/.venvs/ministral3/bin/python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

TRAIN_JSON="${MINISTRAL_TRAIN_JSON:-${REPO_ROOT}/data/lume/api4_prefix_task2_hard_balanced/sft_true_prefix_no_instruction_train.json}"
VAL_JSON="${MINISTRAL_VAL_JSON:-${REPO_ROOT}/data/lume/api4_prefix_task2_hard_balanced/sft_true_prefix_no_instruction_val.json}"
MODEL_PATH="${MINISTRAL_MODEL_PATH:-${REPO_ROOT}/models/ministral-3-8b-base}"
OUTPUT_DIR="${MINISTRAL_OUTPUT_DIR:-${REPO_ROOT}/checkpoints/ffn_edit/ministral3-8b-base_lora/lume_task2_hard_balanced_r64}"
FINAL_MODEL_DIR="${MINISTRAL_FINAL_MODEL_DIR:-${REPO_ROOT}/models/ministral-3-8b-base/lume_task2_hard_balanced_qlora}"
LOG_FILE="${MINISTRAL_LOG_FILE:-${REPO_ROOT}/logs/ffn_edit/train/run_ministral3_8b_base_lume_task2_hard_balanced_qlora_$(date +%Y%m%d_%H%M%S).log}"
RESUME_FROM_CHECKPOINT="${MINISTRAL_RESUME_FROM_CHECKPOINT:-}"

mkdir -p "${OUTPUT_DIR}" "${FINAL_MODEL_DIR}" "$(dirname "${LOG_FILE}")"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[START] $(date '+%F %T %Z') ministral3_8b_base_lume_hard_balanced_qlora"
echo "[INFO] train=${TRAIN_JSON}"
echo "[INFO] val=${VAL_JSON}"
echo "[INFO] model=${MODEL_PATH}"
echo "[INFO] output=${OUTPUT_DIR}"
echo "[INFO] final_adapter=${FINAL_MODEL_DIR}"
echo "[INFO] python=${MINISTRAL_PYTHON}"
echo "[INFO] impact=adapter-only training; base weights unchanged"
nvidia-smi || true

"${MINISTRAL_PYTHON}" - <<PY
import importlib.metadata as metadata
import json
from collections import Counter
from pathlib import Path

required = [
    "${TRAIN_JSON}",
    "${VAL_JSON}",
    "${MODEL_PATH}/config.json",
    "${MODEL_PATH}/model.safetensors.index.json",
    "${MODEL_PATH}/tokenizer.json",
    "${MINISTRAL_PYTHON}",
]
missing = [p for p in required if not Path(p).exists()]
if missing:
    raise SystemExit("missing required files: " + ", ".join(missing))
for pkg in ("transformers", "mistral-common", "peft", "bitsandbytes", "accelerate"):
    print(f"[INFO] {pkg}={metadata.version(pkg)}", flush=True)
rows = json.load(open("${TRAIN_JSON}", encoding="utf-8"))
counts = Counter(row.get("pii_type") for row in rows if isinstance(row, dict))
print("[INFO] train_pii_counts=" + json.dumps(dict(sorted(counts.items())), ensure_ascii=False), flush=True)
if len(counts) < 5 or any(value < 700 for value in counts.values()):
    raise SystemExit("hard-balanced train data is not sufficiently multi-type")
PY

resume_args=()
if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
  resume_args+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

cd "${REPO_ROOT}/src/ffn_edit/train"
"${MINISTRAL_PYTHON}" -u -m ffn_edit.train.accelerate_cli launch \
  --num_processes 1 \
  --num_machines 1 \
  --main_process_port 12483 \
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
  --lora_target_modules 'regex:.*language_model\.layers\.[0-9]+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$' \
  --num_train_epochs 3 \
  --max_train_steps 1400 \
  --checkpointing_steps 350 \
  --keep_last_checkpoints 2 \
  --eval_steps 350 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1.2e-4 \
  --lr_scheduler_type cosine \
  --num_warmup_steps 100 \
  --max_grad_norm 0.25 \
  --weight_decay 0.0 \
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
