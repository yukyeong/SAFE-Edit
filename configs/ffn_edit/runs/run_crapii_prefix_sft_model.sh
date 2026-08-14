#!/usr/bin/env bash
# Train one model on CRAPII API4-style prefix-completion data.
# Input: data/crapii/api4_prefix/sft_true_prefix_no_instruction_{train,val}.json.
# Output: model-specific LoRA/QLoRA adapter under checkpoints/ and models/.
# Impact: adapter-only training; base model weights are not modified.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${REPO_ROOT}"

MODEL_KIND="${CRAPII_MODEL_KIND:?Set CRAPII_MODEL_KIND to one of: llama, qwen, ministral}"
TRAIN_DIR="${REPO_ROOT}/srcs/ffn_edit/train"
ACCELERATE_CLI="${REPO_ROOT}/srcs/ffn_edit/utils/accelerate_cli.py"
PYTHON_BIN="${CRAPII_PYTHON:-python}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

TRAIN_JSON="${CRAPII_TRAIN_JSON:-${REPO_ROOT}/data/crapii/api4_prefix/sft_true_prefix_no_instruction_train.json}"
VAL_JSON="${CRAPII_VAL_JSON:-${REPO_ROOT}/data/crapii/api4_prefix/sft_true_prefix_no_instruction_val.json}"

USE_4BIT=0
ATTN_IMPL="eager"
PORT=12374
LORA_TARGETS="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
LORA_R=64
LORA_ALPHA=128
LORA_DROPOUT=0.0
EPOCHS=8
CKPT_STEPS=250
GRAD_ACCUM=16
LR=2e-4
WARMUP_STEPS=30
BLOCK_SIZE=768
DATALOADER_WORKERS=2

case "${MODEL_KIND}" in
  llama)
    MODEL_LABEL="llama3_8b"
    MODEL_PATH="${CRAPII_MODEL_PATH:-${REPO_ROOT}/models/llama3-8B/baseline}"
    OUTPUT_DIR="${CRAPII_OUTPUT_DIR:-${REPO_ROOT}/checkpoints/ffn_edit/llama3-8b_lora/crapii_prefix_strong_r64_e8}"
    FINAL_MODEL_DIR="${CRAPII_FINAL_MODEL_DIR:-${REPO_ROOT}/models/llama3-8B/crapii_prefix_qlora_strong_r64_e8}"
    LOG_FILE="${CRAPII_LOG_FILE:-${REPO_ROOT}/logs/ffn_edit/train/run_crapii_llama3_8b_prefix_qlora_strong_$(date +%Y%m%d_%H%M%S).log}"
    USE_4BIT=1
    PORT=12374
    ;;
  qwen)
    MODEL_LABEL="qwen3_8b_base"
    MODEL_PATH="${CRAPII_MODEL_PATH:-${REPO_ROOT}/models/qwen3-8b-base}"
    OUTPUT_DIR="${CRAPII_OUTPUT_DIR:-${REPO_ROOT}/checkpoints/ffn_edit/qwen3-8b_lora/crapii_prefix_strong_r64_e8}"
    FINAL_MODEL_DIR="${CRAPII_FINAL_MODEL_DIR:-${REPO_ROOT}/models/qwen3-8b-base/crapii_prefix_qlora_strong_r64_e8}"
    LOG_FILE="${CRAPII_LOG_FILE:-${REPO_ROOT}/logs/ffn_edit/train/run_crapii_qwen3_8b_base_prefix_qlora_strong_$(date +%Y%m%d_%H%M%S).log}"
    USE_4BIT=1
    PORT=12376
    ;;
  ministral)
    MODEL_LABEL="ministral3_8b_base"
    MODEL_PATH="${CRAPII_MODEL_PATH:-${REPO_ROOT}/models/ministral-3-8b-base}"
    OUTPUT_DIR="${CRAPII_OUTPUT_DIR:-${REPO_ROOT}/checkpoints/ffn_edit/ministral3-8b-base_lora/crapii_prefix_strong_r64_e8}"
    FINAL_MODEL_DIR="${CRAPII_FINAL_MODEL_DIR:-${REPO_ROOT}/models/ministral-3-8b-base/crapii_prefix_qlora_strong_r64_e8}"
    LOG_FILE="${CRAPII_LOG_FILE:-${REPO_ROOT}/logs/ffn_edit/train/run_crapii_ministral3_8b_base_prefix_qlora_strong_$(date +%Y%m%d_%H%M%S).log}"
    PYTHON_BIN="${CRAPII_PYTHON:-${REPO_ROOT}/.venvs/ministral3/bin/python}"
    USE_4BIT=1
    PORT=12377
    LORA_TARGETS='regex:.*language_model\.layers\.[0-9]+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$'
    BLOCK_SIZE=512
    WARMUP_STEPS=30
    ;;
  *)
    echo "[crapii_sft] invalid CRAPII_MODEL_KIND=${MODEL_KIND}; expected llama, qwen, ministral" >&2
    exit 2
    ;;
esac

LORA_R="${CRAPII_LORA_R:-${LORA_R}}"
LORA_ALPHA="${CRAPII_LORA_ALPHA:-${LORA_ALPHA}}"
LORA_DROPOUT="${CRAPII_LORA_DROPOUT:-${LORA_DROPOUT}}"
EPOCHS="${CRAPII_EPOCHS:-${EPOCHS}}"
CKPT_STEPS="${CRAPII_CKPT_STEPS:-${CKPT_STEPS}}"
GRAD_ACCUM="${CRAPII_GRAD_ACCUM:-${GRAD_ACCUM}}"
LR="${CRAPII_LR:-${LR}}"
WARMUP_STEPS="${CRAPII_WARMUP_STEPS:-${WARMUP_STEPS}}"
BLOCK_SIZE="${CRAPII_BLOCK_SIZE:-${BLOCK_SIZE}}"
DATALOADER_WORKERS="${CRAPII_DATALOADER_WORKERS:-${DATALOADER_WORKERS}}"

mkdir -p "${OUTPUT_DIR}" "${FINAL_MODEL_DIR}" "$(dirname "${LOG_FILE}")"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[crapii_sft] START $(date '+%F %T %Z') model=${MODEL_KIND}"
echo "[crapii_sft] MODEL_PATH=${MODEL_PATH}"
echo "[crapii_sft] PYTHON_BIN=${PYTHON_BIN}"
echo "[crapii_sft] TRAIN_JSON=${TRAIN_JSON}"
echo "[crapii_sft] VAL_JSON=${VAL_JSON}"
echo "[crapii_sft] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[crapii_sft] FINAL_MODEL_DIR=${FINAL_MODEL_DIR}"
echo "[crapii_sft] impact=adapter-only training; base weights unchanged"
echo "[crapii_sft] config=r${LORA_R} alpha${LORA_ALPHA} dropout${LORA_DROPOUT} lr${LR} epochs${EPOCHS} effective_batch${GRAD_ACCUM} block_size${BLOCK_SIZE} use_4bit=${USE_4BIT}"

python - <<PY
import json
from pathlib import Path
required = [Path("${TRAIN_JSON}"), Path("${VAL_JSON}"), Path("${MODEL_PATH}"), Path("${PYTHON_BIN}")]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit("missing required paths: " + ", ".join(missing))
for label, path in [("train", Path("${TRAIN_JSON}")), ("val", Path("${VAL_JSON}"))]:
    rows = json.loads(path.read_text())
    if not rows:
        raise SystemExit(f"{label} dataset is empty: {path}")
    missing_fields = {"input", "output"} - set(rows[0])
    if missing_fields:
        raise SystemExit(f"{label} missing fields: {sorted(missing_fields)}")
    empty_inputs = sum(not str(row.get("input", "")).strip() for row in rows)
    empty_outputs = sum(not str(row.get("output", "")).strip() for row in rows)
    if empty_inputs or empty_outputs:
        raise SystemExit(f"{label} has empty input/output rows: input={empty_inputs} output={empty_outputs}")
    print(f"[crapii_sft] {label}_rows={len(rows)} pii_types={len({row.get('pii_type') for row in rows})}", flush=True)
PY

cmd=(
  "${PYTHON_BIN}" -u "${ACCELERATE_CLI}" launch
  --num_processes 1
  --num_machines 1
  --main_process_port "${PORT}"
  run_clm_no_trainer.py
  --model_name_or_path "${MODEL_PATH}"
  --train_file "${TRAIN_JSON}"
  --validation_file "${VAL_JSON}"
  --config_name "${MODEL_PATH}"
  --tokenizer_name "${MODEL_PATH}"
  --sft_plain_completion
  --use_lora
  --lora_r "${LORA_R}"
  --lora_alpha "${LORA_ALPHA}"
  --lora_dropout "${LORA_DROPOUT}"
  --lora_target_modules "${LORA_TARGETS}"
  --num_train_epochs "${EPOCHS}"
  --checkpointing_steps "${CKPT_STEPS}"
  --keep_last_checkpoints 2
  --per_device_train_batch_size 1
  --per_device_eval_batch_size 2
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  --learning_rate "${LR}"
  --lr_scheduler_type cosine
  --num_warmup_steps "${WARMUP_STEPS}"
  --max_grad_norm 0.3
  --block_size "${BLOCK_SIZE}"
  --group_by_length
  --output_dir "${OUTPUT_DIR}"
  --torch_dtype bfloat16
  --low_cpu_mem_usage
  --gradient_checkpointing
  --dataloader_num_workers "${DATALOADER_WORKERS}"
  --log_train_samples 2
)

if [ "${USE_4BIT}" = "1" ]; then
  cmd+=(--load_in_4bit --bnb_4bit_use_double_quant)
fi
if [ "${ATTN_IMPL}" = "sdpa" ]; then
  export TRANSFORMERS_ATTENTION_IMPLEMENTATION=sdpa
fi

cd "${TRAIN_DIR}"
"${cmd[@]}"

echo "[crapii_sft] training finished; syncing adapter to FINAL_MODEL_DIR=${FINAL_MODEL_DIR}"
rsync -a --delete --exclude "step_*" --exclude "epoch_*" --exclude "train.pid" "${OUTPUT_DIR}/" "${FINAL_MODEL_DIR}/"
test -s "${FINAL_MODEL_DIR}/adapter_config.json"
echo "[crapii_sft] synced to ${FINAL_MODEL_DIR}"
echo "[crapii_sft] DONE $(date '+%F %T %Z') model=${MODEL_KIND}"
