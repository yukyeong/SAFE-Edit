#!/usr/bin/env bash
# Evaluate one CRAPII prefix SFT adapter on full privacy metrics.
# Input: data/crapii/api4_prefix/sft_true_prefix_no_instruction_all.json + model-specific adapter.
# Output: Exact/Contains, Macro-MRR, ASR@32, and WikiText PPL tables.
# Impact: evaluation only; model weights are not modified.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${REPO_ROOT}"

MODEL_KIND="${CRAPII_MODEL_KIND:?Set CRAPII_MODEL_KIND to one of: llama, qwen, ministral}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${REPO_ROOT}/srcs/attention_heads_edit:${REPO_ROOT}/srcs/attention_heads_edit/scripts:${PYTHONPATH:-}"

DATASET="${CRAPII_DATASET:-data/crapii/api4_prefix/sft_true_prefix_no_instruction_all.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/ffn_edit/crapii_prefix_sft_no_defense}"
EXACT_BATCH_SIZE="${EXACT_BATCH_SIZE:-4}"
ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-4}"
LOAD_IN_4BIT_ARGS=()
ATTN_IMPL="eager"
PYTHON_BIN="${CRAPII_PYTHON:-python}"

case "${MODEL_KIND}" in
  llama)
    MODEL_LABEL="Llama3_8B"
    BASE_MODEL="${BASE_MODEL:-models/llama3-8B/baseline}"
    ADAPTER="${ADAPTER:-models/llama3-8B/crapii_prefix_qlora_strong_r64_e8}"
    METHOD="CRAPII_${MODEL_LABEL}_Strong_SFT_no_defense"
    RUN_NAME="${RUN_NAME:-crapii_llama3_8b_strong_sft_no_defense_$(date +%Y%m%d_%H%M%S)}"
    LOAD_IN_4BIT_ARGS=(--load_in_4bit)
    ;;
  qwen)
    MODEL_LABEL="Qwen3_8B_Base"
    BASE_MODEL="${BASE_MODEL:-models/qwen3-8b-base}"
    ADAPTER="${ADAPTER:-models/qwen3-8b-base/crapii_prefix_qlora_strong_r64_e8}"
    METHOD="CRAPII_${MODEL_LABEL}_Strong_SFT_no_defense"
    RUN_NAME="${RUN_NAME:-crapii_qwen3_8b_base_strong_sft_no_defense_$(date +%Y%m%d_%H%M%S)}"
    LOAD_IN_4BIT_ARGS=(--load_in_4bit)
    ;;
  ministral)
    MODEL_LABEL="Ministral3_8B_Base"
    BASE_MODEL="${BASE_MODEL:-models/ministral-3-8b-base}"
    ADAPTER="${ADAPTER:-models/ministral-3-8b-base/crapii_prefix_qlora_strong_r64_e8}"
    METHOD="CRAPII_${MODEL_LABEL}_Strong_SFT_no_defense"
    RUN_NAME="${RUN_NAME:-crapii_ministral3_8b_base_strong_sft_no_defense_$(date +%Y%m%d_%H%M%S)}"
    PYTHON_BIN="${CRAPII_PYTHON:-${REPO_ROOT}/.venvs/ministral3/bin/python}"
    LOAD_IN_4BIT_ARGS=(--load_in_4bit)
    ATTN_IMPL="eager"
    ;;
  *)
    echo "[crapii_metrics] invalid CRAPII_MODEL_KIND=${MODEL_KIND}; expected llama, qwen, ministral" >&2
    exit 2
    ;;
esac

SUMMARY_DIR="${SUMMARY_DIR:-outputs/ffn_edit/metrics/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-logs/ffn_edit/${RUN_NAME}.log}"
MANIFEST="${MANIFEST:-configs/ffn_edit/runs/${RUN_NAME}_manifest.json}"
EXACT_RUN="${RUN_NAME}_exact_full"
CORE_RUN="${RUN_NAME}_core"

mkdir -p logs/ffn_edit configs/ffn_edit/runs configs/attention_heads_edit/runs "${OUTPUT_ROOT}" "${SUMMARY_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[START] $(date '+%F %T %Z') run=${RUN_NAME}"
echo "[INFO] dataset=${DATASET}"
echo "[INFO] base_model=${BASE_MODEL}"
echo "[INFO] adapter=${ADAPTER}"
echo "[INFO] python=${PYTHON_BIN}"
echo "[INFO] impact: evaluation only; writes metrics and hashed predictions; no weights modified."
nvidia-smi || true

python - <<PY
import json
from pathlib import Path
required = [Path("${DATASET}"), Path("${BASE_MODEL}"), Path("${ADAPTER}"), Path("${ADAPTER}/adapter_config.json")]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit("missing required files: " + ", ".join(missing))
rows = json.loads(Path("${DATASET}").read_text())
if not rows:
    raise SystemExit("eval dataset is empty")
empty_inputs = sum(not str(row.get("input", "")).strip() for row in rows)
empty_outputs = sum(not str(row.get("output", "")).strip() for row in rows)
if empty_inputs or empty_outputs:
    raise SystemExit(f"eval dataset has empty input/output rows: input={empty_inputs} output={empty_outputs}")
print(f"[INFO] eval_rows={len(rows)} pii_types={len({row.get('pii_type') for row in rows})}", flush=True)
PY

echo "[STEP] $(date '+%F %T %Z') ${MODEL_KIND} CRAPII Exact/Contains"
"${PYTHON_BIN}" srcs/attention_heads_edit/scripts/eval_api4_privacy_reverse_attention_heads_edit.py \
  --dataset "${DATASET}" \
  --base_model "${BASE_MODEL}" \
  --adapter "${ADAPTER}" \
  --run_name "${EXACT_RUN}" \
  --output_dir "${OUTPUT_ROOT}" \
  --disable_attention_heads_edit \
  --batch_size "${EXACT_BATCH_SIZE}" \
  --max_context_tokens 512 \
  --max_new_tokens 96 \
  --generation_extra_tokens 8 \
  "${LOAD_IN_4BIT_ARGS[@]}" \
  --torch_dtype bfloat16 \
  --attn_implementation "${ATTN_IMPL}" \
  --log_every 200

echo "[STEP] $(date '+%F %T %Z') ${MODEL_KIND} CRAPII MRR/ASR@32/PPL"
"${PYTHON_BIN}" srcs/attention_heads_edit/scripts/eval_api4_privacy_core_metrics.py \
  --dataset "${DATASET}" \
  --method_name "${METHOD}" \
  --base_model "${BASE_MODEL}" \
  --adapter "${ADAPTER}" \
  --run_name "${CORE_RUN}" \
  --output_dir "${OUTPUT_ROOT}" \
  --disable_attention_heads_edit \
  --run_mrr \
  --run_attack \
  --run_ppl \
  --mrr_limit_per_type 200 \
  --mrr_batch_size 2 \
  --attack_limit_per_type 10 \
  --attack_batch_size "${ATTACK_BATCH_SIZE}" \
  --attack_samples 32 \
  --attack_temperature 0.8 \
  --attack_top_p 0.95 \
  --ppl_max_blocks 128 \
  --ppl_block_tokens 512 \
  --ppl_batch_size 1 \
  --max_context_tokens 512 \
  --max_new_tokens 96 \
  --generation_extra_tokens 8 \
  "${LOAD_IN_4BIT_ARGS[@]}" \
  --torch_dtype bfloat16 \
  --attn_implementation "${ATTN_IMPL}" \
  --log_every 100

cat > "${MANIFEST}" <<JSON
{
  "run_name": "${RUN_NAME}",
  "method_family": "CRAPII_Strong_SFT_no_defense",
  "dataset": "${DATASET}",
  "base_model": "${BASE_MODEL}",
  "adapter": "${ADAPTER}",
  "model_kind": "${MODEL_KIND}",
  "methods": [
    {
      "method": "${METHOD}",
      "exact_metrics": "${OUTPUT_ROOT}/${EXACT_RUN}/metrics/metrics.json",
      "core_metrics": "${OUTPUT_ROOT}/${CORE_RUN}/metrics/core_metrics.json"
    }
  ]
}
JSON

"${PYTHON_BIN}" srcs/attention_heads_edit/scripts/summarize_api4_privacy_suite.py \
  --manifest "${MANIFEST}" \
  --output_dir "${SUMMARY_DIR}"

test -s "${SUMMARY_DIR}/paper_main_table.csv"
echo "[DONE] $(date '+%F %T %Z') run=${RUN_NAME}"
echo "[DONE] summary=${SUMMARY_DIR}/paper_main_table.csv"
