#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${REPO_ROOT}/srcs/attention_heads_edit:${REPO_ROOT}/srcs/attention_heads_edit/scripts:${REPO_ROOT}/srcs/ffn_edit/utils:${REPO_ROOT}/srcs/ffn_edit/eval:${PYTHONPATH:-}"

RUN_NAME="${RUN_NAME:-api4_ffn_edit256_then_attention_heads_edit_top30_alpha_sweep_$(date +%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/ffn_edit/ffn_edit256_then_attention_heads_edit_top30_alpha_sweep}"
SUMMARY_DIR="outputs/ffn_edit/metrics/${RUN_NAME}"
LOG_FILE="logs/ffn_edit/${RUN_NAME}.log"
MANIFEST="configs/ffn_edit/runs/${RUN_NAME}_manifest.json"

BASE_MODEL="${BASE_MODEL:-models/llama3-8B/baseline}"
ADAPTER="${ADAPTER:-models/llama3-8B/api4_prefix_plain_qlora}"
DATASET="${DATASET:-data/api4_200k/sft_true_prefix_no_instruction_all.json}"
ATTENTION_HEADS_EDIT_HEAD_CONFIG="${ATTENTION_HEADS_EDIT_HEAD_CONFIG:-configs/attention_heads_edit/runs/api4_attention_heads_edit_top100_a015_fixed_topN_top30.json}"
ALPHAS="${ALPHAS:-0.01 0.05 0.1 0.15 0.2 0.3 0.5}"

FFN_EDIT_KN_CONFIG="${FFN_EDIT_KN_CONFIG:-outputs/ffn_edit/ffn_edit_only_recheck_64_1024/api4_ffn_edit_only_recheck_64_1024/attribution/kn/kn_bag-api4_ffn_edit_only_recheck_64_1024.json}"
FFN_EDIT_ERASE_NUM="${FFN_EDIT_ERASE_NUM:-256}"

BASE_EXACT="${BASE_EXACT:-outputs/attention_heads_edit/api4_repro_baseline_exact/metrics/metrics.json}"
BASE_CORE="${BASE_CORE:-outputs/attention_heads_edit/api4_repro_baseline_core/metrics/core_metrics.json}"
FFN_EDIT256_EXACT="${FFN_EDIT256_EXACT:-outputs/ffn_edit/ffn_edit_only_recheck_64_1024/api4_ffn_edit_repro_erase256_exact_full/metrics/metrics.json}"
FFN_EDIT256_CORE="${FFN_EDIT256_CORE:-outputs/ffn_edit/ffn_edit_only_recheck_64_1024/api4_ffn_edit_repro_erase256_core/metrics/core_metrics.json}"

mkdir -p logs/ffn_edit configs/ffn_edit/runs "${OUTPUT_ROOT}" "${SUMMARY_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[START] $(date -Is) run=${RUN_NAME}"
echo "[INFO] FFN Edit-first experiment: reuse standalone FFN Edit positions, then apply fixed top30 AttentionHeadsEdit."
echo "[INFO] ffn_edit_kn_config=${FFN_EDIT_KN_CONFIG}"
echo "[INFO] ffn_edit_erase_num=${FFN_EDIT_ERASE_NUM}"
echo "[INFO] attention_heads_edit_head_config=${ATTENTION_HEADS_EDIT_HEAD_CONFIG}"
echo "[INFO] alphas=${ALPHAS}"
echo "[INFO] No FFN Edit attribution is rerun in this script."
nvidia-smi || true

python - <<PY
import json
from pathlib import Path
p = Path("${FFN_EDIT_KN_CONFIG}")
if not p.exists():
    raise SystemExit(f"missing ffn_edit kn config: {p}")
raw = json.loads(p.read_text())
seen = []
keys = set()
for item in raw:
    candidates = item if isinstance(item, list) and item and isinstance(item[0], list) else [item]
    for candidate in candidates:
        if isinstance(candidate, list) and len(candidate) >= 2:
            key = (int(candidate[0]), int(candidate[1]))
            if key not in keys:
                keys.add(key)
                seen.append(key)
            if len(seen) >= int("${FFN_EDIT_ERASE_NUM}"):
                break
    if len(seen) >= int("${FFN_EDIT_ERASE_NUM}"):
        break
print("[INFO] loaded standalone FFN Edit positions", len(seen), "from", p)
if len(seen) < int("${FFN_EDIT_ERASE_NUM}"):
    raise SystemExit(f"expected at least ${FFN_EDIT_ERASE_NUM} unique positions, got {len(seen)}")
PY

METHODS_JSON=""
for ALPHA in ${ALPHAS}; do
  SAFE_ALPHA="${ALPHA/./p}"
  METHOD="FFN_Edit256_then_Attention_Heads_Edit_top30_alpha_${SAFE_ALPHA}"
  EXACT_RUN="${RUN_NAME}_a${SAFE_ALPHA}_exact_full"
  CORE_RUN="${RUN_NAME}_a${SAFE_ALPHA}_core"

  echo "[STEP] $(date -Is) alpha=${ALPHA} Exact/Contains with standalone FFN Edit256 + top30 Attention_Heads_Edit"
  python srcs/attention_heads_edit/scripts/eval_api4_privacy_reverse_attention_heads_edit.py \
    --dataset "${DATASET}" \
    --base_model "${BASE_MODEL}" \
    --adapter "${ADAPTER}" \
    --head_config "${ATTENTION_HEADS_EDIT_HEAD_CONFIG}" \
    --run_name "${EXACT_RUN}" \
    --output_dir "${OUTPUT_ROOT}" \
    --batch_size 4 \
    --alpha "${ALPHA}" \
    --scale_position include_down \
    --ffn_edit_kn_config "${FFN_EDIT_KN_CONFIG}" \
    --ffn_edit_erase_num "${FFN_EDIT_ERASE_NUM}" \
    --max_context_tokens 768 \
    --max_new_tokens 64 \
    --generation_extra_tokens 8 \
    --load_in_4bit \
    --torch_dtype bfloat16 \
    --attn_implementation eager \
    --log_every 500

  echo "[STEP] $(date -Is) alpha=${ALPHA} MRR/ASR@32/PPL with standalone FFN Edit256 + top30 Attention_Heads_Edit"
  python srcs/attention_heads_edit/scripts/eval_api4_privacy_core_metrics.py \
    --dataset "${DATASET}" \
    --base_model "${BASE_MODEL}" \
    --adapter "${ADAPTER}" \
    --method_name "${METHOD}" \
    --run_name "${CORE_RUN}" \
    --output_dir "${OUTPUT_ROOT}" \
    --head_config "${ATTENTION_HEADS_EDIT_HEAD_CONFIG}" \
    --alpha "${ALPHA}" \
    --scale_position include_down \
    --ffn_edit_kn_config "${FFN_EDIT_KN_CONFIG}" \
    --ffn_edit_erase_num "${FFN_EDIT_ERASE_NUM}" \
    --run_mrr \
    --run_attack \
    --run_ppl \
    --mrr_limit_per_type 200 \
    --mrr_batch_size 2 \
    --attack_limit_per_type 10 \
    --attack_batch_size 4 \
    --attack_samples 32 \
    --attack_temperature 0.8 \
    --attack_top_p 0.95 \
    --ppl_max_blocks 128 \
    --ppl_block_tokens 512 \
    --ppl_batch_size 1 \
    --max_context_tokens 768 \
    --max_new_tokens 64 \
    --generation_extra_tokens 8 \
    --load_in_4bit \
    --torch_dtype bfloat16 \
    --attn_implementation eager \
    --log_every 100

  METHODS_JSON="${METHODS_JSON},
    {
      \"method\": \"${METHOD}\",
      \"exact_metrics\": \"${OUTPUT_ROOT}/${EXACT_RUN}/metrics/metrics.json\",
      \"core_metrics\": \"${OUTPUT_ROOT}/${CORE_RUN}/metrics/core_metrics.json\",
      \"ffn_edit_erase_num\": ${FFN_EDIT_ERASE_NUM},
      \"ffn_edit_kn_config\": \"${FFN_EDIT_KN_CONFIG}\",
      \"attention_heads_edit_alpha\": ${ALPHA}
    }"
done

cat > "${MANIFEST}" <<JSON
{
  "run_name": "${RUN_NAME}",
  "method_family": "FFN_Edit256_then_Attention_Heads_Edit_top30_alpha_sweep",
  "order": "standalone FFN Edit positions first, then fixed top30 Attention Heads Edit at evaluation",
  "attention_heads_edit_head_config": "${ATTENTION_HEADS_EDIT_HEAD_CONFIG}",
  "attention_heads_edit_alpha_list": "${ALPHAS}",
  "scale_position": "include_down",
  "ffn_edit_kn_config": "${FFN_EDIT_KN_CONFIG}",
  "ffn_edit_erase_num": ${FFN_EDIT_ERASE_NUM},
  "methods": [
    {
      "method": "Base",
      "exact_metrics": "${BASE_EXACT}",
      "core_metrics": "${BASE_CORE}"
    },
    {
      "method": "FFN_EDIT_recheck_erase256",
      "exact_metrics": "${FFN_EDIT256_EXACT}",
      "core_metrics": "${FFN_EDIT256_CORE}"
    }${METHODS_JSON}
  ]
}
JSON

python srcs/attention_heads_edit/scripts/summarize_api4_privacy_suite.py \
  --manifest "${MANIFEST}" \
  --output_dir "${SUMMARY_DIR}"

echo "[DONE] $(date -Is) run=${RUN_NAME}"
echo "[DONE] summary=${SUMMARY_DIR}/paper_main_table.csv"
