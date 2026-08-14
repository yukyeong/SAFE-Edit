#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${REPO_ROOT}/srcs/attention_heads_edit:${REPO_ROOT}/srcs/attention_heads_edit/scripts:${REPO_ROOT}/srcs/ffn_edit/utils:${REPO_ROOT}/srcs/ffn_edit/eval:${PYTHONPATH:-}"

RUN_NAME="${RUN_NAME:-api4_ffn_edit_only_recheck_64_1024_$(date +%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/ffn_edit/ffn_edit_only_recheck_64_1024}"
ATTR_DIR="${OUTPUT_ROOT}/${RUN_NAME}/attribution"
SUMMARY_DIR="outputs/ffn_edit/metrics/${RUN_NAME}"
LOG_FILE="logs/ffn_edit/${RUN_NAME}.log"
MANIFEST="configs/ffn_edit/runs/${RUN_NAME}_manifest.json"

BASE_EXACT="${BASE_EXACT:-outputs/attention_heads_edit/api4_repro_baseline_exact/metrics/metrics.json}"
BASE_CORE="${BASE_CORE:-outputs/attention_heads_edit/api4_repro_baseline_core/metrics/core_metrics.json}"

BASE_MODEL="${BASE_MODEL:-models/llama3-8B/baseline}"
ADAPTER="${ADAPTER:-models/llama3-8B/api4_prefix_plain_qlora}"
DATASET="${DATASET:-data/api4_200k/sft_true_prefix_no_instruction_all.json}"
PRIV_DATA="${PRIV_DATA:-data/api4_200k/privacy_data_api4_all.json}"

FFN_EDIT_TOP_K="${FFN_EDIT_TOP_K:-1024}"
ERASE_LIST="${ERASE_LIST:-64 128 256 512 1024}"
FFN_EDIT_BATCH_SIZE="${FFN_EDIT_BATCH_SIZE:-4}"
FFN_EDIT_NUM_BATCH="${FFN_EDIT_NUM_BATCH:-4}"
FFN_EDIT_LAYER_INDICES="${FFN_EDIT_LAYER_INDICES:-0,29,30,31}"
FFN_EDIT_LIMIT_BAGS="${FFN_EDIT_LIMIT_BAGS:-0}"

mkdir -p logs/ffn_edit configs/ffn_edit/runs "${OUTPUT_ROOT}" "${ATTR_DIR}" "${SUMMARY_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[START] $(date -Is) run=${RUN_NAME}"
echo "[INFO] standalone FFN Edit recheck; Attention Heads Edit disabled for attribution and all evaluations"
echo "[INFO] output_root=${OUTPUT_ROOT}"
echo "[INFO] ffn_edit_top_k=${FFN_EDIT_TOP_K} erase_list=${ERASE_LIST}"
echo "[INFO] layer_indices=${FFN_EDIT_LAYER_INDICES} batch_size=${FFN_EDIT_BATCH_SIZE} num_batch=${FFN_EDIT_NUM_BATCH}"
echo "[INFO] No legacy 90G *.priv.jsonl is read or written."
nvidia-smi || true

limit_args=()
if [ "${FFN_EDIT_LIMIT_BAGS}" != "0" ]; then
  limit_args+=(--limit_bags "${FFN_EDIT_LIMIT_BAGS}")
fi

echo "[STEP 1] $(date -Is) compact standalone FFN Edit attribution top_k=${FFN_EDIT_TOP_K}"
python -u srcs/ffn_edit/eval/compact_attribution_llama.py \
  --priv_data_path "${PRIV_DATA}" \
  --model_name_or_path "${BASE_MODEL}" \
  --adapter_dir "${ADAPTER}" \
  --output_dir "${ATTR_DIR}" \
  --output_prefix "${RUN_NAME}" \
  --rel "${RUN_NAME}" \
  --max_seq_length 128 \
  --batch_size "${FFN_EDIT_BATCH_SIZE}" \
  --num_batch "${FFN_EDIT_NUM_BATCH}" \
  --layer_indices "${FFN_EDIT_LAYER_INDICES}" \
  --threshold_ratio 0.1 \
  --top_k "${FFN_EDIT_TOP_K}" \
  --save_every 25 \
  --progress_every 25 \
  "${limit_args[@]}"

FFN_EDIT_KN_CONFIG="${ATTR_DIR}/kn/kn_bag-${RUN_NAME}.json"
python - <<PY
import json
from pathlib import Path
p = Path("${FFN_EDIT_KN_CONFIG}")
j = json.loads(p.read_text())
seen = []
keys = set()
for item in j:
    candidates = item if isinstance(item, list) and item and isinstance(item[0], list) else [item]
    for candidate in candidates:
        if isinstance(candidate, list) and len(candidate) >= 2:
            key = (int(candidate[0]), int(candidate[1]))
            if key not in keys:
                keys.add(key)
                seen.append(key)
print("[INFO] ffn_edit_kn_config", p, "bags", len(j), "unique_positions", len(seen), "bytes", p.stat().st_size)
if len(seen) < ${FFN_EDIT_TOP_K}:
    raise SystemExit(f"expected at least ${FFN_EDIT_TOP_K} unique positions, got {len(seen)}")
PY

METHODS_JSON=""
for ERASE_NUM in ${ERASE_LIST}; do
  METHOD="FFN_EDIT_recheck_erase${ERASE_NUM}"
  EXACT_RUN="${RUN_NAME}_erase${ERASE_NUM}_exact_full"
  CORE_RUN="${RUN_NAME}_erase${ERASE_NUM}_core"

  echo "[STEP] $(date -Is) erase=${ERASE_NUM} Exact/Contains"
  python srcs/attention_heads_edit/scripts/eval_api4_privacy_reverse_attention_heads_edit.py \
    --dataset "${DATASET}" \
    --base_model "${BASE_MODEL}" \
    --adapter "${ADAPTER}" \
    --run_name "${EXACT_RUN}" \
    --output_dir "${OUTPUT_ROOT}" \
    --batch_size 4 \
    --disable_attention_heads_edit \
    --ffn_edit_kn_config "${FFN_EDIT_KN_CONFIG}" \
    --ffn_edit_erase_num "${ERASE_NUM}" \
    --max_context_tokens 768 \
    --max_new_tokens 64 \
    --generation_extra_tokens 8 \
    --load_in_4bit \
    --torch_dtype bfloat16 \
    --attn_implementation eager \
    --log_every 500

  echo "[STEP] $(date -Is) erase=${ERASE_NUM} MRR/ASR@32/PPL"
  python srcs/attention_heads_edit/scripts/eval_api4_privacy_core_metrics.py \
    --dataset "${DATASET}" \
    --base_model "${BASE_MODEL}" \
    --adapter "${ADAPTER}" \
    --method_name "${METHOD}" \
    --run_name "${CORE_RUN}" \
    --output_dir "${OUTPUT_ROOT}" \
    --disable_attention_heads_edit \
    --ffn_edit_kn_config "${FFN_EDIT_KN_CONFIG}" \
    --ffn_edit_erase_num "${ERASE_NUM}" \
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
      \"ffn_edit_erase_num\": ${ERASE_NUM},
      \"ffn_edit_kn_config\": \"${FFN_EDIT_KN_CONFIG}\"
    }"
done

cat > "${MANIFEST}" <<JSON
{
  "run_name": "${RUN_NAME}",
  "method_family": "FFN_EDIT_only_recheck_64_1024",
  "attention_heads_edit_disabled": true,
  "ffn_edit_top_k": ${FFN_EDIT_TOP_K},
  "erase_list": "${ERASE_LIST}",
  "ffn_edit_kn_config": "${FFN_EDIT_KN_CONFIG}",
  "ffn_edit_attribution": {
    "privacy_data": "${PRIV_DATA}",
    "model_name_or_path": "${BASE_MODEL}",
    "adapter_dir": "${ADAPTER}",
    "layer_indices": "${FFN_EDIT_LAYER_INDICES}",
    "batch_size": ${FFN_EDIT_BATCH_SIZE},
    "num_batch": ${FFN_EDIT_NUM_BATCH},
    "threshold_ratio": 0.1,
    "attention_heads_edit_aware": false,
    "raw_priv_jsonl_bypassed": true,
    "single_top1024_prefix_reused_for_all_erase_nums": true
  },
  "methods": [
    {
      "method": "Base",
      "exact_metrics": "${BASE_EXACT}",
      "core_metrics": "${BASE_CORE}"
    }${METHODS_JSON}
  ]
}
JSON

python srcs/attention_heads_edit/scripts/summarize_api4_privacy_suite.py \
  --manifest "${MANIFEST}" \
  --output_dir "${SUMMARY_DIR}"

echo "[DONE] $(date -Is) run=${RUN_NAME}"
echo "[DONE] summary=${SUMMARY_DIR}/paper_main_table.csv"
