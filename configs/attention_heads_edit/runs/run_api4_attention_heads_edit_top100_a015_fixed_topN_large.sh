#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${REPO_ROOT}/srcs/attention_heads_edit:${REPO_ROOT}/srcs/attention_heads_edit/scripts:${PYTHONPATH:-}"

RUN_NAME="${RUN_NAME:-api4_attention_heads_edit_top100_a015_fixed_topN_large_$(date +%H%M%S)}"
DATASET="${DATASET:-data/api4_200k/sft_true_prefix_no_instruction_all.json}"
BASE_MODEL="${BASE_MODEL:-models/llama3-8B/baseline}"
ADAPTER="${ADAPTER:-models/llama3-8B/api4_prefix_plain_qlora}"
BASE_EXACT="${BASE_EXACT:-outputs/attention_heads_edit/api4_repro_baseline_exact/metrics/metrics.json}"
BASE_CORE="${BASE_CORE:-outputs/attention_heads_edit/api4_repro_baseline_core/metrics/core_metrics.json}"
LOG_FILE="logs/attention_heads_edit/${RUN_NAME}.log"
SUMMARY_DIR="outputs/attention_heads_edit/metrics/${RUN_NAME}"
MANIFEST="configs/attention_heads_edit/runs/${RUN_NAME}_manifest.json"

mkdir -p logs/attention_heads_edit outputs/attention_heads_edit/metrics configs/attention_heads_edit/runs "${SUMMARY_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[START] $(date -Is) run=${RUN_NAME}"
echo "[INFO] alpha=0.15 scale_position=include_down fixed_topN=30,35,40,45"
nvidia-smi || true

METHODS_JSON=""
for N in 30 35 40 45; do
  METHOD="Attention_Heads_Edit_top100_a015_fixed_top${N}"
  HEAD_CONFIG="configs/attention_heads_edit/runs/api4_attention_heads_edit_top100_a015_fixed_topN_top${N}.json"
  EXACT_RUN="${RUN_NAME}_top${N}_exact_full"
  CORE_RUN="${RUN_NAME}_top${N}_core"

  echo "[STEP] $(date -Is) top${N} Exact/Contains head_config=${HEAD_CONFIG}"
  python srcs/attention_heads_edit/scripts/eval_api4_privacy_reverse_attention_heads_edit.py \
    --dataset "${DATASET}" \
    --base_model "${BASE_MODEL}" \
    --adapter "${ADAPTER}" \
    --head_config "${HEAD_CONFIG}" \
    --run_name "${EXACT_RUN}" \
    --batch_size 4 \
    --alpha 0.15 \
    --scale_position include_down \
    --max_context_tokens 768 \
    --max_new_tokens 64 \
    --generation_extra_tokens 8 \
    --load_in_4bit \
    --torch_dtype bfloat16 \
    --attn_implementation eager \
    --log_every 500

  echo "[STEP] $(date -Is) top${N} MRR/ASR/PPL"
  python srcs/attention_heads_edit/scripts/eval_api4_privacy_core_metrics.py \
    --dataset "${DATASET}" \
    --base_model "${BASE_MODEL}" \
    --adapter "${ADAPTER}" \
    --method_name "${METHOD}" \
    --run_name "${CORE_RUN}" \
    --head_config "${HEAD_CONFIG}" \
    --alpha 0.15 \
    --scale_position include_down \
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
      \"exact_metrics\": \"outputs/attention_heads_edit/${EXACT_RUN}/metrics/metrics.json\",
      \"core_metrics\": \"outputs/attention_heads_edit/${CORE_RUN}/metrics/core_metrics.json\"
    }"
done

cat > "${MANIFEST}" <<JSON
{
  "run_name": "${RUN_NAME}",
  "alpha": 0.15,
  "scale_position": "include_down",
  "selection": "top100 alpha=0.15 fixed top-N by intervention-aligned single-head benefit_sum",
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
