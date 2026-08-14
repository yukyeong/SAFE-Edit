#!/usr/bin/env bash
# Input: clean CRAPII train/all rows, Ministral CRAPII adapter, and existing clean-data FFN Edit16 ranking.
# Output: GQA-balanced Attention Heads Edit candidates, two-fold causal head config, and FFN Edit16->Attention Heads Edit full metrics.
# Impact: FFN Edit16 and model weights are unchanged; only evaluation-time Attention Heads Edit configs/metrics are written.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${REPO_ROOT}/srcs/attention_heads_edit:${REPO_ROOT}/srcs/attention_heads_edit/scripts:${REPO_ROOT}/srcs/ffn_edit/utils:${REPO_ROOT}/srcs/ffn_edit/eval:${PYTHONPATH:-}"

PYBIN="${PYBIN:-${REPO_ROOT}/.venvs/ministral3/bin/python}"
BASE_MODEL="${BASE_MODEL:-models/ministral-3-8b-base}"
ADAPTER="${ADAPTER:-models/ministral-3-8b-base/crapii_prefix_qlora_strong_r64_e8}"
TRAIN_DS="${TRAIN_DS:-data/crapii/api4_prefix_no_copy/sft_true_prefix_no_instruction_train.json}"
EVAL_DS="${EVAL_DS:-data/crapii/api4_prefix_no_copy/sft_true_prefix_no_instruction_all.json}"
FFN_EDIT_KN="${FFN_EDIT_KN:-}"
BASELINE_EXACT="${BASELINE_EXACT:-}"
BASELINE_CORE="${BASELINE_CORE:-}"
FFN_EDIT_EXACT="${FFN_EDIT_EXACT:-}"
FFN_EDIT_CORE="${FFN_EDIT_CORE:-}"

RUN_STAMP="${RUN_STAMP:-ffn_edit16_causalcv_tail16}"
RUN_NAME="${RUN_NAME:-crapii_ministral3_8b_ffn_edit16_causalcv_attention_heads_edit_top3_a0p02_${RUN_STAMP}}"
OUTPUT_ROOT="outputs/ffn_edit/crapii_ministral_causalcv_attention_heads_edit/${RUN_NAME}"
SUMMARY_DIR="outputs/ffn_edit/metrics/${RUN_NAME}"
LOG_FILE="logs/ffn_edit/${RUN_NAME}.log"
MANIFEST="configs/ffn_edit/runs/${RUN_NAME}_manifest.json"

ATTN_RUN="${RUN_NAME}_attn_nexttoken_adaptive_tail16_gqa_balanced_top256"
CANDIDATE_HEADS="configs/attention_heads_edit/runs/${ATTN_RUN}_top256.json"
SCREEN_RUN="${RUN_NAME}_post_ffn_edit16_causalcv"
HEAD_CONFIG="configs/attention_heads_edit/runs/${SCREEN_RUN}_causalcv_top3_alpha0p02.json"
JOINT_EXACT_RUN="${RUN_NAME}_joint_exact_full"
JOINT_CORE_RUN="${RUN_NAME}_joint_core"
JOINT_METHOD="CRAPII_Ministral3_8B_Base_clean_FFN_Edit16_then_causalCV_Attention_Heads_Edit_top3_alpha_0p02_tail16"

mkdir -p logs/ffn_edit configs/ffn_edit/runs configs/attention_heads_edit/runs outputs/ffn_edit/metrics "${OUTPUT_ROOT}" "${SUMMARY_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

wait_gpu_idle() {
  while true; do
    local used
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ' || echo 0)"
    if [ "${used:-99999}" -lt 1000 ]; then
      break
    fi
    echo "[WAIT] $(date -Is) GPU busy: ${used} MiB"
    sleep 120
  done
}

echo "[START] $(date -Is) run=${RUN_NAME}"
echo "[INFO] fixed_ffn_edit=16 existing_kn=${FFN_EDIT_KN}"
echo "[INFO] Attention Heads Edit=candidate256,layer8,kv_group1,causal_preselect32,source_disjoint_cv,top3,alpha0.02,adaptive_tail16"

"${PYBIN}" - <<PY
from pathlib import Path
required = [
    "${TRAIN_DS}", "${EVAL_DS}", "${BASE_MODEL}/config.json", "${ADAPTER}/adapter_config.json",
    "${FFN_EDIT_KN}", "${BASELINE_EXACT}", "${BASELINE_CORE}", "${FFN_EDIT_EXACT}", "${FFN_EDIT_CORE}",
]
missing = [path for path in required if not Path(path).exists() or Path(path).stat().st_size == 0]
if missing:
    raise SystemExit("missing required files: " + ", ".join(missing))
print("[INFO] prerequisites_ok", flush=True)
PY

echo "[STEP 1/4] $(date -Is) locate GQA-balanced Attention Heads Edit candidate pool"
wait_gpu_idle
rm -f "${CANDIDATE_HEADS}"
"${PYBIN}" srcs/attention_heads_edit/scripts/locate_api4_attention_heads.py \
  --dataset "${TRAIN_DS}" \
  --base_model "${BASE_MODEL}" \
  --adapter "${ADAPTER}" \
  --run_name "${ATTN_RUN}" \
  --top_k 256 \
  --max_heads_per_layer 8 \
  --max_heads_per_kv_group 1 \
  --max_samples 0 \
  --batch_size 1 \
  --ranking_metric first_mass \
  --query_mode next_token \
  --steering_span_mode adaptive_tail \
  --steering_tail_tokens 16 \
  --max_context_tokens 512 \
  --load_in_4bit \
  --torch_dtype bfloat16 \
  --attn_implementation eager \
  --metrics_dir outputs/ffn_edit/metrics \
  --config_dir configs/attention_heads_edit/runs \
  --log_every 50
test -s "${CANDIDATE_HEADS}"

echo "[STEP 2/4] $(date -Is) causal first-token preselect and source-disjoint Exact CV"
wait_gpu_idle
rm -f "${HEAD_CONFIG}"
"${PYBIN}" srcs/attention_heads_edit/scripts/select_attention_heads_causal_cv_after_ffn_edit.py \
  --dataset "${TRAIN_DS}" \
  --candidate_heads "${CANDIDATE_HEADS}" \
  --base_model "${BASE_MODEL}" \
  --adapter "${ADAPTER}" \
  --ffn_edit_kn_config "${FFN_EDIT_KN}" \
  --ffn_edit_erase_num 16 \
  --run_name "${SCREEN_RUN}" \
  --output_dir "${OUTPUT_ROOT}" \
  --config_dir configs/attention_heads_edit/runs \
  --candidate_limit 256 \
  --causal_preselect_limit 32 \
  --top_k 3 \
  --screen_limit_per_type 20 \
  --per_type_min_samples 5 \
  --alpha 0.02 \
  --scale_position include_down \
  --steering_span_mode adaptive_tail \
  --steering_tail_tokens 16 \
  --prefix_sizes 1,2,3 \
  --batch_size 1 \
  --max_context_tokens 512 \
  --max_new_tokens 96 \
  --generation_extra_tokens 8 \
  --load_in_4bit \
  --torch_dtype bfloat16 \
  --attn_implementation eager \
  --log_every 16
test -s "${HEAD_CONFIG}"

echo "[STEP 3/4] $(date -Is) FFN Edit16 then causal-CV Attention Heads Edit Exact/Contains"
wait_gpu_idle
"${PYBIN}" srcs/attention_heads_edit/scripts/eval_api4_privacy_reverse_attention_heads_edit.py \
  --dataset "${EVAL_DS}" \
  --base_model "${BASE_MODEL}" \
  --adapter "${ADAPTER}" \
  --head_config "${HEAD_CONFIG}" \
  --run_name "${JOINT_EXACT_RUN}" \
  --output_dir "${OUTPUT_ROOT}" \
  --batch_size 1 \
  --alpha 0.02 \
  --scale_position include_down \
  --steering_span_mode adaptive_tail \
  --steering_tail_tokens 16 \
  --ffn_edit_kn_config "${FFN_EDIT_KN}" \
  --ffn_edit_erase_num 16 \
  --max_context_tokens 512 \
  --max_new_tokens 96 \
  --generation_extra_tokens 8 \
  --load_in_4bit \
  --torch_dtype bfloat16 \
  --attn_implementation eager \
  --log_every 100

echo "[STEP 4/4] $(date -Is) MRR/ASR/PPL and summary"
"${PYBIN}" srcs/attention_heads_edit/scripts/eval_api4_privacy_core_metrics.py \
  --dataset "${EVAL_DS}" \
  --method_name "${JOINT_METHOD}" \
  --base_model "${BASE_MODEL}" \
  --adapter "${ADAPTER}" \
  --head_config "${HEAD_CONFIG}" \
  --run_name "${JOINT_CORE_RUN}" \
  --output_dir "${OUTPUT_ROOT}" \
  --alpha 0.02 \
  --scale_position include_down \
  --steering_span_mode adaptive_tail \
  --steering_tail_tokens 16 \
  --ppl_attention_heads_edit_mode none \
  --ffn_edit_kn_config "${FFN_EDIT_KN}" \
  --ffn_edit_erase_num 16 \
  --run_mrr \
  --run_attack \
  --run_ppl \
  --mrr_limit_per_type 200 \
  --mrr_batch_size 1 \
  --attack_limit_per_type 10 \
  --attack_batch_size 1 \
  --attack_samples 32 \
  --attack_temperature 0.8 \
  --attack_top_p 0.95 \
  --ppl_max_blocks 128 \
  --ppl_block_tokens 512 \
  --ppl_batch_size 1 \
  --max_context_tokens 512 \
  --max_new_tokens 96 \
  --generation_extra_tokens 8 \
  --load_in_4bit \
  --torch_dtype bfloat16 \
  --attn_implementation eager \
  --log_every 50

cat > "${MANIFEST}" <<JSON
{
  "run_name": "${RUN_NAME}",
  "dataset": "CRAPII_no_copy",
  "order": "existing clean-data FFN Edit16 then independently relocated causal-CV Attention_Heads_Edit",
  "ffn_edit_kn_config": "${FFN_EDIT_KN}",
  "ffn_edit_erase_num": 16,
  "attention_heads_edit": {
    "head_config": "${HEAD_CONFIG}",
    "candidate_top_k": 256,
    "causal_preselect_k": 32,
    "selected_top_k": 3,
    "alpha": 0.02,
    "steering_span_mode": "adaptive_tail",
    "steering_tail_tokens": 16,
    "cross_validation": "two source-disjoint folds"
  },
  "methods": [
    {"method": "CRAPII_Ministral3_8B_Base_Strong_SFT_no_defense", "exact_metrics": "${BASELINE_EXACT}", "core_metrics": "${BASELINE_CORE}"},
    {"method": "CRAPII_Ministral3_8B_Base_Strong_SFT_FFN_Edit16_fixeddata", "exact_metrics": "${FFN_EDIT_EXACT}", "core_metrics": "${FFN_EDIT_CORE}"},
    {"method": "${JOINT_METHOD}", "exact_metrics": "${OUTPUT_ROOT}/${JOINT_EXACT_RUN}/metrics/metrics.json", "core_metrics": "${OUTPUT_ROOT}/${JOINT_CORE_RUN}/metrics/core_metrics.json"}
  ]
}
JSON

"${PYBIN}" srcs/attention_heads_edit/scripts/summarize_api4_privacy_suite.py \
  --manifest "${MANIFEST}" \
  --output_dir "${SUMMARY_DIR}"

echo "[DONE] $(date -Is) summary=${SUMMARY_DIR}/paper_main_table.csv"
cat "${SUMMARY_DIR}/paper_main_table.csv"
