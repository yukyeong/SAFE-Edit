#!/usr/bin/env bash
# Run one LUME SFT adapter through optimized FFN Edit128 -> AttentionHeadsEdit.
# Input: processed LUME train/eval JSON, model-specific LoRA adapter, model-local FFN Edit KN ranking.
# Output: exact-safe Attention Heads Edit head config and full Exact/Contains, MRR, ASR@32, PPL paper metrics.
# Impact: evaluation-time FFN Edit/Attention Heads Edit only; no model weights or datasets are modified.

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../_common.sh"


MODEL_KIND="${LUME_MODEL_KIND:?Set LUME_MODEL_KIND to one of: llama, qwen, ministral}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

RUN_STAMP="${RUN_STAMP:-lume_exactsafe_joint}"
TRAIN_DS="${TRAIN_DS:-data/lume/api4_prefix_task2/sft_true_prefix_no_instruction_scenario_clean_train.json}"
SCREEN_DS="${SCREEN_DS:-data/lume/api4_prefix_task2/sft_true_prefix_no_instruction_scenario_clean_val.json}"
EVAL_DS="${EVAL_DS:-data/lume/api4_prefix_task2/sft_true_prefix_no_instruction_scenario_clean_all.json}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-outputs/ffn_edit/lume_ffn_edit_then_attention_heads_edit_exactsafe}"
GPU_IDLE_MEM_MB="${GPU_IDLE_MEM_MB:-1000}"
ATTN_IMPL="${ATTN_IMPL:-eager}"
FFN_EDIT_ERASE_NUM="${FFN_EDIT_ERASE_NUM:-128}"
SCALE_POSITION="${SCALE_POSITION:-include_down}"
STEERING_SPAN_MODE="${STEERING_SPAN_MODE:-tail_tokens}"
STEERING_TAIL_TOKENS="${STEERING_TAIL_TOKENS:-8}"
QUERY_MODE="${QUERY_MODE:-next_token}"
PYBIN="${LUME_PYTHON:-python}"

case "${MODEL_KIND}" in
  llama)
    MODEL_LABEL="Llama3_8B"
    MODEL_TAG="llama3_8b"
    BASE_MODEL="${BASE_MODEL:-models/llama3-8B/baseline}"
    ADAPTER="${ADAPTER:-models/llama3-8B/lume_task2_prefix_plain_qlora}"
    FFN_EDIT_KN="${FFN_EDIT_KN:-outputs/ffn_edit/lume_task2_ablation/lume_task2_ffn_edit256_and_attention_heads_edit_top30_a001/ffn_edit256_attribution/kn/kn_bag-lume_task2_ffn_edit256_and_attention_heads_edit_top30_a001_ffn_edit_only.json}"
    BASELINE_METHOD="LUME_scenario_clean_Llama3_8B_hard_balanced_no_defense"
    FFN_EDIT_METHOD="LUME_scenario_clean_Llama3_8B_hard_balanced_FFN_Edit${FFN_EDIT_ERASE_NUM}_exactsafe_ref"
    DEFAULT_CANDIDATE_TOP_K=64
    DEFAULT_SELECT_TOP_K=30
    DEFAULT_EVAL_ALPHA=0.01
    DEFAULT_SCREEN_LIMIT_PER_TYPE=20
    DEFAULT_MAX_CONTEXT=768
    EXACT_BATCH_SIZE="${EXACT_BATCH_SIZE:-2}"
    MRR_BATCH_SIZE="${MRR_BATCH_SIZE:-2}"
    ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-2}"
    BASELINE_EXACT="${BASELINE_EXACT:-}"
    BASELINE_CORE="${BASELINE_CORE:-}"
    FFN_EDIT_EXACT="${FFN_EDIT_EXACT:-}"
    FFN_EDIT_CORE="${FFN_EDIT_CORE:-}"
    ;;
  qwen)
    MODEL_LABEL="Qwen3_8B_Base"
    MODEL_TAG="qwen3_8b_base"
    BASE_MODEL="${BASE_MODEL:-models/qwen3-8b-base}"
    ADAPTER="${ADAPTER:-models/qwen3-8b-base/lume_task2_hard_balanced_qlora}"
    FFN_EDIT_KN="${FFN_EDIT_KN:-outputs/ffn_edit/lume_task2_qwen3_8b_hard_balanced_ffn_edit128_self_attention_heads_edit_joint_only_top30/lume_task2_qwen3_8b_hard_balanced_ffn_edit128_self_attention_heads_edit_joint_only_top30_a0p01_qwen_joint_only_ffn_edit128_self_attention_heads_edit/attribution/kn/kn_bag-lume_task2_qwen3_8b_hard_balanced_ffn_edit128_self_attention_heads_edit_joint_only_top30_a0p01_qwen_joint_only_ffn_edit128_self_attention_heads_edit.json}"
    BASELINE_METHOD="LUME_scenario_clean_Qwen3_8B_Base_hard_balanced_no_defense"
    FFN_EDIT_METHOD="LUME_scenario_clean_Qwen3_8B_Base_hard_balanced_FFN_Edit${FFN_EDIT_ERASE_NUM}_intervention"
    DEFAULT_CANDIDATE_TOP_K=80
    DEFAULT_SELECT_TOP_K=20
    DEFAULT_EVAL_ALPHA=0.02
    DEFAULT_SCREEN_LIMIT_PER_TYPE=20
    DEFAULT_MAX_CONTEXT=768
    EXACT_BATCH_SIZE="${EXACT_BATCH_SIZE:-2}"
    MRR_BATCH_SIZE="${MRR_BATCH_SIZE:-2}"
    ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-2}"
    BASELINE_EXACT="${BASELINE_EXACT:-}"
    BASELINE_CORE="${BASELINE_CORE:-}"
    FFN_EDIT_EXACT="${FFN_EDIT_EXACT:-}"
    FFN_EDIT_CORE="${FFN_EDIT_CORE:-}"
    ;;
  ministral)
    MODEL_LABEL="Ministral3_8B_Base"
    MODEL_TAG="ministral3_8b_base"
    PYBIN="${LUME_PYTHON:-${REPO_ROOT}/.venvs/ministral3/bin/python}"
    BASE_MODEL="${BASE_MODEL:-models/ministral-3-8b-base}"
    ADAPTER="${ADAPTER:-models/ministral-3-8b-base/lume_task2_hard_balanced_qlora}"
    FFN_EDIT_KN="${FFN_EDIT_KN:-outputs/ffn_edit/lume_task2_ministral3_8b_base_hard_balanced_ffn_edit128_self_attention_heads_edit_top30/lume_task2_ministral3_8b_base_hard_balanced_ffn_edit128_self_attention_heads_edit_top30_a0p01/attribution/kn/kn_bag-lume_task2_ministral3_8b_base_hard_balanced_ffn_edit128_self_attention_heads_edit_top30_a0p01.json}"
    BASELINE_METHOD="LUME_scenario_clean_Ministral3_8B_Base_hard_balanced_no_defense"
    FFN_EDIT_METHOD="LUME_scenario_clean_Ministral3_8B_Base_hard_balanced_FFN_Edit${FFN_EDIT_ERASE_NUM}_intervention"
    DEFAULT_CANDIDATE_TOP_K=64
    DEFAULT_SELECT_TOP_K=30
    DEFAULT_EVAL_ALPHA=0.01
    DEFAULT_SCREEN_LIMIT_PER_TYPE=30
    DEFAULT_MAX_CONTEXT=768
    EXACT_BATCH_SIZE="${EXACT_BATCH_SIZE:-1}"
    MRR_BATCH_SIZE="${MRR_BATCH_SIZE:-1}"
    ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-1}"
    BASELINE_EXACT="${BASELINE_EXACT:-}"
    BASELINE_CORE="${BASELINE_CORE:-}"
    FFN_EDIT_EXACT="${FFN_EDIT_EXACT:-}"
    FFN_EDIT_CORE="${FFN_EDIT_CORE:-}"
    ;;
  *)
    echo "[lume_exactsafe] invalid LUME_MODEL_KIND=${MODEL_KIND}; expected llama, qwen, ministral" >&2
    exit 2
    ;;
esac

CANDIDATE_TOP_K="${CANDIDATE_TOP_K:-${DEFAULT_CANDIDATE_TOP_K}}"
SELECT_TOP_K="${SELECT_TOP_K:-${DEFAULT_SELECT_TOP_K}}"
EVAL_ALPHA="${EVAL_ALPHA:-${DEFAULT_EVAL_ALPHA}}"
SCREEN_LIMIT_PER_TYPE="${SCREEN_LIMIT_PER_TYPE:-${DEFAULT_SCREEN_LIMIT_PER_TYPE}}"
SCREEN_MAX_SAMPLES="${SCREEN_MAX_SAMPLES:-0}"
MAX_CONTEXT="${MAX_CONTEXT:-${DEFAULT_MAX_CONTEXT}}"
ATTN_MAX_SAMPLES="${ATTN_MAX_SAMPLES:-1000}"
MAX_HEADS_PER_LAYER="${MAX_HEADS_PER_LAYER:-0}"
MIN_EXACT_GAIN="${MIN_EXACT_GAIN:-0.0}"
MIN_SCORE_GAIN="${MIN_SCORE_GAIN:-0.0}"
PREFIX_SIZES="${PREFIX_SIZES:-1,3,5,10,20,30,40,60}"
ENFORCE_PER_TYPE_NONWORSENING="${ENFORCE_PER_TYPE_NONWORSENING:-0}"
PER_TYPE_MIN_SAMPLES="${PER_TYPE_MIN_SAMPLES:-10}"
PREFER_SMALLEST_EXACT_PREFIX="${PREFER_SMALLEST_EXACT_PREFIX:-0}"
FORCE_RELOCATE="${FORCE_RELOCATE:-0}"
FORCE_SCREEN="${FORCE_SCREEN:-0}"
FORCE_BASELINE_EVAL="${FORCE_BASELINE_EVAL:-0}"
FORCE_FFN_EDIT_EVAL="${FORCE_FFN_EDIT_EVAL:-0}"

SAFE_ALPHA="${EVAL_ALPHA//./p}"
RUN_NAME="${RUN_NAME:-lume_${MODEL_TAG}_ffn_edit${FFN_EDIT_ERASE_NUM}_attention_heads_edit_exactsafe_top${SELECT_TOP_K}_a${SAFE_ALPHA}_${RUN_STAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT_BASE}/${RUN_NAME}"
SUMMARY_DIR="${SUMMARY_DIR:-outputs/ffn_edit/metrics/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-logs/ffn_edit/${RUN_NAME}.log}"
MANIFEST="${MANIFEST:-outputs/logs/manifests/${RUN_NAME}_manifest.json}"
ATTN_RUN="${RUN_NAME}_attn_${QUERY_MODE}_${STEERING_SPAN_MODE}${STEERING_TAIL_TOKENS}_firstmass_top${CANDIDATE_TOP_K}"
SCREEN_RUN="${RUN_NAME}_post_ffn_edit_exactscreen"
CANDIDATE_HEADS="${CANDIDATE_HEADS:-outputs/attention_heads_edit/heads/${ATTN_RUN}_top${CANDIDATE_TOP_K}.json}"
HEAD_CONFIG="${HEAD_CONFIG:-outputs/attention_heads_edit/heads/${SCREEN_RUN}_exactsafe_top${SELECT_TOP_K}_alpha${SAFE_ALPHA}.json}"

BASELINE_EXACT_RUN="${RUN_NAME}_baseline_exact_full"
BASELINE_CORE_RUN="${RUN_NAME}_baseline_core"
FFN_EDIT_EXACT_RUN="${RUN_NAME}_ffn_edit${FFN_EDIT_ERASE_NUM}_exact_full"
FFN_EDIT_CORE_RUN="${RUN_NAME}_ffn_edit${FFN_EDIT_ERASE_NUM}_core"
JOINT_EXACT_RUN="${RUN_NAME}_ffn_edit${FFN_EDIT_ERASE_NUM}_then_exactsafe_attention_heads_edit_exact_full"
JOINT_CORE_RUN="${RUN_NAME}_ffn_edit${FFN_EDIT_ERASE_NUM}_then_exactsafe_attention_heads_edit_core"
JOINT_METHOD="LUME_scenario_clean_${MODEL_LABEL}_hard_balanced_FFN_Edit${FFN_EDIT_ERASE_NUM}_then_exactsafe_Attention_Heads_Edit_top${SELECT_TOP_K}_alpha_${SAFE_ALPHA}"

if [ -z "${BASELINE_EXACT}" ]; then BASELINE_EXACT="${OUTPUT_ROOT}/${BASELINE_EXACT_RUN}/metrics/metrics.json"; FORCE_BASELINE_EVAL=1; fi
if [ -z "${BASELINE_CORE}" ]; then BASELINE_CORE="${OUTPUT_ROOT}/${BASELINE_CORE_RUN}/metrics/core_metrics.json"; FORCE_BASELINE_EVAL=1; fi
if [ -z "${FFN_EDIT_EXACT}" ]; then FFN_EDIT_EXACT="${OUTPUT_ROOT}/${FFN_EDIT_EXACT_RUN}/metrics/metrics.json"; FORCE_FFN_EDIT_EVAL=1; fi
if [ -z "${FFN_EDIT_CORE}" ]; then FFN_EDIT_CORE="${OUTPUT_ROOT}/${FFN_EDIT_CORE_RUN}/metrics/core_metrics.json"; FORCE_FFN_EDIT_EVAL=1; fi

mkdir -p logs/ffn_edit outputs/logs/manifests outputs/attention_heads_edit/heads "${OUTPUT_ROOT}" "${SUMMARY_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

wait_gpu_idle() {
  while true; do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ' || echo 0)"
    if [ "${used:-99999}" -lt "${GPU_IDLE_MEM_MB}" ]; then
      echo "[INFO] $(date -Is) GPU idle enough: ${used} MiB"
      break
    fi
    echo "[WAIT] $(date -Is) GPU busy: ${used} MiB (need < ${GPU_IDLE_MEM_MB})"
    sleep 120
  done
}

echo "[START] $(date -Is) run=${RUN_NAME}"
echo "[INFO] model=${MODEL_LABEL} task=FFN_Edit${FFN_EDIT_ERASE_NUM}->exact-safe-Attention_Heads_Edit"
echo "[INFO] hyperparams: candidate_top=${CANDIDATE_TOP_K} select_top=${SELECT_TOP_K} alpha=${EVAL_ALPHA} screen_limit_per_type=${SCREEN_LIMIT_PER_TYPE}"
echo "[INFO] Attention Heads Edit semantics: query_mode=${QUERY_MODE} steering_span=${STEERING_SPAN_MODE} tail_tokens=${STEERING_TAIL_TOKENS}"
echo "[INFO] Attention Heads Edit selection: max_heads_per_layer=${MAX_HEADS_PER_LAYER} min_exact_gain=${MIN_EXACT_GAIN} prefix_sizes=${PREFIX_SIZES} per_type_guard=${ENFORCE_PER_TYPE_NONWORSENING}"
echo "[INFO] input_train=${TRAIN_DS}"
echo "[INFO] input_screen=${SCREEN_DS}"
echo "[INFO] input_eval=${EVAL_DS}"
echo "[INFO] ffn_edit_kn=${FFN_EDIT_KN}"
echo "[INFO] impact=no checkpoint writes; metrics/logs/configs only"

"${PYBIN}" - <<PY
from pathlib import Path
required = [
    "${TRAIN_DS}", "${SCREEN_DS}", "${EVAL_DS}", "${BASE_MODEL}/config.json",
    "${ADAPTER}/adapter_config.json", "${FFN_EDIT_KN}",
]
missing = [p for p in required if not Path(p).exists() or Path(p).stat().st_size == 0]
if missing:
    raise SystemExit("missing required files: " + ", ".join(missing))
print("[INFO] prerequisites_ok", flush=True)
PY

if [ "${FORCE_RELOCATE}" = "1" ]; then
  rm -f "${CANDIDATE_HEADS}"
fi

if [ ! -s "${CANDIDATE_HEADS}" ]; then
  echo "[STEP 1/5] $(date -Is) locate candidate Attention Heads Edit heads top${CANDIDATE_TOP_K} on ${MODEL_LABEL}"
  wait_gpu_idle
  "${PYBIN}" -m attention_heads_edit.locate.locate_api4_attention_heads \
    --dataset "${TRAIN_DS}" \
    --base_model "${BASE_MODEL}" \
    --adapter "${ADAPTER}" \
    --run_name "${ATTN_RUN}" \
    --top_k "${CANDIDATE_TOP_K}" \
    --max_heads_per_layer "${MAX_HEADS_PER_LAYER}" \
    --max_samples "${ATTN_MAX_SAMPLES}" \
    --batch_size 1 \
    --ranking_metric first_mass \
    --query_mode "${QUERY_MODE}" \
    --steering_span_mode "${STEERING_SPAN_MODE}" \
    --steering_tail_tokens "${STEERING_TAIL_TOKENS}" \
    --max_context_tokens "${MAX_CONTEXT}" \
    --load_in_4bit \
    --torch_dtype bfloat16 \
    --attn_implementation "${ATTN_IMPL}" \
    --metrics_dir outputs/ffn_edit/metrics \
    --config_dir outputs/attention_heads_edit/heads \
    --log_every 50
else
  echo "[STEP 1/5] $(date -Is) reuse candidate heads ${CANDIDATE_HEADS}"
fi
test -s "${CANDIDATE_HEADS}"

if [ "${FORCE_SCREEN}" = "1" ]; then
  rm -f "${HEAD_CONFIG}"
fi

if [ ! -s "${HEAD_CONFIG}" ]; then
  echo "[STEP 2/5] $(date -Is) post-FFN Edit exact-safe screen for ${MODEL_LABEL}"
  wait_gpu_idle
  SCREEN_SAFETY_ARGS=()
  if [ "${ENFORCE_PER_TYPE_NONWORSENING}" = "1" ]; then
    SCREEN_SAFETY_ARGS+=(--enforce_per_type_nonworsening)
  fi
  if [ "${PREFER_SMALLEST_EXACT_PREFIX}" = "1" ]; then
    SCREEN_SAFETY_ARGS+=(--prefer_smallest_exact_prefix)
  fi
  "${PYBIN}" -m attention_heads_edit.locate.select_attention_heads_by_exact_after_ffn_edit \
    --dataset "${SCREEN_DS}" \
    --candidate_heads "${CANDIDATE_HEADS}" \
    --base_model "${BASE_MODEL}" \
    --adapter "${ADAPTER}" \
    --ffn_edit_kn_config "${FFN_EDIT_KN}" \
    --ffn_edit_erase_num "${FFN_EDIT_ERASE_NUM}" \
    --run_name "${SCREEN_RUN}" \
    --output_dir "${OUTPUT_ROOT}" \
    --config_dir outputs/attention_heads_edit/heads \
    --top_k "${SELECT_TOP_K}" \
    --screen_max_samples "${SCREEN_MAX_SAMPLES}" \
    --screen_limit_per_type "${SCREEN_LIMIT_PER_TYPE}" \
    --candidate_limit "${CANDIDATE_TOP_K}" \
    --min_exact_gain "${MIN_EXACT_GAIN}" \
    --min_score_gain "${MIN_SCORE_GAIN}" \
    --prefix_sizes "${PREFIX_SIZES}" \
    --per_type_min_samples "${PER_TYPE_MIN_SAMPLES}" \
    --alpha "${EVAL_ALPHA}" \
    --scale_position "${SCALE_POSITION}" \
    --steering_span_mode "${STEERING_SPAN_MODE}" \
    --steering_tail_tokens "${STEERING_TAIL_TOKENS}" \
    --batch_size "${EXACT_BATCH_SIZE}" \
    --max_context_tokens "${MAX_CONTEXT}" \
    --max_new_tokens 96 \
    --generation_extra_tokens 8 \
    --load_in_4bit \
    --torch_dtype bfloat16 \
    --attn_implementation "${ATTN_IMPL}" \
    "${SCREEN_SAFETY_ARGS[@]}" \
    --log_every 10
else
  echo "[STEP 2/5] $(date -Is) reuse exact-safe head config ${HEAD_CONFIG}"
fi
test -s "${HEAD_CONFIG}"

"${PYBIN}" - <<PY
import json
from pathlib import Path
cfg = json.loads(Path("${HEAD_CONFIG}").read_text(encoding="utf-8"))
heads = sum(len(v) for v in cfg.values() if isinstance(v, list))
print(f"[INFO] exactsafe_head_config=${HEAD_CONFIG} heads={heads}", flush=True)
PY

if [ "${FORCE_BASELINE_EVAL}" = "1" ] || [ ! -s "${BASELINE_EXACT}" ] || [ ! -s "${BASELINE_CORE}" ]; then
  echo "[STEP 3/5] $(date -Is) ${MODEL_LABEL} baseline full metrics"
  wait_gpu_idle
  "${PYBIN}" -m attention_heads_edit.eval.eval_api4_privacy_reverse_attention_heads_edit \
    --dataset "${EVAL_DS}" \
    --base_model "${BASE_MODEL}" \
    --adapter "${ADAPTER}" \
    --disable_attention_heads_edit \
    --run_name "${BASELINE_EXACT_RUN}" \
    --output_dir "${OUTPUT_ROOT}" \
    --batch_size "${EXACT_BATCH_SIZE}" \
    --max_context_tokens "${MAX_CONTEXT}" \
    --max_new_tokens 96 \
    --generation_extra_tokens 8 \
    --load_in_4bit \
    --torch_dtype bfloat16 \
    --attn_implementation "${ATTN_IMPL}" \
    --log_every 100
  "${PYBIN}" -m attention_heads_edit.eval.eval_api4_privacy_core_metrics \
    --dataset "${EVAL_DS}" \
    --method_name "${BASELINE_METHOD}" \
    --base_model "${BASE_MODEL}" \
    --adapter "${ADAPTER}" \
    --disable_attention_heads_edit \
    --run_name "${BASELINE_CORE_RUN}" \
    --output_dir "${OUTPUT_ROOT}" \
    --run_mrr \
    --run_attack \
    --run_ppl \
    --mrr_limit_per_type 200 \
    --mrr_batch_size "${MRR_BATCH_SIZE}" \
    --attack_limit_per_type 10 \
    --attack_batch_size "${ATTACK_BATCH_SIZE}" \
    --attack_samples 32 \
    --attack_temperature 0.8 \
    --attack_top_p 0.95 \
    --ppl_max_blocks 128 \
    --ppl_block_tokens 512 \
    --ppl_batch_size 1 \
    --max_context_tokens "${MAX_CONTEXT}" \
    --max_new_tokens 96 \
    --generation_extra_tokens 8 \
    --load_in_4bit \
    --torch_dtype bfloat16 \
    --attn_implementation "${ATTN_IMPL}" \
    --log_every 50
  BASELINE_EXACT="${OUTPUT_ROOT}/${BASELINE_EXACT_RUN}/metrics/metrics.json"
  BASELINE_CORE="${OUTPUT_ROOT}/${BASELINE_CORE_RUN}/metrics/core_metrics.json"
else
  echo "[STEP 3/5] $(date -Is) reuse baseline metrics"
fi

if [ "${FORCE_FFN_EDIT_EVAL}" = "1" ] || [ ! -s "${FFN_EDIT_EXACT}" ] || [ ! -s "${FFN_EDIT_CORE}" ]; then
  echo "[STEP 4/5] $(date -Is) ${MODEL_LABEL} standalone FFN Edit${FFN_EDIT_ERASE_NUM} full metrics"
  wait_gpu_idle
  "${PYBIN}" -m attention_heads_edit.eval.eval_api4_privacy_reverse_attention_heads_edit \
    --dataset "${EVAL_DS}" \
    --base_model "${BASE_MODEL}" \
    --adapter "${ADAPTER}" \
    --disable_attention_heads_edit \
    --run_name "${FFN_EDIT_EXACT_RUN}" \
    --output_dir "${OUTPUT_ROOT}" \
    --batch_size "${EXACT_BATCH_SIZE}" \
    --ffn_edit_kn_config "${FFN_EDIT_KN}" \
    --ffn_edit_erase_num "${FFN_EDIT_ERASE_NUM}" \
    --max_context_tokens "${MAX_CONTEXT}" \
    --max_new_tokens 96 \
    --generation_extra_tokens 8 \
    --load_in_4bit \
    --torch_dtype bfloat16 \
    --attn_implementation "${ATTN_IMPL}" \
    --log_every 100
  "${PYBIN}" -m attention_heads_edit.eval.eval_api4_privacy_core_metrics \
    --dataset "${EVAL_DS}" \
    --method_name "${FFN_EDIT_METHOD}" \
    --base_model "${BASE_MODEL}" \
    --adapter "${ADAPTER}" \
    --run_name "${FFN_EDIT_CORE_RUN}" \
    --output_dir "${OUTPUT_ROOT}" \
    --disable_attention_heads_edit \
    --ffn_edit_kn_config "${FFN_EDIT_KN}" \
    --ffn_edit_erase_num "${FFN_EDIT_ERASE_NUM}" \
    --run_mrr \
    --run_attack \
    --run_ppl \
    --mrr_limit_per_type 200 \
    --mrr_batch_size "${MRR_BATCH_SIZE}" \
    --attack_limit_per_type 10 \
    --attack_batch_size "${ATTACK_BATCH_SIZE}" \
    --attack_samples 32 \
    --attack_temperature 0.8 \
    --attack_top_p 0.95 \
    --ppl_max_blocks 128 \
    --ppl_block_tokens 512 \
    --ppl_batch_size 1 \
    --max_context_tokens "${MAX_CONTEXT}" \
    --max_new_tokens 96 \
    --generation_extra_tokens 8 \
    --load_in_4bit \
    --torch_dtype bfloat16 \
    --attn_implementation "${ATTN_IMPL}" \
    --log_every 50
  FFN_EDIT_EXACT="${OUTPUT_ROOT}/${FFN_EDIT_EXACT_RUN}/metrics/metrics.json"
  FFN_EDIT_CORE="${OUTPUT_ROOT}/${FFN_EDIT_CORE_RUN}/metrics/core_metrics.json"
else
  echo "[STEP 4/5] $(date -Is) reuse FFN Edit metrics"
fi

echo "[STEP 5/5] $(date -Is) ${MODEL_LABEL} FFN_Edit${FFN_EDIT_ERASE_NUM}->exact-safe Attention Heads Edit full metrics"
wait_gpu_idle
"${PYBIN}" -m attention_heads_edit.eval.eval_api4_privacy_reverse_attention_heads_edit \
  --dataset "${EVAL_DS}" \
  --base_model "${BASE_MODEL}" \
  --adapter "${ADAPTER}" \
  --head_config "${HEAD_CONFIG}" \
  --run_name "${JOINT_EXACT_RUN}" \
  --output_dir "${OUTPUT_ROOT}" \
  --batch_size "${EXACT_BATCH_SIZE}" \
  --alpha "${EVAL_ALPHA}" \
  --scale_position "${SCALE_POSITION}" \
  --steering_span_mode "${STEERING_SPAN_MODE}" \
  --steering_tail_tokens "${STEERING_TAIL_TOKENS}" \
  --ffn_edit_kn_config "${FFN_EDIT_KN}" \
  --ffn_edit_erase_num "${FFN_EDIT_ERASE_NUM}" \
  --max_context_tokens "${MAX_CONTEXT}" \
  --max_new_tokens 96 \
  --generation_extra_tokens 8 \
  --load_in_4bit \
  --torch_dtype bfloat16 \
  --attn_implementation "${ATTN_IMPL}" \
  --log_every 100

"${PYBIN}" -m attention_heads_edit.eval.eval_api4_privacy_core_metrics \
  --dataset "${EVAL_DS}" \
  --method_name "${JOINT_METHOD}" \
  --base_model "${BASE_MODEL}" \
  --adapter "${ADAPTER}" \
  --run_name "${JOINT_CORE_RUN}" \
  --output_dir "${OUTPUT_ROOT}" \
  --head_config "${HEAD_CONFIG}" \
  --alpha "${EVAL_ALPHA}" \
  --scale_position "${SCALE_POSITION}" \
  --steering_span_mode "${STEERING_SPAN_MODE}" \
  --steering_tail_tokens "${STEERING_TAIL_TOKENS}" \
  --ppl_attention_heads_edit_mode none \
  --ffn_edit_kn_config "${FFN_EDIT_KN}" \
  --ffn_edit_erase_num "${FFN_EDIT_ERASE_NUM}" \
  --run_mrr \
  --run_attack \
  --run_ppl \
  --mrr_limit_per_type 200 \
  --mrr_batch_size "${MRR_BATCH_SIZE}" \
  --attack_limit_per_type 10 \
  --attack_batch_size "${ATTACK_BATCH_SIZE}" \
  --attack_samples 32 \
  --attack_temperature 0.8 \
  --attack_top_p 0.95 \
  --ppl_max_blocks 128 \
  --ppl_block_tokens 512 \
  --ppl_batch_size 1 \
  --max_context_tokens "${MAX_CONTEXT}" \
  --max_new_tokens 96 \
  --generation_extra_tokens 8 \
  --load_in_4bit \
  --torch_dtype bfloat16 \
  --attn_implementation "${ATTN_IMPL}" \
  --log_every 50

cat > "${MANIFEST}" <<JSON
{
  "run_name": "${RUN_NAME}",
  "dataset": "LUME_scenario_clean",
  "model": "${MODEL_LABEL}",
  "order": "FFN_Edit${FFN_EDIT_ERASE_NUM}_first_then_exact_safe_Attention_Heads_Edit",
  "selection": "candidate attention heads are screened after FFN Edit by greedy exact/contains leakage",
  "base_model": "${BASE_MODEL}",
  "adapter": "${ADAPTER}",
  "train_dataset": "${TRAIN_DS}",
  "screen_dataset": "${SCREEN_DS}",
  "eval_dataset": "${EVAL_DS}",
  "ffn_edit_kn_config": "${FFN_EDIT_KN}",
  "ffn_edit_erase_num": ${FFN_EDIT_ERASE_NUM},
  "candidate_heads": "${CANDIDATE_HEADS}",
  "head_config": "${HEAD_CONFIG}",
  "candidate_top_k": ${CANDIDATE_TOP_K},
  "max_heads_per_layer": ${MAX_HEADS_PER_LAYER},
  "select_top_k": ${SELECT_TOP_K},
  "attention_heads_edit_alpha": ${EVAL_ALPHA},
  "scale_position": "${SCALE_POSITION}",
  "query_mode": "${QUERY_MODE}",
  "steering_span_mode": "${STEERING_SPAN_MODE}",
  "steering_tail_tokens": ${STEERING_TAIL_TOKENS},
  "min_exact_gain": ${MIN_EXACT_GAIN},
  "enforce_per_type_nonworsening": ${ENFORCE_PER_TYPE_NONWORSENING},
  "per_type_min_samples": ${PER_TYPE_MIN_SAMPLES},
  "prefer_smallest_exact_prefix": ${PREFER_SMALLEST_EXACT_PREFIX},
  "impact": "evaluation-time intervention only; no checkpoint or dataset mutation",
  "methods": [
    {
      "method": "${BASELINE_METHOD}",
      "exact_metrics": "${BASELINE_EXACT}",
      "core_metrics": "${BASELINE_CORE}"
    },
    {
      "method": "${FFN_EDIT_METHOD}",
      "exact_metrics": "${FFN_EDIT_EXACT}",
      "core_metrics": "${FFN_EDIT_CORE}"
    },
    {
      "method": "${JOINT_METHOD}",
      "exact_metrics": "${OUTPUT_ROOT}/${JOINT_EXACT_RUN}/metrics/metrics.json",
      "core_metrics": "${OUTPUT_ROOT}/${JOINT_CORE_RUN}/metrics/core_metrics.json"
    }
  ]
}
JSON

"${PYBIN}" -m attention_heads_edit.eval.summarize_api4_privacy_suite \
  --manifest "${MANIFEST}" \
  --output_dir "${SUMMARY_DIR}"

test -s "${SUMMARY_DIR}/paper_main_table.csv"
echo "[DONE] $(date -Is) summary=${SUMMARY_DIR}/paper_main_table.csv"
cat "${SUMMARY_DIR}/paper_main_table.csv"
