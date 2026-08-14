#!/usr/bin/env bash
# Run one CRAPII SFT adapter through:
#   1) fresh model-local FFN Edit128 localization on corrected CRAPII privacy bags,
#   2) standalone FFN Edit128 full privacy metrics,
#   3) model-local next-token trigger-span Attention Heads Edit localization,
#   4) post-FFN Edit validation screening and FFN Edit -> Attention Heads Edit full privacy metrics.
# Input: data/crapii/api4_prefix privacy/eval JSON plus a model-specific CRAPII LoRA/QLoRA adapter.
# Output: configs/metrics/logs only; no model weights are modified.

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../_common.sh"


MODEL_KIND="${CRAPII_MODEL_KIND:?Set CRAPII_MODEL_KIND to one of: llama, qwen, ministral}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

RUN_STAMP="${RUN_STAMP:-fixeddata_independent}"
DATASET_LABEL="${DATASET_LABEL:-CRAPII}"
PRIV_DATA="${PRIV_DATA:-data/crapii/api4_prefix/privacy_data_crapii_all.json}"
TRAIN_DATASET="${TRAIN_DATASET:-data/crapii/api4_prefix/sft_true_prefix_no_instruction_train.json}"
SCREEN_DATASET="${SCREEN_DATASET:-data/crapii/api4_prefix/sft_true_prefix_no_instruction_train.json}"
EVAL_DATASET="${EVAL_DATASET:-data/crapii/api4_prefix/sft_true_prefix_no_instruction_all.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/ffn_edit/crapii_ffn_edit128_independent_attention_heads_edit}"

FFN_EDIT_ERASE_NUM="${FFN_EDIT_ERASE_NUM:-128}"
FFN_EDIT_RANKING_METRIC="${FFN_EDIT_RANKING_METRIC:-count_score}"
REF_TOTAL_HEADS="${REF_TOTAL_HEADS:-1024}"
REF_CANDIDATE_TOP_K="${REF_CANDIDATE_TOP_K:-96}"
REF_SELECT_TOP_K="${REF_SELECT_TOP_K:-30}"
ATTENTION_HEADS_EDIT_ALPHA="${ATTENTION_HEADS_EDIT_ALPHA:-0.1}"
SCALE_POSITION="${SCALE_POSITION:-include_down}"
STEERING_SPAN_MODE="${STEERING_SPAN_MODE:-tail_tokens}"
STEERING_TAIL_TOKENS="${STEERING_TAIL_TOKENS:-8}"
QUERY_MODE="${QUERY_MODE:-next_token}"
ATTN_MAX_SAMPLES="${ATTN_MAX_SAMPLES:-0}"
MAX_CONTEXT="${MAX_CONTEXT:-512}"
SCREEN_MAX_SAMPLES="${SCREEN_MAX_SAMPLES:-0}"
SCREEN_LIMIT_PER_TYPE="${SCREEN_LIMIT_PER_TYPE:-20}"
MISSING_BASELINE_POLICY="${MISSING_BASELINE_POLICY:-fail}"
GPU_IDLE_MEM_MB="${GPU_IDLE_MEM_MB:-1000}"
FORCE_FFN_EDIT_RELOCATE="${FORCE_FFN_EDIT_RELOCATE:-0}"
FORCE_ATTENTION_HEADS_EDIT_RELOCATE="${FORCE_ATTENTION_HEADS_EDIT_RELOCATE:-0}"
FORCE_BASELINE_EVAL="${FORCE_BASELINE_EVAL:-0}"
REQUIRE_NO_TARGET_IN_PREFIX="${REQUIRE_NO_TARGET_IN_PREFIX:-0}"
MAX_HEADS_PER_LAYER="${MAX_HEADS_PER_LAYER:-0}"
MIN_EXACT_GAIN="${MIN_EXACT_GAIN:-0.0}"
MIN_SCORE_GAIN="${MIN_SCORE_GAIN:-0.0}"
PREFIX_SIZES="${PREFIX_SIZES:-1,3,5,10,20,30,40,60}"
ENFORCE_PER_TYPE_NONWORSENING="${ENFORCE_PER_TYPE_NONWORSENING:-0}"
PER_TYPE_MIN_SAMPLES="${PER_TYPE_MIN_SAMPLES:-10}"
PREFER_SMALLEST_EXACT_PREFIX="${PREFER_SMALLEST_EXACT_PREFIX:-0}"

PYTHON_BIN="${CRAPII_PYTHON:-python}"
ATTRIBUTION_SCRIPT="-m ffn_edit.locate.compact_attribution_gemma"
ATTRIBUTION_EXTRA_ARGS=()
ATTRIBUTION_RANKING_ARGS=(--ranking_metric "${FFN_EDIT_RANKING_METRIC}")
LOAD_IN_4BIT_ARGS=(--load_in_4bit)
ATTN_IMPL="${ATTN_IMPL:-eager}"
EXACT_BATCH_SIZE="${EXACT_BATCH_SIZE:-}"
MRR_BATCH_SIZE="${MRR_BATCH_SIZE:-}"
ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-}"
PPL_BATCH_SIZE="${PPL_BATCH_SIZE:-1}"

case "${MODEL_KIND}" in
  llama)
    MODEL_LABEL="Llama3_8B"
    MODEL_TAG="llama3_8b"
    BASE_MODEL="${BASE_MODEL:-models/llama3-8B/baseline}"
    ADAPTER="${ADAPTER:-models/llama3-8B/crapii_prefix_qlora_strong_r64_e8}"
    BASELINE_RUN_NAME="${BASELINE_RUN_NAME:-crapii_llama3_8b_strong_sft_no_defense}"
    BASELINE_METHOD="CRAPII_Llama3_8B_Strong_SFT_no_defense"
    ATTRIBUTION_SCRIPT="-m ffn_edit.locate.compact_attribution_llama"
    ATTRIBUTION_EXTRA_ARGS=()
    ATTRIBUTION_RANKING_ARGS=()
    FFN_EDIT_RANKING_METRIC="count"
    LOAD_IN_4BIT_ARGS=(--load_in_4bit)
    EXACT_BATCH_SIZE="${EXACT_BATCH_SIZE:-1}"
    MRR_BATCH_SIZE="${MRR_BATCH_SIZE:-1}"
    ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-1}"
    ;;
  qwen)
    MODEL_LABEL="Qwen3_8B_Base"
    MODEL_TAG="qwen3_8b_base"
    BASE_MODEL="${BASE_MODEL:-models/qwen3-8b-base}"
    ADAPTER="${ADAPTER:-models/qwen3-8b-base/crapii_prefix_qlora_strong_r64_e8}"
    BASELINE_RUN_NAME="${BASELINE_RUN_NAME:-crapii_qwen3_8b_base_strong_sft_no_defense}"
    BASELINE_METHOD="CRAPII_Qwen3_8B_Base_Strong_SFT_no_defense"
    ATTRIBUTION_EXTRA_ARGS=(--ffn_edit_model_kind qwen --torch_dtype bfloat16 --attn_implementation "${ATTN_IMPL}" --load_in_4bit)
    EXACT_BATCH_SIZE="${EXACT_BATCH_SIZE:-2}"
    MRR_BATCH_SIZE="${MRR_BATCH_SIZE:-2}"
    ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-2}"
    ;;
  ministral)
    MODEL_LABEL="Ministral3_8B_Base"
    MODEL_TAG="ministral3_8b_base"
    BASE_MODEL="${BASE_MODEL:-models/ministral-3-8b-base}"
    ADAPTER="${ADAPTER:-models/ministral-3-8b-base/crapii_prefix_qlora_strong_r64_e8}"
    BASELINE_RUN_NAME="${BASELINE_RUN_NAME:-crapii_ministral3_8b_base_strong_sft_no_defense}"
    BASELINE_METHOD="CRAPII_Ministral3_8B_Base_Strong_SFT_no_defense"
    PYTHON_BIN="${CRAPII_PYTHON:-${REPO_ROOT}/.venvs/ministral3/bin/python}"
    ATTRIBUTION_EXTRA_ARGS=(--ffn_edit_model_kind ministral --torch_dtype bfloat16 --attn_implementation "${ATTN_IMPL}" --load_in_4bit)
    EXACT_BATCH_SIZE="${EXACT_BATCH_SIZE:-1}"
    MRR_BATCH_SIZE="${MRR_BATCH_SIZE:-1}"
    ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-1}"
    ;;
  *)
    echo "[crapii_ffn_edit] invalid CRAPII_MODEL_KIND=${MODEL_KIND}; expected llama, qwen, ministral" >&2
    exit 2
    ;;
esac

BASELINE_METHOD="${DATASET_LABEL}_${MODEL_LABEL}_Strong_SFT_no_defense"

read -r MODEL_LAYERS MODEL_HEADS MODEL_TOTAL_HEADS CANDIDATE_TOP_K SELECT_TOP_K <<EOF
$("${PYTHON_BIN}" - <<PY
import json, os
from pathlib import Path
raw = json.loads(Path("${BASE_MODEL}/config.json").read_text())
cfg = raw.get("text_config", raw)
layers = int(cfg["num_hidden_layers"])
heads = int(cfg["num_attention_heads"])
total = layers * heads
ref_total = int("${REF_TOTAL_HEADS}")
ref_cand = int("${REF_CANDIDATE_TOP_K}")
ref_sel = int("${REF_SELECT_TOP_K}")
cand = int(os.environ.get("CANDIDATE_TOP_K") or max(ref_cand, round(ref_cand * total / ref_total)))
sel = int(os.environ.get("SELECT_TOP_K") or max(ref_sel, round(ref_sel * total / ref_total)))
sel = min(sel, cand)
print(layers, heads, total, cand, sel)
PY
)
EOF

SAFE_ALPHA="${ATTENTION_HEADS_EDIT_ALPHA//./p}"
RUN_NAME="${RUN_NAME:-crapii_${MODEL_TAG}_ffn_edit${FFN_EDIT_ERASE_NUM}_independent_attention_heads_edit_top${SELECT_TOP_K}_a${SAFE_ALPHA}_${RUN_STAMP}}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_NAME}"
ATTR_DIR="${RUN_ROOT}/attribution"
SUMMARY_DIR="${SUMMARY_DIR:-outputs/ffn_edit/metrics/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-logs/ffn_edit/${RUN_NAME}.log}"
MANIFEST="${MANIFEST:-outputs/logs/manifests/${RUN_NAME}_manifest.json}"

KN_CONFIG="${KN_CONFIG:-${ATTR_DIR}/kn/kn_bag-${RUN_NAME}.json}"
ATTN_RUN="${RUN_NAME}_attn_${QUERY_MODE}_${STEERING_SPAN_MODE}${STEERING_TAIL_TOKENS}_firstmass_top${CANDIDATE_TOP_K}"
SCREEN_RUN="${RUN_NAME}_post_ffn_edit_exactsafe_screen"
CANDIDATE_HEADS="${CANDIDATE_HEADS:-outputs/attention_heads_edit/heads/${ATTN_RUN}_top${CANDIDATE_TOP_K}.json}"
HEAD_CONFIG="${HEAD_CONFIG:-outputs/attention_heads_edit/heads/${SCREEN_RUN}_exactsafe_top${SELECT_TOP_K}_alpha${SAFE_ALPHA}.json}"

BASELINE_EXACT="${BASELINE_EXACT:-outputs/ffn_edit/crapii_prefix_sft_no_defense/${BASELINE_RUN_NAME}_exact_full/metrics/metrics.json}"
BASELINE_CORE="${BASELINE_CORE:-outputs/ffn_edit/crapii_prefix_sft_no_defense/${BASELINE_RUN_NAME}_core/metrics/core_metrics.json}"
BASELINE_EXACT_RUN="${RUN_NAME}_baseline_exact_full"
BASELINE_CORE_RUN="${RUN_NAME}_baseline_core"
if [ "${FORCE_BASELINE_EVAL}" = "1" ] || [ "${MISSING_BASELINE_POLICY}" = "evaluate" ]; then
  BASELINE_EXACT="${RUN_ROOT}/${BASELINE_EXACT_RUN}/metrics/metrics.json"
  BASELINE_CORE="${RUN_ROOT}/${BASELINE_CORE_RUN}/metrics/core_metrics.json"
fi

FFN_EDIT_METHOD="${DATASET_LABEL}_${MODEL_LABEL}_Strong_SFT_FFN_Edit${FFN_EDIT_ERASE_NUM}_fixeddata"
JOINT_METHOD="${DATASET_LABEL}_${MODEL_LABEL}_Strong_SFT_FFN_Edit${FFN_EDIT_ERASE_NUM}_then_TriggerSpan_Attention_Heads_Edit_top${SELECT_TOP_K}_alpha_${SAFE_ALPHA}_fixeddata"
FFN_EDIT_EXACT_RUN="${RUN_NAME}_ffn_edit${FFN_EDIT_ERASE_NUM}_exact_full"
FFN_EDIT_CORE_RUN="${RUN_NAME}_ffn_edit${FFN_EDIT_ERASE_NUM}_core"
JOINT_EXACT_RUN="${RUN_NAME}_ffn_edit${FFN_EDIT_ERASE_NUM}_then_independent_attention_heads_edit_a${SAFE_ALPHA}_exact_full"
JOINT_CORE_RUN="${RUN_NAME}_ffn_edit${FFN_EDIT_ERASE_NUM}_then_independent_attention_heads_edit_a${SAFE_ALPHA}_core"

mkdir -p logs/ffn_edit outputs/logs/manifests outputs/attention_heads_edit/heads "${RUN_ROOT}" "${SUMMARY_DIR}" "${ATTR_DIR}"
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
echo "[INFO] model_kind=${MODEL_KIND} model=${MODEL_LABEL}"
echo "[INFO] order=FFN_Edit${FFN_EDIT_ERASE_NUM}-only; independent candidate locate; post-FFN Edit validation screen; FFN_Edit${FFN_EDIT_ERASE_NUM}->Attention_Heads_Edit"
echo "[INFO] model_heads=${MODEL_LAYERS}x${MODEL_HEADS}=${MODEL_TOTAL_HEADS}; candidate_top=${CANDIDATE_TOP_K}; select_top=${SELECT_TOP_K}; alpha=${ATTENTION_HEADS_EDIT_ALPHA}"
echo "[INFO] Attention Heads Edit semantics: query_mode=${QUERY_MODE} steering_span=${STEERING_SPAN_MODE} tail_tokens=${STEERING_TAIL_TOKENS}"
echo "[INFO] Attention Heads Edit selection: max_heads_per_layer=${MAX_HEADS_PER_LAYER} min_exact_gain=${MIN_EXACT_GAIN} prefix_sizes=${PREFIX_SIZES} per_type_guard=${ENFORCE_PER_TYPE_NONWORSENING}"
echo "[INFO] priv_data=${PRIV_DATA}"
echo "[INFO] eval_dataset=${EVAL_DATASET}"
echo "[INFO] train_dataset=${TRAIN_DATASET}"
echo "[INFO] screen_dataset=${SCREEN_DATASET}"
echo "[INFO] base_model=${BASE_MODEL}"
echo "[INFO] adapter=${ADAPTER}"
echo "[INFO] python=${PYTHON_BIN}"
echo "[INFO] outputs=${RUN_ROOT}"

"${PYTHON_BIN}" - <<PY
import json
from collections import Counter
from pathlib import Path

required = [
    "${PRIV_DATA}",
    "${TRAIN_DATASET}",
    "${SCREEN_DATASET}",
    "${EVAL_DATASET}",
    "${BASE_MODEL}/config.json",
    "${ADAPTER}/adapter_config.json",
]
if "${MISSING_BASELINE_POLICY}" == "fail" and "${FORCE_BASELINE_EVAL}" != "1":
    required.extend(["${BASELINE_EXACT}", "${BASELINE_CORE}"])
missing = [p for p in required if not Path(p).exists()]
if missing:
    raise SystemExit("missing required files: " + ", ".join(missing))

bags = json.loads(Path("${PRIV_DATA}").read_text(encoding="utf-8"))
if not isinstance(bags, list) or len(bags) < 100:
    raise SystemExit(f"privacy bags unexpectedly small or invalid: {type(bags)} len={len(bags) if isinstance(bags, list) else 'NA'}")
bad = []
for idx, bag in enumerate(bags[: min(200, len(bags))]):
    if not isinstance(bag, list) or not bag or not isinstance(bag[0], list) or len(bag[0]) < 2:
        bad.append(idx)
        continue
    full, secret = str(bag[0][0]), str(bag[0][1])
    if secret not in full:
        bad.append(idx)
if bad:
    raise SystemExit(f"privacy_data format check failed at sample indices: {bad[:10]}")

rows = json.loads(Path("${EVAL_DATASET}").read_text(encoding="utf-8"))
if "${REQUIRE_NO_TARGET_IN_PREFIX}" == "1":
    contaminated = [idx for idx, row in enumerate(rows) if str(row.get("output", "")) in str(row.get("input", ""))]
    if contaminated:
        raise SystemExit(f"target appears in input prefix at indices: {contaminated[:10]}")
counts = Counter(row.get("pii_type") for row in rows if isinstance(row, dict))
print("[INFO] privacy_bags=" + str(len(bags)), flush=True)
print("[INFO] eval_pii_counts=" + json.dumps(dict(sorted(counts.items())), ensure_ascii=False), flush=True)
if len(rows) < 100:
    raise SystemExit("eval dataset unexpectedly small")
PY

nvidia-smi || true

if [ "${FORCE_FFN_EDIT_RELOCATE}" = "1" ]; then
  rm -f "${KN_CONFIG}"
fi
if [ ! -s "${KN_CONFIG}" ]; then
  echo "[STEP 1/5] $(date -Is) Locate ${MODEL_LABEL} FFN_Edit${FFN_EDIT_ERASE_NUM} neurons on corrected CRAPII privacy bags"
  wait_gpu_idle
  "${PYTHON_BIN}" -u ${ATTRIBUTION_SCRIPT} \
    --priv_data_path "${PRIV_DATA}" \
    --model_name_or_path "${BASE_MODEL}" \
    --adapter_dir "${ADAPTER}" \
    --output_dir "${ATTR_DIR}" \
    --output_prefix "${RUN_NAME}" \
    --rel "${RUN_NAME}" \
    --max_seq_length 128 \
    --gpus 0 \
    --batch_size 2 \
    --num_batch 10 \
    --threshold_ratio 0.1 \
    --top_k "${FFN_EDIT_ERASE_NUM}" \
    --save_every 20 \
    --progress_every 20 \
    "${ATTRIBUTION_RANKING_ARGS[@]}" \
    "${ATTRIBUTION_EXTRA_ARGS[@]}"
else
  echo "[INFO] $(date -Is) Reusing existing FFN Edit KN config: ${KN_CONFIG}"
fi

"${PYTHON_BIN}" - <<PY
from pathlib import Path
p = Path("${KN_CONFIG}")
if not p.exists() or p.stat().st_size == 0:
    raise SystemExit(f"missing FFN Edit KN config: {p}")
print("[INFO] ffn_edit_kn_config=" + str(p), flush=True)
PY

if [ "${FORCE_BASELINE_EVAL}" = "1" ] || [ ! -s "${BASELINE_EXACT}" ] || [ ! -s "${BASELINE_CORE}" ]; then
  echo "[STEP 2/6] $(date -Is) ${MODEL_LABEL} baseline metrics on the current CRAPII evaluation set"
  wait_gpu_idle
  "${PYTHON_BIN}" -m attention_heads_edit.eval.eval_api4_privacy_reverse_attention_heads_edit \
    --dataset "${EVAL_DATASET}" \
    --base_model "${BASE_MODEL}" \
    --adapter "${ADAPTER}" \
    --run_name "${BASELINE_EXACT_RUN}" \
    --output_dir "${RUN_ROOT}" \
    --disable_attention_heads_edit \
    --batch_size "${EXACT_BATCH_SIZE}" \
    --max_context_tokens "${MAX_CONTEXT}" \
    --max_new_tokens 96 \
    --generation_extra_tokens 8 \
    "${LOAD_IN_4BIT_ARGS[@]}" \
    --torch_dtype bfloat16 \
    --attn_implementation "${ATTN_IMPL}" \
    --log_every 100
  "${PYTHON_BIN}" -m attention_heads_edit.eval.eval_api4_privacy_core_metrics \
    --dataset "${EVAL_DATASET}" \
    --method_name "${BASELINE_METHOD}_no_copy_eval" \
    --base_model "${BASE_MODEL}" \
    --adapter "${ADAPTER}" \
    --run_name "${BASELINE_CORE_RUN}" \
    --output_dir "${RUN_ROOT}" \
    --disable_attention_heads_edit \
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
    --ppl_batch_size "${PPL_BATCH_SIZE}" \
    --max_context_tokens "${MAX_CONTEXT}" \
    --max_new_tokens 96 \
    --generation_extra_tokens 8 \
    "${LOAD_IN_4BIT_ARGS[@]}" \
    --torch_dtype bfloat16 \
    --attn_implementation "${ATTN_IMPL}" \
    --log_every 50
fi

echo "[STEP 3/6] $(date -Is) ${MODEL_LABEL} standalone FFN Edit${FFN_EDIT_ERASE_NUM} full metrics"
wait_gpu_idle
"${PYTHON_BIN}" -m attention_heads_edit.eval.eval_api4_privacy_reverse_attention_heads_edit \
  --dataset "${EVAL_DATASET}" \
  --base_model "${BASE_MODEL}" \
  --adapter "${ADAPTER}" \
  --run_name "${FFN_EDIT_EXACT_RUN}" \
  --output_dir "${RUN_ROOT}" \
  --disable_attention_heads_edit \
  --ffn_edit_kn_config "${KN_CONFIG}" \
  --ffn_edit_erase_num "${FFN_EDIT_ERASE_NUM}" \
  --batch_size "${EXACT_BATCH_SIZE}" \
  --max_context_tokens "${MAX_CONTEXT}" \
  --max_new_tokens 96 \
  --generation_extra_tokens 8 \
  "${LOAD_IN_4BIT_ARGS[@]}" \
  --torch_dtype bfloat16 \
  --attn_implementation "${ATTN_IMPL}" \
  --log_every 100

"${PYTHON_BIN}" -m attention_heads_edit.eval.eval_api4_privacy_core_metrics \
  --dataset "${EVAL_DATASET}" \
  --method_name "${FFN_EDIT_METHOD}" \
  --base_model "${BASE_MODEL}" \
  --adapter "${ADAPTER}" \
  --run_name "${FFN_EDIT_CORE_RUN}" \
  --output_dir "${RUN_ROOT}" \
  --disable_attention_heads_edit \
  --ffn_edit_kn_config "${KN_CONFIG}" \
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
  --ppl_batch_size "${PPL_BATCH_SIZE}" \
  --max_context_tokens "${MAX_CONTEXT}" \
  --max_new_tokens 96 \
  --generation_extra_tokens 8 \
  "${LOAD_IN_4BIT_ARGS[@]}" \
  --torch_dtype bfloat16 \
  --attn_implementation "${ATTN_IMPL}" \
  --log_every 50

if [ "${FORCE_ATTENTION_HEADS_EDIT_RELOCATE}" = "1" ]; then
  rm -f "${CANDIDATE_HEADS}" "${HEAD_CONFIG}"
fi

if [ ! -s "${CANDIDATE_HEADS}" ]; then
  echo "[STEP 4/6] $(date -Is) Locate independent Attention Heads Edit candidate pool top${CANDIDATE_TOP_K} WITHOUT FFN_Edit"
  wait_gpu_idle
  "${PYTHON_BIN}" -m attention_heads_edit.locate.locate_api4_attention_heads \
    --dataset "${TRAIN_DATASET}" \
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
    "${LOAD_IN_4BIT_ARGS[@]}" \
    --torch_dtype bfloat16 \
    --attn_implementation "${ATTN_IMPL}" \
    --metrics_dir outputs/ffn_edit/metrics \
    --config_dir outputs/attention_heads_edit/heads \
    --log_every 50
else
  echo "[INFO] $(date -Is) Reusing existing independent candidate heads: ${CANDIDATE_HEADS}"
fi

echo "[STEP 5/6] $(date -Is) Post-FFN Edit exact-safe Attention Heads Edit screen with per-type limits"
wait_gpu_idle
SCREEN_SAFETY_ARGS=()
if [ "${ENFORCE_PER_TYPE_NONWORSENING}" = "1" ]; then
  SCREEN_SAFETY_ARGS+=(--enforce_per_type_nonworsening)
fi
if [ "${PREFER_SMALLEST_EXACT_PREFIX}" = "1" ]; then
  SCREEN_SAFETY_ARGS+=(--prefer_smallest_exact_prefix)
fi
"${PYTHON_BIN}" -m attention_heads_edit.locate.select_attention_heads_by_exact_after_ffn_edit \
  --dataset "${SCREEN_DATASET}" \
  --base_model "${BASE_MODEL}" \
  --adapter "${ADAPTER}" \
  --candidate_heads "${CANDIDATE_HEADS}" \
  --ffn_edit_kn_config "${KN_CONFIG}" \
  --ffn_edit_erase_num "${FFN_EDIT_ERASE_NUM}" \
  --run_name "${SCREEN_RUN}" \
  --output_dir "${RUN_ROOT}" \
  --config_dir outputs/attention_heads_edit/heads \
  --alpha "${ATTENTION_HEADS_EDIT_ALPHA}" \
  --scale_position "${SCALE_POSITION}" \
  --steering_span_mode "${STEERING_SPAN_MODE}" \
  --steering_tail_tokens "${STEERING_TAIL_TOKENS}" \
  --top_k "${SELECT_TOP_K}" \
  --candidate_limit "${CANDIDATE_TOP_K}" \
  --min_exact_gain "${MIN_EXACT_GAIN}" \
  --min_score_gain "${MIN_SCORE_GAIN}" \
  --prefix_sizes "${PREFIX_SIZES}" \
  --per_type_min_samples "${PER_TYPE_MIN_SAMPLES}" \
  --screen_max_samples "${SCREEN_MAX_SAMPLES}" \
  --screen_limit_per_type "${SCREEN_LIMIT_PER_TYPE}" \
  --batch_size "${EXACT_BATCH_SIZE}" \
  --max_context_tokens "${MAX_CONTEXT}" \
  --max_new_tokens 96 \
  --generation_extra_tokens 8 \
  "${LOAD_IN_4BIT_ARGS[@]}" \
  --torch_dtype bfloat16 \
  --attn_implementation "${ATTN_IMPL}" \
  "${SCREEN_SAFETY_ARGS[@]}" \
  --log_every 10

test -s "${HEAD_CONFIG}"
echo "[INFO] selected_head_config=${HEAD_CONFIG}"

echo "[STEP 6/6] $(date -Is) ${MODEL_LABEL} FFN_Edit${FFN_EDIT_ERASE_NUM} THEN independent Attention Heads Edit full metrics"
wait_gpu_idle
"${PYTHON_BIN}" -m attention_heads_edit.eval.eval_api4_privacy_reverse_attention_heads_edit \
  --dataset "${EVAL_DATASET}" \
  --base_model "${BASE_MODEL}" \
  --adapter "${ADAPTER}" \
  --head_config "${HEAD_CONFIG}" \
  --run_name "${JOINT_EXACT_RUN}" \
  --output_dir "${RUN_ROOT}" \
  --batch_size "${EXACT_BATCH_SIZE}" \
  --alpha "${ATTENTION_HEADS_EDIT_ALPHA}" \
  --scale_position "${SCALE_POSITION}" \
  --steering_span_mode "${STEERING_SPAN_MODE}" \
  --steering_tail_tokens "${STEERING_TAIL_TOKENS}" \
  --ffn_edit_kn_config "${KN_CONFIG}" \
  --ffn_edit_erase_num "${FFN_EDIT_ERASE_NUM}" \
  --max_context_tokens "${MAX_CONTEXT}" \
  --max_new_tokens 96 \
  --generation_extra_tokens 8 \
  "${LOAD_IN_4BIT_ARGS[@]}" \
  --torch_dtype bfloat16 \
  --attn_implementation "${ATTN_IMPL}" \
  --log_every 100

"${PYTHON_BIN}" -m attention_heads_edit.eval.eval_api4_privacy_core_metrics \
  --dataset "${EVAL_DATASET}" \
  --method_name "${JOINT_METHOD}" \
  --base_model "${BASE_MODEL}" \
  --adapter "${ADAPTER}" \
  --run_name "${JOINT_CORE_RUN}" \
  --output_dir "${RUN_ROOT}" \
  --head_config "${HEAD_CONFIG}" \
  --alpha "${ATTENTION_HEADS_EDIT_ALPHA}" \
  --scale_position "${SCALE_POSITION}" \
  --steering_span_mode "${STEERING_SPAN_MODE}" \
  --steering_tail_tokens "${STEERING_TAIL_TOKENS}" \
  --ppl_attention_heads_edit_mode none \
  --ffn_edit_kn_config "${KN_CONFIG}" \
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
  --ppl_batch_size "${PPL_BATCH_SIZE}" \
  --max_context_tokens "${MAX_CONTEXT}" \
  --max_new_tokens 96 \
  --generation_extra_tokens 8 \
  "${LOAD_IN_4BIT_ARGS[@]}" \
  --torch_dtype bfloat16 \
  --attn_implementation "${ATTN_IMPL}" \
  --log_every 50

cat > "${MANIFEST}" <<JSON
{
  "run_name": "${RUN_NAME}",
  "method_family": "${DATASET_LABEL}_${MODEL_LABEL}_FFN_EDIT_and_FFN_EDIT_then_TriggerSpan_Attention_Heads_Edit",
  "order": "model-local FFN_Edit${FFN_EDIT_ERASE_NUM}; standalone FFN Edit evaluation; independent next-token trigger-span candidate localization; post-FFN Edit validation screening; FFN Edit then Attention Heads Edit evaluation",
  "dataset": "${EVAL_DATASET}",
  "train_dataset_for_attention_heads_edit_locate": "${TRAIN_DATASET}",
  "screen_dataset_for_attention_heads_edit": "${SCREEN_DATASET}",
  "privacy_data": "${PRIV_DATA}",
  "base_model": "${BASE_MODEL}",
  "adapter": "${ADAPTER}",
  "model_kind": "${MODEL_KIND}",
  "baseline_reference": {
    "method": "${BASELINE_METHOD}",
    "exact_metrics": "${BASELINE_EXACT}",
    "core_metrics": "${BASELINE_CORE}"
  },
  "ffn_edit": {
    "kn_config": "${KN_CONFIG}",
    "erase_num": ${FFN_EDIT_ERASE_NUM},
    "ranking_metric": "${FFN_EDIT_RANKING_METRIC}",
    "privacy_data_format": "corrected_nested_bags"
  },
  "attention_heads_edit": {
    "head_config": "${HEAD_CONFIG}",
    "head_source": "model-local next-token trigger-span localization on train, exact-safe screening after FFN Edit on validation",
    "candidate_top_k": ${CANDIDATE_TOP_K},
    "max_heads_per_layer": ${MAX_HEADS_PER_LAYER},
    "top_k": ${SELECT_TOP_K},
    "top_k_rule": "scaled_from_llama_top30_by_total_query_heads",
    "alpha": ${ATTENTION_HEADS_EDIT_ALPHA},
    "scale_position": "${SCALE_POSITION}",
    "query_mode": "${QUERY_MODE}",
    "steering_span_mode": "${STEERING_SPAN_MODE}",
    "steering_tail_tokens": ${STEERING_TAIL_TOKENS},
    "min_exact_gain": ${MIN_EXACT_GAIN},
    "enforce_per_type_nonworsening": ${ENFORCE_PER_TYPE_NONWORSENING},
    "per_type_min_samples": ${PER_TYPE_MIN_SAMPLES},
    "prefer_smallest_exact_prefix": ${PREFER_SMALLEST_EXACT_PREFIX}
  },
  "methods": [
    {
      "method": "${BASELINE_METHOD}",
      "exact_metrics": "${BASELINE_EXACT}",
      "core_metrics": "${BASELINE_CORE}",
      "reference": "strong_sft_no_defense"
    },
    {
      "method": "${FFN_EDIT_METHOD}",
      "exact_metrics": "${RUN_ROOT}/${FFN_EDIT_EXACT_RUN}/metrics/metrics.json",
      "core_metrics": "${RUN_ROOT}/${FFN_EDIT_CORE_RUN}/metrics/core_metrics.json"
    },
    {
      "method": "${JOINT_METHOD}",
      "exact_metrics": "${RUN_ROOT}/${JOINT_EXACT_RUN}/metrics/metrics.json",
      "core_metrics": "${RUN_ROOT}/${JOINT_CORE_RUN}/metrics/core_metrics.json"
    }
  ]
}
JSON

"${PYTHON_BIN}" -m attention_heads_edit.eval.summarize_api4_privacy_suite \
  --manifest "${MANIFEST}" \
  --output_dir "${SUMMARY_DIR}"

test -s "${SUMMARY_DIR}/paper_main_table.csv"
echo "[DONE] $(date -Is) run=${RUN_NAME}"
echo "[DONE] summary=${SUMMARY_DIR}/paper_main_table.csv"
echo "[DONE] manifest=${MANIFEST}"
cat "${SUMMARY_DIR}/paper_main_table.csv"
