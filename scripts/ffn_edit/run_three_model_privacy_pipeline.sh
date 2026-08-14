#!/usr/bin/env bash
# Fresh model-local FFN Edit localization, FFN-Edit-only evaluation, independent Attention Heads Edit localization,
# and FFN Edit -> Attention Heads Edit full metrics for LUME or CRAPII.

set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <lume|crapii> <llama|qwen|ministral>" >&2
  exit 2
fi

DATASET_KIND="$1"
MODEL_KIND="$2"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

USER_FFN_EDIT_ERASE_NUM="${FFN_EDIT_ERASE_NUM:-}"
USER_CANDIDATE_TOP_K="${CANDIDATE_TOP_K:-}"
USER_MAX_HEADS_PER_LAYER="${MAX_HEADS_PER_LAYER:-}"
USER_SELECT_TOP_K="${SELECT_TOP_K:-}"
USER_ATTENTION_HEADS_EDIT_ALPHA="${ATTENTION_HEADS_EDIT_ALPHA:-}"
USER_STEERING_SPAN_MODE="${STEERING_SPAN_MODE:-}"
USER_STEERING_TAIL_TOKENS="${STEERING_TAIL_TOKENS:-}"

PYTHON_BIN="python"
MAX_CONTEXT=768
CANDIDATE_TOP_K=96
MAX_HEADS_PER_LAYER=0
SELECT_TOP_K=30
ATTENTION_HEADS_EDIT_ALPHA=0.01
STEERING_SPAN_MODE=tail_tokens
STEERING_TAIL_TOKENS=8
ENFORCE_PER_TYPE_NONWORSENING=1
PREFER_SMALLEST_EXACT_PREFIX=1

case "${MODEL_KIND}" in
  llama)
    BASE_MODEL="models/llama3-8B/baseline"
    ;;
  qwen)
    BASE_MODEL="models/qwen3-8b-base"
    ;;
  ministral)
    BASE_MODEL="models/ministral-3-8b-base"
    PYTHON_BIN="${MINISTRAL_PYTHON:-${REPO_ROOT}/.venvs/ministral3/bin/python}"
    ;;
  *)
    echo "Unknown model: ${MODEL_KIND}" >&2
    exit 2
    ;;
esac

case "${DATASET_KIND}" in
  lume)
    PRIV_DATA="data/lume/api4_prefix_task2/privacy_data_lume_task2_scenario_clean_all.json"
    TRAIN_DATASET="data/lume/api4_prefix_task2/sft_true_prefix_no_instruction_scenario_clean_train.json"
    SCREEN_DATASET="data/lume/api4_prefix_task2/sft_true_prefix_no_instruction_scenario_clean_val.json"
    EVAL_DATASET="data/lume/api4_prefix_task2/sft_true_prefix_no_instruction_scenario_clean_all.json"
    case "${MODEL_KIND}" in
      llama)
        ADAPTER="models/llama3-8B/lume_task2_prefix_plain_qlora"
        FFN_EDIT_ERASE_NUM=128
        SELECT_TOP_K=10
        ;;
      qwen)
        ADAPTER="models/qwen3-8b-base/lume_task2_hard_balanced_qlora"
        FFN_EDIT_ERASE_NUM=32
        CANDIDATE_TOP_K=128
        MAX_HEADS_PER_LAYER=4
        SELECT_TOP_K=5
        STEERING_SPAN_MODE=adaptive_tail
        ;;
      ministral)
        ADAPTER="models/ministral-3-8b-base/lume_task2_hard_balanced_qlora"
        FFN_EDIT_ERASE_NUM=16
        CANDIDATE_TOP_K=128
        MAX_HEADS_PER_LAYER=4
        SELECT_TOP_K=32
        ATTENTION_HEADS_EDIT_ALPHA=0.1
        ;;
    esac
    ;;
  crapii)
    PRIV_DATA="data/crapii/api4_prefix/privacy_data_crapii_all.json"
    TRAIN_DATASET="data/crapii/api4_prefix/sft_true_prefix_no_instruction_train.json"
    SCREEN_DATASET="data/crapii/api4_prefix/sft_true_prefix_no_instruction_val.json"
    EVAL_DATASET="data/crapii/api4_prefix/sft_true_prefix_no_instruction_all.json"
    MAX_CONTEXT=512
    ATTENTION_HEADS_EDIT_ALPHA=0.1
    case "${MODEL_KIND}" in
      llama)
        ADAPTER="models/llama3-8B/crapii_prefix_qlora_strong_r64_e8"
        FFN_EDIT_ERASE_NUM=32
        ;;
      qwen)
        ADAPTER="models/qwen3-8b-base/crapii_prefix_qlora_strong_r64_e8"
        FFN_EDIT_ERASE_NUM=32
        CANDIDATE_TOP_K=128
        MAX_HEADS_PER_LAYER=4
        SELECT_TOP_K=34
        ;;
      ministral)
        ADAPTER="models/ministral-3-8b-base/crapii_prefix_qlora_strong_r64_e8"
        FFN_EDIT_ERASE_NUM=16
        CANDIDATE_TOP_K=256
        MAX_HEADS_PER_LAYER=8
        SELECT_TOP_K=3
        ATTENTION_HEADS_EDIT_ALPHA=0.02
        STEERING_SPAN_MODE=adaptive_tail
        STEERING_TAIL_TOKENS=16
        ;;
    esac
    ;;
  *)
    echo "Unknown dataset: ${DATASET_KIND}" >&2
    exit 2
    ;;
esac

FFN_EDIT_ERASE_NUM="${USER_FFN_EDIT_ERASE_NUM:-${FFN_EDIT_ERASE_NUM}}"
CANDIDATE_TOP_K="${USER_CANDIDATE_TOP_K:-${CANDIDATE_TOP_K}}"
MAX_HEADS_PER_LAYER="${USER_MAX_HEADS_PER_LAYER:-${MAX_HEADS_PER_LAYER}}"
SELECT_TOP_K="${USER_SELECT_TOP_K:-${SELECT_TOP_K}}"
ATTENTION_HEADS_EDIT_ALPHA="${USER_ATTENTION_HEADS_EDIT_ALPHA:-${ATTENTION_HEADS_EDIT_ALPHA}}"
STEERING_SPAN_MODE="${USER_STEERING_SPAN_MODE:-${STEERING_SPAN_MODE}}"
STEERING_TAIL_TOKENS="${USER_STEERING_TAIL_TOKENS:-${STEERING_TAIL_TOKENS}}"

RUN_STAMP="${RUN_STAMP:-repro_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/ffn_edit/repro_${DATASET_KIND}_${MODEL_KIND}}"

CRAPII_MODEL_KIND="${MODEL_KIND}" \
CRAPII_PYTHON="${PYTHON_BIN}" \
DATASET_LABEL="${DATASET_KIND^^}" \
RUN_STAMP="${RUN_STAMP}" \
PRIV_DATA="${PRIV_DATA}" \
TRAIN_DATASET="${TRAIN_DATASET}" \
SCREEN_DATASET="${SCREEN_DATASET}" \
EVAL_DATASET="${EVAL_DATASET}" \
BASE_MODEL="${BASE_MODEL}" \
ADAPTER="${ADAPTER}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
FFN_EDIT_ERASE_NUM="${FFN_EDIT_ERASE_NUM}" \
FORCE_FFN_EDIT_RELOCATE="${FORCE_FFN_EDIT_RELOCATE:-1}" \
FORCE_ATTENTION_HEADS_EDIT_RELOCATE="${FORCE_ATTENTION_HEADS_EDIT_RELOCATE:-1}" \
FORCE_BASELINE_EVAL="${FORCE_BASELINE_EVAL:-1}" \
MISSING_BASELINE_POLICY=evaluate \
CANDIDATE_TOP_K="${CANDIDATE_TOP_K}" \
MAX_HEADS_PER_LAYER="${MAX_HEADS_PER_LAYER}" \
SELECT_TOP_K="${SELECT_TOP_K}" \
ATTENTION_HEADS_EDIT_ALPHA="${ATTENTION_HEADS_EDIT_ALPHA}" \
STEERING_SPAN_MODE="${STEERING_SPAN_MODE}" \
STEERING_TAIL_TOKENS="${STEERING_TAIL_TOKENS}" \
MAX_CONTEXT="${MAX_CONTEXT}" \
ENFORCE_PER_TYPE_NONWORSENING="${ENFORCE_PER_TYPE_NONWORSENING}" \
PREFER_SMALLEST_EXACT_PREFIX="${PREFER_SMALLEST_EXACT_PREFIX}" \
bash configs/ffn_edit/runs/run_crapii_ffn_edit128_then_attention_heads_edit_intervention_model.sh
