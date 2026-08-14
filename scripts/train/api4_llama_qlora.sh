#!/usr/bin/env bash
# Full plain-prefix QLoRA (run from repo root)
#   bash scripts/train/api4_llama_qlora.sh

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../_common.sh"

export LLAMA_TRAIN_JSON="${LLAMA_TRAIN_JSON:-${REPO_ROOT}/data/api4_200k/sft_true_prefix_no_instruction.json}"
export LLAMA_VAL_JSON="${LLAMA_VAL_JSON:-${REPO_ROOT}/data/api4_200k/sft_true_prefix_no_instruction_val.json}"
export LLAMA_MODEL_PATH="${LLAMA_MODEL_PATH:-${REPO_ROOT}/models/llama3-8B/baseline}"
export LLAMA_OUTPUT_DIR="${LLAMA_OUTPUT_DIR:-${REPO_ROOT}/checkpoints/ffn_edit/llama3-8b_lora/api4_prefix_plain}"
export LLAMA_FINAL_MODEL_DIR="${LLAMA_FINAL_MODEL_DIR:-${REPO_ROOT}/models/llama3-8B/api4_prefix_plain_qlora}"
export LLAMA_LOG_FILE="${LLAMA_LOG_FILE:-${REPO_ROOT}/logs/ffn_edit/train/run_llama3_api4_prefix_qlora_full_$(date +%Y%m%d_%H%M%S).log}"
export LLAMA_RESUME_FROM_CHECKPOINT="${LLAMA_RESUME_FROM_CHECKPOINT:-}"

exec bash "${REPO_ROOT}/scripts/train/run_llama3_api4_prefix_qlora.sh"
