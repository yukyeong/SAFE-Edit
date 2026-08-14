#!/usr/bin/env bash
# Evaluate the fine-tuned Llama3 API4 baseline on all privacy and utility metrics.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${REPO_ROOT}/srcs/attention_heads_edit:${REPO_ROOT}/srcs/attention_heads_edit/scripts:${REPO_ROOT}/srcs/ffn_edit/utils:${PYTHONPATH:-}"

DATASET="${DATASET:-data/api4_200k/sft_true_prefix_no_instruction_all.json}"
BASE_MODEL="${BASE_MODEL:-models/llama3-8B/baseline}"
ADAPTER="${ADAPTER:-models/llama3-8B/api4_prefix_plain_qlora}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/attention_heads_edit}"
EXACT_RUN="${EXACT_RUN:-api4_repro_baseline_exact}"
CORE_RUN="${CORE_RUN:-api4_repro_baseline_core}"

python srcs/attention_heads_edit/scripts/eval_api4_privacy_reverse_attention_heads_edit.py \
  --dataset "${DATASET}" \
  --base_model "${BASE_MODEL}" \
  --adapter "${ADAPTER}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${EXACT_RUN}" \
  --disable_attention_heads_edit \
  --batch_size 2 \
  --load_in_4bit \
  --torch_dtype bfloat16 \
  --attn_implementation eager \
  --max_context_tokens 768 \
  --max_new_tokens 64 \
  --generation_extra_tokens 8 \
  --log_every 200

python srcs/attention_heads_edit/scripts/eval_api4_privacy_core_metrics.py \
  --dataset "${DATASET}" \
  --method_name API4_Llama3_8B_SFT_no_defense \
  --base_model "${BASE_MODEL}" \
  --adapter "${ADAPTER}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${CORE_RUN}" \
  --disable_attention_heads_edit \
  --run_mrr \
  --run_attack \
  --run_ppl \
  --mrr_limit_per_type 200 \
  --mrr_batch_size 2 \
  --attack_limit_per_type 10 \
  --attack_batch_size 2 \
  --attack_samples 32 \
  --ppl_max_blocks 128 \
  --ppl_block_tokens 512 \
  --ppl_batch_size 1 \
  --load_in_4bit \
  --torch_dtype bfloat16 \
  --attn_implementation eager \
  --max_context_tokens 768 \
  --max_new_tokens 64 \
  --generation_extra_tokens 8 \
  --log_every 100

echo "[DONE] exact=${OUTPUT_DIR}/${EXACT_RUN}/metrics/metrics.json"
echo "[DONE] core=${OUTPUT_DIR}/${CORE_RUN}/metrics/core_metrics.json"
