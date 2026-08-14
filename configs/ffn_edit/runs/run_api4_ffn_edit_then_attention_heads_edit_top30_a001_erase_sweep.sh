#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${REPO_ROOT}/srcs/attention_heads_edit:${REPO_ROOT}/srcs/attention_heads_edit/scripts:${REPO_ROOT}/srcs/ffn_edit/utils:${REPO_ROOT}/srcs/ffn_edit/eval:${PYTHONPATH:-}"

RUN_NAME="${RUN_NAME:-api4_ffn_edit_then_attention_heads_edit_top30_a001_erase_sweep_$(date +%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/ffn_edit/ffn_edit_then_attention_heads_edit_top30_a001_erase_sweep}"
SUMMARY_DIR="outputs/ffn_edit/metrics/${RUN_NAME}"
LOG_FILE="logs/ffn_edit/${RUN_NAME}.log"
MANIFEST="configs/ffn_edit/runs/${RUN_NAME}_manifest.json"
TIMING_CSV="${SUMMARY_DIR}/runtime_by_step.csv"

BASE_MODEL="${BASE_MODEL:-models/llama3-8B/baseline}"
ADAPTER="${ADAPTER:-models/llama3-8B/api4_prefix_plain_qlora}"
DATASET="${DATASET:-data/api4_200k/sft_true_prefix_no_instruction_all.json}"
ATTENTION_HEADS_EDIT_HEAD_CONFIG="${ATTENTION_HEADS_EDIT_HEAD_CONFIG:-configs/attention_heads_edit/runs/api4_attention_heads_edit_top100_a015_fixed_topN_top30.json}"
ATTENTION_HEADS_EDIT_ALPHA="${ATTENTION_HEADS_EDIT_ALPHA:-0.01}"
ERASE_LIST="${ERASE_LIST:-64 128 256 512 1024}"

FFN_EDIT_KN_CONFIG="${FFN_EDIT_KN_CONFIG:-outputs/ffn_edit/ffn_edit_only_recheck_64_1024/api4_ffn_edit_only_recheck_64_1024/attribution/kn/kn_bag-api4_ffn_edit_only_recheck_64_1024.json}"

BASE_EXACT="${BASE_EXACT:-outputs/attention_heads_edit/api4_repro_baseline_exact/metrics/metrics.json}"
BASE_CORE="${BASE_CORE:-outputs/attention_heads_edit/api4_repro_baseline_core/metrics/core_metrics.json}"
FFN_EDIT_RECHECK_SUMMARY="${FFN_EDIT_RECHECK_SUMMARY:-outputs/ffn_edit/metrics/api4_ffn_edit_repro/paper_main_table.csv}"

mkdir -p logs/ffn_edit configs/ffn_edit/runs "${OUTPUT_ROOT}" "${SUMMARY_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[START] $(date -Is) run=${RUN_NAME}"
echo "[INFO] FFN Edit-first erase sweep: reuse standalone FFN Edit positions, then apply fixed top30 AttentionHeadsEdit."
echo "[INFO] ffn_edit_kn_config=${FFN_EDIT_KN_CONFIG}"
echo "[INFO] erase_list=${ERASE_LIST}"
echo "[INFO] attention_heads_edit_head_config=${ATTENTION_HEADS_EDIT_HEAD_CONFIG}"
echo "[INFO] attention_heads_edit_alpha=${ATTENTION_HEADS_EDIT_ALPHA} scale_position=include_down"
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
print("[INFO] loaded unique standalone FFN Edit positions", len(seen), "from", p)
needed = max(int(x) for x in "${ERASE_LIST}".split())
if len(seen) < needed:
    raise SystemExit(f"expected at least {needed} unique positions, got {len(seen)}")
PY

echo "method,stage,start_epoch,end_epoch,duration_seconds" > "${TIMING_CSV}"

record_timing() {
  local method="$1"
  local stage="$2"
  local start_epoch="$3"
  local end_epoch="$4"
  echo "${method},${stage},${start_epoch},${end_epoch},$((end_epoch - start_epoch))" >> "${TIMING_CSV}"
}

METHODS_JSON=""
for ERASE_NUM in ${ERASE_LIST}; do
  METHOD="FFN_Edit${ERASE_NUM}_then_Attention_Heads_Edit_top30_alpha_0p01"
  EXACT_RUN="${RUN_NAME}_erase${ERASE_NUM}_exact_full"
  CORE_RUN="${RUN_NAME}_erase${ERASE_NUM}_core"
  EXACT_METRICS="${OUTPUT_ROOT}/${EXACT_RUN}/metrics/metrics.json"
  CORE_METRICS="${OUTPUT_ROOT}/${CORE_RUN}/metrics/core_metrics.json"

  if [ -s "${EXACT_METRICS}" ]; then
    echo "[SKIP] $(date -Is) erase=${ERASE_NUM} Exact/Contains exists: ${EXACT_METRICS}"
  else
    echo "[STEP] $(date -Is) erase=${ERASE_NUM} Exact/Contains with standalone FFN Edit${ERASE_NUM} + top30 Attention Heads Edit alpha=${ATTENTION_HEADS_EDIT_ALPHA}"
    start_epoch="$(date +%s)"
    python srcs/attention_heads_edit/scripts/eval_api4_privacy_reverse_attention_heads_edit.py \
      --dataset "${DATASET}" \
      --base_model "${BASE_MODEL}" \
      --adapter "${ADAPTER}" \
      --head_config "${ATTENTION_HEADS_EDIT_HEAD_CONFIG}" \
      --run_name "${EXACT_RUN}" \
      --output_dir "${OUTPUT_ROOT}" \
      --batch_size 4 \
      --alpha "${ATTENTION_HEADS_EDIT_ALPHA}" \
      --scale_position include_down \
      --ffn_edit_kn_config "${FFN_EDIT_KN_CONFIG}" \
      --ffn_edit_erase_num "${ERASE_NUM}" \
      --max_context_tokens 768 \
      --max_new_tokens 64 \
      --generation_extra_tokens 8 \
      --load_in_4bit \
      --torch_dtype bfloat16 \
      --attn_implementation eager \
      --log_every 500
    end_epoch="$(date +%s)"
    record_timing "${METHOD}" "exact_contains" "${start_epoch}" "${end_epoch}"
  fi

  if [ -s "${CORE_METRICS}" ]; then
    echo "[SKIP] $(date -Is) erase=${ERASE_NUM} MRR/ASR@32/PPL exists: ${CORE_METRICS}"
  else
    echo "[STEP] $(date -Is) erase=${ERASE_NUM} MRR/ASR@32/PPL with standalone FFN Edit${ERASE_NUM} + top30 Attention Heads Edit alpha=${ATTENTION_HEADS_EDIT_ALPHA}"
    start_epoch="$(date +%s)"
    python srcs/attention_heads_edit/scripts/eval_api4_privacy_core_metrics.py \
      --dataset "${DATASET}" \
      --base_model "${BASE_MODEL}" \
      --adapter "${ADAPTER}" \
      --method_name "${METHOD}" \
      --run_name "${CORE_RUN}" \
      --output_dir "${OUTPUT_ROOT}" \
      --head_config "${ATTENTION_HEADS_EDIT_HEAD_CONFIG}" \
      --alpha "${ATTENTION_HEADS_EDIT_ALPHA}" \
      --scale_position include_down \
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
    end_epoch="$(date +%s)"
    record_timing "${METHOD}" "mrr_asr_ppl" "${start_epoch}" "${end_epoch}"
  fi

  METHODS_JSON="${METHODS_JSON},
    {
      \"method\": \"${METHOD}\",
      \"exact_metrics\": \"${EXACT_METRICS}\",
      \"core_metrics\": \"${CORE_METRICS}\",
      \"ffn_edit_erase_num\": ${ERASE_NUM},
      \"ffn_edit_kn_config\": \"${FFN_EDIT_KN_CONFIG}\",
      \"attention_heads_edit_alpha\": ${ATTENTION_HEADS_EDIT_ALPHA}
    }"
done

python - <<PY
import csv
import json
from pathlib import Path

summary = Path("${FFN_EDIT_RECHECK_SUMMARY}")
methods = []
if summary.exists():
    with summary.open() as handle:
        rows = list(csv.DictReader(handle))
    exact_dir = Path("outputs/ffn_edit/ffn_edit_only_recheck_64_1024")
    run = "api4_ffn_edit_only_recheck_64_1024"
    for erase in "${ERASE_LIST}".split():
        methods.append({
            "method": f"FFN_EDIT_recheck_erase{erase}",
            "exact_metrics": str(exact_dir / f"{run}_erase{erase}_exact_full/metrics/metrics.json"),
            "core_metrics": str(exact_dir / f"{run}_erase{erase}_core/metrics/core_metrics.json"),
            "ffn_edit_erase_num": int(erase),
            "reference": "standalone_ffn_edit_recheck",
        })

manifest = {
    "run_name": "${RUN_NAME}",
    "method_family": "FFN_EDIT_then_Attention_Heads_Edit_top30_alpha0p01_erase_sweep",
    "order": "standalone FFN Edit positions first, then fixed top30 Attention Heads Edit at evaluation",
    "attention_heads_edit_head_config": "${ATTENTION_HEADS_EDIT_HEAD_CONFIG}",
    "attention_heads_edit_alpha": float("${ATTENTION_HEADS_EDIT_ALPHA}"),
    "scale_position": "include_down",
    "ffn_edit_kn_config": "${FFN_EDIT_KN_CONFIG}",
    "erase_list": "${ERASE_LIST}",
    "timing_csv": "${TIMING_CSV}",
    "methods": [
        {
            "method": "Base",
            "exact_metrics": "${BASE_EXACT}",
            "core_metrics": "${BASE_CORE}",
        },
        *methods,
    ],
}
manifest["methods"].extend(json.loads("[" + """${METHODS_JSON}""".lstrip(",") + "]") if """${METHODS_JSON}""".strip() else [])
Path("${MANIFEST}").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\\n")
PY

python srcs/attention_heads_edit/scripts/summarize_api4_privacy_suite.py \
  --manifest "${MANIFEST}" \
  --output_dir "${SUMMARY_DIR}"

echo "[DONE] $(date -Is) run=${RUN_NAME}"
echo "[DONE] summary=${SUMMARY_DIR}/paper_main_table.csv"
echo "[DONE] timing=${TIMING_CSV}"
