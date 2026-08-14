#!/usr/bin/env bash
# Calibrate one model's FFN Edit erase count with a fixed exact-leakage subset.
# Input: processed LUME/CRAPII eval JSON, model-specific adapter, model-local FFN Edit KN ranking.
# Output: sweep CSV under outputs/ffn_edit/metrics and metrics JSON for every tested count.
# Impact: evaluation-time neuron erasure only; no model weights, checkpoints, or datasets are modified.

set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../_common.sh"


DATASET_KIND="${DATASET_KIND:?Set DATASET_KIND to one of: lume, crapii}"
MODEL_KIND="${MODEL_KIND:?Set MODEL_KIND to one of: llama, qwen, ministral}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

RUN_STAMP="${RUN_STAMP:-ffn_edit_exact_sweep}"
GPU_IDLE_MEM_MB="${GPU_IDLE_MEM_MB:-1000}"
ATTN_IMPL="${ATTN_IMPL:-eager}"
SWEEP_LIMIT_PER_TYPE="${SWEEP_LIMIT_PER_TYPE:-80}"
PYTHON_BIN="${PYTHON_BIN:-}"
LOAD_IN_4BIT_ARGS=(--load_in_4bit)

case "${DATASET_KIND}" in
  lume)
    EVAL_DATASET="${EVAL_DATASET:-data/lume/api4_prefix_task2/sft_true_prefix_no_instruction_scenario_clean_all.json}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/ffn_edit/lume_ffn_edit_exact_sweep}"
    MAX_CONTEXT="${MAX_CONTEXT:-768}"
    DEFAULT_COUNTS_LLAMA="64 128"
    DEFAULT_COUNTS_QWEN="8 16 32 48 64"
    DEFAULT_COUNTS_MINISTRAL="4 8 16 32 64"
    ;;
  crapii)
    EVAL_DATASET="${EVAL_DATASET:-data/crapii/api4_prefix/sft_true_prefix_no_instruction_all.json}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/ffn_edit/crapii_ffn_edit_exact_sweep}"
    MAX_CONTEXT="${MAX_CONTEXT:-512}"
    DEFAULT_COUNTS_LLAMA="16 32 64"
    DEFAULT_COUNTS_QWEN="4 8 16 24 32"
    DEFAULT_COUNTS_MINISTRAL="1 2 4 8 12 16"
    ;;
  *)
    echo "[ffn_edit_sweep] invalid DATASET_KIND=${DATASET_KIND}; expected lume or crapii" >&2
    exit 2
    ;;
esac

case "${MODEL_KIND}" in
  llama)
    MODEL_TAG="llama3_8b"
    MODEL_LABEL="Llama3_8B"
    PYTHON_BIN="${PYTHON_BIN:-python}"
    BASE_MODEL="${BASE_MODEL:-models/llama3-8B/baseline}"
    if [ "${DATASET_KIND}" = "lume" ]; then
      ADAPTER="${ADAPTER:-models/llama3-8B/lume_task2_prefix_plain_qlora}"
      KN_CONFIG="${KN_CONFIG:-outputs/ffn_edit/lume_task2_ablation/lume_task2_ffn_edit256_and_attention_heads_edit_top30_a001/ffn_edit256_attribution/kn/kn_bag-lume_task2_ffn_edit256_and_attention_heads_edit_top30_a001_ffn_edit_only.json}"
    else
      ADAPTER="${ADAPTER:-models/llama3-8B/crapii_prefix_qlora_strong_r64_e8}"
      KN_CONFIG="${KN_CONFIG:-outputs/ffn_edit/crapii_ffn_edit128_self_attention_heads_edit_top30/crapii_llama_ffn_edit128_self_attention_heads_edit_top30_a0p01/attribution/kn/kn_bag-crapii_llama_ffn_edit128_self_attention_heads_edit_top30_a0p01.json}"
    fi
    COUNTS="${COUNTS:-${DEFAULT_COUNTS_LLAMA}}"
    EXACT_BATCH_SIZE="${EXACT_BATCH_SIZE:-2}"
    ;;
  qwen)
    MODEL_TAG="qwen3_8b_base"
    MODEL_LABEL="Qwen3_8B_Base"
    PYTHON_BIN="${PYTHON_BIN:-python}"
    BASE_MODEL="${BASE_MODEL:-models/qwen3-8b-base}"
    if [ "${DATASET_KIND}" = "lume" ]; then
      ADAPTER="${ADAPTER:-models/qwen3-8b-base/lume_task2_hard_balanced_qlora}"
      KN_CONFIG="${KN_CONFIG:-outputs/ffn_edit/lume_task2_qwen3_8b_hard_balanced_ffn_edit128_self_attention_heads_edit_joint_only_top30/lume_task2_qwen3_8b_hard_balanced_ffn_edit128_self_attention_heads_edit_joint_only_top30_a0p01_qwen_joint_only_ffn_edit128_self_attention_heads_edit/attribution/kn/kn_bag-lume_task2_qwen3_8b_hard_balanced_ffn_edit128_self_attention_heads_edit_joint_only_top30_a0p01_qwen_joint_only_ffn_edit128_self_attention_heads_edit.json}"
    else
      ADAPTER="${ADAPTER:-models/qwen3-8b-base/crapii_prefix_qlora_strong_r64_e8}"
      KN_CONFIG="${KN_CONFIG:-outputs/ffn_edit/crapii_ffn_edit128_self_attention_heads_edit_top30/crapii_qwen_ffn_edit128_self_attention_heads_edit_top30_a0p01/attribution/kn/kn_bag-crapii_qwen_ffn_edit128_self_attention_heads_edit_top30_a0p01.json}"
    fi
    COUNTS="${COUNTS:-${DEFAULT_COUNTS_QWEN}}"
    EXACT_BATCH_SIZE="${EXACT_BATCH_SIZE:-2}"
    ;;
  ministral)
    MODEL_TAG="ministral3_8b_base"
    MODEL_LABEL="Ministral3_8B_Base"
    PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venvs/ministral3/bin/python}"
    BASE_MODEL="${BASE_MODEL:-models/ministral-3-8b-base}"
    if [ "${DATASET_KIND}" = "lume" ]; then
      ADAPTER="${ADAPTER:-models/ministral-3-8b-base/lume_task2_hard_balanced_qlora}"
      KN_CONFIG="${KN_CONFIG:-outputs/ffn_edit/lume_task2_ministral3_8b_base_hard_balanced_ffn_edit128_self_attention_heads_edit_top30/lume_task2_ministral3_8b_base_hard_balanced_ffn_edit128_self_attention_heads_edit_top30_a0p01/attribution/kn/kn_bag-lume_task2_ministral3_8b_base_hard_balanced_ffn_edit128_self_attention_heads_edit_top30_a0p01.json}"
    else
      ADAPTER="${ADAPTER:-models/ministral-3-8b-base/crapii_prefix_qlora_strong_r64_e8}"
      KN_CONFIG="${KN_CONFIG:-outputs/ffn_edit/crapii_ffn_edit128_self_attention_heads_edit_top30/crapii_ministral_ffn_edit128_self_attention_heads_edit_top30_a0p01/attribution/kn/kn_bag-crapii_ministral_ffn_edit128_self_attention_heads_edit_top30_a0p01.json}"
    fi
    COUNTS="${COUNTS:-${DEFAULT_COUNTS_MINISTRAL}}"
    EXACT_BATCH_SIZE="${EXACT_BATCH_SIZE:-1}"
    ;;
  *)
    echo "[ffn_edit_sweep] invalid MODEL_KIND=${MODEL_KIND}; expected llama, qwen, ministral" >&2
    exit 2
    ;;
esac

RUN_NAME_PREFIX="${DATASET_KIND}_${MODEL_TAG}_ffn_edit_exact_sweep_${RUN_STAMP}_limit${SWEEP_LIMIT_PER_TYPE}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_NAME_PREFIX}"
SUMMARY_CSV="${SUMMARY_CSV:-outputs/ffn_edit/metrics/${RUN_NAME_PREFIX}.csv}"
LOG_FILE="${LOG_FILE:-logs/ffn_edit/${RUN_NAME_PREFIX}.log}"
MANIFEST="${MANIFEST:-outputs/logs/manifests/${RUN_NAME_PREFIX}_manifest.json}"
BASELINE_RUN="${RUN_NAME_PREFIX}_baseline"
BASELINE_METRICS="${RUN_ROOT}/${BASELINE_RUN}/metrics/metrics.json"

mkdir -p logs/ffn_edit outputs/logs/manifests outputs/ffn_edit/metrics "${RUN_ROOT}"
exec > >(tee -a "${LOG_FILE}") 2>&1

wait_gpu_idle() {
  while true; do
    used_memory="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ' || echo 0)"
    if [ "${used_memory:-99999}" -lt "${GPU_IDLE_MEM_MB}" ]; then
      echo "[INFO] $(date -Is) GPU idle enough: ${used_memory} MiB"
      break
    fi
    echo "[WAIT] $(date -Is) GPU busy: ${used_memory} MiB (need < ${GPU_IDLE_MEM_MB})"
    sleep 120
  done
}

write_summary_csv() {
  "${PYTHON_BIN}" - <<PY
import csv
import json
from pathlib import Path

baseline_path = Path("${BASELINE_METRICS}")
summary_path = Path("${SUMMARY_CSV}")
run_root = Path("${RUN_ROOT}")
counts = [int(item) for item in "${COUNTS}".split()]

def read_exact(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    macro = payload.get("macro", {})
    return {
        "exact": macro.get("normalized_exact_match_mean"),
        "contains": macro.get("normalized_contains_target_mean"),
        "total_count": macro.get("total_count"),
    }

baseline = read_exact(baseline_path)
if baseline is None or baseline["exact"] is None:
    raise SystemExit(f"missing baseline metrics: {baseline_path}")

rows = [{
    "dataset_kind": "${DATASET_KIND}",
    "model_kind": "${MODEL_KIND}",
    "model_label": "${MODEL_LABEL}",
    "ffn_edit_erase_num": 0,
    "baseline_exact_leakage_rate": baseline["exact"],
    "exact_leakage_rate": baseline["exact"],
    "contains_leakage_rate": baseline["contains"],
    "absolute_exact_drop": 0.0,
    "relative_exact_drop": 0.0,
    "total_count": baseline["total_count"],
    "metrics_json": str(baseline_path),
}]
for count in counts:
    run_name = f"${RUN_NAME_PREFIX}_ffn_edit{count}"
    metrics_path = run_root / run_name / "metrics" / "metrics.json"
    exact_payload = read_exact(metrics_path)
    if exact_payload is None or exact_payload["exact"] is None:
        continue
    absolute_drop = baseline["exact"] - exact_payload["exact"]
    relative_drop = absolute_drop / baseline["exact"] if baseline["exact"] else 0.0
    rows.append({
        "dataset_kind": "${DATASET_KIND}",
        "model_kind": "${MODEL_KIND}",
        "model_label": "${MODEL_LABEL}",
        "ffn_edit_erase_num": count,
        "baseline_exact_leakage_rate": baseline["exact"],
        "exact_leakage_rate": exact_payload["exact"],
        "contains_leakage_rate": exact_payload["contains"],
        "absolute_exact_drop": absolute_drop,
        "relative_exact_drop": relative_drop,
        "total_count": exact_payload["total_count"],
        "metrics_json": str(metrics_path),
    })
summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"[SUMMARY] wrote {summary_path}", flush=True)
PY
}

echo "[START] $(date -Is) run=${RUN_NAME_PREFIX}"
echo "[INFO] dataset_kind=${DATASET_KIND} model=${MODEL_LABEL}"
echo "[INFO] eval_dataset=${EVAL_DATASET}"
echo "[INFO] base_model=${BASE_MODEL}"
echo "[INFO] adapter=${ADAPTER}"
echo "[INFO] kn_config=${KN_CONFIG}"
echo "[INFO] counts=${COUNTS} limit_per_type=${SWEEP_LIMIT_PER_TYPE}"
echo "[INFO] impact=no checkpoint/dataset writes; exact subset metrics only"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

required = [
    Path("${EVAL_DATASET}"),
    Path("${BASE_MODEL}") / "config.json",
    Path("${ADAPTER}") / "adapter_config.json",
    Path("${KN_CONFIG}"),
]
missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
if missing:
    raise SystemExit("missing required files: " + ", ".join(missing))

kn_raw = json.loads(Path("${KN_CONFIG}").read_text(encoding="utf-8"))
positions = []
seen = set()
for item in kn_raw:
    candidates = item if isinstance(item, list) and item and isinstance(item[0], list) else [item]
    for candidate in candidates:
        if not isinstance(candidate, list) or len(candidate) < 2:
            continue
        key = (int(candidate[0]), int(candidate[1]))
        if key in seen:
            continue
        seen.add(key)
        positions.append(key)
max_count = max(int(item) for item in "${COUNTS}".split())
if len(positions) < max_count:
    raise SystemExit(f"not enough FFN Edit positions: have={len(positions)} need={max_count}")
print(f"[VALID] kn_positions={len(positions)} max_count={max_count}", flush=True)
PY

cat > "${MANIFEST}" <<JSON
{
  "run_name": "${RUN_NAME_PREFIX}",
  "dataset_kind": "${DATASET_KIND}",
  "model_kind": "${MODEL_KIND}",
  "model_label": "${MODEL_LABEL}",
  "eval_dataset": "${EVAL_DATASET}",
  "base_model": "${BASE_MODEL}",
  "adapter": "${ADAPTER}",
  "kn_config": "${KN_CONFIG}",
  "counts": "${COUNTS}",
  "sweep_limit_per_type": ${SWEEP_LIMIT_PER_TYPE},
  "summary_csv": "${SUMMARY_CSV}",
  "impact": "evaluation-time FFN Edit exact sweep only; no checkpoint or dataset mutation"
}
JSON

if [ ! -s "${BASELINE_METRICS}" ]; then
  echo "[STEP] $(date -Is) baseline exact subset"
  wait_gpu_idle
  "${PYTHON_BIN}" -m attention_heads_edit.eval.eval_api4_privacy_reverse_attention_heads_edit \
    --dataset "${EVAL_DATASET}" \
    --base_model "${BASE_MODEL}" \
    --adapter "${ADAPTER}" \
    --run_name "${BASELINE_RUN}" \
    --output_dir "${RUN_ROOT}" \
    --disable_attention_heads_edit \
    --limit_per_type "${SWEEP_LIMIT_PER_TYPE}" \
    --batch_size "${EXACT_BATCH_SIZE}" \
    --max_context_tokens "${MAX_CONTEXT}" \
    --max_new_tokens 96 \
    --generation_extra_tokens 8 \
    "${LOAD_IN_4BIT_ARGS[@]}" \
    --torch_dtype bfloat16 \
    --attn_implementation "${ATTN_IMPL}" \
    --log_every 100
else
  echo "[STEP] $(date -Is) reuse baseline ${BASELINE_METRICS}"
fi

for ffn_edit_count in ${COUNTS}; do
  ffn_edit_run="${RUN_NAME_PREFIX}_ffn_edit${ffn_edit_count}"
  ffn_edit_metrics="${RUN_ROOT}/${ffn_edit_run}/metrics/metrics.json"
  if [ -s "${ffn_edit_metrics}" ]; then
    echo "[STEP] $(date -Is) reuse FFN_Edit${ffn_edit_count} ${ffn_edit_metrics}"
    write_summary_csv
    continue
  fi
  echo "[STEP] $(date -Is) exact subset ${MODEL_LABEL} FFN_Edit${ffn_edit_count}"
  wait_gpu_idle
  "${PYTHON_BIN}" -m attention_heads_edit.eval.eval_api4_privacy_reverse_attention_heads_edit \
    --dataset "${EVAL_DATASET}" \
    --base_model "${BASE_MODEL}" \
    --adapter "${ADAPTER}" \
    --run_name "${ffn_edit_run}" \
    --output_dir "${RUN_ROOT}" \
    --disable_attention_heads_edit \
    --ffn_edit_kn_config "${KN_CONFIG}" \
    --ffn_edit_erase_num "${ffn_edit_count}" \
    --limit_per_type "${SWEEP_LIMIT_PER_TYPE}" \
    --batch_size "${EXACT_BATCH_SIZE}" \
    --max_context_tokens "${MAX_CONTEXT}" \
    --max_new_tokens 96 \
    --generation_extra_tokens 8 \
    "${LOAD_IN_4BIT_ARGS[@]}" \
    --torch_dtype bfloat16 \
    --attn_implementation "${ATTN_IMPL}" \
    --log_every 100
  write_summary_csv
done

write_summary_csv
echo "[DONE] $(date -Is) summary=${SUMMARY_CSV}"
cat "${SUMMARY_CSV}"
