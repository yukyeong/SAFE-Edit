#!/usr/bin/env bash
# Run one CRAPII SFT adapter through standalone FFN Edit only.
# Input: processed CRAPII prefix/eval JSON, model-specific CRAPII LoRA/QLoRA adapter,
#        and a model-local FFN Edit neuron ranking JSON.
# Output: exact/contains, MRR, ASR@32, PPL metrics plus a manifest under outputs/logs/configs.
# Impact: evaluation-time neuron erasure only; no model weights or datasets are modified.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${REPO_ROOT}"

MODEL_KIND="${CRAPII_MODEL_KIND:?Set CRAPII_MODEL_KIND to one of: llama, qwen, ministral}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${REPO_ROOT}/srcs/attention_heads_edit:${REPO_ROOT}/srcs/attention_heads_edit/scripts:${REPO_ROOT}/srcs/ffn_edit/utils:${REPO_ROOT}/srcs/ffn_edit/eval:${PYTHONPATH:-}"

RUN_STAMP="${RUN_STAMP:-crapii_adaptive_ffn_edit_only}"
EVAL_DATASET="${EVAL_DATASET:-data/crapii/api4_prefix/sft_true_prefix_no_instruction_all.json}"
PRIV_DATA="${PRIV_DATA:-data/crapii/api4_prefix/privacy_data_crapii_all.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/ffn_edit/crapii_ffn_edit_adaptive_only}"
MAX_CONTEXT="${MAX_CONTEXT:-512}"
ATTN_IMPL="${ATTN_IMPL:-eager}"
GPU_IDLE_MEM_MB="${GPU_IDLE_MEM_MB:-1000}"
PYTHON_BIN="${CRAPII_PYTHON:-python}"

LOAD_IN_4BIT_ARGS=(--load_in_4bit)
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
    DEFAULT_FFN_EDIT_ERASE_NUM=32
    KN_CONFIG="${KN_CONFIG:-outputs/ffn_edit/crapii_ffn_edit128_self_attention_heads_edit_top30/crapii_llama_ffn_edit128_self_attention_heads_edit_top30_a0p01/attribution/kn/kn_bag-crapii_llama_ffn_edit128_self_attention_heads_edit_top30_a0p01.json}"
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
    DEFAULT_FFN_EDIT_ERASE_NUM=96
    KN_CONFIG="${KN_CONFIG:-outputs/ffn_edit/crapii_ffn_edit128_self_attention_heads_edit_top30/crapii_qwen_ffn_edit128_self_attention_heads_edit_top30_a0p01/attribution/kn/kn_bag-crapii_qwen_ffn_edit128_self_attention_heads_edit_top30_a0p01.json}"
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
    DEFAULT_FFN_EDIT_ERASE_NUM=48
    KN_CONFIG="${KN_CONFIG:-outputs/ffn_edit/crapii_ffn_edit128_self_attention_heads_edit_top30/crapii_ministral_ffn_edit128_self_attention_heads_edit_top30_a0p01/attribution/kn/kn_bag-crapii_ministral_ffn_edit128_self_attention_heads_edit_top30_a0p01.json}"
    EXACT_BATCH_SIZE="${EXACT_BATCH_SIZE:-1}"
    MRR_BATCH_SIZE="${MRR_BATCH_SIZE:-1}"
    ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-1}"
    ;;
  *)
    echo "[crapii_ffn_edit] invalid CRAPII_MODEL_KIND=${MODEL_KIND}; expected llama, qwen, ministral" >&2
    exit 2
    ;;
esac

FFN_EDIT_ERASE_NUM="${FFN_EDIT_ERASE_NUM:-${DEFAULT_FFN_EDIT_ERASE_NUM}}"
RUN_NAME="${RUN_NAME:-crapii_${MODEL_TAG}_ffn_edit${FFN_EDIT_ERASE_NUM}_adaptive_only_${RUN_STAMP}}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_NAME}"
SUMMARY_DIR="${SUMMARY_DIR:-outputs/ffn_edit/metrics/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-logs/ffn_edit/${RUN_NAME}.log}"
MANIFEST="${MANIFEST:-configs/ffn_edit/runs/${RUN_NAME}_manifest.json}"
BASELINE_EXACT="${BASELINE_EXACT:-outputs/ffn_edit/crapii_prefix_sft_no_defense/${BASELINE_RUN_NAME}_exact_full/metrics/metrics.json}"
BASELINE_CORE="${BASELINE_CORE:-outputs/ffn_edit/crapii_prefix_sft_no_defense/${BASELINE_RUN_NAME}_core/metrics/core_metrics.json}"
FFN_EDIT_METHOD="CRAPII_${MODEL_LABEL}_Strong_SFT_FFN_Edit${FFN_EDIT_ERASE_NUM}_adaptive"
FFN_EDIT_EXACT_RUN="${RUN_NAME}_exact_full"
FFN_EDIT_CORE_RUN="${RUN_NAME}_core"

mkdir -p logs/ffn_edit configs/ffn_edit/runs "${RUN_ROOT}" "${SUMMARY_DIR}"
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
echo "[INFO] task=CRAPII standalone FFN Edit; no Attention Heads Edit; no weight writes"
echo "[INFO] ffn_edit_erase_num=${FFN_EDIT_ERASE_NUM} kn_config=${KN_CONFIG}"
echo "[INFO] eval_dataset=${EVAL_DATASET}"
echo "[INFO] privacy_data=${PRIV_DATA}"
echo "[INFO] base_model=${BASE_MODEL}"
echo "[INFO] adapter=${ADAPTER}"
echo "[INFO] python=${PYTHON_BIN}"

"${PYTHON_BIN}" - <<PY
import json
from collections import Counter
from pathlib import Path

paths = {
    "eval_dataset": Path("${EVAL_DATASET}"),
    "privacy_data": Path("${PRIV_DATA}"),
    "base_model_config": Path("${BASE_MODEL}") / "config.json",
    "adapter_config": Path("${ADAPTER}") / "adapter_config.json",
    "baseline_exact": Path("${BASELINE_EXACT}"),
    "baseline_core": Path("${BASELINE_CORE}"),
    "ffn_edit_kn_config": Path("${KN_CONFIG}"),
}
missing = [f"{name}={path}" for name, path in paths.items() if not path.exists() or path.stat().st_size == 0]
if missing:
    raise SystemExit("missing required files: " + "; ".join(missing))

rows = json.loads(paths["eval_dataset"].read_text(encoding="utf-8"))
if not isinstance(rows, list) or not rows:
    raise SystemExit("empty eval dataset")
bad_rows = [
    idx for idx, row in enumerate(rows[: min(200, len(rows))])
    if not isinstance(row, dict) or not row.get("input") or not row.get("output") or not row.get("pii_type")
]
if bad_rows:
    raise SystemExit(f"bad eval dataset rows: {bad_rows[:10]}")
pii_counts = Counter(str(row.get("pii_type")) for row in rows)

bags = json.loads(paths["privacy_data"].read_text(encoding="utf-8"))
bad_bags = []
for idx, bag in enumerate(bags[: min(500, len(bags))]):
    if not isinstance(bag, list) or not bag or not isinstance(bag[0], list) or len(bag[0]) < 2:
        bad_bags.append(idx)
        continue
    full, secret = str(bag[0][0]), str(bag[0][1])
    if not full or not secret or secret not in full:
        bad_bags.append(idx)
if bad_bags:
    raise SystemExit(f"bad privacy bags: {bad_bags[:10]}")

kn_raw = json.loads(paths["ffn_edit_kn_config"].read_text(encoding="utf-8"))
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
if len(positions) < int("${FFN_EDIT_ERASE_NUM}"):
    raise SystemExit(f"not enough FFN Edit positions: have={len(positions)} need=${FFN_EDIT_ERASE_NUM}")

raw_cfg = json.loads(paths["base_model_config"].read_text(encoding="utf-8"))
cfg = raw_cfg.get("text_config", raw_cfg)
arch = {
    "model_type": raw_cfg.get("model_type", cfg.get("model_type")),
    "text_model_type": cfg.get("model_type"),
    "num_hidden_layers": cfg.get("num_hidden_layers"),
    "num_attention_heads": cfg.get("num_attention_heads"),
    "num_key_value_heads": cfg.get("num_key_value_heads"),
    "hidden_size": cfg.get("hidden_size"),
    "intermediate_size": cfg.get("intermediate_size"),
}
print("[VALID] rows=", len(rows), "pii_counts=", dict(sorted(pii_counts.items())), flush=True)
print("[VALID] privacy_bags=", len(bags), "kn_positions=", len(positions), "using_top=", "${FFN_EDIT_ERASE_NUM}", flush=True)
print("[VALID] arch=", arch, flush=True)
PY

echo "[STEP 1/3] $(date -Is) ${MODEL_LABEL} FFN_Edit${FFN_EDIT_ERASE_NUM} exact/contains"
wait_gpu_idle
"${PYTHON_BIN}" srcs/attention_heads_edit/scripts/eval_api4_privacy_reverse_attention_heads_edit.py \
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

echo "[STEP 2/3] $(date -Is) ${MODEL_LABEL} FFN_Edit${FFN_EDIT_ERASE_NUM} MRR/ASR/PPL"
wait_gpu_idle
"${PYTHON_BIN}" srcs/attention_heads_edit/scripts/eval_api4_privacy_core_metrics.py \
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

echo "[STEP 3/3] $(date -Is) summarize ${MODEL_LABEL}"
"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

manifest = {
    "run_name": "${RUN_NAME}",
    "dataset": "CRAPII",
    "task": "standalone_ffn_edit_adaptive_only",
    "input": {
        "eval_dataset": "${EVAL_DATASET}",
        "privacy_data": "${PRIV_DATA}",
        "base_model": "${BASE_MODEL}",
        "adapter": "${ADAPTER}",
        "kn_config": "${KN_CONFIG}",
    },
    "impact": "evaluation-time neuron erasure only; no checkpoint or dataset mutation",
    "ffn_edit": {
        "erase_num": int("${FFN_EDIT_ERASE_NUM}"),
        "ranking_source": "${KN_CONFIG}",
    },
    "methods": [
        {
            "method": "${BASELINE_METHOD}",
            "exact_metrics": "${BASELINE_EXACT}",
            "core_metrics": "${BASELINE_CORE}",
        },
        {
            "method": "${FFN_EDIT_METHOD}",
            "exact_metrics": "${RUN_ROOT}/${FFN_EDIT_EXACT_RUN}/metrics/metrics.json",
            "core_metrics": "${RUN_ROOT}/${FFN_EDIT_CORE_RUN}/metrics/core_metrics.json",
        },
    ],
}
path = Path("${MANIFEST}")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[INFO] wrote manifest={path}", flush=True)
PY

"${PYTHON_BIN}" srcs/attention_heads_edit/scripts/summarize_api4_privacy_suite.py \
  --manifest "${MANIFEST}" \
  --output_dir "${SUMMARY_DIR}"

test -s "${SUMMARY_DIR}/paper_main_table.csv"
echo "[DONE] $(date -Is) summary=${SUMMARY_DIR}/paper_main_table.csv"
cat "${SUMMARY_DIR}/paper_main_table.csv"
