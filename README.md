# SAFE-Edit: LLM Privacy Memorization, FFN Edit, and Attention Heads Edit

This repository provides single-GPU (RTX 4090) reproducible code for privacy memorization experiments covering:

- Llama3-8B QLoRA fine-tuning on API4-200K, full privacy metrics, and FFN Edit / Attention Heads Edit ablations.
- Llama3-8B, Qwen3-8B-Base, and Ministral-3-8B-Base fine-tuning on LUME and CRAPII.
- Per-model FFN Edit neuron localization, independent Attention Heads Edit head localization, then `FFN Edit -> Attention Heads Edit` joint evaluation.
- Exact, Contains, Macro-MRR, ASR@32, and WikiText-103 PPL.

Only code, static configs, and reproduction notes are versioned. Data, base models, adapters, checkpoints, logs, predictions, and metrics are ignored by `.gitignore`.

## 1. Experiment Matrix and Pipeline

| Dataset | Model | Fine-tune | FFN Edit only | FFN Edit -> Attention Heads Edit | API4 sweep |
|---|---|---:|---:|---:|---:|
| API4-200K | Llama3-8B | yes | yes | yes | yes |
| LUME | Llama3-8B / Qwen3-8B-Base / Ministral-3-8B-Base | yes | yes | yes | no |
| CRAPII | Llama3-8B / Qwen3-8B-Base / Ministral-3-8B-Base | yes | yes | yes | no |

Joint pipeline order:

1. Relocate FFN Edit neurons on the current model and dataset.
2. Evaluate FFN Edit only.
3. With FFN Edit disabled, independently locate Attention Heads Edit candidates for the current model.
4. Screen candidate heads on the post-FFN-Edit validation set.
5. Evaluate `FFN Edit -> Attention Heads Edit` without cross-model neuron/head remapping.

## 2. Layout

```text
configs/heads/                    Static API4 Attention Heads Edit JSON configs
scripts/_common.sh                Shared REPO_ROOT + PYTHONPATH=src
scripts/prepare/                  Dataset preparation CLIs
scripts/train/                    QLoRA / SFT entrypoints
scripts/eval/                     Baseline, sweep, and joint evaluation entrypoints
scripts/pipelines/                Multi-model end-to-end pipelines
src/ffn_edit/                     FFN Edit hooks, locate, and training code
src/attention_heads_edit/         Attention Heads Edit core, locate, and eval
outputs/                          Runtime metrics, manifests, generated heads (not versioned)
logs/                             Runtime logs (not versioned)
```

## 3. Environment

Validated on Ubuntu, Python 3.12, CUDA 12.8, RTX 4090 24GB. Training/eval default to QLoRA/4-bit, BF16, and gradient checkpointing. Attention Heads Edit localization requires eager attention.

```bash
git clone https://github.com/yukyeong/SAFE-Edit.git
cd SAFE-Edit

python3.12 -m venv .venvs/repro
source .venvs/repro/bin/activate
pip install -U pip
pip install -r requirements.txt
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

python3.12 -m venv .venvs/ministral3
.venvs/ministral3/bin/pip install -U pip
.venvs/ministral3/bin/pip install -r requirements-ministral.txt
```

Bash entrypoints under `scripts/` source `scripts/_common.sh`, which sets `REPO_ROOT` and `PYTHONPATH=src`. For direct `python -m ...` calls, export `PYTHONPATH` as above.

Use `requirements.txt` for Llama3 and Qwen3. Ministral needs a separate venv (`requirements-ministral.txt`) because its `mistral3` wrapper requires Transformers 5.0.0rc0, which conflicts with the 4.57.3 pin above.

## 4. Download Models

Run `hf auth login` first. Llama3 also requires gated-model access.

```bash
hf download meta-llama/Meta-Llama-3-8B --local-dir models/llama3-8B/baseline
hf download Qwen/Qwen3-8B-Base --local-dir models/qwen3-8b-base
hf download mistralai/Ministral-3-8B-Base-2512 --local-dir models/ministral-3-8b-base
```

Scripts read the paths above by default and can be overridden with `BASE_MODEL` / `*_MODEL_PATH`.

## 5. Prepare Data

### 5.1 Unified Format

Training/eval JSON is a list of objects. Each row must at least contain:

```json
{
  "input": "prefix immediately before a private value",
  "output": "private completion",
  "pii_type": "EMAIL",
  "source_id": "document-level split id",
  "entity_index": 0
}
```

FFN Edit privacy bags are lists of bags, each shaped like `[[input + output, output]]`. Real PII should not be committed; default evaluation stores hashed predictions only.

### 5.2 API4-200K

API4-200K is not redistributed here due to licensing. Prepare:

```text
data/api4_200k/sft_true_prefix_no_instruction.json
data/api4_200k/sft_true_prefix_no_instruction_val.json
data/api4_200k/sft_true_prefix_no_instruction_all.json
```

```bash
python scripts/prepare/prepare_api4_privacy_bags.py \
  --dataset data/api4_200k/sft_true_prefix_no_instruction_all.json \
  --output data/api4_200k/privacy_data_api4_all.json
python -m attention_heads_edit.eval.validate_api4_edit_dataset \
  --dataset data/api4_200k/sft_true_prefix_no_instruction_all.json \
  --privacy_data data/api4_200k/privacy_data_api4_all.json \
  --require_privacy_alignment
```

### 5.3 LUME

```bash
git clone --depth 1 https://github.com/amazon-science/lume-llm-unlearning.git \
  data/lume/lume-llm-unlearning
python scripts/prepare/prepare_lume_api4_prefix.py
python scripts/prepare/prepare_lume_hard_balanced.py
python scripts/prepare/prepare_lume_edit_subsets.py
python -m attention_heads_edit.eval.validate_api4_edit_dataset \
  --dataset data/lume/api4_prefix_task2/sft_true_prefix_no_instruction_scenario_clean_all.json \
  --privacy_data data/lume/api4_prefix_task2/privacy_data_lume_task2_scenario_clean_all.json \
  --expected_pii_types ADDRESS,DOB,EMAIL,PHONENUMBER,SSN \
  --require_privacy_alignment
```

`hard_balanced` uses seed 42: ADDRESS/DOB/PHONENUMBER/SSN are repeated 3x and EMAIL stays 1x to avoid EMAIL-only memorization.

### 5.4 CRAPII

```bash
hf download metaboulie/Tidied-PII-Detection-Kaggle-7k \
  --repo-type dataset \
  --local-dir data/crapii/tidied-pii-detection-kaggle-7k
python scripts/prepare/prepare_crapii_api4_prefix.py
python -m attention_heads_edit.eval.validate_api4_edit_dataset \
  --dataset data/crapii/api4_prefix/sft_true_prefix_no_instruction_all.json \
  --privacy_data data/crapii/api4_prefix/privacy_data_crapii_all.json \
  --require_privacy_alignment
```

CRAPII splits by document into train/val/heldout. For no-copy diagnostics:

```bash
python scripts/prepare/prepare_crapii_api4_prefix.py \
  --output_dir data/crapii/api4_prefix_no_copy \
  --drop_target_in_prefix
```

### 5.5 PPL Data

```bash
python scripts/prepare/prepare_wikitext_utility.py
```

Writes `data/external/wikitext-103-raw-v1/validation.jsonl`. Missing PPL data makes the full PPL stage fail instead of silently emitting incomplete metrics.

## 6. Fine-tuning

### 6.1 Llama3-8B / API4-200K

```bash
bash scripts/train/api4_llama_qlora.sh
tail -f logs/ffn_edit/train/run_llama3_api4_prefix_qlora_full_*.log
```

This launches background training and syncs the final adapter to `models/llama3-8B/api4_prefix_plain_qlora/`. Base weights are unchanged.

### 6.2 Three Models / LUME

```bash
bash scripts/train/lume_llama_qlora.sh
bash scripts/train/lume_qwen_qlora.sh
MINISTRAL_PYTHON="$PWD/.venvs/ministral3/bin/python" \
  bash scripts/train/lume_ministral_qlora.sh
```

Final adapters:

```text
models/llama3-8B/lume_task2_prefix_plain_qlora
models/qwen3-8b-base/lume_task2_hard_balanced_qlora
models/ministral-3-8b-base/lume_task2_hard_balanced_qlora
```

### 6.3 Three Models / CRAPII

```bash
for model in llama qwen ministral; do
  CRAPII_MODEL_KIND="$model" \
  CRAPII_PYTHON="$([ "$model" = ministral ] && echo "$PWD/.venvs/ministral3/bin/python" || command -v python)" \
    bash scripts/train/crapii_sft_model.sh
done
```

These configs use 4-bit QLoRA for 24GB GPUs. Ministral LoRA targets use nested `language_model.layers.*` regexes.

## 7. API4 Full Ablation

Evaluate the fine-tuned baseline first:

```bash
bash scripts/eval/api4_baseline_metrics.sh
```

### 7.1 FFN Edit Count Sweep

```bash
RUN_NAME=api4_ffn_edit_repro \
ERASE_LIST="64 128 256 512 1024" \
  bash scripts/eval/api4_ffn_only_sweep.sh

export FFN_EDIT_KN_CONFIG="outputs/ffn_edit/ffn_edit_only_recheck_64_1024/api4_ffn_edit_repro/attribution/kn/kn_bag-api4_ffn_edit_repro.json"
```

Outputs include KN rankings, per-erase full metrics, and `outputs/ffn_edit/metrics/api4_ffn_edit_repro/paper_main_table.csv`.

### 7.2 Attention Heads Edit: Fixed Alpha, Vary Top-k

```bash
bash scripts/eval/api4_heads_topk_small.sh  # top 10/15/20
bash scripts/eval/api4_heads_topk_large.sh  # top 30/35/40/45
```

### 7.3 Attention Heads Edit: Fixed Top40, Vary Alpha

```bash
ALPHAS="0.01 0.05 0.1 0.15 0.2 0.3 0.5" \
  bash scripts/eval/api4_heads_alpha_sweep.sh
```

### 7.4 Joint: Fixed Attention Heads Edit, Vary FFN Edit Count

```bash
FFN_EDIT_KN_CONFIG="$FFN_EDIT_KN_CONFIG" \
FFN_EDIT_RECHECK_SUMMARY=outputs/ffn_edit/metrics/api4_ffn_edit_repro/paper_main_table.csv \
ERASE_LIST="64 128 256 512 1024" \
  bash scripts/eval/api4_joint_erase_sweep.sh
```

### 7.5 Joint: Fixed FFN Edit256/top30, Vary Attention Heads Edit Alpha

```bash
FFN_EDIT_KN_CONFIG="$FFN_EDIT_KN_CONFIG" \
ALPHAS="0.01 0.05 0.1 0.15 0.2 0.3 0.5" \
  bash scripts/eval/api4_joint_alpha_sweep.sh
```

Static top-k head JSON files under `configs/heads/api4_top100_a015_top*.json` are Llama3 API4 only. Qwen3 and Ministral3 must relocate heads locally.

## 8. LUME/CRAPII Three-Model Experiments

Unified entrypoint from scratch: FFN Edit localization -> FFN Edit-only full metrics -> independent Attention Heads Edit localization -> post-FFN-Edit validation screening -> joint full metrics.

```bash
for dataset in lume crapii; do
  for model in llama qwen ministral; do
    bash scripts/pipelines/three_model_privacy.sh "$dataset" "$model"
  done
done
```

Default settings:

| Dataset / Model | FFN Edit | Attention Heads Edit top-k | alpha | span |
|---|---:|---:|---:|---|
| LUME / Llama3 | 128 | 10 | 0.01 | tail 8 |
| LUME / Qwen3 | 32 | 5 | 0.01 | adaptive tail 8 |
| LUME / Ministral3 | 16 | 32 | 0.10 | tail 8 |
| CRAPII / Llama3 | 32 | 30 | 0.10 | tail 8 |
| CRAPII / Qwen3 | 32 | 34 | 0.10 | tail 8 |
| CRAPII / Ministral3 | 16 | 3 | 0.02 | adaptive tail 16 |

Override with environment variables, for example:

```bash
FFN_EDIT_ERASE_NUM=16 SELECT_TOP_K=10 ATTENTION_HEADS_EDIT_ALPHA=0.02 \
  bash scripts/pipelines/three_model_privacy.sh lume qwen
```

## 9. Metrics and Outputs

- `metrics.json`: Exact/Contains by PII type and macro.
- `core_metrics.json`: Macro-MRR, ASR@32, WikiText-103 PPL.
- `paper_main_table.csv`: baseline, FFN Edit-only, and joint summaries.
- `manifest.json`: model/adapter/data/neuron/head and hyperparameter snapshot.

Primary outputs:

```text
outputs/ffn_edit/metrics/<run>/paper_main_table.csv
outputs/attention_heads_edit/metrics/<run>/paper_main_table.csv
outputs/logs/manifests/<run>_manifest.json
outputs/attention_heads_edit/heads/<run>_*.json
logs/ffn_edit/<run>.log
```

Lower privacy metrics usually mean less leakage; lower PPL usually means better utility. WikiText PPL defaults to prefix-continuation scoring (`--ppl_score_region continuation`, `--ppl_protocol prefix_continuation_v1`): 384 prefix tokens plus 128 continuation tokens, with NLL only on the continuation. `--ppl_attention_heads_edit_mode none` leaves Attention Heads Edit off for that PPL. `aligned_span` applies the same prefix-tail span used for leakage. `full_block` scores the whole block and is rejected when the score region is continuation.

## 10. Architecture Notes

- **Llama3**: localize decoder MLP `down_proj` input channels; Attention Heads Edit uses 32x32 query heads.
- **Qwen3**: keep the official forward and erase gated-MLP channels via pre-hooks; Attention Heads Edit ranks local query heads under eager attention with QK-Norm/GQA.
- **Ministral3**: target nested `language_model.layers` inside the `mistral3` wrapper; use the Transformers 5 environment and constrain GQA candidate concentration.
- **next-token query**: score the next-token query over the prefix/trigger span for prefix-completion leakage; this does not change the FFN Edit-then-Attention-Heads-Edit order.

## 11. Smoke Checks

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m compileall -q src scripts/prepare
bash -n scripts/train/*.sh scripts/eval/*.sh scripts/pipelines/*.sh scripts/_common.sh
python scripts/prepare/prepare_api4_privacy_bags.py --help
python -m attention_heads_edit.eval.validate_api4_edit_dataset --help
```

If a process is waiting for GPU memory, logs periodically report usage. Missing data, adapters, WikiText, or KN/head configs cause hard failures instead of incomplete “success” metrics.
