#!/usr/bin/env python
# coding=utf-8
"""
Causal Language Modeling (CLM) training script for Llama3-8B.
"""

import argparse
import glob
import json
import logging
import math
import os
import random
import re
import shutil
from pathlib import Path
from itertools import chain

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
try:
    from transformers.trainer_pt_utils import LengthGroupedSampler
except ImportError:
    LengthGroupedSampler = None
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
    default_data_collator,
    BitsAndBytesConfig,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

check_min_version("4.21.0")
require_version("datasets>=1.8.0", "To fix: pip install -r requirements.txt (repo root)")

logger = get_logger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
ALLOWED_LOCAL_DATASET_EXTENSIONS = {"csv", "json", "jsonl", "txt"}


def _expand_data_source(spec):
    """
    Expand a local dataset spec into one or more files.
    Supported forms:
    - single file path
    - directory path (loads all supported files directly under it)
    - glob pattern
    - comma-separated mix of the above
    """
    if spec is None:
        return None

    items = [item.strip() for item in spec.split(",") if item.strip()]
    resolved = []

    for item in items:
        if any(ch in item for ch in "*?[]"):
            matches = sorted(glob.glob(item))
            if not matches:
                raise ValueError(f"No files matched dataset pattern: {item}")
            resolved.extend(matches)
            continue

        path = Path(item)
        if path.is_dir():
            matches = sorted(
                str(child) for child in path.iterdir()
                if child.is_file() and child.suffix.lower().lstrip(".") in ALLOWED_LOCAL_DATASET_EXTENSIONS
            )
            if not matches:
                raise ValueError(
                    f"Dataset directory {item} does not contain any supported files "
                    f"({', '.join(sorted(ALLOWED_LOCAL_DATASET_EXTENSIONS))})."
                )
            resolved.extend(matches)
            continue

        if not path.exists():
            raise ValueError(f"Dataset path does not exist: {item}")
        resolved.append(str(path))

    deduped = []
    seen = set()
    for path in resolved:
        ext = Path(path).suffix.lower().lstrip(".")
        if ext not in ALLOWED_LOCAL_DATASET_EXTENSIONS:
            raise ValueError(
                f"Unsupported dataset file extension for {path}. "
                f"Expected one of: {', '.join(sorted(ALLOWED_LOCAL_DATASET_EXTENSIONS))}."
            )
        if path not in seen:
            seen.add(path)
            deduped.append(path)

    if not deduped:
        raise ValueError(f"No dataset files resolved from: {spec}")

    return deduped[0] if len(deduped) == 1 else deduped


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help=(
            "Training data source. Supports a single file, a directory of files, a glob pattern, "
            "or a comma-separated list."
        ),
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help=(
            "Validation data source. Supports a single file, a directory of files, a glob pattern, "
            "or a comma-separated list."
        ),
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help=(
            "Column to use as the CLM training text. If unset, the script prefers `text`, then `source_text`, "
            "then falls back to the first column. Ignored when `--sft_llama3_chat` is set."
        ),
    )
    parser.add_argument(
        "--sft_llama3_chat",
        action="store_true",
        help=(
            "JSON SFT: rows contain instruction/input/output. Build Llama3 wire-format user/assistant segments "
            "(BOS + <|redacted_*_header_id|> + <|eot_id|>) and mask loss to assistant completion only. "
            "Compatible with base tokenizer without chat_template."
        ),
    )
    parser.add_argument(
        "--sft_plain_completion",
        action="store_true",
        help=(
            "JSON SFT for plain continuation: rows contain input/output. Build input+output without chat headers "
            "and mask loss on input tokens, so only output tokens contribute to training. An EOS token is appended "
            "to the output and included in the loss when the tokenizer has one."
        ),
    )
    parser.add_argument(
        "--sft_instruction_field",
        type=str,
        default="instruction",
        help="Column name for instruction text when using `--sft_llama3_chat`.",
    )
    parser.add_argument(
        "--sft_input_field",
        type=str,
        default="input",
        help="Column name for prefix/input text when using `--sft_llama3_chat` or `--sft_plain_completion`.",
    )
    parser.add_argument(
        "--sft_output_field",
        type=str,
        default="output",
        help="Column name for supervised completion when using `--sft_llama3_chat` or `--sft_plain_completion`.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--adapter_name_or_path",
        type=str,
        default=None,
        help=(
            "Optional path to an existing PEFT/LoRA adapter to continue training. "
            "This adapter is loaded on top of `model_name_or_path` and kept trainable."
        ),
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping. Set <= 0 to disable gradient clipping.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps. If set, OVERRIDES num_train_epochs (epoch count becomes derived). Prefer setting only num_train_epochs to avoid overtraining.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="The number of worker processes to use for train/eval DataLoaders.",
    )
    parser.add_argument(
        "--group_by_length",
        action="store_true",
        help="Group training batches by similar sequence lengths to reduce padding waste.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training continues from a checkpoint.",
    )
    parser.add_argument(
        "--keep_last_checkpoints",
        type=int,
        default=2,
        help=(
            "Keep at most the latest N checkpoint directories under `output_dir` (step_* / epoch_*). "
            "After each successful save, older directories are removed (oldest deleted first). "
            "Before saving, one extra old checkpoint may be removed so peak disk use stays near N checkpoints. "
            "Set <=0 to disable pruning."
        ),
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"mlflow"`, `"clearml"` and `"all"`.'
            ' Use `"all"` (default) to report to all integrations.'
            " Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="Maximum sequence length. When None, block_size is used (recommended: set block_size only to avoid silent override).",
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            " If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default=None,
        help=(
            "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the dtype will be automatically derived from the model's weights."
        ),
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Whether or not to use 8-bit quantization for model loading.",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Whether or not to use 4-bit quantization (QLoRA-style) for model loading. Requires --use_lora.",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Enable LoRA/PEFT training (parameter-efficient fine-tuning). Required for k-bit quantized training.",
    )
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank (r).")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help=(
            "Comma-separated list of target module names for LoRA, or `regex:<pattern>` for full-name regex "
            "selection. Regex mode is useful for Mistral3/Ministral3 text-only adapters to avoid the vision tower."
        ),
    )
    parser.add_argument(
        "--bnb_4bit_quant_type",
        type=str,
        default="nf4",
        choices=["nf4", "fp4"],
        help="4-bit quantization type for bitsandbytes.",
    )
    parser.add_argument(
        "--bnb_4bit_use_double_quant",
        action="store_true",
        help="Enable nested/double quantization for 4-bit (saves memory).",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower training.",
    )
    parser.add_argument(
        "--skip_first_batches_on_resume",
        action="store_true",
        help=(
            "If set, when resuming from a `step_*` checkpoint, skip the first N batches in the dataloader to reach "
            "the exact in-epoch batch. This can be extremely slow for large N. By default this is disabled and we "
            "rely on Accelerate's restored dataloader/sampler and RNG states."
        ),
    )
    parser.add_argument(
        "--per_document_sequences",
        action="store_true",
        help=(
            "If set, each training example is one document (one line) truncated/padded to block_size, instead of "
            "concatenating all texts and splitting into fixed chunks. Use this for better memorization of "
            "privacy-related content (phone, name) so the model sees full context in one sequence."
        ),
    )
    parser.add_argument(
        "--per_doc_truncation_side",
        type=str,
        default="right",
        choices=["left", "right"],
        help="For per_document_sequences: 'right' = keep last block_size tokens (tail, default); 'left' = keep first block_size tokens (head). Use 'left' when supervision is at document start.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="Run evaluation every N optimization steps. If set, evaluation runs every eval_steps instead of only at end of each epoch (reduces eval frequency for large step counts).",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Optional cap on the number of training examples after preprocessing. Useful for faster stage-2 continuation runs.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Optional cap on the number of evaluation examples after preprocessing.",
    )
    parser.add_argument(
        "--log_train_samples",
        type=int,
        default=3,
        help="How many random training samples to log after preprocessing. Set to 0 to disable sample logging.",
    )
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            args.train_file = _expand_data_source(args.train_file)
        if args.validation_file is not None:
            args.validation_file = _expand_data_source(args.validation_file)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def _cleanup_old_checkpoints(output_dir: str | None, keep_last: int | None) -> None:
    """
    Delete older checkpoint directories under output_dir, keeping only the latest `keep_last`.
    Supports both `step_*` and `epoch_*` naming patterns.
    """
    if output_dir is None or keep_last is None or keep_last <= 0:
        return

    base = Path(output_dir)
    if not base.exists() or not base.is_dir():
        return

    ckpt_dirs: list[tuple[str, int, Path]] = []
    for p in base.iterdir():
        if not p.is_dir():
            continue
        m_step = re.fullmatch(r"step_(\d+)", p.name)
        if m_step:
            ckpt_dirs.append(("step", int(m_step.group(1)), p))
            continue
        m_epoch = re.fullmatch(r"epoch_(\d+)", p.name)
        if m_epoch:
            ckpt_dirs.append(("epoch", int(m_epoch.group(1)), p))

    if len(ckpt_dirs) <= keep_last:
        return

    # Sort by (kind, index) to keep latest checkpoints within the same naming scheme.
    # If you mix step_* and epoch_* in the same output_dir, this will keep the latest N across both
    # after sorting by kind then index; in typical runs you will only have one pattern.
    ckpt_dirs.sort(key=lambda x: (x[0], x[1]))
    to_delete = ckpt_dirs[:-keep_last]
    for kind, idx, p in to_delete:
        logger.info(f"Removing old checkpoint directory: {p} ({kind}={idx})")
        shutil.rmtree(p, ignore_errors=False)


def _is_no_space_left_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "no space left on device" in message
        or "os error 28" in message
        or "errno 28" in message
    )


def _is_lora_resume_load_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "unexpected key(s) in state_dict" in message
        or "error(s) in loading state_dict for peftmodel" in message
    )


def _normalize_local_dataset_extension(path: str) -> str:
    extension = Path(path).suffix.lower().lstrip(".")
    if extension == "txt":
        return "text"
    if extension == "jsonl":
        return "json"
    return extension


def _pick_sample_path(path_or_paths):
    if isinstance(path_or_paths, (list, tuple)):
        return path_or_paths[0]
    return path_or_paths


def _resolve_text_column(column_names, requested_column=None):
    if requested_column:
        if requested_column not in column_names:
            raise ValueError(
                f"`--text_column {requested_column}` was requested, but available columns are: {', '.join(column_names)}"
            )
        return requested_column

    for candidate in ("text", "source_text"):
        if candidate in column_names:
            return candidate
    return column_names[0]


def _resolve_block_size(args, tokenizer):
    if args.block_size is not None:
        block_size = args.block_size
    elif args.max_seq_length is not None:
        block_size = args.max_seq_length
    else:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({block_size}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 1024
    if block_size > tokenizer.model_max_length:
        logger.warning(
            f"The block_size passed ({block_size}) is larger than the maximum length for the model"
            f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
        )
        block_size = min(block_size, tokenizer.model_max_length)
    return block_size


def _llama3_sft_encode_with_masked_labels(tokenizer, instruction: str, input_text: str, assistant: str):
    """
    Llama3-style user/assistant transcript; labels are -100 on user/prompt tokens, else token id for assistant span.
    """
    user = f"{(instruction or '').strip()}\n{(input_text or '').strip()}".strip()
    assistant = (assistant or "").strip()
    bos = "<|begin_of_text|>"
    uhdr = "<|start_header_id|>user<|end_header_id|>\n\n"
    ahdr = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    eot = "<|eot_id|>"
    user_block = uhdr + user + eot
    asst_block = ahdr + assistant + eot
    full_text = bos + user_block + asst_block
    prompt_text = bos + user_block + ahdr
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    if len(full_ids) < len(prompt_ids) or full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "Llama3 SFT: prompt token prefix does not match full sequence start; "
            "check that the tokenizer matches Llama3 special tokens."
        )
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    return full_ids, labels


def _plain_completion_encode_with_masked_labels(tokenizer, input_text: str, output_text: str):
    """
    Plain continuation transcript; labels are -100 on prefix/input tokens and token ids on output tokens.
    EOS is supervised so generation can learn to stop after the secret instead of continuing arbitrary text.
    """
    input_text = input_text or ""
    output_text = output_text or ""
    prompt_ids = tokenizer.encode(input_text, add_special_tokens=False)
    output_ids = tokenizer.encode(output_text, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        output_ids = output_ids + [tokenizer.eos_token_id]
    full_ids = prompt_ids + output_ids
    labels = [-100] * len(prompt_ids) + output_ids
    return full_ids, labels


def _resize_token_embeddings_if_needed(model, tokenizer):
    """
    Align model embeddings to the tokenizer before PEFT wrapping.
    Resizing after LoRA/QLoRA wrapping can cause PEFT to save full embedding/lm_head
    tensors in the adapter and complicate reloads for vocab-mismatched models.
    """
    try:
        current_vocab = model.get_input_embeddings().num_embeddings
    except Exception:
        current_vocab = None
    tokenizer_vocab = len(tokenizer)
    if current_vocab is not None and current_vocab != tokenizer_vocab:
        model.resize_token_embeddings(tokenizer_vocab)
    else:
        logger.info(f"Skipping resize_token_embeddings: model_vocab={current_vocab}, tokenizer_vocab={tokenizer_vocab}")


def _resolve_resume_checkpoint_path(args):
    if args.resume_from_checkpoint is None:
        return None
    if args.resume_from_checkpoint != "":
        return args.resume_from_checkpoint
    dirs = [f.path for f in os.scandir(args.output_dir) if f.is_dir()]
    if not dirs:
        raise ValueError(f"No checkpoint directories found under output_dir={args.output_dir}")
    dirs.sort(key=os.path.getctime)
    return dirs[-1]


def main():
    args = parse_args()
    resume_checkpoint_path = _resolve_resume_checkpoint_path(args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will pick up all the trackers in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Load datasets from --dataset_name or local --train_file/--validation_file.
    if args.dataset_name is not None:
        # Download and load a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
            )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        sample_path = _pick_sample_path(args.train_file if args.train_file else args.validation_file)
        extension = _normalize_local_dataset_extension(sample_path)
        if accelerator.is_main_process:
            train_inputs = data_files.get("train")
            valid_inputs = data_files.get("validation")
            logger.info(
                "Resolved local dataset inputs: train=%s validation=%s",
                train_inputs if isinstance(train_inputs, str) else len(train_inputs or []),
                valid_inputs if isinstance(valid_inputs, str) else len(valid_inputs or []),
            )
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                **dataset_args
            )

    # Load pretrained model and tokenizer.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a model config from scratch. This is not recommended.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not possible with this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # Set pad_token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Safety guard:
    # Full-parameter fine-tuning with an 8-bit/4-bit quantized model is unstable and commonly produces NaNs.
    # We only allow k-bit loading when LoRA/PEFT is enabled (k-bit training / QLoRA).
    if args.adapter_name_or_path and not args.use_lora:
        raise ValueError("`--adapter_name_or_path` requires `--use_lora`.")
    if (args.load_in_8bit or args.load_in_4bit) and not args.use_lora:
        raise ValueError(
            "Invalid configuration: --load_in_8bit/--load_in_4bit is enabled, but --use_lora is not set.\n"
            "This script does full-parameter AdamW updates by default, and k-bit quantization + full finetuning is "
            "highly unstable (NaNs) and/or infeasible.\n"
            "Fix: add --use_lora (recommended), or remove --load_in_8bit/--load_in_4bit for full finetuning."
        )
    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("Invalid configuration: choose only one of --load_in_8bit or --load_in_4bit.")

    if args.model_name_or_path:
        model_kwargs = {}
        if args.low_cpu_mem_usage:
            model_kwargs["low_cpu_mem_usage"] = True
        if accelerator.device.type == "cuda":
            model_kwargs["attn_implementation"] = "sdpa"

        # Model loading: either standard (full precision) or k-bit (LoRA/PEFT only).
        if args.load_in_8bit or args.load_in_4bit:
            # k-bit loading for LoRA/PEFT training (QLoRA-style).
            compute_dtype = None
            if args.torch_dtype is None:
                compute_dtype = torch.bfloat16
            elif args.torch_dtype == "auto":
                # Prefer bf16 compute for stability.
                compute_dtype = torch.bfloat16
            elif args.torch_dtype == "float16":
                logger.warning(
                    "WARNING: Converting float16 to bfloat16 for better training stability (k-bit training compute dtype)."
                )
                compute_dtype = torch.bfloat16
            else:
                compute_dtype = getattr(torch, args.torch_dtype)

            quantization_config = BitsAndBytesConfig(
                load_in_8bit=bool(args.load_in_8bit),
                load_in_4bit=bool(args.load_in_4bit),
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=args.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=bool(args.bnb_4bit_use_double_quant),
            )
            model_kwargs["quantization_config"] = quantization_config
            # Put the whole model on this process's device (single-GPU in this repo's scripts).
            model_kwargs["device_map"] = {"": accelerator.process_index}

            logger.info(
                f"Loading model in {'8-bit' if args.load_in_8bit else '4-bit'} mode for LoRA/PEFT training "
                f"(compute dtype={compute_dtype})."
            )

            model_auto_cls = AutoModelForImageTextToText if getattr(config, "model_type", None) == "mistral3" else AutoModelForCausalLM
            model = model_auto_cls.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                **model_kwargs,
            )
        else:
            # Standard full-precision loading (full finetuning path).
            if args.torch_dtype is not None:
                if args.torch_dtype == "auto":
                    model_kwargs["torch_dtype"] = "auto"
                elif args.torch_dtype == "float16":
                    # Convert float16 to bfloat16 for better training stability
                    logger.warning(
                        "WARNING: Converting float16 to bfloat16 for better training stability. "
                        "bfloat16 is more stable than float16 for training large models."
                    )
                    model_kwargs["torch_dtype"] = torch.bfloat16
                else:
                    model_kwargs["torch_dtype"] = getattr(torch, args.torch_dtype)

            model_auto_cls = AutoModelForImageTextToText if getattr(config, "model_type", None) == "mistral3" else AutoModelForCausalLM
            model = model_auto_cls.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                **model_kwargs,
            )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    # Resize embeddings before PEFT/LoRA wrapping. This is especially important for
    # Qwen/Gemma checkpoints whose tokenizer length differs from config.vocab_size.
    _resize_token_embeddings_if_needed(model, tokenizer)

    # LoRA/PEFT preparation (optional)
    if args.use_lora:
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

        # Prepare model for k-bit training (casts norms to fp32, enables input grads, etc.)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

        checkpoint_adapter_path = None
        if resume_checkpoint_path is not None:
            adapter_config = os.path.join(resume_checkpoint_path, "adapter_config.json")
            adapter_model = os.path.join(resume_checkpoint_path, "adapter_model.safetensors")
            if os.path.isfile(adapter_config) and os.path.isfile(adapter_model):
                checkpoint_adapter_path = resume_checkpoint_path

        adapter_source = checkpoint_adapter_path or args.adapter_name_or_path
        if adapter_source:
            if checkpoint_adapter_path:
                logger.info("Loading LoRA adapter from resume checkpoint: %s", checkpoint_adapter_path)
            else:
                logger.info("Loading existing LoRA adapter for continued training: %s", args.adapter_name_or_path)
            model = PeftModel.from_pretrained(
                model,
                adapter_source,
                is_trainable=True,
            )
        else:
            if args.lora_target_modules.startswith("regex:"):
                target_modules = args.lora_target_modules[len("regex:") :]
                if "\\\\." in target_modules:
                    normalized_target_modules = target_modules.replace("\\\\.", "\\.")
                    logger.warning(
                        "Normalized double-escaped LoRA regex from %s to %s",
                        target_modules,
                        normalized_target_modules,
                    )
                    target_modules = normalized_target_modules
                target_matches = [name for name, _module in model.named_modules() if re.fullmatch(target_modules, name)]
                logger.info(
                    "LoRA regex target matched %d modules; examples=%s",
                    len(target_matches),
                    target_matches[:8],
                )
            else:
                target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
            lora_cfg = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_cfg)
        if accelerator.is_main_process:
            try:
                model.print_trainable_parameters()
            except Exception:
                pass

        # Gradient checkpointing requires disabling KV cache.
        if getattr(model, "config", None) is not None:
            model.config.use_cache = False
    else:
        # Enable gradient checkpointing to save memory (optional)
        if args.gradient_checkpointing:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled to save memory")
            else:
                logger.warning("Model does not support gradient checkpointing")
            if getattr(model, "config", None) is not None:
                model.config.use_cache = False
        else:
            logger.info("Gradient checkpointing disabled for faster training")

    # Preprocessing the datasets.
    column_names = raw_datasets["train"].column_names
    block_size = _resolve_block_size(args, tokenizer)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    if args.sft_llama3_chat and args.sft_plain_completion:
        raise ValueError("Use only one of `--sft_llama3_chat` and `--sft_plain_completion`.")

    if args.sft_llama3_chat:
        ins_f, inp_f, out_f = args.sft_instruction_field, args.sft_input_field, args.sft_output_field
        for label, field in (("instruction", ins_f), ("input", inp_f), ("output", out_f)):
            if field not in column_names:
                raise ValueError(
                    f"`--sft_llama3_chat` requires column `{field}` ({label}); available: {', '.join(column_names)}"
                )
        trunc_side = getattr(args, "per_doc_truncation_side", "right")

        def sft_llama3_map(examples):
            batch_ids, batch_labels, batch_masks = [], [], []
            batch_len = len(examples[ins_f])
            for i in range(batch_len):
                out = examples[out_f][i]
                if out is None or (isinstance(out, str) and not str(out).strip()):
                    raise ValueError(
                        f"Empty `{out_f}` in training row (batch index {i}). "
                        "SFT requires non-empty assistant completion."
                    )
                ids, labels = _llama3_sft_encode_with_masked_labels(
                    tokenizer,
                    str(examples[ins_f][i]),
                    str(examples[inp_f][i]),
                    str(out),
                )
                if len(ids) >= block_size:
                    if trunc_side == "right":
                        ids = ids[-block_size:]
                        labels = labels[-block_size:]
                    else:
                        ids = ids[:block_size]
                        labels = labels[:block_size]
                batch_ids.append(ids)
                batch_labels.append(labels)
                batch_masks.append([1] * len(ids))
            return {"input_ids": batch_ids, "labels": batch_labels, "attention_mask": batch_masks}

        with accelerator.main_process_first():
            lm_datasets = raw_datasets.map(
                sft_llama3_map,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc=f"Llama3 SFT (masked labels), truncate to {block_size} ({trunc_side})",
            )
        logger.info(
            "Using `--sft_llama3_chat`: instruction/input/output -> Llama3 wire-format; "
            "loss only on assistant completion; dynamic padding in the collator."
        )
    elif args.sft_plain_completion:
        inp_f, out_f = args.sft_input_field, args.sft_output_field
        for label, field in (("input", inp_f), ("output", out_f)):
            if field not in column_names:
                raise ValueError(
                    f"`--sft_plain_completion` requires column `{field}` ({label}); available: {', '.join(column_names)}"
                )
        trunc_side = getattr(args, "per_doc_truncation_side", "right")

        def sft_plain_completion_map(examples):
            batch_ids, batch_labels, batch_masks = [], [], []
            batch_len = len(examples[inp_f])
            for i in range(batch_len):
                out = examples[out_f][i]
                if out is None or (isinstance(out, str) and not str(out).strip()):
                    raise ValueError(
                        f"Empty `{out_f}` in training row (batch index {i}). "
                        "Plain completion SFT requires non-empty output."
                    )
                ids, labels = _plain_completion_encode_with_masked_labels(
                    tokenizer,
                    str(examples[inp_f][i]),
                    str(out),
                )
                if len(ids) >= block_size:
                    if trunc_side == "right":
                        ids = ids[-block_size:]
                        labels = labels[-block_size:]
                    else:
                        ids = ids[:block_size]
                        labels = labels[:block_size]
                if all(label == -100 for label in labels):
                    continue
                batch_ids.append(ids)
                batch_labels.append(labels)
                batch_masks.append([1] * len(ids))
            return {"input_ids": batch_ids, "labels": batch_labels, "attention_mask": batch_masks}

        with accelerator.main_process_first():
            lm_datasets = raw_datasets.map(
                sft_plain_completion_map,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc=f"Plain completion SFT (masked labels), truncate to {block_size} ({trunc_side})",
            )
        logger.info(
            "Using `--sft_plain_completion`: input/output -> plain input+output; "
            "loss only on output completion plus EOS; dynamic padding in the collator."
        )
    else:
        text_column_name = _resolve_text_column(column_names, args.text_column)
        logger.info(f"Using text column `{text_column_name}` from columns: {column_names}")

        def tokenize_function(examples):
            return tokenizer(examples[text_column_name])

        with accelerator.main_process_first():
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

    if not args.sft_llama3_chat and not args.sft_plain_completion and args.per_document_sequences:
        # Per-document mode: one training example per row; truncate to block_size but delay padding to the
        # collator so batches are padded only to their local max length instead of always to block_size.
        trunc_side = getattr(args, "per_doc_truncation_side", "right")
        def per_doc_truncate(examples):
            input_ids_list = examples["input_ids"]
            new_input_ids = []
            new_labels = []
            new_attention_mask = []
            for ids in input_ids_list:
                if len(ids) >= block_size:
                    if trunc_side == "right":
                        ids = ids[-block_size:]
                    else:
                        ids = ids[:block_size]
                mask = [1] * len(ids)
                labels = ids.copy()
                new_input_ids.append(ids)
                new_labels.append(labels)
                new_attention_mask.append(mask)
            return {"input_ids": new_input_ids, "labels": new_labels, "attention_mask": new_attention_mask}

        with accelerator.main_process_first():
            lm_datasets = tokenized_datasets.map(
                per_doc_truncate,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                desc=f"Per-document truncate to {block_size} ({trunc_side})",
                remove_columns=tokenized_datasets["train"].column_names,
            )
        logger.info(
            f"Using per-document sequences (one example per document, max_len={block_size}, truncation={trunc_side}). "
            "Document boundaries preserved and padding is deferred to batch collation for better throughput."
        )
    elif not args.sft_llama3_chat and not args.sft_plain_completion:
        # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
        # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
        # to preprocess.
        with accelerator.main_process_first():
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    if args.max_train_samples is not None and args.max_train_samples > 0 and len(train_dataset) > args.max_train_samples:
        train_dataset = train_dataset.select(range(args.max_train_samples))
        logger.info("Capped training dataset to %s examples via --max_train_samples", len(train_dataset))
    if args.max_eval_samples is not None and args.max_eval_samples > 0 and len(eval_dataset) > args.max_eval_samples:
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        logger.info("Capped evaluation dataset to %s examples via --max_eval_samples", len(eval_dataset))

    if args.log_train_samples > 0 and len(train_dataset) > 0:
        sample_count = min(args.log_train_samples, len(train_dataset))
        for index in random.sample(range(len(train_dataset)), sample_count):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    def causal_lm_dynamic_padding_collator(features):
        pad_to_multiple_of = 8 if accelerator.device.type == "cuda" else None
        max_len = max(len(feature["input_ids"]) for feature in features)
        if pad_to_multiple_of is not None and max_len % pad_to_multiple_of != 0:
            max_len = ((max_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

        input_ids = []
        attention_masks = []
        labels = []
        for feature in features:
            ids = feature["input_ids"]
            feature_labels = feature["labels"]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_token_id] * pad_len)
            attention_masks.append([1] * len(ids) + [0] * pad_len)
            labels.append(feature_labels + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    use_dynamic_padding_collator = args.per_document_sequences or args.sft_llama3_chat or args.sft_plain_completion
    data_collator = causal_lm_dynamic_padding_collator if use_dynamic_padding_collator else default_data_collator
    dataloader_kwargs = {
        "num_workers": max(0, args.dataloader_num_workers),
        "pin_memory": accelerator.device.type == "cuda",
    }
    if dataloader_kwargs["num_workers"] > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = 4

    train_dataloader_kwargs = dict(dataloader_kwargs)
    if args.group_by_length and use_dynamic_padding_collator and LengthGroupedSampler is not None:
        lengths = [len(example["input_ids"]) for example in train_dataset]
        train_sampler = LengthGroupedSampler(
            batch_size=args.per_device_train_batch_size,
            dataset=train_dataset,
            lengths=lengths,
            model_input_name="input_ids",
        )
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            collate_fn=data_collator,
            batch_size=args.per_device_train_batch_size,
            **train_dataloader_kwargs,
        )
        logger.info("Using LengthGroupedSampler to reduce dynamic padding overhead.")
    else:
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=data_collator,
            batch_size=args.per_device_train_batch_size,
            **train_dataloader_kwargs,
        )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
        **dataloader_kwargs,
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    # Include common LayerNorm/RMSNorm parameter name variants for Llama-like models.
    no_decay = ["bias", "layer_norm.weight", "layernorm.weight", "norm.weight", "ln_f.weight"]

    named_trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if accelerator.is_main_process:
        logger.info(f"Trainable parameters: {len(named_trainable_params)} tensors")

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in named_trainable_params if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in named_trainable_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    # For LoRA/k-bit training, use a memory-friendly optimizer from bitsandbytes.
    if args.use_lora and (args.load_in_8bit or args.load_in_4bit):
        import bitsandbytes as bnb

        optimizer = bnb.optim.PagedAdamW8bit(optimizer_grouped_parameters, lr=args.learning_rate)
        logger.info("Using bitsandbytes PagedAdamW8bit optimizer for LoRA/k-bit training")
    else:
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    else:
        logger.warning(
            "max_train_steps is set; num_train_epochs will be derived from it and may be large, "
            "which can cause overtraining and unstable LR decay. Prefer setting only num_train_epochs."
        )
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    def run_evaluation():
        """Run evaluation over eval_dataloader; use actual batch sizes for correct loss aggregation (last batch may be smaller)."""
        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            if not torch.isfinite(loss):
                continue
            batch_size = batch["input_ids"].shape[0]
            gathered = accelerator.gather_for_metrics((loss.repeat(batch_size),))
            # gather_for_metrics may return a tuple when given a tuple; ensure we have a tensor for torch.cat
            if isinstance(gathered, tuple):
                gathered = gathered[0]
            losses.append(gathered)
        model.train()
        if len(losses) == 0:
            return None, None
        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses).item()
            perplexity = math.exp(eval_loss) if math.isfinite(eval_loss) else float("inf")
        except OverflowError:
            perplexity = float("inf")
            eval_loss = None
        return eval_loss, perplexity

    def save_lora_checkpoint_artifacts(output_dir):
        if not args.use_lora:
            return
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)

    def save_checkpoint(output_dir):
        try:
            if accelerator.is_main_process:
                # Pre-prune before writing the next checkpoint so we don't fail on disk-full
                # while temporarily holding N old checkpoints plus the new one.
                _cleanup_old_checkpoints(
                    args.output_dir,
                    max((args.keep_last_checkpoints or 0) - 1, 0) if args.keep_last_checkpoints is not None else None,
                )
            accelerator.wait_for_everyone()
            accelerator.save_state(output_dir)
            accelerator.wait_for_everyone()
            save_lora_checkpoint_artifacts(output_dir)
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                _cleanup_old_checkpoints(args.output_dir, args.keep_last_checkpoints)
            return True
        except Exception as exc:
            if _is_no_space_left_error(exc):
                logger.error(
                    "Checkpoint save failed because the target disk is full: %s. "
                    "Training will continue without stopping, but this checkpoint is unavailable.",
                    exc,
                )
                if accelerator.is_main_process and output_dir and os.path.isdir(output_dir):
                    shutil.rmtree(output_dir, ignore_errors=True)
                return False
            raise

    # We need to initialize the trackers we use, and also store our configuration.
    # We initialize the trackers only on the main process because `accelerator.log`
    # only logs on the main process anyway.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("clm_no_trainer", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Max sequence length = {block_size}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous checkpoint
    if args.resume_from_checkpoint is not None:
        resume_path = resume_checkpoint_path
        accelerator.print(f"Resumed from checkpoint: {resume_path}")
        path = os.path.basename(resume_path)
        try:
            if args.use_lora and (args.load_in_8bit or args.load_in_4bit):
                accelerator.load_state(resume_path, strict=False)
            else:
                accelerator.load_state(resume_path)
        except RuntimeError as exc:
            if args.use_lora and (args.load_in_8bit or args.load_in_4bit) and _is_lora_resume_load_error(exc):
                logger.warning(
                    "QLoRA resume encountered incompatible full-model keys in %s. "
                    "Falling back to adapter-only resume from the checkpoint directory.",
                    resume_path,
                )
                adapter_config = os.path.join(resume_path, "adapter_config.json")
                adapter_model = os.path.join(resume_path, "adapter_model.safetensors")
                if not (os.path.isfile(adapter_config) and os.path.isfile(adapter_model)):
                    raise RuntimeError(
                        f"Checkpoint {resume_path} does not contain adapter files required for LoRA resume."
                    ) from exc
            else:
                raise
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to get `steps`
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step = resume_step % len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        if args.resume_from_checkpoint and args.skip_first_batches_on_resume and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            # NOTE: When resuming with `--skip_first_batches_on_resume`, we already created `active_dataloader`
            # via `accelerator.skip_first_batches(train_dataloader, resume_step)`, so we must NOT skip again here.
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                
                # Check for NaN/Inf in loss before backward pass
                if not torch.isfinite(loss):
                    logger.error(
                        f"WARNING: Non-finite loss detected at step {completed_steps}, epoch {epoch}, step {step}! "
                        f"Loss value: {loss.item()}. Skipping this batch."
                    )
                    # Skip this batch and continue
                    optimizer.zero_grad()
                    continue
                
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                
                accelerator.backward(loss)
                
                # Gradient clipping for stability - only clip when gradients are synced
                # This prevents clipping on every micro-step during gradient accumulation
                if accelerator.sync_gradients:
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # Check for NaN/Inf gradients before clipping
                        grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        # `grad_norm` may be a Python float or a torch.Tensor (possibly on GPU).
                        if isinstance(grad_norm, torch.Tensor):
                            is_finite_grad_norm = torch.isfinite(grad_norm.detach()).all().item()
                        else:
                            is_finite_grad_norm = math.isfinite(float(grad_norm))

                        if not is_finite_grad_norm:
                            logger.error(
                                f"WARNING: Non-finite gradient norm detected at step {completed_steps}! "
                                f"Gradient norm: {grad_norm}. Skipping optimizer step."
                            )
                            optimizer.zero_grad()
                            continue
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        save_checkpoint(output_dir)

                if completed_steps >= args.max_train_steps:
                    break

                if getattr(args, "eval_steps", None) and completed_steps % args.eval_steps == 0:
                    eval_loss, perplexity = run_evaluation()
                    if perplexity is not None:
                        logger.info(f"step {completed_steps}: perplexity: {perplexity}")
                    if args.with_tracking and eval_loss is not None:
                        accelerator.log({"perplexity": perplexity, "eval_loss": eval_loss}, step=completed_steps)

        if getattr(args, "eval_steps", None) is None:
            eval_loss, perplexity = run_evaluation()
            if perplexity is not None:
                logger.info(f"epoch {epoch}: perplexity: {perplexity}")
                if args.with_tracking:
                    accelerator.log(
                        {
                            "perplexity": perplexity,
                            "eval_loss": eval_loss,
                            "train_loss": total_loss.item() / len(train_dataloader),
                            "epoch": epoch,
                            "step": completed_steps,
                        },
                        step=completed_steps,
                    )
            else:
                logger.warning(f"WARNING: No valid evaluation losses at epoch {epoch}. Skipping eval log.")

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            save_checkpoint(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
