"""
Llama3 CLM (Causal Language Modeling) runner for FFN Edit
Privacy-neuron attribution for Llama3-8B
"""

import logging
import argparse
import math
import os
import torch
import random
import numpy as np
import json, jsonlines
import pickle
import time

import transformers
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from custom_llama import LlamaForCausalLMWithEditing, patch_llama_model
from ffn_edit_pii_utils import find_token_subsequence
import torch.nn.functional as F

# set logger
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def example2feature(example, max_seq_length, tokenizer):
    """
    Convert an example into input features for Causal LM.
    example:
    - legacy: ['My phone number is 1 2 3 4 5 6 7 8 9 0']
    - PII span: ['full text with secret', 'secret span']
    """
    full_text = example[0]
    target_tokens = example[1:] if len(example) > 1 else []

    tokenized = tokenizer(full_text, add_special_tokens=False, padding=False)
    input_ids = tokenized["input_ids"]

    # CLM: predict the next token. Use the last usable position in the sequence.
    # Store original length before padding
    original_len = len(input_ids)

    target_token_id = None
    secret_located = False
    if target_tokens:
        target_token_str = target_tokens[0] if len(target_tokens) == 1 else " ".join(str(tok) for tok in target_tokens)
        encoded = tokenizer.encode(target_token_str, add_special_tokens=False)
        if encoded:
            target_token_id = encoded[0]
            secret_start = find_token_subsequence(input_ids, encoded)
            if secret_start is not None and secret_start > 0:
                prefix_ids = input_ids[:secret_start]
                if len(prefix_ids) > max_seq_length:
                    prefix_ids = prefix_ids[-max_seq_length:]
                input_ids = prefix_ids
                original_len = len(input_ids)
                secret_located = True

    if secret_located and original_len >= 1:
        tgt_pos = original_len - 1
    elif original_len >= 2:
        tgt_pos = original_len - 2
    elif original_len >= 1:
        tgt_pos = 0
    else:
        tgt_pos = 0

    # Determine target_token_id before padding (to avoid padding token issues)
    if target_token_id is None and original_len >= 2:
        if tgt_pos < original_len:
            target_token_id = tokenized["input_ids"][tgt_pos + 1] if tgt_pos + 1 < len(tokenized["input_ids"]) else tokenized["input_ids"][-1]
        else:
            target_token_id = tokenized["input_ids"][-1]
    elif target_token_id is None and original_len == 1:
        target_token_id = input_ids[0]

    if len(input_ids) > max_seq_length:
        input_ids = input_ids[-max_seq_length:]
        original_len = len(input_ids)
        tgt_pos = original_len - 1

    real_len = len(input_ids)
    padding_length = max_seq_length - real_len
    if padding_length > 0:
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length

    # Must match the truncated sequence; do not use the full pre-truncation tokenized length.
    attention_mask = [1] * real_len + [0] * padding_length

    assert len(input_ids) == max_seq_length
    assert len(attention_mask) == max_seq_length

    features = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    tokens_info = {
        "tokens": tokenizer.convert_ids_to_tokens(input_ids),
        "gold_obj": target_token_id,
        "target_pos": tgt_pos,
        "pred_obj": None,
    }
    return features, tokens_info


def scaled_input(emb, batch_size, num_batch):
    """
    Create scaled inputs for integrated gradients
    emb: (1, intermediate_size) - FFN intermediate output
    """
    baseline = torch.zeros_like(emb)  # (1, intermediate_size)

    num_points = batch_size * num_batch
    step = (emb - baseline) / num_points  # (1, intermediate_size)

    res = torch.cat([torch.add(baseline, step * i) for i in range(num_points)], dim=0)  # (num_points, intermediate_size)
    return res, step[0]


def convert_to_triplet_ig(ig_list):
    """
    Convert integrated gradients to triplet format
    ig_list: list of lists, shape (num_layers, intermediate_size)
    """
    ig_triplet = []
    ig = np.array(ig_list)  # (num_layers, intermediate_size)
    max_ig = ig.max()
    for i in range(ig.shape[0]):
        for j in range(ig.shape[1]):
            if ig[i][j] >= max_ig * 0.1:
                ig_triplet.append([i, j, ig[i][j]])
    return ig_triplet


CKPT_PREFIX = "step2_ckpt"
MAX_CKPT_KEEP = 3


def _ckpt_path(output_dir, output_prefix, n):
    return os.path.join(output_dir, f"{output_prefix}.{CKPT_PREFIX}.{n}.json")


def load_step2_checkpoint(output_dir, output_prefix):
    """Load latest checkpoint; return completed_bags (0 if none). Keeps at most 3 checkpoints."""
    best_count = 0
    best_mtime = 0
    for n in range(1, MAX_CKPT_KEEP + 1):
        p = _ckpt_path(output_dir, output_prefix, n)
        if not os.path.isfile(p):
            continue
        try:
            with open(p, "r") as f:
                data = json.load(f)
            c = data.get("completed_bags", 0)
            mtime = os.path.getmtime(p)
            if c > best_count or mtime > best_mtime:
                best_count = c
                best_mtime = mtime
        except Exception:
            continue
    return best_count


def save_step2_checkpoint(output_dir, output_prefix, completed_bags):
    """Save checkpoint and rotate to keep only the latest MAX_CKPT_KEEP (3)."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    data = {"completed_bags": completed_bags, "timestamp": ts, "output_prefix": output_prefix}
    temp_path = os.path.join(output_dir, f"{output_prefix}.{CKPT_PREFIX}.tmp.json")
    with open(temp_path, "w") as f:
        json.dump(data, f, indent=2)
    # Rotate: .1 <- .2, .2 <- .3, .3 <- temp; then remove temp
    for n in range(1, MAX_CKPT_KEEP):
        cur = _ckpt_path(output_dir, output_prefix, n)
        nxt = _ckpt_path(output_dir, output_prefix, n + 1)
        if os.path.isfile(nxt):
            if os.path.isfile(cur):
                os.remove(cur)
            os.rename(nxt, cur)
    dest = _ckpt_path(output_dir, output_prefix, MAX_CKPT_KEEP)
    if os.path.isfile(dest):
        os.remove(dest)
    os.rename(temp_path, dest)


def main():
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument("--priv_data_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data path. Should be .json file for the CLM task. ")
    parser.add_argument("--model_name_or_path",
                        type=str,
                        default="meta-llama/Meta-Llama-3-8B",
                        help="Path to pretrained model or model identifier from huggingface.co/models.",
                        required=False,
    )
    parser.add_argument("--adapter_dir",
                        type=str,
                        default=None,
                        help="Optional path to PEFT/LoRA adapter. If set, load on top of base model.",
    )
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--output_prefix",
                        default=None,
                        type=str,
                        required=True,
                        help="The output prefix to indentify each running of experiment. ")

    # Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after tokenization. \n"
                            "Sequences longer than this will be truncated, and sequences shorter \n"
                            "than this will be padded.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--gpus",
                        type=str,
                        default='0',
                        help="available gpus id")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--use_slow_tokenizer",
                        action="store_true",
                        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    # parameters about integrated grad
    parser.add_argument("--batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for cut.")
    parser.add_argument("--num_batch",
                        default=10,
                        type=int,
                        help="Num batch of an example.")
    parser.add_argument("--layer_start",
                        default=None,
                        type=int,
                        help="Optional first transformer layer to scan, inclusive. Defaults to all layers.")
    parser.add_argument("--layer_end",
                        default=None,
                        type=int,
                        help="Optional last transformer layer to scan, inclusive. Defaults to all layers.")
    parser.add_argument("--layer_indices",
                        default=None,
                        type=str,
                        help="Optional comma-separated layer indices to scan. Overrides layer_start/layer_end.")

    # parse arguments
    args = parser.parse_args()

    # set device
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        n_gpu = 0
    elif len(args.gpus) == 1:
        device = torch.device("cuda:%s" % args.gpus)
        n_gpu = 1
    else:
        # !!! to implement multi-gpus
        pass
    print("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(n_gpu > 1)))

    # set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # save args
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(args.output_dir, args.output_prefix + '.args.json'), 'w'), sort_keys=True, indent=2)

    # init tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load pre-trained Llama3 model
    print("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()
    
    # Load model and patch it
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if n_gpu > 0 else None,
    )
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    
    # Optional: load PEFT adapter (e.g. LoRA); patch is in-place so no merge needed
    if getattr(args, 'adapter_dir', None):
        from peft import PeftModel
        print(f"Loading adapter from {args.adapter_dir}", flush=True)
        model = PeftModel.from_pretrained(model, args.adapter_dir)
        print("Adapter loaded.", flush=True)
    
    # Patch model for FFN Edit editing support (in-place, no state_dict copy)
    print("Patching model for FFN Edit...", flush=True)
    model = patch_llama_model(model)
    print("Patch done.", flush=True)
    if n_gpu > 0 and getattr(model, "device_map", None) is None:
        model.to(device)
    
    # Get intermediate size from config (Llama3 uses mlp intermediate size)
    intermediate_size = getattr(config, 'intermediate_size', config.hidden_size * 4)  # Default to 4x hidden_size for Llama
    num_hidden_layers = config.num_hidden_layers
    if args.layer_indices:
        layer_indices = sorted({int(x.strip()) for x in args.layer_indices.split(",") if x.strip()})
    else:
        layer_start = 0 if args.layer_start is None else args.layer_start
        layer_end = num_hidden_layers - 1 if args.layer_end is None else args.layer_end
        layer_indices = list(range(layer_start, layer_end + 1))
    layer_indices = [idx for idx in layer_indices if 0 <= idx < num_hidden_layers]
    if not layer_indices:
        raise ValueError("No valid layers selected for attribution.")
    print(f"Scanning layers: {layer_indices}", flush=True)

    # data parallel
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    # prepare eval set
    with open(args.priv_data_path, 'r') as f:
        eval_bag_list_all = json.load(f)
    eval_bag_list_perrel = []
    for bag_idx, eval_bag in enumerate(eval_bag_list_all):
        eval_bag_list_perrel.append(eval_bag)

    # evaluate each privacy text
    # record running time
    tic = time.perf_counter()
    out_path = os.path.join(args.output_dir, args.output_prefix + '.priv' + '.jsonl')
    resume_count = load_step2_checkpoint(args.output_dir, args.output_prefix)
    if resume_count > 0:
        print(f"Resuming from checkpoint: {resume_count} bags already done.", flush=True)
    print('start processing, while dataset is ', len(eval_bag_list_perrel), flush=True)
    count = resume_count
    file_mode = 'a' if resume_count > 0 else 'w'
    skipped_so_far = 0  # when resuming, skip first resume_count non-empty bags
    with jsonlines.open(out_path, file_mode) as fw:
        for bag_idx, priv_texts in enumerate(eval_bag_list_perrel):
            if not priv_texts:
                logger.warning(f"Empty bag at index {bag_idx}, skipping.")
                continue
            if resume_count > 0 and skipped_so_far < resume_count:
                skipped_so_far += 1
                continue
            res_dict_bag = []

            sum_list = [[0 for i in range(intermediate_size)] for j in range(num_hidden_layers)]
            _, tokens_info = example2feature(priv_texts[0], args.max_seq_length, tokenizer)

            for ex_idx, eval_example in enumerate(priv_texts):
                eval_features, tokens_info = example2feature(eval_example, args.max_seq_length, tokenizer)
                # convert features to long type tensors
                input_ids = torch.tensor(eval_features['input_ids'], dtype=torch.long).unsqueeze(0)
                attention_mask = torch.tensor(eval_features['attention_mask'], dtype=torch.long).unsqueeze(0)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                # record real input length
                input_len = int(attention_mask[0].sum())

                # record target position (last token position for next token prediction)
                tgt_pos = tokens_info['target_pos']

                # record various results
                res_dict = {
                    'ig_gold': [],
                }

                for tgt_layer in layer_indices:
                    if tgt_layer == layer_indices[0] or (tgt_layer + 1) % 8 == 0 or tgt_layer == layer_indices[-1]:
                        print(f"  bag {bag_idx} ex {ex_idx} layer {tgt_layer}/{num_hidden_layers-1}", flush=True)
                    # Forward pass to get FFN weights
                    ffn_weights, logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        tgt_pos=tgt_pos,
                        tgt_layer=tgt_layer
                    )  # (1, intermediate_size), (1, n_vocab)
                    
                    if ffn_weights is None:
                        logger.warning(f"FFN weights is None at layer {tgt_layer}, skipping...")
                        # Create dummy weights
                        ffn_weights = torch.zeros((1, intermediate_size), device=device)
                    
                    pred_label = int(torch.argmax(logits[0, :]))  # scalar
                    gold_label = tokens_info['gold_obj']
                    if gold_label is None:
                        next_pos = min(tgt_pos + 1, input_ids.shape[1] - 1)
                        gold_label = input_ids[0, next_pos].item()
                    
                    # Create scaled inputs for integrated gradients
                    scaled_weights, weights_step = scaled_input(ffn_weights, args.batch_size, args.num_batch)  # (num_points, intermediate_size), (intermediate_size)
                    scaled_weights.requires_grad_(True)

                    # integrated grad at the gold label for each layer
                    ig_gold = None
                    for batch_idx in range(args.num_batch):
                        batch_weights = scaled_weights[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
                        # Create labels tensor for gradient computation
                        batch_size = batch_weights.shape[0]
                        batch_input_ids = input_ids.repeat(batch_size, 1)
                        batch_attention_mask = attention_mask.repeat(batch_size, 1)
                        labels = batch_input_ids.clone()
                        labels[:, tgt_pos] = gold_label

                        tgt_prob, grad = model(
                            input_ids=batch_input_ids,
                            attention_mask=batch_attention_mask,
                            tgt_pos=tgt_pos,
                            tgt_layer=tgt_layer,
                            tmp_score=batch_weights,
                            labels=labels
                        )  # (batch, n_vocab), (batch, intermediate_size)
                        
                        grad = grad.sum(dim=0)  # (intermediate_size)
                        ig_gold = grad if ig_gold is None else torch.add(ig_gold, grad)  # (intermediate_size)
                    
                    ig_gold = ig_gold * weights_step  # (intermediate_size)
                    res_dict['ig_gold'].append(ig_gold.tolist())  # (layer_num, intermediate_size)

                print(f"  bag {bag_idx} ex {ex_idx} done.", flush=True)
                # sum integrated grad
                for i in range(len(res_dict['ig_gold'])):
                    for j in range(len(res_dict['ig_gold'][i])):
                        sum_list[i][j] += res_dict['ig_gold'][i][j]
   
            res_dict = convert_to_triplet_ig(sum_list)
            res_dict_bag.append([tokens_info, res_dict])

            fw.write(res_dict_bag)
            count += 1
            save_step2_checkpoint(args.output_dir, args.output_prefix, count)
            if count % 10 == 0 or count <= 2:
                print('Has processed ', count, ' bags', flush=True)
        # record running time
        toc = time.perf_counter()
        print(f"***** Private texts have been processed. Costing time: {toc - tic:0.4f} seconds *****")


if __name__ == "__main__":
    main()
