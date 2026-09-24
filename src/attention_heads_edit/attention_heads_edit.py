"""Attention Heads Edit Implementation"""
import torch
import abc, inspect, json
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, cast, overload, Tuple

import transformers
from attention_heads_edit.utils import tokenizer_utils
from attention_heads_edit.utils.typing import (
    Model,
    Dataset,
    Device,
    ModelInput,
    ModelOutput,
    StrSequence,
    Tokenizer,
    TokenizerOffsetMapping,
)


def _model_config_value(model: Model, key: str, default: Any = None) -> Any:
    config = getattr(model, "config", None)
    candidates = []
    if config is not None:
        candidates.append(config)
        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            candidates.append(text_config)
        if hasattr(config, "get_text_config"):
            try:
                candidates.append(config.get_text_config())
            except Exception:
                pass
    seen = set()
    for candidate in candidates:
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        value = getattr(candidate, key, None)
        if value is not None:
            return value
    return default


def get_decoder_layers(model: Model) -> torch.nn.ModuleList:
    roots = [
        model,
        getattr(model, "base_model", None),
        getattr(getattr(model, "base_model", None), "model", None),
        getattr(model, "model", None),
        getattr(getattr(model, "model", None), "model", None),
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(getattr(getattr(model, "model", None), "language_model", None), "model", None),
        getattr(model, "language_model", None),
        getattr(getattr(model, "language_model", None), "model", None),
    ]
    seen = set()
    for root in roots:
        if root is None or id(root) in seen:
            continue
        seen.add(id(root))
        if isinstance(root, torch.nn.ModuleList):
            return root
        layers = getattr(root, "layers", None)
        if isinstance(layers, torch.nn.ModuleList):
            return layers
    raise RuntimeError("Cannot locate decoder layers for Attention Heads Edit attention steering.")


class AttentionHeadsEdit(abc.ABC):
    """
    Create Attention Heads Edit to steer attentions of transformer models.

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be steered.
        tokenizer ([`transformers.PreTrainedTokenizer`]): The model's tokenizer.
        head_config (`dict`): The config to control which attention heads to be steered.
        alpha (`float`): The scaling coefficient of attention steering.
        scale_position (`str`): To upweight the scores of highlighted tokens (`include`),
            or to downweight those of unselected tokens (`exclude`).

    Returns:
        AttentionHeadsEdit: AttentionHeadsEdit steerer that can register the pre-forward hooks on target models.

    """

    ATTN_MODULE_NAME = {
        "gptj": "transformer.h.{}.attn",
        "llama": "model.layers.{}.self_attn",
        "mistral": "model.layers.{}.self_attn",
        "gemma": "model.layers.{}.self_attn",
        "gemma3": "model.language_model.layers.{}.self_attn",
        "qwen3": "model.layers.{}.self_attn",
        "ministral3": "model.language_model.layers.{}.self_attn",
        "phi3mini": "model.layers.{}.self_attn"
    }
    ATTENTION_MASK_ARGIDX = {
        "gptj": 2,
        "llama": 1,
        "mistral": 1,
        "gemma": 1,
        "gemma3": 2,
        "qwen3": 2,
        "ministral3": 2,
    }
    def __init__(
        self,
        model: Model,
        tokenizer: Tokenizer,
        head_config: dict|list|None = None,
        alpha: float = 0.01,
        scale_position: str = "exclude",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.setup_model(model)
        attn_impl = _model_config_value(model, "_attn_implementation")
        if self.model_name in {"gemma3", "qwen3", "ministral3"} and attn_impl not in {None, "eager"}:
            raise ValueError(
                f"{self.model_name} Attention Heads Edit requires attn_implementation='eager' so per-query-head "
                "attention-mask steering is applied reliably."
            )

        self.alpha = alpha
        self.scale_position = scale_position
        self.setup_head_config(head_config)

        if self.scale_position not in {"include", "exclude", "generation", "include_down"}:
            raise ValueError(f"Unsupported Attention Heads Edit scale_position: {self.scale_position}")
        if self.alpha <= 0:
            raise ValueError("Attention Heads Edit alpha must be positive")
        self.scale_constant = None
        self.hook_call_count = 0
        self.edit_call_count = 0
        self.edited_mask_element_count = 0
        self.observed_attention_head_dims: set[int] = set()
        self.observed_attention_modules: set[str] = set()

    def setup_model(self, model):
        """Obtain the model type and complete the configuration."""
        self.num_hidden_layer = None
        if isinstance(model, transformers.LlamaForCausalLM):
            self.model_name = "llama"
            self.num_attn_head = model.config.num_attention_heads
            self.num_hidden_layer = _model_config_value(model, "num_hidden_layers")
        elif isinstance(model, transformers.GPTJForCausalLM):
            self.model_name = "gptj"
            self.num_attn_head = model.config.n_head
            self.num_hidden_layer = _model_config_value(model, "n_layer")
        elif isinstance(model, transformers.MistralForCausalLM):
            self.model_name = "mistral"
            self.num_attn_head = model.config.num_attention_heads
            self.num_hidden_layer = _model_config_value(model, "num_hidden_layers")
        elif isinstance(model, transformers.GemmaForCausalLM):
            self.model_name = "gemma"
            self.num_attn_head = model.config.num_attention_heads
            self.num_hidden_layer = _model_config_value(model, "num_hidden_layers")
        elif (
            model.__class__.__name__ in {"Gemma2ForCausalLM", "Gemma3ForCausalLM", "Gemma3ForConditionalGeneration"}
            or str(_model_config_value(model, "model_type", "")).startswith("gemma3")
        ):
            self.model_name = "gemma3"
            num_attn_head = _model_config_value(model, "num_attention_heads")
            num_hidden_layer = _model_config_value(model, "num_hidden_layers")
            if num_attn_head is None or num_hidden_layer is None:
                raise ValueError("Gemma3 Attention Heads Edit requires num_attention_heads and num_hidden_layers in text_config.")
            self.num_attn_head = int(num_attn_head)
            self.num_hidden_layer = int(num_hidden_layer)
        elif (
            model.__class__.__name__ == "Qwen3ForCausalLM"
            or getattr(getattr(model, "config", None), "model_type", None) == "qwen3"
        ):
            self.model_name = "qwen3"
            self.num_attn_head = model.config.num_attention_heads
            self.num_hidden_layer = _model_config_value(model, "num_hidden_layers")
        elif (
            model.__class__.__name__ == "Mistral3ForConditionalGeneration"
            or str(_model_config_value(model, "model_type", "")).startswith("mistral3")
            or str(_model_config_value(model, "model_type", "")).startswith("ministral3")
        ):
            self.model_name = "ministral3"
            num_attn_head = _model_config_value(model, "num_attention_heads")
            num_hidden_layer = _model_config_value(model, "num_hidden_layers")
            if num_attn_head is None or num_hidden_layer is None:
                raise ValueError("Ministral3 Attention Heads Edit requires num_attention_heads and num_hidden_layers in text_config.")
            self.num_attn_head = int(num_attn_head)
            self.num_hidden_layer = int(num_hidden_layer)
        elif model.__class__.__name__ == "Phi3ForCausalLM":
            self.model_name = "phi3mini"
            self.num_attn_head = model.config.num_attention_heads
            self.num_hidden_layer = _model_config_value(model, "num_hidden_layers")
        else:
            raise ValueError("Unimplemented Model Type.")

    def setup_head_config(self, head_config):
        """
        Config the attention heads to be steered.

        If `head_config` is `list` of layer index, Attention Heads Edit will steer the entire layers.
        """
        if isinstance(head_config, dict):
            self.head_config = {
                int(layer_idx): sorted({int(head_idx) for head_idx in head_ids})
                for layer_idx, head_ids in head_config.items()
            }
            self.all_layers_idx = [int(key) for key in head_config]
        elif isinstance(head_config, list):
            self.all_layers_idx = [int(v) for v in head_config]
            self.head_config = {
                idx:list(range(self.num_attn_head)) for idx in self.all_layers_idx
            }
        else:
            raise ValueError(f"Incorrect head config: {head_config}")
        if not self.head_config or not any(self.head_config.values()):
            raise ValueError("Attention Heads Edit head_config must contain at least one attention head")
        invalid_layers = [
            layer_idx
            for layer_idx in self.all_layers_idx
            if self.num_hidden_layer is not None and not (0 <= int(layer_idx) < int(self.num_hidden_layer))
        ]
        if invalid_layers:
            raise ValueError(
                f"Attention Heads Edit head_config contains invalid layer ids {invalid_layers}; "
                f"model has {self.num_hidden_layer} decoder layers."
            )
        invalid_heads = [
            (layer_idx, head_idx)
            for layer_idx, head_ids in self.head_config.items()
            for head_idx in head_ids
            if not (0 <= int(head_idx) < int(self.num_attn_head))
        ]
        if invalid_heads:
            raise ValueError(
                f"Attention Heads Edit head_config contains invalid head ids {invalid_heads}; "
                f"model has {self.num_attn_head} query attention heads."
            )

    def _maybe_batch(self, text: str | StrSequence) -> StrSequence:
        """Batch the text if it is not already batched."""
        if isinstance(text, str):
            return [text]
        return text

    def token_ranges_from_batch(
        self,
        strings: str | StrSequence,
        substrings: str | StrSequence,
        offsets_mapping: Sequence[TokenizerOffsetMapping],
        occurrence: int = 0,
    ) -> torch.Tensor:
        """Return shape (batch_size, 2) tensor of token ranges for (str, substr) pairs."""
        strings = self._maybe_batch(strings)
        substrings = self._maybe_batch(substrings)
        if len(strings) != len(substrings):
            raise ValueError(
                f"got {len(strings)} strings but only {len(substrings)} substrings"
            )
        return torch.tensor(
            [
                tokenizer_utils.find_token_range(
                    string, substring, offset_mapping=offset_mapping, occurrence=occurrence
                )
                for string, substring, offset_mapping in zip(
                    strings, substrings, offsets_mapping
                )
            ]
        )

    def edit_attention_mask(
        self,
        module: torch.nn.Module,
        input_args: tuple,
        input_kwargs: dict,
        head_idx: list[int],
        token_range: torch.Tensor,
        input_len: int,
    ):
        """
        The hook function registerred pre-forward for attention models.

        Args:
            module ([`torch.nn.Module`]): The registerred attention modules.
            input_args (`tuple`): The positional arguments of forward function.
            input_kwargs (`dict`): The keyword arguments of forward function.
            head_idx (`list[int]`): The index of heads to be steered.
            token_range (`torch.Tensor`): A B*2 tensor,
                suggesting the index range of hightlight tokens.
            input_len (`int`): The length L of inputs.

        Returns:
            tuple, dict: return the modified `attention_mask`,
                while not changing other input arguments.
        """
        if "attention_mask" in input_kwargs:
            attention_mask = input_kwargs['attention_mask'].clone()
        elif input_args is not None:
            arg_idx = self.ATTENTION_MASK_ARGIDX[self.model_name]
            attention_mask = input_args[arg_idx].clone()
        else:
            raise ValueError(f"Not found attention masks in {str(module)}")

        bsz, head_dim, tgt_len, src_len = attention_mask.size()
        dtype, device = attention_mask.dtype, attention_mask.device
        if head_dim != self.num_attn_head:
            attention_mask = attention_mask.expand(
                bsz, self.num_attn_head, tgt_len, src_len
            ).clone()
        if not self.scale_constant:
            self.scale_constant = torch.Tensor([self.alpha]).to(dtype).to(device).log()

        for bi, (ti,tj) in enumerate(token_range.tolist()):
            if self.scale_position == "include":
                attention_mask[bi, head_idx, :, ti:tj] += self.scale_constant
            else:
                attention_mask[bi, head_idx, :, :ti] += self.scale_constant
                attention_mask[bi, head_idx, :, tj:input_len] += self.scale_constant

        if self.model_name in ["llama", "mistral", "gemma", "phi3mini"]:
            attention_mask.old_size = attention_mask.size
            attention_mask.size = lambda:(bsz, 1, tgt_len, src_len)

        if "attention_mask" in input_kwargs:
            input_kwargs['attention_mask'] = attention_mask
            return input_args, input_kwargs
        else:
            return (*input_args[:arg_idx], attention_mask, *input_args[arg_idx+1:]), input_kwargs

    def edit_multisection_attention(
        self,
        module: torch.nn.Module,
        input_args: tuple,
        input_kwargs: dict,
        head_idx: list[int],
        token_ranges: list[torch.Tensor],
        input_len: int,
    ):
        """
        The hook function registerred pre-forward for attention models.

        Args:
            module ([`torch.nn.Module`]): The registerred attention modules.
            input_args (`tuple`): The positional arguments of forward function.
            input_kwargs (`dict`): The keyword arguments of forward function.
            head_idx (`list[int]`): The index of heads to be steered.
            token_ranges (`torch.Tensor`): A list of B*2 tensors,
                suggesting the index range of hightlight tokens of multiple sections.
            input_len (`int`): The length L of inputs.

        Returns:
            tuple, dict: return the modified `attention_mask`,
                while not changing other input arguments.
        """
        if "attention_mask" in input_kwargs:
            attention_mask = input_kwargs['attention_mask']
        elif input_args is not None:
            parameter_names = list(inspect.signature(module.forward).parameters)
            arg_idx = (
                parameter_names.index("attention_mask")
                if "attention_mask" in parameter_names
                else self.ATTENTION_MASK_ARGIDX[self.model_name]
            )
            if arg_idx >= len(input_args):
                raise ValueError(
                    f"Cannot read positional attention_mask at index {arg_idx} from "
                    f"{module.__class__.__name__} with {len(input_args)} arguments"
                )
            attention_mask = input_args[arg_idx]
        else:
            raise ValueError(f"Not found attention masks in {str(module)}")

        self.hook_call_count += 1
        self.observed_attention_modules.add(module.__class__.__name__)
        if not isinstance(attention_mask, torch.Tensor):
            raise TypeError(
                f"Attention Heads Edit requires a tensor attention mask, got {type(attention_mask).__name__} "
                f"from {module.__class__.__name__}. Use eager attention."
            )
        if attention_mask.ndim != 4:
            raise ValueError(
                f"Attention Heads Edit requires a 4D additive attention mask, got shape {tuple(attention_mask.shape)}"
            )
        module_config = getattr(module, "config", None)
        module_attn_impl = getattr(module_config, "_attn_implementation", None)
        if module_attn_impl not in {None, "eager"}:
            raise ValueError(
                f"Attention Heads Edit hooked {module.__class__.__name__} with attention implementation "
                f"{module_attn_impl!r}; eager is required."
            )
        module_query_heads = int(
            getattr(module, "num_heads", None)
            or getattr(module_config, "num_attention_heads", None)
            or self.num_attn_head
        )
        if module_query_heads != self.num_attn_head:
            raise ValueError(
                f"Attention Heads Edit query-head mismatch: model={self.num_attn_head}, "
                f"module={module_query_heads} on {module.__class__.__name__}"
            )

        attention_mask = attention_mask.clone().detach()
        bsz, head_dim, tgt_len, src_len = attention_mask.size()
        self.observed_attention_head_dims.add(int(head_dim))
        dtype, device = attention_mask.dtype, attention_mask.device
        if head_dim == 1:
            attention_mask = attention_mask.expand(
                bsz, self.num_attn_head, tgt_len, src_len
            ).clone()
        elif head_dim != self.num_attn_head:
            raise ValueError(
                f"Attention Heads Edit attention-mask head dimension must be 1 or {self.num_attn_head}, "
                f"got {head_dim}."
            )
        if self.scale_constant is None or self.scale_constant.device != device or self.scale_constant.dtype != dtype:
            self.scale_constant = torch.Tensor([self.alpha]).to(dtype).to(device).log()

        edited_elements = 0
        for token_range in token_ranges:
            for bi, (ti,tj) in enumerate(token_range.tolist()):
                if not (0 <= ti < tj <= src_len):
                    raise ValueError(
                        f"Attention Heads Edit token range [{ti}, {tj}) is invalid for source length {src_len}"
                    )
                if self.scale_position == "include":
                    attention_mask[bi, head_idx, :, ti:tj] += self.scale_constant
                    edited_elements += len(head_idx) * tgt_len * (tj - ti)
                elif self.scale_position == "exclude":
                    attention_mask[bi, head_idx, :, :ti] += self.scale_constant
                    attention_mask[bi, head_idx, :, tj:input_len] += self.scale_constant
                    edited_elements += len(head_idx) * tgt_len * (
                        ti + max(0, min(input_len, src_len) - tj)
                    )
                elif self.scale_position == "include_down":
                    attention_mask[bi, head_idx, :, ti:tj] += self.scale_constant
                    edited_elements += len(head_idx) * tgt_len * (tj - ti)
                elif self.scale_position == "generation":
                    attention_mask[bi, head_idx, :, :input_len] += self.scale_constant
                    edited_elements += len(head_idx) * tgt_len * min(input_len, src_len)
                else:
                    raise ValueError(f"Unexcepted {self.scale_position}.")
        if self.scale_position == "include":
            attention_mask[:, head_idx, :, :input_len] -= self.scale_constant

        if self.model_name in ["llama", "mistral", "phi3mini"]:
            attention_mask.old_size = attention_mask.size
            attention_mask.size = lambda:(bsz, 1, tgt_len, src_len)

        self.edit_call_count += 1
        self.edited_mask_element_count += edited_elements

        if "attention_mask" in input_kwargs:
            input_kwargs['attention_mask'] = attention_mask
            return input_args, input_kwargs
        else:
            return (*input_args[:arg_idx], attention_mask, *input_args[arg_idx+1:]), input_kwargs


    @contextmanager
    def apply_steering(
        self,
        model: Model,
        strings: list,
        substrings: list,
        model_input: ModelInput,
        offsets_mapping: Sequence[TokenizerOffsetMapping],
        occurrence: int = 0,
        token_ranges: Optional[Any] = None,
    ):
        """
        The function of context manager to register the pre-forward hook on `model`.

        Args:
            model ([`transformers.PreTrainedModel`]): The transformer model to be steered.
            strings (`list[str]`): The input strings.
            substrings (`list[list[str]]` or list[str]): The highlighted input spans for each string.
            model_input (`transformers.BatchEncoding`): The batched model inputs.
            offsets_mapping (`TokenizerOffsetMapping`): The offset mapping outputed by
                the tokenizer when encoding the `strings`.
            token_ranges: Optional explicit token ranges. If set, substring search is skipped.
                Each item is a (batch, 2) tensor of [start, end) token indices.
        """
        if token_ranges is None:
            if isinstance(substrings[0], str):
                substrings = [substrings]

            token_ranges = []
            for sections in substrings:
                token_range = self.token_ranges_from_batch(
                    strings, sections, offsets_mapping, occurrence=occurrence,
                )
                token_ranges.append(token_range)
        elif isinstance(token_ranges, torch.Tensor):
            token_ranges = [token_ranges]

        registered_hooks = []
        for layer_idx in self.all_layers_idx:
            module = self.get_attention_module(model, layer_idx)
            # Prepare the hook function with partial arguments being fixed.
            # Pass the head_idx, token_range, input_len for each attention module in advance.
            hook_func = partial(
                self.edit_multisection_attention,
                head_idx = self.head_config[layer_idx],
                token_ranges = token_ranges,
                input_len = model_input['input_ids'].size(-1)
            )
            registered_hook = module.register_forward_pre_hook(hook_func, with_kwargs=True)
            registered_hooks.append(registered_hook)
        try:
            yield model
        except Exception as error:
            raise error
        finally:
            for registered_hook in registered_hooks:
                registered_hook.remove()

    def get_attention_module(self, model: Model, layer_idx: int) -> torch.nn.Module:
        name_template = self.ATTN_MODULE_NAME.get(self.model_name)
        if name_template is not None:
            name = name_template.format(layer_idx)
            try:
                return model.get_submodule(name)
            except AttributeError:
                pass
        layers = get_decoder_layers(model)
        if not (0 <= int(layer_idx) < len(layers)):
            raise ValueError(f"Layer index {layer_idx} is out of range for {len(layers)} decoder layers.")
        layer = layers[int(layer_idx)]
        attention = getattr(layer, "self_attn", None)
        if attention is None:
            raise RuntimeError(f"Cannot locate self_attn on decoder layer {layer_idx}.")
        return attention

    def runtime_stats(self) -> dict[str, Any]:
        """Return architecture and hook evidence collected during steering."""
        return {
            "attention_heads_edit_model_name": self.model_name,
            "attention_heads_edit_num_query_heads": int(self.num_attn_head),
            "attention_heads_edit_hook_call_count": int(self.hook_call_count),
            "attention_heads_edit_edit_call_count": int(self.edit_call_count),
            "attention_heads_edit_edited_mask_element_count": int(self.edited_mask_element_count),
            "attention_heads_edit_observed_attention_head_dims": sorted(self.observed_attention_head_dims),
            "attention_heads_edit_observed_attention_modules": sorted(self.observed_attention_modules),
        }


    def inputs_from_batch(
        self,
        text: str | StrSequence,
        tokenizer: Tokenizer|None = None,
        device: Optional[Device] = None,
    ) -> tuple[ModelInput, Sequence[TokenizerOffsetMapping]]:
        """Precompute model inputs."""
        if tokenizer is None:
            tokenizer = self.tokenizer
        with tokenizer_utils.set_padding_side(tokenizer, padding_side="left"):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=False,
                padding="longest",
                return_offsets_mapping=True,
            )
            offset_mapping = inputs.pop("offset_mapping")
        if device is not None:
            inputs = inputs.to(device)
        return inputs, offset_mapping

    @classmethod
    def load_head_config(cls, file:str|Path):
        """Load the `head_config` from JSON file."""
        with open(file, "r") as f:
            head_config = json.load(f)
        return head_config
