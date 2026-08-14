"""Qwen3 FFN Edit hooks.

Qwen3 is a dense decoder-only Transformer with a gated MLP:

    down_proj(act_fn(gate_proj(x)) * up_proj(x))

For FFN Edit, the privacy neuron activation is the tensor entering
``mlp.down_proj``.  This module deliberately keeps the official
Qwen3ForCausalLM forward implementation intact and installs forward pre-hooks
on down_proj modules instead of subclassing or copying Qwen decoder layers.
"""

from __future__ import annotations

import re
from typing import Any

import torch

from custom_gemma import GemmaFfnEditController, _label_at_pos


_LAYER_PATTERNS = (
    re.compile(r"(?:^|\.)model\.layers\.(\d+)\.mlp\.down_proj$"),
    re.compile(r"(?:^|\.)base_model\.model\.model\.layers\.(\d+)\.mlp\.down_proj$"),
    re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.down_proj$"),
)


def _extract_layer_index(module_name: str) -> int | None:
    for pattern in _LAYER_PATTERNS:
        match = pattern.search(module_name)
        if match:
            return int(match.group(1))
    if not module_name.endswith("mlp.down_proj"):
        return None
    matches = re.findall(r"(?:^|\.)layers\.(\d+)\.", module_name)
    return int(matches[-1]) if matches else None


def find_qwen_down_proj_modules(model) -> list[tuple[int, str, torch.nn.Module]]:
    """Return ``(layer_idx, module_name, module)`` for every Qwen MLP down_proj."""

    modules: list[tuple[int, str, torch.nn.Module]] = []
    seen: set[int] = set()
    for name, module in model.named_modules():
        layer_idx = _extract_layer_index(name)
        if layer_idx is None:
            continue
        module_id = id(module)
        if module_id in seen:
            continue
        seen.add(module_id)
        modules.append((layer_idx, name, module))
    modules.sort(key=lambda item: item[0])
    return modules


class QwenFfnEditController(GemmaFfnEditController):
    """Stateful hook controller for Qwen FFN Edit runs."""

    def install(self) -> None:
        modules = find_qwen_down_proj_modules(self.model)
        if not modules:
            raise AttributeError("Could not find Qwen MLP down_proj modules; expected names ending with mlp.down_proj.")
        for layer_idx, name, module in modules:
            self.layer_names[layer_idx] = name
            self.handles.append(module.register_forward_pre_hook(self._make_hook(layer_idx)))


def patch_qwen_model_for_ffn_edit(model):
    """Patch a Qwen/Qwen3 model in-place to support FFN Edit forward kwargs."""

    if getattr(model, "_qwen_ffn_edit_patched", False):
        return model

    controller = QwenFfnEditController(model)
    controller.install()
    original_forward = model.forward

    def ffn_edit_forward(
        *args,
        tgt_pos=None,
        tgt_layer=None,
        tmp_score=None,
        imp_pos=None,
        imp_op=None,
        labels=None,
        **kwargs,
    ):
        ffn_edit_mode = (
            tgt_pos is not None
            or tgt_layer is not None
            or tmp_score is not None
            or imp_pos is not None
            or imp_op is not None
        )
        controller.configure(
            imp_pos=imp_pos,
            imp_op=imp_op,
            tgt_layer=tgt_layer,
            tgt_pos=tgt_pos,
            tmp_score=tmp_score,
        )
        if ffn_edit_mode:
            kwargs["use_cache"] = False
        try:
            forward_labels = None if (tgt_layer is not None and tgt_pos is not None) else labels
            outputs = original_forward(*args, labels=forward_labels, **kwargs)
            if tgt_layer is None or tgt_pos is None:
                return outputs

            logits = getattr(outputs, "logits", None)
            if logits is None and isinstance(outputs, (tuple, list)):
                logits = next((item for item in outputs if isinstance(item, torch.Tensor) and item.dim() == 3), None)
            if logits is None:
                raise ValueError("Could not read logits from Qwen model output.")

            pos = controller.last_tgt_pos
            if pos is None:
                pos = max(0, min(int(tgt_pos), logits.shape[1] - 1))
            logits_at_pos = logits[:, pos, :]
            ffn_weights = controller.last_ffn_weights
            if tmp_score is not None and labels is not None:
                label = _label_at_pos(labels.to(logits.device), pos)
                tgt_prob = torch.nn.functional.softmax(logits_at_pos, dim=-1)
                scalar = tgt_prob[:, label].sum()
                grad = torch.autograd.grad(scalar, tmp_score, retain_graph=False, create_graph=False)[0]
                return tgt_prob, grad
            return ffn_weights, logits_at_pos
        finally:
            controller.clear()

    model.forward = ffn_edit_forward
    model._qwen_ffn_edit_patched = True
    model._qwen_ffn_edit_controller = controller
    return model


def install_static_qwen_ffn_edit_hooks(model, positions: list[list[int]], imp_op: str = "remove"):
    """Install static Qwen FFN Edit erasure hooks without replacing model.forward."""

    if getattr(model, "_qwen_static_ffn_edit_patched", False):
        controller = model._qwen_static_ffn_edit_controller
        controller.configure(imp_pos=positions, imp_op=imp_op)
        return model

    controller = QwenFfnEditController(model)
    controller.install()
    controller.configure(imp_pos=positions, imp_op=imp_op)
    model._qwen_static_ffn_edit_patched = True
    model._qwen_static_ffn_edit_controller = controller
    return model


def zero_qwen_down_proj_columns(model: Any, positions: list[list[int]]) -> int:
    """Permanently zero Qwen-style down_proj columns for merged BF16 models."""

    modules = {layer_idx: module for layer_idx, _name, module in find_qwen_down_proj_modules(model)}
    if not modules:
        raise AttributeError("Could not find Qwen-style MLP down_proj modules for permanent editing.")

    edited = 0
    grouped: dict[int, list[int]] = {}
    for layer_idx, neuron_idx in positions:
        grouped.setdefault(int(layer_idx), []).append(int(neuron_idx))

    with torch.no_grad():
        for layer_idx, neuron_ids in grouped.items():
            module = modules.get(layer_idx)
            if module is None:
                continue
            weight = module.weight
            valid = [idx for idx in sorted(set(neuron_ids)) if 0 <= idx < weight.shape[1]]
            if not valid:
                continue
            weight[:, valid] = 0
            edited += len(valid)
    return edited
