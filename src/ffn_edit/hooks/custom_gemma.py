"""Gemma/Gemma3 FFN Edit hooks.

The FFN Edit neuron for Gemma's gated MLP is the activation entering
``mlp.down_proj``:

    act_fn(gate_proj(x)) * up_proj(x)

This module patches a loaded Gemma/Gemma3 causal LM or PEFT model in-place so
callers can pass the same lightweight FFN Edit kwargs used by the Llama runner:
``imp_pos``, ``imp_op``, ``tgt_layer``, ``tgt_pos`` and ``tmp_score``.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F


_LAYER_PATTERNS = (
    re.compile(r"(?:^|\.)language_model\.layers\.(\d+)\.mlp\.down_proj$"),
    re.compile(r"(?:^|\.)model\.layers\.(\d+)\.mlp\.down_proj$"),
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


def find_gemma_down_proj_modules(model) -> list[tuple[int, str, torch.nn.Module]]:
    """Return ``(layer_idx, module_name, module)`` for every Gemma MLP down_proj."""

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


@dataclass
class GemmaFfnEditController:
    """Stateful hook controller for Gemma FFN Edit runs."""

    model: Any
    handles: list[Any] = field(default_factory=list)
    layer_names: dict[int, str] = field(default_factory=dict)
    positions_by_layer: dict[int, list[int]] = field(default_factory=dict)
    imp_op: str | None = None
    tgt_layer: int | None = None
    tgt_pos: int | None = None
    tmp_score: torch.Tensor | None = None
    last_ffn_weights: torch.Tensor | None = None
    last_tgt_pos: int | None = None
    hook_call_count: int = 0
    edit_call_count: int = 0
    edited_neuron_count: int = 0

    def install(self) -> None:
        modules = find_gemma_down_proj_modules(self.model)
        if not modules:
            raise AttributeError(
                "Could not find Gemma MLP down_proj modules; expected names ending with mlp.down_proj."
            )
        for layer_idx, name, module in modules:
            self.layer_names[layer_idx] = name
            self.handles.append(module.register_forward_pre_hook(self._make_hook(layer_idx)))

    def configure(
        self,
        *,
        imp_pos: list[list[int]] | None = None,
        imp_op: str | None = None,
        tgt_layer: int | None = None,
        tgt_pos: int | None = None,
        tmp_score: torch.Tensor | None = None,
    ) -> None:
        if imp_pos is None:
            imp_pos = getattr(self.model, "_ffn_edit_static_imp_pos", None)
        if imp_op is None:
            imp_op = getattr(self.model, "_ffn_edit_static_imp_op", None)
        grouped: dict[int, list[int]] = defaultdict(list)
        for item in imp_pos or []:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            layer_idx, neuron_idx = int(item[0]), int(item[1])
            grouped[layer_idx].append(neuron_idx)
        self.positions_by_layer = {layer: sorted(set(neurons)) for layer, neurons in grouped.items()}
        self.imp_op = imp_op
        self.tgt_layer = int(tgt_layer) if tgt_layer is not None else None
        self.tgt_pos = int(tgt_pos) if tgt_pos is not None else None
        self.tmp_score = tmp_score
        self.last_ffn_weights = None
        self.last_tgt_pos = None

    def clear(self) -> None:
        self.positions_by_layer = {}
        self.imp_op = None
        self.tgt_layer = None
        self.tgt_pos = None
        self.tmp_score = None

    def _make_hook(self, layer_idx: int):
        def hook(_module, inputs):
            if not inputs:
                return inputs
            hidden = inputs[0]
            if not isinstance(hidden, torch.Tensor) or hidden.dim() != 3:
                return inputs
            self.hook_call_count += 1

            changed = False
            edited = hidden

            if self.tgt_layer == layer_idx and self.tgt_pos is not None:
                pos = max(0, min(int(self.tgt_pos), hidden.shape[1] - 1))
                self.last_tgt_pos = pos
                self.last_ffn_weights = hidden[:, pos, :]
                if self.tmp_score is not None:
                    replacement = self.tmp_score.to(device=hidden.device, dtype=hidden.dtype)
                    if replacement.shape[0] != hidden.shape[0]:
                        raise ValueError(
                            f"tmp_score batch {replacement.shape[0]} != hidden batch {hidden.shape[0]}"
                        )
                    edited = edited.clone() if not changed else edited
                    edited[:, pos, :] = replacement
                    changed = True

            neurons = self.positions_by_layer.get(layer_idx)
            if neurons and self.imp_op in {"remove", "enhance"}:
                valid_neurons = [idx for idx in neurons if 0 <= idx < hidden.shape[-1]]
                if valid_neurons:
                    edited = edited.clone() if not changed else edited
                    if self.imp_op == "remove":
                        edited[:, :, valid_neurons] = 0
                    else:
                        edited[:, :, valid_neurons] = edited[:, :, valid_neurons] * 2
                    self.edited_neuron_count += len(valid_neurons)
                    changed = True

            if not changed:
                return inputs
            self.edit_call_count += 1
            return (edited, *inputs[1:])

        return hook


def _label_at_pos(labels: torch.Tensor, pos: int) -> int:
    if labels.dim() == 2:
        return int(labels[0, pos].item())
    return int(labels[pos].item())


def patch_gemma_model_for_ffn_edit(model):
    """Patch a Gemma/Gemma3 model in-place to support FFN Edit forward kwargs."""

    if getattr(model, "_gemma_ffn_edit_patched", False):
        return model

    controller = GemmaFfnEditController(model)
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
                raise ValueError("Could not read logits from Gemma model output.")

            pos = controller.last_tgt_pos
            if pos is None:
                pos = max(0, min(int(tgt_pos), logits.shape[1] - 1))
            logits_at_pos = logits[:, pos, :]
            ffn_weights = controller.last_ffn_weights
            if tmp_score is not None and labels is not None:
                label = _label_at_pos(labels.to(logits.device), pos)
                tgt_prob = F.softmax(logits_at_pos, dim=-1)
                scalar = tgt_prob[:, label].sum()
                grad = torch.autograd.grad(scalar, tmp_score, retain_graph=False, create_graph=False)[0]
                return tgt_prob, grad
            return ffn_weights, logits_at_pos
        finally:
            controller.clear()

    model.forward = ffn_edit_forward
    model._gemma_ffn_edit_patched = True
    model._gemma_ffn_edit_controller = controller
    return model


def install_static_gemma_ffn_edit_hooks(model, positions: list[list[int]], imp_op: str = "remove"):
    """Install static Gemma FFN Edit erasure hooks without replacing model.forward."""

    if getattr(model, "_gemma_static_ffn_edit_patched", False):
        controller = model._gemma_static_ffn_edit_controller
        controller.configure(imp_pos=positions, imp_op=imp_op)
        return model

    controller = GemmaFfnEditController(model)
    controller.install()
    controller.configure(imp_pos=positions, imp_op=imp_op)
    model._gemma_static_ffn_edit_patched = True
    model._gemma_static_ffn_edit_controller = controller
    return model


def gemma_ffn_edit_runtime_stats(model) -> dict[str, int]:
    """Return lightweight runtime counters for static Gemma FFN Edit hooks."""

    controller = getattr(model, "_gemma_static_ffn_edit_controller", None)
    if controller is None:
        return {}
    return {
        "ffn_edit_hook_call_count": int(getattr(controller, "hook_call_count", 0)),
        "ffn_edit_edit_call_count": int(getattr(controller, "edit_call_count", 0)),
        "ffn_edit_edited_neuron_count": int(getattr(controller, "edited_neuron_count", 0)),
    }
