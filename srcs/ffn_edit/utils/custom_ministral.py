"""Ministral3 FFN Edit hooks.

Ministral 3 8B Base is exposed by Transformers as a Mistral3 multimodal
wrapper whose text tower contains Ministral decoder layers:

    language_model.layers[i].mlp.down_proj

For FFN Edit we erase the activation entering ``mlp.down_proj``, matching the
Qwen/Gemma gated-MLP implementation while keeping the official model forward
implementation intact.
"""

from __future__ import annotations

import re

import torch

from custom_gemma import GemmaFfnEditController, _label_at_pos


_LAYER_PATTERNS = (
    re.compile(r"(?:^|\.)language_model\.layers\.(\d+)\.mlp\.down_proj$"),
    re.compile(r"(?:^|\.)model\.language_model\.layers\.(\d+)\.mlp\.down_proj$"),
    re.compile(r"(?:^|\.)base_model\.model\.language_model\.layers\.(\d+)\.mlp\.down_proj$"),
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


def find_ministral_down_proj_modules(model) -> list[tuple[int, str, torch.nn.Module]]:
    """Return ``(layer_idx, module_name, module)`` for Ministral MLP down_proj modules."""

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


class MinistralFfnEditController(GemmaFfnEditController):
    """Stateful hook controller for Ministral3 FFN Edit runs."""

    def install(self) -> None:
        modules = find_ministral_down_proj_modules(self.model)
        if not modules:
            raise AttributeError(
                "Could not find Ministral MLP down_proj modules; expected names ending with mlp.down_proj."
            )
        for layer_idx, name, module in modules:
            self.layer_names[layer_idx] = name
            self.handles.append(module.register_forward_pre_hook(self._make_hook(layer_idx)))


def patch_ministral_model_for_ffn_edit(model):
    """Patch a Ministral3 model in-place to support FFN Edit attribution kwargs."""

    if getattr(model, "_ministral_ffn_edit_patched", False):
        return model

    controller = MinistralFfnEditController(model)
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
                raise ValueError("Could not read logits from Ministral model output.")

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
    model._ministral_ffn_edit_patched = True
    model._ministral_ffn_edit_controller = controller
    return model


def install_static_ministral_ffn_edit_hooks(model, positions: list[list[int]], imp_op: str = "remove"):
    """Install static Ministral FFN Edit erasure hooks without replacing model.forward."""

    if getattr(model, "_ministral_static_ffn_edit_patched", False):
        controller = model._ministral_static_ffn_edit_controller
        controller.configure(imp_pos=positions, imp_op=imp_op)
        return model

    controller = MinistralFfnEditController(model)
    controller.install()
    controller.configure(imp_pos=positions, imp_op=imp_op)
    model._ministral_static_ffn_edit_patched = True
    model._ministral_static_ffn_edit_controller = controller
    return model
