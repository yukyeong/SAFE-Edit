"""PyTorch Custom Llama3 model for FFN Edit."""

import os
import copy
import json
import math
import logging
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from transformers import LlamaForCausalLM, LlamaConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = logging.getLogger(__name__)


def _make_causal_attention_mask(
    batch_size: int,
    seq_length: int,
    attention_mask: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
) -> torch.Tensor:
    """Build 4D causal + padding mask compatible with decoder attention. No dependency on HF internal APIs."""
    seq_length_with_past = seq_length + past_key_values_length
    min_dtype = torch.finfo(dtype).min if dtype.is_floating_point else torch.iinfo(dtype).min
    causal = torch.triu(
        torch.full((seq_length_with_past, seq_length_with_past), min_dtype, dtype=dtype, device=device),
        diagonal=1,
    )
    causal = causal.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
    if attention_mask is not None and attention_mask.dim() == 2:
        mask = attention_mask[:, None, None, :].to(dtype=dtype)
        mask = (1.0 - mask) * min_dtype
        causal = causal + mask
    return causal


def _check_tgt_pos(tgt_pos, seq_len, name="tgt_pos"):
    """Return None if tgt_pos is out of bounds, else tgt_pos. Logs warning on clamp."""
    if tgt_pos is None or seq_len is None:
        return tgt_pos
    if tgt_pos < 0 or tgt_pos >= seq_len:
        logger.warning("%s=%d out of range [0, %d), ignoring for this forward", name, tgt_pos, seq_len)
        return None
    return tgt_pos


class LlamaForCausalLMWithEditing(LlamaForCausalLM):
    """
    Llama3 model with support for neuron editing and integrated gradients.
    Extends LlamaForCausalLM to support FFN Edit functionality.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self._patch_model_layers()
        
    def _patch_model_layers(self):
        """Patch model layers to support FFN Edit editing."""
        # Store original MLP forward methods for each layer
        for i, layer in enumerate(self.model.layers):
            original_mlp_forward = layer.mlp.forward
            
            def make_mlp_forward_wrapper(layer_idx, mlp_layer, orig_forward):
                def patched_mlp_forward(hidden_states, tmp_score=None, tgt_pos=None, imp_pos=None, imp_op=None, tgt_layer=None, **kw):
                    seq_len = hidden_states.shape[1] if hidden_states.dim() >= 2 else 0
                    tgt_pos = _check_tgt_pos(tgt_pos, seq_len, "tgt_pos")
                    gate_proj = mlp_layer.gate_proj(hidden_states)
                    up_proj = mlp_layer.up_proj(hidden_states)
                    intermediate = F.silu(gate_proj) * up_proj
                    if tgt_layer == layer_idx and tgt_pos is not None and tmp_score is None and intermediate.requires_grad:
                        intermediate.retain_grad()
                    if tmp_score is not None and tgt_pos is not None and tgt_layer == layer_idx:
                        intermediate = intermediate.clone()
                        intermediate[:, tgt_pos, :] = tmp_score
                    if imp_pos is not None and imp_op is not None:
                        for edit_layer_idx, edit_pos in imp_pos:
                            if edit_layer_idx == layer_idx:
                                if imp_op == 'remove':
                                    intermediate[:, :, edit_pos] = 0.0
                                elif imp_op == 'enhance':
                                    intermediate[:, :, edit_pos] *= 2.0
                    down_proj = mlp_layer.down_proj(intermediate)
                    if tgt_layer == layer_idx and tgt_pos is not None and tmp_score is None:
                        mlp_layer._last_intermediate = intermediate
                    if imp_op == 'return' and tgt_pos is not None and imp_pos is not None:
                        down_proj.imp_weights = []
                        for edit_layer_idx, edit_pos in imp_pos:
                            if edit_layer_idx == layer_idx and 0 <= tgt_pos < intermediate.shape[1]:
                                down_proj.imp_weights.append(intermediate[0, tgt_pos, edit_pos].item())
                    return down_proj
                
                return patched_mlp_forward
            
            layer.mlp.forward = make_mlp_forward_wrapper(i, layer.mlp, original_mlp_forward)
        
        # Patch decoder layer forward
        for i, layer in enumerate(self.model.layers):
            original_layer_forward = layer.forward
            
            def make_layer_forward_wrapper(layer_idx, decoder_layer, orig_forward):
                def patched_decoder_forward(
                    hidden_states,
                    attention_mask=None,
                    position_ids=None,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                    tgt_pos=None,
                    tgt_layer=None,
                    tmp_score=None,
                    imp_pos=None,
                    imp_op=None,
                    **kwargs,
                ):
                    residual = hidden_states
                    # Self Attention
                    hidden_states = decoder_layer.input_layernorm(hidden_states)
                    attn_outputs = decoder_layer.self_attn(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )
                    hidden_states = attn_outputs[0]
                    hidden_states = residual + hidden_states
                    
                    # Feed Forward Network with FFN Edit support
                    residual = hidden_states
                    hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
                    
                    # Apply MLP with FFN Edit parameters
                    if tgt_layer == layer_idx:
                        ffn_output = decoder_layer.mlp(
                            hidden_states,
                            tmp_score=tmp_score,
                            tgt_pos=tgt_pos,
                            imp_pos=imp_pos,
                            imp_op=imp_op,
                            tgt_layer=tgt_layer
                        )
                        # Store intermediate for extraction in model forward
                        if tgt_pos is not None and tmp_score is None and hasattr(decoder_layer.mlp, '_last_intermediate'):
                            # Already stored in MLP
                            pass
                    else:
                        ffn_output = decoder_layer.mlp(
                            hidden_states,
                            imp_pos=imp_pos,
                            imp_op=imp_op,
                            tgt_layer=tgt_layer
                        )
                    
                    hidden_states = residual + ffn_output
                    if getattr(ffn_output, "imp_weights", None) is not None:
                        hidden_states.imp_weights = getattr(ffn_output, "imp_weights", [])
                    outputs = (hidden_states,)
                    if output_attentions:
                        outputs += (attn_outputs[1],)
                    if use_cache:
                        outputs += (attn_outputs[2],)
                    return outputs
                
                return patched_decoder_forward
            
            layer.forward = make_layer_forward_wrapper(i, layer, original_layer_forward)
        
        # Patch model forward
        original_model_forward = self.model.forward
        
        def patched_model_forward(
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            tgt_pos=None,
            tgt_layer=None,
            tmp_score=None,
            imp_pos=None,
            imp_op=None,
            **kwargs,
        ):
            # Use original forward but pass through FFN Edit parameters
            # We need to manually call layers to pass FFN Edit params
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            use_cache = use_cache if use_cache is not None else self.config.use_cache
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            
            # Prepare inputs
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
            elif input_ids is not None:
                batch_size, seq_length = input_ids.shape[:2]
            elif inputs_embeds is not None:
                batch_size, seq_length, _ = inputs_embeds.shape
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")
            tgt_pos = _check_tgt_pos(tgt_pos, seq_length, "tgt_pos")
            seq_length_with_past = seq_length
            past_key_values_length = 0
            
            if past_key_values is not None:
                past_key_values_length = past_key_values[0][0].shape[2]
                seq_length_with_past = seq_length_with_past + past_key_values_length
            
            if position_ids is None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                position_ids = torch.arange(
                    past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
                )
                position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
            else:
                position_ids = position_ids.view(-1, seq_length).long()
            
            if inputs_embeds is None:
                inputs_embeds = self.model.embed_tokens(input_ids)
            
            # Prepare attention mask
            if attention_mask is None:
                attention_mask = torch.ones(
                    (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
                )
            else:
                attention_mask = attention_mask.to(inputs_embeds.device)
            
            # Create causal mask
            causal_mask = self.model._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )
            
            # Pass through layers manually to inject FFN Edit parameters
            all_hidden_states = () if output_hidden_states else None
            all_self_attns = () if output_attentions else None
            next_decoder_cache = () if use_cache else None
            imp_weights = []
            ffn_weights = None
            
            hidden_states = inputs_embeds
            
            for idx, decoder_layer in enumerate(self.model.layers):
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)
                
                past_key_value = past_key_values[idx] if past_key_values is not None else None
                
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    tgt_pos=tgt_pos,
                    tgt_layer=tgt_layer,
                    tmp_score=tmp_score,
                    imp_pos=imp_pos,
                    imp_op=imp_op,
                )
                
                hidden_states = layer_outputs[0]
                
                # Extract FFN weights at target layer (before residual was added)
                if tgt_layer == idx and tgt_pos is not None and tmp_score is None:
                    if hasattr(decoder_layer.mlp, '_last_intermediate') and decoder_layer.mlp._last_intermediate is not None:
                        intermediate = decoder_layer.mlp._last_intermediate
                        if tgt_pos < intermediate.shape[1]:
                            if intermediate.shape[0] == 1:
                                ffn_weights = intermediate[0, tgt_pos, :].unsqueeze(0)
                            else:
                                ffn_weights = intermediate[:, tgt_pos, :]
                        decoder_layer.mlp._last_intermediate = None
                    else:
                        layer_normed = decoder_layer.post_attention_layernorm(hidden_states)
                        gate_proj = decoder_layer.mlp.gate_proj(layer_normed)
                        up_proj = decoder_layer.mlp.up_proj(layer_normed)
                        intermediate = F.silu(gate_proj) * up_proj
                        if tgt_pos < intermediate.shape[1]:
                            if intermediate.shape[0] == 1:
                                ffn_weights = intermediate[0, tgt_pos, :].unsqueeze(0)
                            else:
                                ffn_weights = intermediate[:, tgt_pos, :]
                
                if use_cache:
                    next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
                
                if output_attentions:
                    all_self_attns += (layer_outputs[1],)
                
                # Collect imp_weights from hidden_states (copied from ffn_output in decoder layer)
                if imp_op == 'return':
                    h = layer_outputs[0]
                    if getattr(h, 'imp_weights', None) is not None and isinstance(h.imp_weights, list):
                        imp_weights.extend(h.imp_weights)
            
            hidden_states = self.model.norm(hidden_states)
            
            # Add last hidden state
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            next_cache = next_decoder_cache if use_cache else None
            if not return_dict:
                result = tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
            else:
                from transformers.modeling_outputs import BaseModelOutputWithPast
                result = BaseModelOutputWithPast(
                    last_hidden_state=hidden_states,
                    past_key_values=next_cache,
                    hidden_states=all_hidden_states,
                    attentions=all_self_attns,
                )
            
            # Attach FFN Edit-specific outputs
            if isinstance(result, tuple):
                # Convert to object-like structure
                class Result:
                    def __init__(self, data):
                        self.data = data
                        self.ffn_weights = ffn_weights
                        self.imp_weights = imp_weights
                    def __getitem__(self, idx):
                        return self.data[idx]
                    def __len__(self):
                        return len(self.data)
                return Result(result)
            else:
                result.ffn_weights = ffn_weights
                result.imp_weights = imp_weights
            
            return result
        
        self.model.forward = patched_model_forward
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        tgt_pos=None,
        tgt_layer=None,
        tmp_score=None,
        imp_pos=None,
        imp_op=None,
        **kwargs,
    ):
        """
        Forward pass with FFN Edit editing support.
        When no FFN Edit params are passed, returns standard CausalLMOutputWithPast (HF compatible).
        """
        ffn_edit_mode = (tgt_pos is not None or tgt_layer is not None or tmp_score is not None
                     or imp_op is not None or imp_pos is not None)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            tgt_pos=tgt_pos,
            tgt_layer=tgt_layer,
            tmp_score=tmp_score,
            imp_pos=imp_pos,
            imp_op=imp_op,
        )

        hidden_states = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
        ffn_weights = getattr(outputs, 'ffn_weights', None)
        imp_weights = getattr(outputs, 'imp_weights', [])
        if tgt_pos is not None and hidden_states.shape[1] > tgt_pos:
            hidden_states = hidden_states[:, tgt_pos:tgt_pos + 1, :]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if tgt_pos is not None and logits.shape[1] == 1:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.squeeze(1), labels[:, tgt_pos] if labels.dim() == 2 else labels)
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_labels.view(-1).to(shift_logits.device),
                )

        # Standard HF return when not in FFN Edit mode, or when FFN Edit is used only as
        # a pure inference-time edit. This keeps `.generate()` and external
        # evaluators compatible while still applying neuron removal/enhancement
        # inside patched MLP layers.
        if not ffn_edit_mode or (imp_op in {"remove", "enhance"} and tmp_score is None and tgt_pos is None and tgt_layer is None):
            if not return_dict:
                return (loss,) + (logits,) if loss is not None else (logits,)
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=getattr(outputs, 'past_key_values', None),
                hidden_states=getattr(outputs, 'hidden_states', None),
                attentions=getattr(outputs, 'attentions', None),
            )

        # FFN Edit-specific returns
        if imp_op == 'return':
            if tmp_score is None:
                return logits, ffn_weights, imp_weights
            if labels is not None and tgt_pos is not None:
                tgt_label = labels[0, tgt_pos] if labels.dim() == 2 else labels[tgt_pos]
                tgt_prob = F.softmax(logits[:, 0, :], dim=-1)
                gradient = torch.autograd.grad(
                    tgt_prob[:, tgt_label], tmp_score, retain_graph=True, create_graph=True
                )[0]
                return tgt_prob, gradient
            return logits, ffn_weights
        if tmp_score is None:
            logits_out = logits.squeeze(1) if logits.shape[1] == 1 else logits
            return ffn_weights, logits_out
        if labels is not None and tgt_pos is not None:
            tgt_label = labels[0, tgt_pos] if labels.dim() == 2 else labels[tgt_pos]
            tgt_prob = F.softmax(logits[:, 0, :], dim=-1)
            gradient = torch.autograd.grad(
                tgt_prob[:, tgt_label], tmp_score, retain_graph=True, create_graph=True
            )[0]
            return tgt_prob, gradient
        return logits, ffn_weights


def _apply_patch_to_inner_layers(inner_model):
    """Apply FFN Edit MLP and layer forward patches to inner LlamaModel (in-place)."""
    for i, layer in enumerate(inner_model.layers):
        original_mlp_forward = layer.mlp.forward

        def make_mlp_fwd(layer_idx, mlp_layer, _orig):
            def patched_mlp_forward(hidden_states, tmp_score=None, tgt_pos=None, imp_pos=None, imp_op=None, tgt_layer=None, **kw):
                seq_len = hidden_states.shape[1] if hidden_states.dim() >= 2 else 0
                tgt_pos = _check_tgt_pos(tgt_pos, seq_len, "tgt_pos")
                gate_proj = mlp_layer.gate_proj(hidden_states)
                up_proj = mlp_layer.up_proj(hidden_states)
                intermediate = F.silu(gate_proj) * up_proj
                if tgt_layer == layer_idx and tgt_pos is not None and tmp_score is None and intermediate.requires_grad:
                    intermediate.retain_grad()
                if tmp_score is not None and tgt_pos is not None and tgt_layer == layer_idx:
                    intermediate = intermediate.clone()
                    intermediate[:, tgt_pos, :] = tmp_score
                if imp_pos is not None and imp_op is not None:
                    for edit_layer_idx, edit_pos in imp_pos:
                        if edit_layer_idx == layer_idx:
                            if imp_op == "remove":
                                intermediate[:, :, edit_pos] = 0.0
                            elif imp_op == "enhance":
                                intermediate[:, :, edit_pos] *= 2.0
                down_proj = mlp_layer.down_proj(intermediate)
                if tgt_layer == layer_idx and tgt_pos is not None and tmp_score is None:
                    mlp_layer._last_intermediate = intermediate
                if imp_op == "return" and tgt_pos is not None and imp_pos is not None:
                    down_proj.imp_weights = []
                    for el, ep in imp_pos:
                        if el == layer_idx and 0 <= tgt_pos < intermediate.shape[1]:
                            down_proj.imp_weights.append(intermediate[0, tgt_pos, ep].item())
                return down_proj
            return patched_mlp_forward

        layer.mlp.forward = make_mlp_fwd(i, layer.mlp, original_mlp_forward)

    for i, layer in enumerate(inner_model.layers):
        original_layer_forward = layer.forward

        def make_layer_fwd(layer_idx, decoder_layer, _orig):
            def patched_decoder_forward(
                hidden_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                past_key_values=None,
                cache_position=None,
                position_embeddings=None,
                output_attentions=False,
                use_cache=False,
                tgt_pos=None,
                tgt_layer=None,
                tmp_score=None,
                imp_pos=None,
                imp_op=None,
                **kwargs,
            ):
                residual = hidden_states
                hidden_states = decoder_layer.input_layernorm(hidden_states)
                attn_kw = dict(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_value or past_key_values,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
                attn_outputs = decoder_layer.self_attn(**attn_kw)
                if isinstance(attn_outputs, tuple):
                    hidden_states = attn_outputs[0]
                else:
                    hidden_states = attn_outputs
                hidden_states = residual + hidden_states
                residual = hidden_states
                hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
                if tgt_layer == layer_idx:
                    ffn_output = decoder_layer.mlp(
                        hidden_states,
                        tmp_score=tmp_score,
                        tgt_pos=tgt_pos,
                        imp_pos=imp_pos,
                        imp_op=imp_op,
                        tgt_layer=tgt_layer,
                    )
                else:
                    ffn_output = decoder_layer.mlp(
                        hidden_states,
                        imp_pos=imp_pos,
                        imp_op=imp_op,
                        tgt_layer=tgt_layer,
                    )
                hidden_states = residual + ffn_output
                if getattr(ffn_output, "imp_weights", None) is not None:
                    hidden_states.imp_weights = getattr(ffn_output, "imp_weights", [])
                outputs = (hidden_states,)
                if output_attentions and isinstance(attn_outputs, tuple) and len(attn_outputs) > 1:
                    outputs += (attn_outputs[1],)
                if use_cache and isinstance(attn_outputs, tuple) and len(attn_outputs) > 2:
                    outputs += (attn_outputs[2],)
                return outputs
            return patched_decoder_forward

        layer.forward = make_layer_fwd(i, layer, original_layer_forward)

    # Patch inner model's forward to accept and pass FFN Edit params (same logic as LlamaForCausalLMWithEditing)
    original_model_forward = inner_model.forward

    def patched_model_forward(
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        tgt_pos=None,
        tgt_layer=None,
        tmp_score=None,
        imp_pos=None,
        imp_op=None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else inner_model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else inner_model.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else inner_model.config.use_cache
        return_dict = return_dict if return_dict is not None else inner_model.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        tgt_pos = _check_tgt_pos(tgt_pos, seq_length, "tgt_pos")
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length + past_key_values_length
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()
        if inputs_embeds is None:
            inputs_embeds = inner_model.embed_tokens(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        else:
            attention_mask = attention_mask.to(inputs_embeds.device)
        causal_mask = _make_causal_attention_mask(
            batch_size,
            seq_length_with_past,
            attention_mask,
            inputs_embeds.dtype,
            inputs_embeds.device,
            past_key_values_length,
        )
        cache_position = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_length,
            device=inputs_embeds.device,
        )
        position_embeddings = getattr(inner_model, "rotary_emb", None)
        if position_embeddings is not None and callable(position_embeddings):
            position_embeddings = position_embeddings(inputs_embeds, position_ids)
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        imp_weights = []
        ffn_weights = None
        hidden_states = inputs_embeds
        for idx, decoder_layer in enumerate(inner_model.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                output_attentions=output_attentions,
                use_cache=use_cache,
                tgt_pos=tgt_pos,
                tgt_layer=tgt_layer,
                tmp_score=tmp_score,
                imp_pos=imp_pos,
                imp_op=imp_op,
            )
            hidden_states = layer_outputs[0]
            if tgt_layer == idx and tgt_pos is not None and tmp_score is None:
                if getattr(decoder_layer.mlp, "_last_intermediate", None) is not None:
                    intermediate = decoder_layer.mlp._last_intermediate
                    if tgt_pos < intermediate.shape[1]:
                        ffn_weights = intermediate[0, tgt_pos, :].unsqueeze(0) if intermediate.shape[0] == 1 else intermediate[:, tgt_pos, :]
                    decoder_layer.mlp._last_intermediate = None
                else:
                    layer_normed = decoder_layer.post_attention_layernorm(hidden_states)
                    gate_proj = decoder_layer.mlp.gate_proj(layer_normed)
                    up_proj = decoder_layer.mlp.up_proj(layer_normed)
                    intermediate = F.silu(gate_proj) * up_proj
                    if tgt_pos < intermediate.shape[1]:
                        ffn_weights = intermediate[0, tgt_pos, :].unsqueeze(0) if intermediate.shape[0] == 1 else intermediate[:, tgt_pos, :]
            if use_cache and len(layer_outputs) > (2 if output_attentions else 1):
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
            if output_attentions and len(layer_outputs) > 1:
                all_self_attns += (layer_outputs[1],)
            if imp_op == "return":
                h0 = layer_outputs[0]
                if getattr(h0, "imp_weights", None) is not None and isinstance(h0.imp_weights, list):
                    imp_weights.extend(h0.imp_weights)
        hidden_states = inner_model.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            result = tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        else:
            from transformers.modeling_outputs import BaseModelOutputWithPast
            result = BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )
        if isinstance(result, tuple):
            class Result:
                def __init__(self, data):
                    self.data = data
                    self.ffn_weights = ffn_weights
                    self.imp_weights = imp_weights
                def __getitem__(self, idx):
                    return self.data[idx]
                def __len__(self):
                    return len(self.data)
            return Result(result)
        result.ffn_weights = ffn_weights
        result.imp_weights = imp_weights
        return result

    inner_model.forward = patched_model_forward


def patch_llama_model_in_place(model):
    """
    Patch an existing LlamaForCausalLM or PeftModel in-place for FFN Edit (no state_dict copy).
    Use this when the model is already loaded (e.g. with device_map or PEFT) to avoid OOM.
    """
    if isinstance(model, LlamaForCausalLMWithEditing):
        return model
    # Resolve outer (LlamaForCausalLM whose forward we replace) and inner (LlamaModel with .layers)
    if hasattr(model, "base_model"):
        outer = model.base_model
    else:
        outer = model
    if hasattr(outer, "layers"):
        inner = outer
        host = model  # config/lm_head/forward live on top-level (e.g. PeftModel or LlamaForCausalLM)
    else:
        inner = getattr(outer, "model", None)
        if inner is None:
            raise AttributeError("Could not find inner model (no .model on outer)")
        while not hasattr(inner, "layers") and hasattr(inner, "model"):
            inner = inner.model
        host = outer
    if not hasattr(inner, "layers"):
        raise AttributeError("Could not find inner LlamaModel with .layers on this model")
    _apply_patch_to_inner_layers(inner)
    config = host.config
    lm_head = host.lm_head
    original_forward = host.forward

    def ffn_edit_forward(
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        tgt_pos=None,
        tgt_layer=None,
        tmp_score=None,
        imp_pos=None,
        imp_op=None,
        **kwargs,
    ):
        ffn_edit_mode = (tgt_pos is not None or tgt_layer is not None or tmp_score is not None
                     or imp_op is not None or imp_pos is not None)
        output_attentions = output_attentions if output_attentions is not None else config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else config.output_hidden_states
        return_dict = return_dict if return_dict is not None else config.use_return_dict
        # Call the patched inner (LlamaModel), not outer.model, so we get ffn_weights and FFN Edit outputs
        outputs = inner(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            tgt_pos=tgt_pos,
            tgt_layer=tgt_layer,
            tmp_score=tmp_score,
            imp_pos=imp_pos,
            imp_op=imp_op,
        )
        # Support: tuple, our Result (.data[0]), BaseModelOutputWithPast.last_hidden_state, or .hidden_states[-1]
        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        elif hasattr(outputs, "data") and isinstance(getattr(outputs, "data", None), tuple):
            hidden_states = outputs.data[0]
        else:
            hidden_states = getattr(outputs, "last_hidden_state", None)
            if hidden_states is None and getattr(outputs, "hidden_states", None):
                hidden_states = outputs.hidden_states[-1]
        if hidden_states is None:
            raise ValueError("Could not get hidden_states from model output")
        ffn_weights = getattr(outputs, "ffn_weights", None)
        if ffn_weights is None and tgt_layer is not None and hasattr(inner, "layers") and 0 <= tgt_layer < len(inner.layers):
            li = getattr(inner.layers[tgt_layer], "_last_intermediate", None)
            if li is not None and isinstance(li, (tuple, list)) and len(li) >= 2:
                ffn_weights = li[1]
        imp_weights = getattr(outputs, "imp_weights", [])
        seq_len = hidden_states.shape[1]
        tgt_pos = _check_tgt_pos(tgt_pos, seq_len, "tgt_pos")
        if tgt_pos is not None:
            hidden_states = hidden_states[:, tgt_pos : tgt_pos + 1, :]
        logits = lm_head(hidden_states)
        loss = None
        if labels is not None:
            if tgt_pos is not None and logits.shape[1] == 1:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.squeeze(1), labels[:, tgt_pos] if labels.dim() == 2 else labels)
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, config.vocab_size), shift_labels.view(-1).to(shift_logits.device))
        if not ffn_edit_mode or (imp_op in {"remove", "enhance"} and tmp_score is None and tgt_pos is None and tgt_layer is None):
            if not return_dict:
                return (loss,) + (logits,) if loss is not None else (logits,)
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=getattr(outputs, "past_key_values", None),
                hidden_states=getattr(outputs, "hidden_states", None),
                attentions=getattr(outputs, "attentions", None),
            )
        if imp_op == "return":
            if tmp_score is None:
                return logits, ffn_weights, imp_weights
            if labels is not None and tgt_pos is not None:
                tgt_label = labels[0, tgt_pos] if labels.dim() == 2 else labels[tgt_pos]
                tgt_prob = F.softmax(logits[:, 0, :], dim=-1)
                scalar = tgt_prob[:, tgt_label].sum()
                # IG only needs d(tgt_prob)/d(tmp_score); create_graph=True keeps the full graph and can OOM on 24GB.
                gradient = torch.autograd.grad(scalar, tmp_score, retain_graph=False, create_graph=False)[0]
                return tgt_prob, gradient
            return logits, ffn_weights
        if tmp_score is None:
            logits_out = logits.squeeze(1) if logits.shape[1] == 1 else logits
            return ffn_weights, logits_out
        if labels is not None and tgt_pos is not None:
            tgt_label = labels[0, tgt_pos] if labels.dim() == 2 else labels[tgt_pos]
            tgt_prob = F.softmax(logits[:, 0, :], dim=-1)
            scalar = tgt_prob[:, tgt_label].sum()
            gradient = torch.autograd.grad(scalar, tmp_score, retain_graph=False, create_graph=False)[0]
            return tgt_prob, gradient
        return logits, ffn_weights

    host.forward = ffn_edit_forward
    return model


def patch_llama_model(model):
    """
    Patch a standard LlamaForCausalLM model to support FFN Edit editing.
    Prefer patch_llama_model_in_place() when using PEFT or device_map to avoid OOM.
    """
    if isinstance(model, LlamaForCausalLMWithEditing):
        return model
    # Use in-place patch to avoid duplicating state_dict (works for plain Llama and PeftModel)
    return patch_llama_model_in_place(model)
