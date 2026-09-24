"""Shared privacy-span selection for Attention Heads Edit localization and evaluation."""

from __future__ import annotations

from typing import Any


STEERING_SPAN_MODES = ("tail_tokens", "adaptive_tail", "full_prompt")
QUERY_MODES = ("next_token", "target_tokens")


def adaptive_tail_count(text_token_count: int, max_tail_tokens: int) -> int:
    """Keep a bounded suffix while leaving at least one text token unscaled."""
    if text_token_count <= 0:
        raise ValueError("text_token_count must be positive")
    if max_tail_tokens <= 0:
        raise ValueError("max_tail_tokens must be positive")
    if text_token_count == 1:
        return 1
    return min(max_tail_tokens, max(1, text_token_count // 2), text_token_count - 1)


def select_privacy_steering_span(
    text: str,
    tokenizer: Any,
    *,
    mode: str = "tail_tokens",
    tail_tokens: int = 8,
) -> str:
    """Select the prompt substring whose attention should be attenuated."""
    if mode not in STEERING_SPAN_MODES:
        raise ValueError(f"Unsupported steering span mode: {mode}")
    if mode == "full_prompt":
        return text
    if tail_tokens <= 0:
        raise ValueError("tail_tokens must be positive")

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"]
    valid_offsets = [
        (int(start), int(end))
        for start, end in offsets
        if int(end) > int(start)
    ]
    if not valid_offsets:
        raise ValueError("Tokenizer returned no valid offsets for steering span selection")
    selected_count = tail_tokens
    if mode == "adaptive_tail":
        selected_count = adaptive_tail_count(len(valid_offsets), tail_tokens)
    selected = valid_offsets[-selected_count:]
    span = text[selected[0][0] : selected[-1][1]]
    if not span or span not in text:
        raise ValueError("Failed to derive a valid privacy steering span")
    return span


def select_privacy_steering_token_range(
    prefix_token_count: int,
    *,
    mode: str = "tail_tokens",
    tail_tokens: int = 8,
) -> tuple[int, int]:
    """Return [start, end) token indices relative to a prefix of prefix_token_count tokens."""
    if mode not in STEERING_SPAN_MODES:
        raise ValueError(f"Unsupported steering span mode: {mode}")
    if prefix_token_count <= 0:
        raise ValueError("prefix_token_count must be positive")
    if mode == "full_prompt":
        return 0, prefix_token_count
    if tail_tokens <= 0:
        raise ValueError("tail_tokens must be positive")
    selected_count = tail_tokens
    if mode == "adaptive_tail":
        selected_count = adaptive_tail_count(prefix_token_count, tail_tokens)
    selected_count = min(selected_count, prefix_token_count)
    return prefix_token_count - selected_count, prefix_token_count


def attention_ranges(
    *,
    prompt_len: int,
    target_len: int,
    steering_span_mode: str,
    steering_tail_tokens: int,
    query_mode: str,
    prompt_text_len: int | None = None,
) -> tuple[int, int, int, int]:
    """Return query and key token ranges used for Attention Heads Edit head localization."""
    if prompt_len <= 0:
        raise ValueError("prompt_len must be positive")
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    if steering_span_mode not in STEERING_SPAN_MODES:
        raise ValueError(f"Unsupported steering span mode: {steering_span_mode}")
    if query_mode not in QUERY_MODES:
        raise ValueError(f"Unsupported query mode: {query_mode}")

    if steering_span_mode == "full_prompt":
        key_start = 0
    else:
        if steering_tail_tokens <= 0:
            raise ValueError("steering_tail_tokens must be positive")
        selected_count = steering_tail_tokens
        if steering_span_mode == "adaptive_tail":
            text_token_count = prompt_text_len if prompt_text_len is not None else prompt_len
            if text_token_count <= 0 or text_token_count > prompt_len:
                raise ValueError("prompt_text_len must be in [1, prompt_len]")
            selected_count = adaptive_tail_count(text_token_count, steering_tail_tokens)
        key_start = max(0, prompt_len - selected_count)
    key_end = prompt_len

    if query_mode == "next_token":
        query_start = prompt_len - 1
        query_end = prompt_len
    else:
        query_start = prompt_len
        query_end = prompt_len + target_len
    return query_start, query_end, key_start, key_end
