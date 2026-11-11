# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Vendored BAGEL inference primitives used by the OpenVINO adapter.

The upstream BAGEL checkpoint currently does not publish Transformers remote
code and the reference implementation imports its model packages as top-level
modules. This module isolates that compatibility layer and installs a portable
SDPA fallback when FlashAttention is unavailable.
"""

import importlib.machinery
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F


_REFERENCE_ROOT = Path(__file__).parent
if str(_REFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_REFERENCE_ROOT))


def _install_flash_attention_fallback() -> None:
    try:
        import flash_attn  # noqa: F401

        return
    except ImportError:
        pass

    flash_attn = types.ModuleType("flash_attn")
    flash_attn.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)

    def flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q=None,
        max_seqlen_k=None,
        causal=False,
        **kwargs,
    ):
        outputs = []
        for i in range(int(cu_seqlens_q.numel() - 1)):
            query_start, query_end = int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1])
            key_start, key_end = int(cu_seqlens_k[i]), int(cu_seqlens_k[i + 1])
            query = q[query_start:query_end].transpose(0, 1).unsqueeze(0)
            key = k[key_start:key_end].transpose(0, 1).unsqueeze(0)
            value = v[key_start:key_end].transpose(0, 1).unsqueeze(0)
            if query.shape[1] != key.shape[1]:
                groups = query.shape[1] // key.shape[1]
                key = key.repeat_interleave(groups, dim=1)
                value = value.repeat_interleave(groups, dim=1)
            output = F.scaled_dot_product_attention(query, key, value, is_causal=causal)
            outputs.append(output.squeeze(0).transpose(0, 1))
        return torch.cat(outputs, dim=0)

    flash_attn.flash_attn_varlen_func = flash_attn_varlen_func
    sys.modules["flash_attn"] = flash_attn


def load_reference_classes():
    """Load BAGEL classes without importing them during normal package startup."""

    _install_flash_attention_fallback()
    from modeling.bagel import (
        Bagel,
        BagelConfig,
        Qwen2Config,
        Qwen2ForCausalLM,
        SiglipVisionConfig,
        SiglipVisionModel,
    )

    return Bagel, BagelConfig, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
