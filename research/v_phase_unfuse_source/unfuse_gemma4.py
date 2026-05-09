"""Unfuse Gemma 4 MoE expert tensors.

Replaces transformers.models.gemma4.modeling_gemma4.Gemma4TextExperts (which
stores weights as 3D fused tensors) with an equivalent module that uses standard
nn.Linear submodules. This unblocks bnb 4-bit quantization, since bitsandbytes
does not support the 3D fused tensor format used by Gemma 4 MoE.

Adapted from bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4 (quantize_gemma4_moe.py)
with NVFP4-specific paths removed.
"""

from __future__ import annotations

import gc
from typing import Optional

import torch
import torch.nn as nn


class UnfusedGemma4Expert(nn.Module):
    """A single Gemma 4 MoE expert with standard nn.Linear modules."""

    def __init__(self, hidden_dim: int, intermediate_dim: int, dtype=None, device=None):
        super().__init__()
        kw = {"bias": False, "dtype": dtype, "device": device}
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, **kw)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, **kw)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, **kw)


class UnfusedGemma4TextExperts(nn.Module):
    """Drop-in replacement for transformers Gemma4TextExperts.

    Same forward signature. Stores experts as nn.ModuleList of UnfusedGemma4Expert
    instead of two 3D parameter tensors (gate_up_proj, down_proj).
    """

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        act_fn,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.act_fn = act_fn
        self.experts = nn.ModuleList([
            UnfusedGemma4Expert(hidden_dim, intermediate_dim, dtype=dtype, device=device)
            for _ in range(num_experts)
        ])

    @classmethod
    def from_fused(cls, fused: nn.Module) -> "UnfusedGemma4TextExperts":
        """Build an unfused version from a transformers Gemma4TextExperts.

        Copies (not aliases) the per-expert slices of the 3D tensors into
        per-expert nn.Linear weights. Caller is responsible for freeing the
        original fused module if desired.
        """
        num_experts = fused.num_experts
        hidden_dim = fused.hidden_dim
        intermediate_dim = fused.intermediate_dim
        act_fn = fused.act_fn

        dtype = fused.gate_up_proj.dtype
        device = fused.gate_up_proj.device

        out = cls(num_experts, hidden_dim, intermediate_dim, act_fn, dtype=dtype, device=device)

        with torch.no_grad():
            for idx in range(num_experts):
                # gate_up_proj[idx] has shape (2*intermediate, hidden); first half is gate, second is up
                out.experts[idx].gate_proj.weight.copy_(
                    fused.gate_up_proj[idx, :intermediate_dim, :]
                )
                out.experts[idx].up_proj.weight.copy_(
                    fused.gate_up_proj[idx, intermediate_dim:, :]
                )
                out.experts[idx].down_proj.weight.copy_(fused.down_proj[idx])

        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Same forward as transformers Gemma4TextExperts but using nn.Linear submodules."""
        final_hidden_states = torch.zeros_like(hidden_states)

        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            expert = self.experts[expert_idx]
            gate = expert.gate_proj(current_state)
            up = expert.up_proj(current_state)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = expert.down_proj(current_hidden_states)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )

        return final_hidden_states


def unfuse_gemma4_model(model: nn.Module, free_after: bool = True) -> nn.Module:
    """Walk a Gemma 4 MoE model and replace every Gemma4TextExperts with an unfused version.

    The replacement happens in-place. After each layer's experts are unfused,
    the original fused module is dereferenced and (if free_after) gc + cuda empty_cache
    are invoked to bound peak memory.

    Returns the same model object for chaining.
    """
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts

    targets = []
    for name, module in model.named_modules():
        if isinstance(module, Gemma4TextExperts):
            targets.append(name)

    for name in targets:
        parent_name, _, attr_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        fused = getattr(parent, attr_name)
        unfused = UnfusedGemma4TextExperts.from_fused(fused)
        setattr(parent, attr_name, unfused)
        del fused
        if free_after:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return model
