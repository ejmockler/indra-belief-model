"""Synthetic forward-equivalence test for unfuse_gemma4.

Constructs a small Gemma4TextExperts module with random weights, applies
unfuse, and verifies the two modules produce identical forward outputs
for the same input (within fp32/bf16 numerical tolerance).

Run on the ROCm host with the v-phase venv active:

    source ~/venvs/v-phase/bin/activate
    python research/v_phase_unfuse_source/test_unfuse_equivalence.py
"""

import sys
from pathlib import Path

import torch
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts

sys.path.insert(0, str(Path(__file__).parent))
from unfuse_gemma4 import UnfusedGemma4TextExperts


class _DummyConfig:
    """Minimal config matching what Gemma4TextExperts expects.

    `_experts_implementation` is required by transformers' `@use_experts_implementation`
    decorator (which wraps Gemma4TextExperts.forward to dispatch to a named impl).
    """

    def __init__(self, num_experts, hidden_size, moe_intermediate_size,
                 hidden_activation="gelu_pytorch_tanh", experts_implementation="eager"):
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.hidden_activation = hidden_activation
        self._experts_implementation = experts_implementation


def _build_random_inputs(num_tokens, hidden_dim, num_experts, top_k, dtype, device, seed=0):
    """Build (hidden_states, top_k_index, top_k_weights) for a forward pass."""
    g = torch.Generator(device=device).manual_seed(seed)
    hidden = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device, generator=g)
    # random top-k experts per token, normalized weights
    indices = torch.stack([
        torch.randperm(num_experts, generator=g, device=device)[:top_k]
        for _ in range(num_tokens)
    ])
    raw_w = torch.rand(num_tokens, top_k, dtype=dtype, device=device, generator=g)
    weights = raw_w / raw_w.sum(dim=-1, keepdim=True)
    return hidden, indices, weights


def test_forward_equivalence(num_experts=4, hidden_dim=64, intermediate_dim=128,
                              num_tokens=16, top_k=2, dtype=torch.float32,
                              device=None, atol=1e-5, rtol=1e-4):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = _DummyConfig(num_experts, hidden_dim, intermediate_dim)
    fused = Gemma4TextExperts(cfg).to(device=device, dtype=dtype)

    # randomize the params (Gemma4TextExperts inits with empty)
    with torch.no_grad():
        torch.manual_seed(42)
        fused.gate_up_proj.copy_(torch.randn_like(fused.gate_up_proj) * 0.02)
        fused.down_proj.copy_(torch.randn_like(fused.down_proj) * 0.02)

    # build an unfused copy
    unfused = UnfusedGemma4TextExperts.from_fused(fused)

    # forward both with the same input
    hidden, indices, weights = _build_random_inputs(
        num_tokens, hidden_dim, num_experts, top_k, dtype, device
    )

    with torch.no_grad():
        out_fused = fused(hidden.clone(), indices.clone(), weights.clone())
        out_unfused = unfused(hidden.clone(), indices.clone(), weights.clone())

    err_max = (out_fused - out_unfused).abs().max().item()
    err_mean = (out_fused - out_unfused).abs().mean().item()
    print(f"  config: num_experts={num_experts} hidden={hidden_dim} inter={intermediate_dim} "
          f"tokens={num_tokens} top_k={top_k} dtype={dtype}")
    print(f"  out shapes: fused={tuple(out_fused.shape)} unfused={tuple(out_unfused.shape)}")
    print(f"  err: max={err_max:.2e} mean={err_mean:.2e} (tolerance: atol={atol})")
    torch.testing.assert_close(out_fused, out_unfused, atol=atol, rtol=rtol)
    print("  PASS")


def test_unfuse_walker():
    """Verify unfuse_gemma4_model walks a small synthetic model and replaces all Gemma4TextExperts."""
    import torch.nn as nn
    from unfuse_gemma4 import unfuse_gemma4_model

    cfg = _DummyConfig(num_experts=4, hidden_size=64, moe_intermediate_size=128)

    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = Gemma4TextExperts(cfg)

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([FakeLayer() for _ in range(3)])

    model = FakeModel()
    n_before = sum(1 for m in model.modules() if isinstance(m, Gemma4TextExperts))
    n_unfused_before = sum(1 for m in model.modules() if isinstance(m, UnfusedGemma4TextExperts))
    print(f"  before: {n_before} Gemma4TextExperts, {n_unfused_before} Unfused")

    unfuse_gemma4_model(model, free_after=False)

    n_after = sum(1 for m in model.modules() if isinstance(m, Gemma4TextExperts))
    n_unfused_after = sum(1 for m in model.modules() if isinstance(m, UnfusedGemma4TextExperts))
    print(f"  after:  {n_after} Gemma4TextExperts, {n_unfused_after} Unfused")
    assert n_after == 0, f"expected 0 fused, got {n_after}"
    assert n_unfused_after == 3, f"expected 3 unfused, got {n_unfused_after}"
    print("  PASS")


if __name__ == "__main__":
    print("=== test_forward_equivalence (fp32, CPU/GPU) ===")
    test_forward_equivalence(dtype=torch.float32, atol=1e-5, rtol=1e-4)

    print("\n=== test_forward_equivalence (bf16, GPU) ===")
    if torch.cuda.is_available():
        test_forward_equivalence(dtype=torch.bfloat16, atol=5e-3, rtol=5e-3, device="cuda")
    else:
        print("  SKIP (no CUDA/HIP available)")

    print("\n=== test_forward_equivalence (large config, fp32) ===")
    test_forward_equivalence(
        num_experts=128, hidden_dim=256, intermediate_dim=512,
        num_tokens=64, top_k=4, dtype=torch.float32, atol=1e-5, rtol=1e-4
    )

    print("\n=== test_unfuse_walker ===")
    test_unfuse_walker()

    print("\n=== ALL TESTS PASSED ===")
