"""NVFP4 quantization for Gemma 4 26B MoE — all-in-one.

Handles:
1. Registering Gemma4TextExperts quantization plugin (fused 3D → nn.Linear)
2. Excluding vision encoder from quantization
3. Quantizing with modelopt NVFP4
4. Exporting with correct key names for vLLM (moe.experts.E.proj format)

Usage:
    python quantize_gemma4_moe.py
    python quantize_gemma4_moe.py --qformat nvfp4_awq
"""

import argparse
import copy
import glob
import json
import os
import re
import time
from collections import defaultdict

import torch
import torch.nn as nn

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn import QuantModule, QuantModuleRegistry
from modelopt.torch.utils.dataset_utils import create_forward_loop
from safetensors.torch import load_file, save_file


# ── Gemma4 Expert QuantModule ────────────────────────────────────────────────

class _Gemma4ExpertModule(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)


class _QuantGemma4TextExperts(QuantModule):
    def _setup(self):
        from accelerate import init_empty_weights
        dtype, device = self.gate_up_proj.dtype, self.gate_up_proj.device

        def _copy_weight(module, weight):
            module.to_empty(device=device)
            with torch.no_grad():
                module.weight.data = weight.detach().data.to(dtype=dtype, device=device)

        expert_dim = self.intermediate_dim
        with init_empty_weights():
            expert_modules = nn.ModuleList([
                _Gemma4ExpertModule(self.hidden_dim, expert_dim)
                for _ in range(self.num_experts)
            ])

        for idx in range(self.num_experts):
            _copy_weight(expert_modules[idx].gate_proj, self.gate_up_proj[idx, :expert_dim, :])
            _copy_weight(expert_modules[idx].up_proj, self.gate_up_proj[idx, expert_dim:, :])
            _copy_weight(expert_modules[idx].down_proj, self.down_proj[idx])

        delattr(self, "gate_up_proj")
        delattr(self, "down_proj")
        for idx in range(self.num_experts):
            self.add_module(str(idx), expert_modules[idx])

    def __len__(self):
        return self.num_experts

    def __iter__(self):
        for idx in range(self.num_experts):
            yield getattr(self, str(idx))

    def __getitem__(self, idx):
        return getattr(self, str(int(idx)))

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            with torch.no_grad():
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            expert = self[expert_idx]
            gate = expert.gate_proj(current_state)
            up = expert.up_proj(current_state)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = expert.down_proj(current_hidden_states)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states


# ── Post-export key fixer ────────────────────────────────────────────────────

def fix_keys_for_vllm(model_dir):
    """Rename exported keys to match vLLM's Gemma4 modelopt weight loader.

    modelopt exports:  model.language_model.layers.N.experts.E.gate_proj.weight
    vLLM expects:      model.language_model.layers.N.moe.experts.E.gate_proj.weight
    """
    shard_files = sorted(glob.glob(f"{model_dir}/model-*.safetensors"))
    print(f"  Fixing keys for vLLM ({len(shard_files)} shards)...")

    all_tensors = {}
    for f in shard_files:
        all_tensors.update(load_file(f))

    new_tensors = {}
    renamed = 0
    for key, tensor in all_tensors.items():
        # Add moe. prefix before experts
        new_key = re.sub(r"\.experts\.(\d+)\.", r".moe.experts.\1.", key)
        if new_key != key:
            renamed += 1
        new_tensors[new_key] = tensor

    print(f"  Renamed {renamed} keys (added moe. prefix)")

    # Save
    for f in shard_files:
        os.remove(f)

    keys_by_size = sorted(new_tensors.items(),
                          key=lambda x: x[1].numel() * x[1].element_size(), reverse=True)
    shards, current, size = [], {}, 0
    for k, t in keys_by_size:
        b = t.numel() * t.element_size()
        if size + b > 8e9 and current:
            shards.append(current); current, size = {}, 0
        current[k] = t; size += b
    if current:
        shards.append(current)

    index = {"metadata": {"total_size": sum(t.numel() * t.element_size() for t in new_tensors.values())},
             "weight_map": {}}
    for i, s in enumerate(shards):
        fn = f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"
        save_file(s, f"{model_dir}/{fn}")
        for k in s:
            index["weight_map"][k] = fn

    with open(f"{model_dir}/model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    print(f"  {len(new_tensors)} keys in {len(shards)} shards")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Quantize Gemma 4 26B A4B to NVFP4")
    parser.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    parser.add_argument("--output", default="Gemma-4-26B-A4B-it-NVFP4")
    parser.add_argument("--qformat", default="nvfp4", help="nvfp4, nvfp4_awq")
    parser.add_argument("--calib-samples", type=int, default=4096)
    parser.add_argument("--calib-seq-len", type=int, default=1024)
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  Gemma 4 26B A4B → NVFP4 (all-in-one)")
    print(f"{'='*60}\n")

    # Step 1: Register plugin
    print("[1/6] Registering Gemma4TextExperts plugin...")
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts
    if Gemma4TextExperts not in QuantModuleRegistry._registry:
        QuantModuleRegistry.register({Gemma4TextExperts: "hf.Gemma4TextExperts"})(
            _QuantGemma4TextExperts
        )
        print("  Registered!")
    else:
        print("  Already registered")

    # Step 2: Load
    print("\n[2/6] Loading model...")
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
    )
    print(f"  Loaded in {time.time()-t0:.0f}s")

    # Step 3: Calibration
    print(f"\n[3/6] Preparing calibration ({args.calib_samples} samples)...")
    forward_loop = create_forward_loop(
        model=model, dataset_name="cnn_dailymail", tokenizer=tokenizer,
        batch_size=16, num_samples=args.calib_samples,
        max_sample_length=args.calib_seq_len, device="cuda:0",
    )

    # Step 4: Quantize — exclude vision encoder
    from modelopt.torch.quantization.config import _nvfp4_selective_quant_cfg
    quant_cfg = copy.deepcopy(
        {
            "nvfp4": mtq.NVFP4_DEFAULT_CFG,
            "nvfp4_awq": mtq.NVFP4_AWQ_LITE_CFG,
            "nvfp4_w4a16": _nvfp4_selective_quant_cfg(["*"], weight_only=True),
        }.get(args.qformat, mtq.NVFP4_DEFAULT_CFG)
    )
    quant_cfg["quant_cfg"]["*vision*"] = {"enable": False}
    quant_cfg["quant_cfg"]["*embed_vision*"] = {"enable": False}
    quant_cfg["quant_cfg"]["*multi_modal_projector*"] = {"enable": False}
    # Let experts calibrate naturally — don't force uniform routing
    # More samples (4096) gives better organic coverage than forced activation

    print(f"\n[4/6] Quantizing ({args.qformat}, vision excluded)...")
    t1 = time.time()
    model = mtq.quantize(model, quant_cfg, forward_loop=forward_loop)
    print(f"  Quantized in {time.time()-t1:.0f}s")

    # Step 5: Export
    print(f"\n[5/6] Exporting to {args.output}...")
    os.makedirs(args.output, exist_ok=True)
    from modelopt.torch.export import export_hf_checkpoint
    export_hf_checkpoint(model, dtype=torch.bfloat16, export_dir=args.output)
    tokenizer.save_pretrained(args.output)

    # Copy processor/preprocessor configs from original model
    import shutil
    src_dir = None
    cache_pattern = f"{os.path.expanduser('~')}/.cache/huggingface/hub/models--{args.model.replace('/', '--')}/snapshots/*"
    src_dirs = glob.glob(cache_pattern)
    if src_dirs:
        src_dir = src_dirs[0]
        for cfg_file in ("processor_config.json", "preprocessor_config.json",
                         "special_tokens_map.json", "chat_template.json"):
            src = os.path.join(src_dir, cfg_file)
            if os.path.exists(src):
                shutil.copy(src, args.output)
                print(f"  Copied {cfg_file}")
    print(f"  Tokenizer + configs saved")

    # Step 6: Fix keys for vLLM
    print(f"\n[6/6] Fixing keys for vLLM...")
    fix_keys_for_vllm(args.output)

    total = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, fs in os.walk(args.output) for f in fs
    )

    print(f"\n{'='*60}")
    print(f"  DONE!")
    print(f"  Output:   {args.output}")
    print(f"  Size:     {total/1e9:.2f} GB")
    print(f"  Original: ~49 GB (BF16)")
    print(f"  Ratio:    {49/max(total/1e9,0.1):.1f}x")
    print(f"  Time:     {time.time()-t0:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
