"""Find PEFT-targetable module names in E4B."""
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
m = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-E4B-it", quantization_config=bnb_config, device_map="auto",
)

# Sample one decoder layer's attention to see the wrapping pattern
for name, mod in m.named_modules():
    if "layers.0" in name and ("q_proj" in name or "k_proj" in name or "self_attn" in name):
        print(f"  {name}: {type(mod).__name__}")
    if "layers.1" in name:
        break

# How many of each leaf type?
from collections import Counter
leaf_types = Counter()
for name, mod in m.named_modules():
    if not list(mod.children()):
        leaf_types[type(mod).__name__] += 1
print("\nLeaf module types:")
for t, c in leaf_types.most_common(15):
    print(f"  {t}: {c}")

# Suggest target_modules pattern that hits Linear4bit inside Gemma4ClippableLinear
print("\nLinear4bit module name samples (first 12):")
import bitsandbytes as bnb
for i, (name, mod) in enumerate(m.named_modules()):
    if isinstance(mod, bnb.nn.Linear4bit):
        print(f"  {name}")
        if i > 30:
            break

# Find unique attention-related Linear4bit suffixes
print("\nUnique attention Linear4bit suffixes:")
suffixes = set()
for name, mod in m.named_modules():
    if isinstance(mod, bnb.nn.Linear4bit) and ("attn" in name or "q_proj" in name or "k_proj" in name or "v_proj" in name or "o_proj" in name):
        suffix = ".".join(name.split(".")[-3:])
        suffixes.add(suffix)
for s in sorted(suffixes):
    print(f"  {s}")
