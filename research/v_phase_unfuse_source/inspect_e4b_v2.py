"""Inspect E4B language_model attention modules with type info."""
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

# Print attention module types in language_model layer 0
print("=== language_model layer 0 attention ===")
for name, mod in m.named_modules():
    if "language_model.layers.0.self_attn" in name:
        print(f"  {name}: {type(mod).__name__}")

# Distinguish vision vs language tower q_projs
print("\n=== q_proj instances by tower ===")
import bitsandbytes as bnb
language_q = []
vision_q = []
audio_q = []
for name, mod in m.named_modules():
    if name.endswith(".q_proj") or name.endswith(".q_proj.linear"):
        if "language_model" in name:
            language_q.append((name, type(mod).__name__))
        elif "vision_tower" in name:
            vision_q.append((name, type(mod).__name__))
        elif "audio_tower" in name:
            audio_q.append((name, type(mod).__name__))

print(f"language_model q_proj: {len(language_q)}")
for n, t in language_q[:3]:
    print(f"  {n}: {t}")
print(f"vision_tower q_proj: {len(vision_q)}")
for n, t in vision_q[:3]:
    print(f"  {n}: {t}")
print(f"audio_tower q_proj: {len(audio_q)}")
for n, t in audio_q[:3]:
    print(f"  {n}: {t}")
