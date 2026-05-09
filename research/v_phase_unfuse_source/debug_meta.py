"""Inspect which parameters end up on meta device after low_cpu_mem_usage load."""
import torch
from transformers import AutoModelForCausalLM

print('loading bf16 to CPU...')
m = AutoModelForCausalLM.from_pretrained(
    'google/gemma-4-26B-A4B-it',
    dtype=torch.bfloat16,
    device_map='cpu',
    low_cpu_mem_usage=True,
)
print('loaded.')

meta_params = [(n, tuple(p.shape), p.dtype) for n, p in m.named_parameters() if p.is_meta]
non_meta = [(n, tuple(p.shape), p.device) for n, p in m.named_parameters() if not p.is_meta]
meta_buffers = [(n, tuple(b.shape), b.dtype) for n, b in m.named_buffers() if b.is_meta]

print(f'TOTAL params: {sum(1 for _ in m.named_parameters())}, meta: {len(meta_params)}, non-meta: {len(non_meta)}')
print(f'meta buffers: {len(meta_buffers)}')

print('\nFirst 15 meta params:')
for n, s, d in meta_params[:15]:
    print(f'  {n}: {s} {d}')

print('\nFirst 5 non-meta params:')
for n, s, dev in non_meta[:5]:
    print(f'  {n}: {s} {dev}')

print('\nMeta buffers:')
for n, s, d in meta_buffers[:10]:
    print(f'  {n}: {s} {d}')
