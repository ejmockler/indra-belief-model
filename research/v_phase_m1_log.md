# M1 build log — ROCm training env

Date: 2026-05-05
Host: noot@100.97.101.59 (Radeon RX 7900 XTX, gfx1100, ROCm 6.4.1)

## Phase A: venv (DONE)

- Created `~/venvs/v-phase` (Python 3.10.12)
- Upgraded pip 22.0.2 → 26.1.1
- Installed ninja 1.13.0

## Phase B: PyTorch (DONE)

- Installed `torch==2.9.1+rocm6.4` from `https://download.pytorch.org/whl/rocm6.4`
- Bundled `pytorch-triton-rocm 3.5.1`
- Verified: `torch.cuda.is_available()=True`, device 0 = "Radeon RX 7900 XTX" capability (11, 0)
- bf16 matmul on GPU works

## Phase C: Supporting libs (DONE)

- numpy 2.2.6, transformers 5.8.0, peft 0.19.1, accelerate 1.13.0, datasets 4.8.5, safetensors 0.7.0, sentencepiece 0.2.1, protobuf 7.34.1
- Verified `transformers.models.gemma4.modeling_gemma4.Gemma4TextExperts` exists — M3 unfuse target available

## Phase D: bitsandbytes build (IN PROGRESS — second attempt)

### First attempt: upstream `bitsandbytes-foundation/bitsandbytes:multi-backend-refactor` (commit c3eac42) — FAILED

- cmake configure: clean (HIP 19.0.0, target gfx1100, hipblas 2.4.0)
- make: clean compile (only benign loop-unroll warnings)
- pip install -e .: success
- **Runtime FAIL on import**: `OSError: ... undefined symbol: _Z36__device_stub__kOptimizer32bit1StateI12hip_bfloat16Li1EE...`
- Root cause: missing template instantiations for `kOptimizer32bit1State<hip_bfloat16, 1/2/4>` — the `Li5EE` and `Li0EE` instantiations are present, but indices 1, 2, 4 (which correspond to Adam, Momentum, RMSprop optimizers for bf16) are not emitted by the HIP compiler. The `__device_stub__` host-side wrappers reference them, breaking dlopen.
- This is a regression in the upstream multi-backend-refactor branch's hipification of bf16 optimizer kernels.

### Second attempt: `ROCm/bitsandbytes:rocm_enabled` (commit 4fa939b3, Sep 2025) — IN PROGRESS

- AMD's fork has gfx1100 explicitly in `CMakeLists.txt` default architectures
- Building with `-DBNB_ROCM_ARCH="gfx1100"`

## Decision points hit

- Stayed with PyTorch ROCm 6.4 (matches host ROCm 6.4.1 exactly) — no compatibility concerns
- Used upstream multi-backend-refactor first per official docs; pivoted to AMD fork after symbol-resolution failure

## Next on success

- Phase E: 4-bit roundtrip + Linear4bit forward/backward smoke
- Phase F: install tqdm/rich/extras for trainer if any
- Then: M2 (Llama-3.2 small QLoRA smoke)

## Backup if AMD fork fails too

- Skip bnb 8-bit optimizers entirely (use plain AdamW for LoRA params; doesn't need kernels)
- If even 4-bit Linear has missing symbols, try:
  - Older AMD fork commit
  - `multi-backend-refactor` with `-DBUILD_BNB_OPTIMIZERS=OFF` if such a flag exists (TBD)
  - Last resort: accept E4B pivot — no MoE = no QLoRA pressure, can use bf16 LoRA without bnb
