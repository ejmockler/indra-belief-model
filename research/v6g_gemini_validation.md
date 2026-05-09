# V6g — Gemini model validation on U2 per-probe gold

Date: 2026-05-07
U2 records: 482 | tasks per model: 1600
Composite gate (per probe): saturating lift `micro ≥ mfc + 0.30·(1-mfc)`, macro-recall `≥ 0.65`, min-class-recall `≥ 0.30` (for classes with support `≥ 5`). 1-class probes (mfc=1.0) emit `INSUFFICIENT_GOLD`.

## Per-probe summary (micro / macro / Δ vs most-frequent-class baseline)

| Probe | gemini-3.1-pro-preview |
|---|---|
| relation_axis | micro=0.773 / macro=0.599 / Δmfc=+0.077 ✗ FAIL |
| subject_role | micro=0.826 / macro=0.685 / Δmfc=-0.141 ✗ FAIL |
| object_role | micro=0.874 / macro=0.828 / Δmfc=-0.087 ✗ FAIL |
| scope | micro=0.899 / macro=0.710 / Δmfc=-0.049 ✗ FAIL |
| verify_grounding | — |

## Gate criteria per probe

Each criterion shows pass/fail with the threshold inline. `—` = not applicable for 1-class probes.


### gemini-3.1-pro-preview

| Probe | lift (micro ≥ mfc+Δ·(1-mfc)) | macro-recall ≥ floor | min-class recall | verdict |
|---|---|---|---|---|
| relation_axis | ✗ 0.773 < 0.787 (mfc=0.696+0.30·0.304) | ✗ 0.599 < 0.65 | ✗ < 0.30: direct_sign_mismatch(n=16, r=0.250) | FAIL |
| subject_role | ✗ 0.826 < 0.977 (mfc=0.967+0.30·0.033) | ✓ 0.685 ≥ 0.65 | ✓ all classes (support ≥ 5) recall ≥ 0.30 | FAIL |
| object_role | ✗ 0.874 < 0.973 (mfc=0.961+0.30·0.039) | ✓ 0.828 ≥ 0.65 | ✓ all classes (support ≥ 5) recall ≥ 0.30 | FAIL |
| scope | ✗ 0.899 < 0.964 (mfc=0.948+0.30·0.052) | ✓ 0.710 ≥ 0.65 | ✓ all classes (support ≥ 5) recall ≥ 0.30 | FAIL |
| verify_grounding | — | — | — | — |

## Trivial baseline (most-frequent class) per probe

| Probe | n | top gold classes (support) | mfc baseline acc |
|---|---|---|---|
| relation_axis | 392 | direct_sign_match=273, no_relation=53, direct_axis_mismatch=50, direct_sign_mismatch=16 | 0.696 |
| subject_role | 460 | present_as_subject=445, absent=15 | 0.967 |
| object_role | 460 | present_as_object=442, absent=18 | 0.961 |
| scope | 288 | asserted=273, hedged=8, negated=7 | 0.948 |
| verify_grounding | — | — | — |

## Per-class precision / recall / F1


### gemini-3.1-pro-preview


**relation_axis** (n=392, micro=0.773, macro=0.599, mfc=0.696, err=0)

| Class | Support | Predicted | Precision | Recall | F1 |
|---|---|---|---|---|---|
| direct_sign_match | 273 | 259 | 0.896 | 0.850 | 0.872 |
| no_relation | 53 | 75 | 0.520 | 0.736 | 0.609 |
| direct_axis_mismatch | 50 | 52 | 0.538 | 0.560 | 0.549 |
| direct_sign_mismatch | 16 | 6 | 0.667 | 0.250 | 0.364 |

**subject_role** (n=460, micro=0.826, macro=0.685, mfc=0.967, err=0)

| Class | Support | Predicted | Precision | Recall | F1 |
|---|---|---|---|---|---|
| present_as_subject | 445 | 378 | 0.984 | 0.836 | 0.904 |
| absent | 15 | 34 | 0.235 | 0.533 | 0.327 |
| present_as_decoy | 0 | 22 | 0.000 | — | — |
| present_as_mediator | 0 | 4 | 0.000 | — | — |
| present_as_object | 0 | 22 | 0.000 | — | — |

**object_role** (n=460, micro=0.874, macro=0.828, mfc=0.961, err=0)

| Class | Support | Predicted | Precision | Recall | F1 |
|---|---|---|---|---|---|
| present_as_object | 442 | 391 | 0.992 | 0.878 | 0.932 |
| absent | 18 | 42 | 0.333 | 0.778 | 0.467 |
| present_as_decoy | 0 | 12 | 0.000 | — | — |
| present_as_mediator | 0 | 3 | 0.000 | — | — |
| present_as_subject | 0 | 12 | 0.000 | — | — |

**scope** (n=288, micro=0.899, macro=0.710, mfc=0.948, err=0)

| Class | Support | Predicted | Precision | Recall | F1 |
|---|---|---|---|---|---|
| asserted | 273 | 253 | 0.988 | 0.916 | 0.951 |
| hedged | 8 | 4 | 1.000 | 0.500 | 0.667 |
| negated | 7 | 6 | 0.833 | 0.714 | 0.769 |
| abstain | 0 | 22 | 0.000 | — | — |
| asserted_with_condition | 0 | 3 | 0.000 | — | — |

## Verdict

Composite gate: lift Δ=0.30, macro≥0.65, per-class recall≥0.30 for classes with support≥5. 1-class probes (mfc=1.0) emit INSUFFICIENT_GOLD.

- **gemini-3.1-pro-preview**: FAIL
    - relation_axis: FAIL [lift micro=0.773<0.787 (mfc=0.696, Δ·(1-mfc)=+0.091); macro=0.599<0.65; min-class-recall<0.30: direct_sign_mismatch(n=16,r=0.250)]
    - subject_role: FAIL [lift micro=0.826<0.977 (mfc=0.967, Δ·(1-mfc)=+0.010)]
    - object_role: FAIL [lift micro=0.874<0.973 (mfc=0.961, Δ·(1-mfc)=+0.012)]
    - scope: FAIL [lift micro=0.899<0.964 (mfc=0.948, Δ·(1-mfc)=+0.016)]

## Cost ladder (full 400K corpus, batch API)

| Model | Batch cost | Quality |
|---|---|---|
| gemini-2.5-flash-lite | ~$10 | baseline |
| gemini-2.5-flash | ~$43 | balanced |
| gemini-3.1-pro-preview | ~$200 | top tier |
