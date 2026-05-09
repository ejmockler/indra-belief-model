"""V6g — validate Gemini models as labelers on U2 per-probe gold.

For each U2-gold-labeled holdout record × each of the 5 probes × each of N
Gemini models, render the production probe prompt (system + few-shots +
user message), call Gemini, parse the JSON answer, compare to U2 gold.

Models tested in parallel:
  - gemini-2.5-flash-lite (cheapest baseline)
  - gemini-2.5-flash       (well-balanced)
  - gemini-3.1-flash-preview (newest, preview)

Gate: composite per-probe — lift over most-frequent-class baseline,
      macro-recall floor, and per-class recall floor for classes with
      sufficient support. 1-class probes emit `INSUFFICIENT_GOLD`. See
      research/v6g_gate_design.md and `Gate` dataclass for thresholds.

Auth: requires GEMINI_API_KEY env var.

Output:
  research/v6g_gemini_validation.md   — accuracy table + verdict
  data/v_phase/v6g_responses.jsonl    — raw responses for audit
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from google import genai
from google.genai import types


# ── Composite gate ──
#
# The previous gate (`micro ≥ 0.90 AND macro-recall ≥ 0.75 per probe`) is
# inadequate when classes are heavily imbalanced. See
# research/v6g_gate_design.md for full rationale. Four criteria are now
# evaluated per probe; verdict per probe = AND of criteria, with a special
# escape hatch for 1-class probes.

@dataclass(frozen=True)
class Gate:
    """Composite-gate thresholds for per-probe acceptance.

    delta_mfc:        required fraction of error-rate-headroom that the
                       curator's micro must close above the most-frequent-
                       class baseline. Saturating formula:
                         `micro ≥ mfc + delta_mfc · (1 - mfc)`
                       — at low mfc this is roughly an absolute lift; at
                       high mfc the bar shrinks so the criterion stays
                       reachable. Replaces the previous additive formula
                       which was unreachable when `mfc + delta_mfc > 1.0`.
    macro_floor:      required macro-recall floor.
    min_class_recall: required per-class recall floor for any gold class
                       with `support ≥ min_support`.
    min_support:      a class is held to `min_class_recall` only when its
                       gold support meets this threshold; smaller classes
                       are exempt to avoid noisy CIs forcing failure.
    """

    delta_mfc: float = 0.30
    macro_floor: float = 0.65
    min_class_recall: float = 0.30
    min_support: int = 5


def evaluate_gate(mp: dict, gate: Gate) -> dict:
    """Apply the composite gate to a single probe's metrics.

    Returns a dict with:
      - one_class:        True iff mfc == 1.0 and macro == micro (no minority gold)
      - lift_pass:        micro ≥ mfc + delta_mfc
      - macro_pass:       macro ≥ macro_floor
      - min_class_pass:   recall ≥ min_class_recall for every class with support ≥ min_support
      - failing_classes:  list of (class, support, recall) tuples that violated min_class
      - verdict:          "PASS" | "FAIL" | "INSUFFICIENT_GOLD"
      - reasons:          list of human-readable failure-reason strings (empty on PASS)
    """
    micro = mp["micro"]
    macro = mp["macro"]
    mfc = mp["mfc"]
    per_class = mp["per_class"]

    one_class = (mfc >= 0.999999) and (abs(macro - micro) < 1e-9)
    if one_class:
        return {
            "one_class": True,
            "lift_pass": None,
            "macro_pass": None,
            "min_class_pass": None,
            "failing_classes": [],
            "verdict": "INSUFFICIENT_GOLD",
            "reasons": [],
        }

    lift_target = mfc + gate.delta_mfc * (1.0 - mfc)
    lift_pass = micro >= lift_target
    macro_pass = macro >= gate.macro_floor

    failing_classes: list[tuple[str, int, float]] = []
    for c, d in per_class.items():
        support = d.get("support", 0) or 0
        recall = d.get("recall")
        if support < gate.min_support or recall is None:
            continue
        if recall < gate.min_class_recall:
            failing_classes.append((c, support, recall))
    min_class_pass = len(failing_classes) == 0

    reasons: list[str] = []
    if not lift_pass:
        reasons.append(
            f"lift micro={micro:.3f}<{lift_target:.3f} "
            f"(mfc={mfc:.3f}, Δ·(1-mfc)={gate.delta_mfc * (1.0 - mfc):+.3f})"
        )
    if not macro_pass:
        reasons.append(f"macro={macro:.3f}<{gate.macro_floor:.2f}")
    if not min_class_pass:
        worst = ", ".join(
            f"{c}(n={s},r={r:.3f})" for c, s, r in failing_classes
        )
        reasons.append(
            f"min-class-recall<{gate.min_class_recall:.2f}: {worst}"
        )

    verdict = "PASS" if (lift_pass and macro_pass and min_class_pass) else "FAIL"
    return {
        "one_class": False,
        "lift_pass": lift_pass,
        "macro_pass": macro_pass,
        "min_class_pass": min_class_pass,
        "failing_classes": failing_classes,
        "verdict": verdict,
        "reasons": reasons,
    }

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ── Class indices, must match probes/types.py + grounding.py ──

RELATION_AXIS_CLASSES = ["direct_sign_match", "direct_amount_match", "direct_sign_mismatch",
                          "direct_axis_mismatch", "direct_partner_mismatch", "via_mediator",
                          "via_mediator_partial", "no_relation"]
ROLE_CLASSES = ["present_as_subject", "present_as_object", "present_as_mediator",
                "present_as_decoy", "absent"]
SCOPE_CLASSES = ["asserted", "hedged", "asserted_with_condition", "negated", "abstain"]
GROUNDING_CLASSES = ["mentioned", "equivalent", "not_present", "uncertain"]

PROBES = ["relation_axis", "subject_role", "object_role", "scope", "verify_grounding"]

PROBE_CLASSES: dict[str, list[str]] = {
    "relation_axis": RELATION_AXIS_CLASSES,
    "subject_role": ROLE_CLASSES,
    "object_role": ROLE_CLASSES,
    "scope": SCOPE_CLASSES,
    "verify_grounding": GROUNDING_CLASSES,
}


# ── U2 gold mapping (mirrors V7a) ──
#
# Audit history (see research/v6g_gold_audit.md for full record):
#
# Original mapping forced subject_role/object_role = "absent" for
# gold_tags `no_relation`, `grounding`, `entity_boundaries`. Pro rationales
# showed many of those records actually have the entity present in
# evidence under an alias (TRAIL=TNFSF10, Astrin=SPAG5, INI1=SMARCB1,
# iNOS=NOS2, eIF2alpha=EIF2S1, COX-2=PTGS2, etc.), so forcing "absent"
# poisoned the role-probe metric. The original audit picked Option B —
# set those entries to None (skip from accuracy calc).
#
# Revision 2026-05-07 (Option A — content-conditional resolver):
#   Option B accidentally collapsed role-probe gold to a single class
#   (354 present_as_subject / 354 present_as_object, no minority gold),
#   so the trivial baseline = 1.00 and the composite gate emits
#   INSUFFICIENT_GOLD. To restore measurement we replace `None` with the
#   sentinel `"__RESOLVE__"` for the three target tags and resolve
#   per-record at task-build time via `_resolve_role_gold(...)`:
#
#     1. case-insensitive substring of the entity in evidence_text
#     2. case-insensitive substring of any alias in `_ROLE_ALIAS_MAP`
#        (HGNC↔display-name pairs harvested from Pro's V6g rationales —
#        ~80 entries; see _ROLE_ALIAS_MAP below for the full list)
#     3. miss → "absent" for all three tags. For `grounding` and
#        `entity_boundaries`, this is high-confidence (those tags mean
#        the entity name itself is suspect); for `no_relation` it means
#        the entity wasn't mentioned at all.
#     4. hit → present_as_subject for the subject_role probe,
#        present_as_object for the object_role probe. We do NOT try to
#        detect mediator / decoy roles — that requires per-record
#        curation we don't have. This biases the role-class assumption
#        toward subject/object in present cases, but that matches the
#        original-claim role and the production probe's expected output.
#
# Resolver tally (482 records, 106 under target tags) — see audit doc:
#     no_relation       (N=53):  S→present=49 absent=4 | O→present=51 absent=2
#     grounding         (N=33):  S→present=24 absent=9 | O→present=20 absent=13
#     entity_boundaries (N=20):  S→present=18 absent=2 | O→present=18 absent=2
#   Net: ~31 "absent" gold restored across both role probes — enough to
#   exit the 1-class trap so the composite gate measures something.

_ROLE_RESOLVE_SENTINEL = "__RESOLVE__"

GOLD_TAG_TO_PROBE_CLASS = {
    "correct":           {"relation_axis": "direct_sign_match",     "subject_role": "present_as_subject",   "object_role": "present_as_object",    "scope": "asserted",  "verify_grounding": "mentioned"},
    "polarity":          {"relation_axis": "direct_sign_mismatch",  "subject_role": "present_as_subject",   "object_role": "present_as_object",    "scope": None,         "verify_grounding": "mentioned"},
    "act_vs_amt":        {"relation_axis": "direct_axis_mismatch",  "subject_role": "present_as_subject",   "object_role": "present_as_object",    "scope": None,         "verify_grounding": "mentioned"},
    "wrong_relation":    {"relation_axis": "direct_axis_mismatch",  "subject_role": "present_as_subject",   "object_role": "present_as_object",    "scope": None,         "verify_grounding": "mentioned"},
    "no_relation":       {"relation_axis": "no_relation",           "subject_role": _ROLE_RESOLVE_SENTINEL,  "object_role": _ROLE_RESOLVE_SENTINEL, "scope": None,         "verify_grounding": None},
    "grounding":         {"relation_axis": None,                    "subject_role": _ROLE_RESOLVE_SENTINEL,  "object_role": _ROLE_RESOLVE_SENTINEL, "scope": None,         "verify_grounding": "not_present"},
    "entity_boundaries": {"relation_axis": None,                    "subject_role": _ROLE_RESOLVE_SENTINEL,  "object_role": _ROLE_RESOLVE_SENTINEL, "scope": None,         "verify_grounding": "not_present"},
    "hypothesis":        {"relation_axis": None,                    "subject_role": "present_as_subject",   "object_role": "present_as_object",    "scope": "hedged",     "verify_grounding": "mentioned"},
    "negative_result":   {"relation_axis": None,                    "subject_role": "present_as_subject",   "object_role": "present_as_object",    "scope": "negated",    "verify_grounding": "mentioned"},
    "mod_site":          {"relation_axis": None,                    "subject_role": None,                   "object_role": None,                   "scope": None,         "verify_grounding": "mentioned"},
    "agent_conditions":  {"relation_axis": None,                    "subject_role": None,                   "object_role": None,                   "scope": None,         "verify_grounding": "mentioned"},
}


# ── Static alias map for role-gold resolver ──
#
# Pairs harvested from Pro's V6g rationales on U2 holdout records under
# `no_relation` / `grounding` / `entity_boundaries`. Symmetric: when the
# gold name is the HGNC symbol we list the display aliases the literature
# uses, and vice versa. Used by `_resolve_role_gold` along with a
# punctuation/case-insensitive normalized-substring check (handles
# BMP2↔BMP-2, MMP1↔MMP-1, IL22↔IL-22, etc. without per-pair entries).

_ROLE_ALIAS_MAP: dict[str, list[str]] = {
    # Pro-rationale-attested HGNC↔display pairs (V6g audit 2026-05-07)
    "TNFSF10": ["TRAIL"],            "TRAIL": ["TNFSF10"],
    "SPAG5": ["Astrin"],             "Astrin": ["SPAG5"],
    "SMARCB1": ["INI1", "BAF47", "SNF5", "hSNF5"],
    "INI1": ["SMARCB1"],
    "NOS2": ["iNOS", "NOSII"],       "iNOS": ["NOS2"],
    "EIF2S1": ["eIF2alpha", "eIF2-alpha", "eIF2"],
    "eIF2alpha": ["EIF2S1"],
    "PTGS2": ["COX-2", "COX2", "cyclooxygenase-2"],
    "COX-2": ["PTGS2"],
    "MAPK1": ["ERK2", "ERK", "p42 MAPK", "p42"],
    "MAPK3": ["ERK1", "p44 MAPK", "p44"],
    "MAPK8": ["JNK1", "JNK"],
    "MAPK9": ["JNK2"],
    "MAP3K7": ["TAK1"],              "TAK1": ["MAP3K7"],
    "AKT1": ["AKT", "PKB"],          "AKT": ["AKT1", "PKB"],
    "NFKB1": ["NFkB", "NF-kB", "NF-kappaB", "nuclear factor-kappaB", "nuclear factor-κB", "p50"],
    "NFkappaB": ["NFKB1", "NF-kB", "NF-kappaB", "nuclear factor-kappaB", "nuclear factor-κB", "p50", "p65"],
    "CXCR7": ["ACKR3"],              "ACKR3": ["CXCR7"],
    "PACAP": ["ADCYAP1"],            "ADCYAP1": ["PACAP"],
    "PBK": ["TOPK"],                 "TOPK": ["PBK"],
    "HSPB1": ["Hsp27", "HSP27"],     "Hsp27": ["HSPB1"],
    "ERCC3": ["XPB"],                "XPB": ["ERCC3"],
    "WRNIP1": ["WHIP"],
    "CCN2": ["CTGF"],                "CTGF": ["CCN2"],
    "TNFRSF14": ["HVEM"],            "HVEM": ["TNFRSF14"],
    "KAT5": ["Tip60"],
    "PROM1": ["CD133"],
    "CENPJ": ["CPAP"],
    "PTK6": ["Brk"],
    "PPIE": ["Cyp33"],
    "APBB1": ["Fe65"],
    "NTRK1": ["Trk", "TrkA"],
    "UBE2I": ["Ubc9"],
    "STK3": ["MST2"],
    "NCOR2": ["SMRT"],
    "PRNP": ["PrP", "PrPc", "prion"],
    "PRND": ["Dpl", "doppel"],
    "TNFSF11": ["RANKL", "sRANKL", "OPGL"],
    "PTK2": ["FAK", "focal adhesion kinase"],
    "PTK2B": ["Pyk2"],
    "CFL1": ["cofilin", "cofilin-1"],
    "NOS3": ["eNOS"],
    "NOS1": ["nNOS"],
    "TNNI3": ["cTnI", "cardiac troponin I", "troponin I"],
    "FN1": ["fibronectin", "FN"],
    "Integrins": ["integrin", "αv", "beta6", "αvβ6", "α5β1", "β6 integrin"],
    "NADPH_oxidase": ["Nox", "NOX2", "Nox2", "NADPH oxidase"],
    "CYBB": ["Nox2", "NOX2"],
    "IL20RA": ["IL-20R1", "IL20R1"],
    "LGALS1": ["galectin-1"],
    "KIFC1": ["HSET"],
    "SOSTDC1": ["ectodin", "USAG-1"],
    "PPP1R15A": ["Gadd34", "GADD34"],
    "C4BPA": ["C4BP"],
    "SH2D1A": ["SAP", "SLAM-associated protein"],
    "BCL2L1": ["Bcl-x", "Bcl-xL", "BCL-X"],
    "BCL2": ["Bcl-2", "bcl-2"],
    "TGFB": ["TGF-beta", "TGF-β", "TGFbeta"],
    "TGFB1": ["TbetaRI"],
    "CASP8": ["caspase-8", "caspase 8"],
    "MAPK8IP1": ["JIP1", "JSAP1"],
    "MKNK1": ["Mnk1", "MNK1"],
    "MYBL2": ["B-Myb"],
    # additional pairs from final-miss inspection
    "Interferon": ["IFN", "IFN-α", "IFN-β", "IFN-γ", "interferon", "type I IFN"],
    "CDKN1A": ["p21", "p21 Cip1", "p21Cip1", "p21Waf1"],
    "CDKN1B": ["p27", "p27 Kip1", "p27Kip1"],
    "TP53": ["p53"],
    "POU5F1": ["OCT4", "Oct-4", "Oct3/4"],
    "INSR": ["insulin receptor"],
    "F2RL1": ["PAR-2", "PAR2"],
    "TNFRSF10A": ["DR4"],
    "TNFRSF10B": ["DR5"],
    "PARD3": ["Par3"],
    "PARD6A": ["Par6"],
    "TJP2": ["ZO-2"],
    "TRIM28": ["KAP1"],
    "VCL": ["vinculin"],
    "TNFRSF4": ["OX40"],
    "IL2RA": ["IL-2R", "CD25"],
    "CCNB1": ["cyclin B1", "cycB1"],
}


def _normalize_for_match(s: str) -> str:
    """Lowercase + strip non-alphanumeric. Catches BMP2↔BMP-2, IL22↔IL-22, etc."""
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _entity_in_evidence(name: str | None, evidence: str | None) -> bool:
    """True if `name` (or any aliased form) appears in `evidence`.

    Uses (a) literal case-insensitive substring, (b) normalized substring
    (punctuation/whitespace-insensitive), (c) the same two checks against
    each alias in `_ROLE_ALIAS_MAP[name]`. No regex word boundaries — the
    aim is high recall on aliases that appear inside larger tokens (e.g.,
    "TNFSF10" matches "TNFSF10-induced").
    """
    if not name or not evidence:
        return False
    if name.lower() in evidence.lower():
        return True
    norm_ev = _normalize_for_match(evidence)
    norm_name = _normalize_for_match(name)
    if norm_name and norm_name in norm_ev:
        return True
    for alt in _ROLE_ALIAS_MAP.get(name, []):
        if alt and alt.lower() in evidence.lower():
            return True
        norm_alt = _normalize_for_match(alt)
        if norm_alt and norm_alt in norm_ev:
            return True
    return False


def _resolve_role_gold(probe: str, gold_tag: str, rec: dict) -> str | None:
    """Content-conditional gold resolver for role probes under the three
    "claim-suspect" tags (no_relation, grounding, entity_boundaries).

    Returns:
      - "present_as_subject" / "present_as_object" if the entity (or a
        known alias) appears in the evidence text;
      - "absent" otherwise (with `grounding` and `entity_boundaries`,
        a miss is high-confidence absent because the gold tag itself
        says the entity name is suspect; with `no_relation` it means
        the entity isn't mentioned at all);
      - None for unsupported probe/tag combinations (caller skips).

    See `GOLD_TAG_TO_PROBE_CLASS` comment block for the full audit log.
    """
    if probe not in ("subject_role", "object_role"):
        return None
    if gold_tag not in ("no_relation", "grounding", "entity_boundaries"):
        return None
    entity = rec.get("subject") if probe == "subject_role" else rec.get("object")
    evidence = rec.get("evidence_text") or ""
    if _entity_in_evidence(entity, evidence):
        return "present_as_subject" if probe == "subject_role" else "present_as_object"
    return "absent"


# ── Holdout join (mirrors V7a) ──

def load_holdout_pairs() -> list[dict]:
    """Returns list of {source_hash, stmt_type, subject, object, evidence_text, gold_tag, gold_target}."""
    gold = {}
    with (ROOT / "data" / "benchmark" / "probe_gold_holdout.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            sh = r.get("source_hash")
            if isinstance(sh, int):
                gold[sh] = r
    out = []
    seen = set()
    with (ROOT / "data" / "benchmark" / "holdout_v15_sample.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            sh = r.get("source_hash")
            if not isinstance(sh, int) or sh not in gold or sh in seen:
                continue
            seen.add(sh)
            g = gold[sh]
            out.append({
                "source_hash": sh,
                "stmt_type": r.get("stmt_type") or g.get("claim_stmt_type"),
                "subject": r.get("subject") or g.get("subject"),
                "object": r.get("object") or g.get("object"),
                "evidence_text": (r.get("evidence_text") or "").strip(),
                "source_api": r.get("source_api"),
                "gold_tag": g.get("gold_tag"),
                "gold_target": g.get("gold_target"),
            })
    return out


# ── Prompt builders, one per probe — match production system+few-shots+user pattern ──

def _claim_string(stmt_type: str, subject: str, obj: str) -> str:
    """Render claim as e.g. `Phosphorylation(MAPK1, JUN)`."""
    return f"{stmt_type}({subject}, {obj})"


def build_messages(probe: str, rec: dict, *, use_curator: bool = True) -> tuple[str, list[tuple[str, str]], str]:
    """Returns (system_prompt, few_shots, user_message) for the given probe.

    When `use_curator` is True (default) and the probe has a curator-path
    override registered in `indra_belief.v_phase.curator_prompts`, the
    curator system prompt + few-shots are used in preference to
    production. Probes without a curator override (or all probes when
    `use_curator=False`) fall back to production prompts.
    """
    if probe == "relation_axis":
        from indra_belief.scorers.probes.relation_axis import _SYSTEM_PROMPT, _FEW_SHOTS
    elif probe == "subject_role":
        from indra_belief.scorers.probes.subject_role import _SYSTEM_PROMPT, _FEW_SHOTS
    elif probe == "object_role":
        from indra_belief.scorers.probes.object_role import _SYSTEM_PROMPT, _FEW_SHOTS
    elif probe == "scope":
        from indra_belief.scorers.probes.scope import _SYSTEM_PROMPT, _FEW_SHOTS
    elif probe == "verify_grounding":
        from indra_belief.scorers.grounding import _SYSTEM_PROMPT, _FEW_SHOTS
    else:
        raise ValueError(probe)

    system_prompt: str = _SYSTEM_PROMPT
    few_shots: list[tuple[str, str]] = list(_FEW_SHOTS)
    if use_curator:
        try:
            from indra_belief.v_phase.curator_prompts import (
                CURATOR_FEW_SHOTS,
                CURATOR_SYSTEM_PROMPTS,
            )
        except Exception:
            CURATOR_SYSTEM_PROMPTS = {}  # type: ignore[assignment]
            CURATOR_FEW_SHOTS = {}  # type: ignore[assignment]
        if probe in CURATOR_SYSTEM_PROMPTS:
            system_prompt = CURATOR_SYSTEM_PROMPTS[probe]
        if probe in CURATOR_FEW_SHOTS:
            few_shots = list(CURATOR_FEW_SHOTS[probe])

    claim = _claim_string(rec["stmt_type"], rec["subject"] or "?", rec["object"] or "?")
    evidence = rec["evidence_text"]
    if probe == "subject_role":
        user = f"CLAIM SUBJECT: {rec['subject']}\nEVIDENCE: {evidence}"
    elif probe == "object_role":
        user = f"CLAIM OBJECT: {rec['object']}\nEVIDENCE: {evidence}"
    elif probe == "verify_grounding":
        user = (
            f"Claim entity: {rec['subject']}\n"
            f"Aliases: (none provided)\n\n"
            f"Evidence: \"{evidence}\"\n"
            f"Does the evidence reference this entity?"
        )
    else:
        user = f"CLAIM: {claim}\nEVIDENCE: {evidence}"

    return system_prompt, few_shots, user


# ── Gemini call (one-shot per probe-record) ──

async def call_gemini(client, model: str, system: str, few_shots: list[tuple[str, str]],
                      user: str, semaphore, answer_classes: list[str]) -> dict:
    """Send a single labeling query. Returns parsed JSON or {'error': ...}.

    answer_classes constrains the model's `answer` field to a closed set
    via response_schema (Gemini's enum-on-string).
    """
    contents = []
    for q, a in few_shots:
        contents.append({"role": "user", "parts": [{"text": q}]})
        contents.append({"role": "model", "parts": [{"text": a}]})
    contents.append({"role": "user", "parts": [{"text": user}]})

    response_schema = {
        "type": "OBJECT",
        "properties": {
            "answer":    {"type": "STRING", "enum": list(answer_classes)},
            "rationale": {"type": "STRING"},
        },
        "required": ["answer", "rationale"],
        "propertyOrdering": ["answer", "rationale"],
    }

    async with semaphore:
        for attempt in range(3):
            try:
                cfg_kwargs = dict(
                    system_instruction=system,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                    temperature=0.0,
                    max_output_tokens=4096,
                )
                if "pro" in model:
                    cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=512)
                resp = await asyncio.to_thread(
                    client.models.generate_content,
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(**cfg_kwargs),
                )
                txt = resp.text
                if not txt:
                    return {"error": "empty"}
                try:
                    return json.loads(txt)
                except Exception:
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return {"error": "json_decode", "raw": txt[:300]}
            except Exception as e:
                msg = f"{type(e).__name__}: {str(e)[:200]}"
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return {"error": msg}
        return {"error": "max_retries"}


def extract_class(resp: dict) -> str | None:
    if "error" in resp:
        return None
    for k in ("answer", "label", "class", "verdict", "status"):
        if k in resp and isinstance(resp[k], str):
            return resp[k]
    return None


# ── Metrics ──

def compute_per_probe_metrics(
    tasks: list[dict],
    all_results: dict[str, list[dict]],
    models: list[str],
) -> dict[str, dict[str, dict]]:
    """For each model and probe, compute micro-acc, macro-recall,
    most-frequent-class baseline, and per-class precision/recall/F1/support.

    Macro is unweighted mean of per-class recall over classes that appear in
    gold (support > 0). Records with `pred is None` count as wrong (no TP) but
    still appear in `support` for the gold class — they hurt recall but
    don't move precision (no predicted positive).
    """
    from collections import Counter
    out: dict[str, dict[str, dict]] = {m: {} for m in models}
    for probe in PROBES:
        idxs = [i for i, t in enumerate(tasks) if t["probe"] == probe]
        if not idxs:
            for m in models:
                out[m][probe] = None  # type: ignore[assignment]
            continue
        gold_counts: Counter[str] = Counter(tasks[i]["gold_class"] for i in idxs)
        for m in models:
            tp: Counter[str] = Counter()
            pred_counts: Counter[str] = Counter()
            errors = 0
            for i in idxs:
                pred = extract_class(all_results[m][i]) if all_results[m][i] is not None else None
                gold = tasks[i]["gold_class"]
                if pred is None:
                    errors += 1
                    continue
                pred_counts[pred] += 1
                if pred == gold:
                    tp[gold] += 1
            n = len(idxs)
            classes = sorted(set(gold_counts) | set(pred_counts))
            per_class: dict[str, dict] = {}
            macro_sum = 0.0
            macro_n = 0
            for c in classes:
                support = gold_counts[c]
                pred_support = pred_counts[c]
                recall = (tp[c] / support) if support > 0 else None
                precision = (tp[c] / pred_support) if pred_support > 0 else None
                if recall is not None and precision is not None and (recall + precision) > 0:
                    f1 = 2 * recall * precision / (recall + precision)
                else:
                    f1 = None
                per_class[c] = {
                    "support": support, "pred_support": pred_support,
                    "precision": precision, "recall": recall, "f1": f1,
                }
                if support > 0:
                    macro_sum += recall or 0.0
                    macro_n += 1
            micro = sum(tp.values()) / n if n else 0.0
            macro = macro_sum / macro_n if macro_n else 0.0
            mfc = (max(gold_counts.values()) / n) if gold_counts else 0.0
            out[m][probe] = {
                "n": n,
                "errors": errors,
                "micro": micro,
                "macro": macro,
                "mfc": mfc,
                "per_class": per_class,
            }
    return out


def render_report(
    metrics: dict[str, dict[str, dict]],
    models: list[str],
    n_records: int,
    n_tasks: int,
    *,
    gate: Gate | None = None,
) -> str:
    """Render the V6g markdown report from per-probe metrics."""
    if gate is None:
        gate = Gate()
    parts: list[str] = []
    parts.append("# V6g — Gemini model validation on U2 per-probe gold\n")
    parts.append("Date: 2026-05-07")
    parts.append(f"U2 records: {n_records} | tasks per model: {n_tasks}")
    parts.append(
        "Composite gate (per probe): "
        f"saturating lift `micro ≥ mfc + {gate.delta_mfc:.2f}·(1-mfc)`, "
        f"macro-recall `≥ {gate.macro_floor:.2f}`, "
        f"min-class-recall `≥ {gate.min_class_recall:.2f}` "
        f"(for classes with support `≥ {gate.min_support}`). "
        "1-class probes (mfc=1.0) emit `INSUFFICIENT_GOLD`.\n"
    )

    # Pre-compute gate evaluations once per (model, probe).
    gate_eval: dict[str, dict[str, dict]] = {m: {} for m in models}
    for m in models:
        for probe in PROBES:
            mp = metrics[m].get(probe)
            if not mp:
                continue
            gate_eval[m][probe] = evaluate_gate(mp, gate)

    parts.append("## Per-probe summary (micro / macro / Δ vs most-frequent-class baseline)\n")
    parts.append("| Probe | " + " | ".join(models) + " |")
    parts.append("|---|" + "|".join(["---"] * len(models)) + "|")
    for probe in PROBES:
        row = [probe]
        for m in models:
            mp = metrics[m].get(probe)
            if not mp:
                row.append("—")
                continue
            micro = mp["micro"]; macro = mp["macro"]; mfc = mp["mfc"]
            delta = micro - mfc
            err = f" (err={mp['errors']})" if mp["errors"] else ""
            ge = gate_eval[m].get(probe, {})
            v = ge.get("verdict", "?")
            if v == "PASS":
                flag = "✓ PASS"
            elif v == "INSUFFICIENT_GOLD":
                flag = "○ INSUFFICIENT_GOLD"
            else:
                flag = "✗ FAIL"
            row.append(
                f"micro={micro:.3f} / macro={macro:.3f} / Δmfc={delta:+.3f}{err} {flag}"
            )
        parts.append("| " + " | ".join(row) + " |")

    parts.append("\n## Gate criteria per probe\n")
    parts.append(
        "Each criterion shows pass/fail with the threshold inline. "
        "`—` = not applicable for 1-class probes."
    )
    parts.append("")
    for m in models:
        parts.append(f"\n### {m}\n")
        parts.append(
            "| Probe | lift (micro ≥ mfc+Δ·(1-mfc)) | macro-recall ≥ floor | min-class recall | verdict |"
        )
        parts.append("|---|---|---|---|---|")
        for probe in PROBES:
            mp = metrics[m].get(probe)
            if not mp:
                parts.append(f"| {probe} | — | — | — | — |")
                continue
            ge = gate_eval[m].get(probe, {})
            micro = mp["micro"]; macro = mp["macro"]; mfc = mp["mfc"]
            if ge.get("one_class"):
                lift_cell = "—"
                macro_cell = "—"
                minc_cell = "—"
            else:
                lift_thr = mfc + gate.delta_mfc * (1.0 - mfc)
                lift_mark = "✓" if ge["lift_pass"] else "✗"
                lift_cell = (
                    f"{lift_mark} {micro:.3f} {'≥' if ge['lift_pass'] else '<'} "
                    f"{lift_thr:.3f} (mfc={mfc:.3f}+{gate.delta_mfc:.2f}·{1.0-mfc:.3f})"
                )
                macro_mark = "✓" if ge["macro_pass"] else "✗"
                macro_cell = (
                    f"{macro_mark} {macro:.3f} "
                    f"{'≥' if ge['macro_pass'] else '<'} {gate.macro_floor:.2f}"
                )
                if ge["min_class_pass"]:
                    minc_cell = (
                        f"✓ all classes (support ≥ {gate.min_support}) "
                        f"recall ≥ {gate.min_class_recall:.2f}"
                    )
                else:
                    failing = ", ".join(
                        f"{c}(n={s}, r={r:.3f})"
                        for c, s, r in ge["failing_classes"]
                    )
                    minc_cell = (
                        f"✗ < {gate.min_class_recall:.2f}: {failing}"
                    )
            verdict = ge.get("verdict", "?")
            parts.append(
                f"| {probe} | {lift_cell} | {macro_cell} | {minc_cell} | {verdict} |"
            )

    parts.append("\n## Trivial baseline (most-frequent class) per probe\n")
    parts.append("| Probe | n | top gold classes (support) | mfc baseline acc |")
    parts.append("|---|---|---|---|")
    if models:
        m0 = models[0]
        for probe in PROBES:
            mp = metrics[m0].get(probe)
            if not mp:
                parts.append(f"| {probe} | — | — | — |")
                continue
            ranked = sorted(mp["per_class"].items(), key=lambda kv: -kv[1]["support"])
            dist = ", ".join(f"{c}={d['support']}" for c, d in ranked if d["support"] > 0)
            parts.append(f"| {probe} | {mp['n']} | {dist[:160]} | {mp['mfc']:.3f} |")

    parts.append("\n## Per-class precision / recall / F1\n")
    for m in models:
        parts.append(f"\n### {m}\n")
        for probe in PROBES:
            mp = metrics[m].get(probe)
            if not mp:
                continue
            parts.append(
                f"\n**{probe}** (n={mp['n']}, micro={mp['micro']:.3f}, "
                f"macro={mp['macro']:.3f}, mfc={mp['mfc']:.3f}, err={mp['errors']})\n"
            )
            parts.append("| Class | Support | Predicted | Precision | Recall | F1 |")
            parts.append("|---|---|---|---|---|---|")
            for c, d in sorted(mp["per_class"].items(), key=lambda kv: -kv[1]["support"]):
                p = f"{d['precision']:.3f}" if d["precision"] is not None else "—"
                r = f"{d['recall']:.3f}" if d["recall"] is not None else "—"
                f1 = f"{d['f1']:.3f}" if d["f1"] is not None else "—"
                parts.append(f"| {c} | {d['support']} | {d['pred_support']} | {p} | {r} | {f1} |")

    parts.append("\n## Verdict\n")
    parts.append(
        f"Composite gate: lift Δ={gate.delta_mfc:.2f}, macro≥{gate.macro_floor:.2f}, "
        f"per-class recall≥{gate.min_class_recall:.2f} for classes with "
        f"support≥{gate.min_support}. 1-class probes (mfc=1.0) emit INSUFFICIENT_GOLD.\n"
    )
    for m in models:
        per_probe_verdicts: list[tuple[str, dict]] = []
        for probe in PROBES:
            ge = gate_eval[m].get(probe)
            if ge is None:
                continue
            per_probe_verdicts.append((probe, ge))

        any_fail = any(g["verdict"] == "FAIL" for _, g in per_probe_verdicts)
        any_insuf = any(g["verdict"] == "INSUFFICIENT_GOLD" for _, g in per_probe_verdicts)
        all_pass = all(g["verdict"] == "PASS" for _, g in per_probe_verdicts) if per_probe_verdicts else False

        if any_fail:
            model_verdict = "FAIL"
        elif any_insuf:
            model_verdict = "INSUFFICIENT_GOLD"
        elif all_pass:
            model_verdict = "PASS"
        else:
            model_verdict = "FAIL"

        parts.append(f"- **{m}**: {model_verdict}")
        for probe, ge in per_probe_verdicts:
            v = ge["verdict"]
            if v == "PASS":
                parts.append(f"    - {probe}: PASS")
            elif v == "INSUFFICIENT_GOLD":
                parts.append(
                    f"    - {probe}: INSUFFICIENT_GOLD "
                    f"(mfc=1.0; no minority gold to evaluate against)"
                )
            else:
                reasons = "; ".join(ge["reasons"])
                parts.append(f"    - {probe}: FAIL [{reasons}]")

    parts.append("\n## Cost ladder (full 400K corpus, batch API)\n")
    parts.append("| Model | Batch cost | Quality |")
    parts.append("|---|---|---|")
    parts.append("| gemini-2.5-flash-lite | ~$10 | baseline |")
    parts.append("| gemini-2.5-flash | ~$43 | balanced |")
    parts.append("| gemini-3.1-pro-preview | ~$200 | top tier |")
    return "\n".join(parts) + "\n"


def load_responses_for_report(
    responses_path: Path, models: list[str],
) -> tuple[list[dict], dict[str, list[dict]], int]:
    """Reconstruct (tasks, all_results, n_records) from a v6g_responses.jsonl."""
    tasks: list[dict] = []
    all_results: dict[str, list[dict]] = {m: [] for m in models}
    seen_records: set = set()
    with responses_path.open() as f:
        for line in f:
            r = json.loads(line)
            tasks.append({
                "record_id": r["record_id"],
                "probe": r["probe"],
                "gold_class": r["gold_class"],
            })
            seen_records.add(r["record_id"])
            for m in models:
                all_results[m].append(r["responses"].get(m, {"error": "missing"}))
    return tasks, all_results, len(seen_records)


# ── Main ──

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=[
        "gemini-3.1-pro-preview",
    ])
    parser.add_argument("--limit", type=int, default=482)
    parser.add_argument("--rps", type=int, default=10)
    parser.add_argument(
        "--use-curator-prompts",
        dest="use_curator_prompts",
        action="store_true",
        default=True,
        help=(
            "Use curator-path system prompts + few-shots from "
            "indra_belief.v_phase.curator_prompts when available "
            "(default: True). Falls back to production prompts for "
            "any probe not registered in CURATOR_SYSTEM_PROMPTS."
        ),
    )
    parser.add_argument(
        "--no-curator-prompts",
        dest="use_curator_prompts",
        action="store_false",
        help="Disable curator-path overrides; use production prompts for all probes.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help=(
            "Skip API calls; load existing data/v_phase/v6g_responses.jsonl "
            "and regenerate the markdown report only."
        ),
    )
    parser.add_argument(
        "--gate-delta-mfc",
        type=float,
        default=Gate.delta_mfc,
        help=(
            "Required lift of micro over most-frequent-class baseline "
            f"(default: {Gate.delta_mfc:.2f})."
        ),
    )
    parser.add_argument(
        "--gate-macro-floor",
        type=float,
        default=Gate.macro_floor,
        help=(
            f"Macro-recall floor per probe (default: {Gate.macro_floor:.2f})."
        ),
    )
    parser.add_argument(
        "--gate-min-class-recall",
        type=float,
        default=Gate.min_class_recall,
        help=(
            "Per-class recall floor for any gold class meeting min-support "
            f"(default: {Gate.min_class_recall:.2f})."
        ),
    )
    parser.add_argument(
        "--gate-min-support",
        type=int,
        default=Gate.min_support,
        help=(
            "Min support to enforce per-class recall floor "
            f"(default: {Gate.min_support})."
        ),
    )
    parser.add_argument(
        "--decomposed",
        action="store_true",
        help=(
            "Use the decomposed curator (sub-question chain) for probes "
            "that have a registered scorer in "
            "indra_belief.v_phase.decomposed_curator. Other probes fall back "
            "to single-shot curator prompts."
        ),
    )
    args = parser.parse_args()

    gate = Gate(
        delta_mfc=args.gate_delta_mfc,
        macro_floor=args.gate_macro_floor,
        min_class_recall=args.gate_min_class_recall,
        min_support=args.gate_min_support,
    )

    out_dir = ROOT / "data" / "v_phase"
    raw_path = out_dir / "v6g_responses.jsonl"
    rep_path = ROOT / "research" / "v6g_gemini_validation.md"

    if args.report_only:
        print(f"Report-only mode; reading {raw_path}")
        tasks, all_results, n_records = load_responses_for_report(raw_path, args.models)
        metrics = compute_per_probe_metrics(tasks, all_results, args.models)
        rep_path.write_text(
            render_report(metrics, args.models, n_records, len(tasks), gate=gate)
        )
        print(f"Wrote {rep_path}")
        return 0

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set")
        return 2

    print("Loading U2 holdout pairs...")
    pairs = load_holdout_pairs()[:args.limit]
    print(f"  {len(pairs)} records")
    if args.use_curator_prompts:
        try:
            from indra_belief.v_phase.curator_prompts import (
                CURATOR_SYSTEM_PROMPTS as _CSP,
            )
            print(
                f"  curator-path prompts ENABLED for: {sorted(_CSP)}"
            )
        except Exception:
            print("  curator-path prompts ENABLED but module not importable; using production")
    else:
        print("  curator-path prompts DISABLED; using production prompts for all probes")

    # Optional decomposed-curator import + per-probe registration check
    decomposed_probes: set[str] = set()
    if args.decomposed:
        try:
            from indra_belief.v_phase import decomposed_curator as _dec
            for _p in PROBES:
                if _dec.has_decomposed(_p):
                    decomposed_probes.add(_p)
            print(
                f"  decomposed-curator ENABLED for: {sorted(decomposed_probes) or '<none registered>'}"
            )
        except Exception as _e:
            print(f"  decomposed-curator ENABLED but module not importable: {_e}")

    # Build tasks: (record_id, probe, gold_class, rec, [system, fewshots, user] | decomposed)
    tasks = []
    skipped = defaultdict(int)
    resolver_tally: dict[tuple[str, str, str], int] = defaultdict(int)
    for rec in pairs:
        gold_tag = rec.get("gold_tag", "correct")
        probe_map = GOLD_TAG_TO_PROBE_CLASS.get(gold_tag, {})
        for probe in PROBES:
            gold_class = probe_map.get(probe)
            if gold_class == _ROLE_RESOLVE_SENTINEL:
                gold_class = _resolve_role_gold(probe, gold_tag, rec)
                if gold_class is not None:
                    resolver_tally[(probe, gold_tag, gold_class)] += 1
            if gold_class is None:
                skipped[f"{probe}:no_gold_for_tag_{gold_tag}"] += 1
                continue
            t = {
                "record_id": rec["source_hash"],
                "probe": probe,
                "gold_class": gold_class,
                "rec": rec,
                "decomposed": probe in decomposed_probes,
            }
            if not t["decomposed"]:
                try:
                    sys_, fs, usr = build_messages(
                        probe, rec, use_curator=args.use_curator_prompts
                    )
                except Exception as e:
                    skipped[f"{probe}:render_{type(e).__name__}"] += 1
                    continue
                t["system"] = sys_
                t["fewshots"] = fs
                t["user"] = usr
            tasks.append(t)

    if resolver_tally:
        print("  role-gold resolver tally (Option A, content-conditional):")
        for (probe, tag, cls), n in sorted(resolver_tally.items()):
            print(f"    {probe:>13s}  tag={tag:<18s}  -> {cls:<20s}  ({n})")

    print(f"  built {len(tasks)} tasks (skipped {sum(skipped.values())} for no-gold-mapping)")

    client = genai.Client(api_key=api_key)
    semaphore = asyncio.Semaphore(args.rps)

    out_dir.mkdir(parents=True, exist_ok=True)

    def _make_call_subq(model: str):
        async def call_subq(system_prompt: str, few_shots: list[tuple[str, str]],
                            user_msg: str, class_space: list[str]) -> dict:
            return await call_gemini(
                client, model, system_prompt, few_shots, user_msg,
                semaphore, class_space,
            )
        return call_subq

    async def indexed_call(idx: int, model: str, t: dict) -> tuple[int, dict]:
        if t.get("decomposed"):
            from indra_belief.v_phase import decomposed_curator as _dec
            try:
                r = await _dec.score_decomposed(
                    t["probe"], t["rec"], _make_call_subq(model)
                )
            except Exception as e:
                r = {"error": f"decomposed_{type(e).__name__}: {str(e)[:200]}"}
        else:
            r = await call_gemini(client, model, t["system"], t["fewshots"], t["user"],
                                   semaphore, PROBE_CLASSES[t["probe"]])
        return idx, r

    all_results = {}
    for model in args.models:
        print(f"\n=== {model} ({len(tasks)} calls) ===")
        t0 = time.time()
        coros = [indexed_call(i, model, t) for i, t in enumerate(tasks)]
        results: list[dict] = [None] * len(tasks)  # type: ignore[list-item]
        done = 0
        for fut in asyncio.as_completed(coros):
            idx, r = await fut
            results[idx] = r
            done += 1
            if done % 200 == 0:
                rate = done / max(time.time() - t0, 0.01)
                print(f"  {done}/{len(tasks)} ({rate:.1f}/s, {time.time()-t0:.0f}s)")
        all_results[model] = results
        print(f"  done in {time.time()-t0:.0f}s")

    # Save raw responses
    with raw_path.open("w") as f:
        for i, t in enumerate(tasks):
            row = {
                "record_id": t["record_id"], "probe": t["probe"], "gold_class": t["gold_class"],
                "responses": {m: all_results[m][i] for m in args.models},
            }
            f.write(json.dumps(row) + "\n")
    print(f"\nWrote {raw_path}")

    # Compute metrics + render report
    metrics = compute_per_probe_metrics(tasks, all_results, args.models)
    rep_path.write_text(
        render_report(metrics, args.models, len(pairs), len(tasks), gate=gate)
    )
    print(f"Wrote {rep_path}")

    print("\n=== SUMMARY ===")
    for m in args.models:
        print(f"  {m}:")
        for probe in PROBES:
            mp = metrics[m].get(probe)
            if not mp:
                continue
            print(
                f"    {probe}: micro={mp['micro']:.3f} macro={mp['macro']:.3f} "
                f"mfc={mp['mfc']:.3f} (n={mp['n']}, err={mp['errors']})"
            )


if __name__ == "__main__":
    asyncio.run(main())
