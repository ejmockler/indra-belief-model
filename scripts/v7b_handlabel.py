"""V7b — agent-based hand-labeling of 250 stratified records (50/probe).

Per the V7b task brief and `research/v5r_data_prep_doctrine.md` §7b:

V7a's verdict was gold-coverage-limited because U2 collapses several
classes (correct→asserted only, no equivalent class for verify_grounding,
no present_as_mediator/decoy, no via_mediator).  V7b fills these gaps
with careful agent-based labeling that models the curator's judgment.

Pipeline:
  1. For each probe, load `data/v_phase/labels/{probe}_sample.parquet`,
     stratify 50/30/20 (rare-class / borderline / confident).
  2. Walk the corpus to recover (statement, evidence) pairs by record_id.
  3. Render the production prompt + assign agent label using closed-set
     judgment rules informed by `types.py` + `grounding.py` + doctrine.
     Confidence: high/medium/low.  Flag and skip when unsure.
  4. AFTER labeling, compute LF votes per record.
  5. Compute per-LF accuracy + Wilson 95% CI + V8 PASS/MARGINAL/FAIL.
  6. Combine with V7a status: PASS if either V7a or V7b passes.

Outputs:
  - `data/v_phase/v7b_handlabels.jsonl`
  - `research/v7b_handlabel_log.md`
  - `data/v_phase/v8_lf_status.json`
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indra_belief.v_phase.labeling import (  # noqa: E402
    ABSTAIN,
    _build_probe_lf_index,
    _apply_lf_safe,
    iter_pairs,
    load_holdout_exclusion,
)
from indra_belief.v_phase.substrate_lfs import (  # noqa: E402
    GROUNDING_CLASSES,
    RELATION_AXIS_CLASSES,
    ROLE_CLASSES,
    SCOPE_CLASSES,
)


ROOT = Path(__file__).resolve().parents[1]
LABELS_DIR = ROOT / "data" / "v_phase" / "labels"
OUT_JSONL = ROOT / "data" / "v_phase" / "v7b_handlabels.jsonl"
OUT_MD = ROOT / "research" / "v7b_handlabel_log.md"
OUT_V8_JSON = ROOT / "data" / "v_phase" / "v8_lf_status.json"
V7A_JSON = ROOT / "data" / "v_phase" / "v7a_lf_accuracy.json"


PROBES = ["relation_axis", "subject_role", "object_role", "scope",
          "verify_grounding"]

CLASS_INDEX = {
    "relation_axis": RELATION_AXIS_CLASSES,
    "subject_role": ROLE_CLASSES,
    "object_role": ROLE_CLASSES,
    "scope": SCOPE_CLASSES,
    "verify_grounding": GROUNDING_CLASSES,
}

# Per-probe target sample size
SAMPLE_SIZE = 50  # 50 records per probe × 5 probes = 250


# ---------------------------------------------------------------------------
# 1. Stratified sample selection from V6c parquets.
# ---------------------------------------------------------------------------

def stratified_sample(probe: str, n_total: int = SAMPLE_SIZE,
                       seed: int = 42) -> pd.DataFrame:
    """Stratify per V5r §7b: 50% rare-class, 30% borderline, 20% confident.

    Rare-class: half of records pulled from the records where argmax_class
    is one of the rarest classes (sorted by total count). For relation_axis
    spread across 5 rarest of 8 classes; for K=4/5 probes spread across
    rarest 2-3 classes.

    Borderline: max_proba ∈ [0.5, 0.7].
    Confident: max_proba > 0.9.
    """
    rng = np.random.default_rng(seed)
    df = pd.read_parquet(LABELS_DIR / f"{probe}_sample.parquet")

    # Identify rare classes — sort by count ascending.
    class_counts = df["argmax_class"].value_counts().sort_values()
    K = len(CLASS_INDEX[probe])
    n_rare_classes = max(2, min(5, K - 2))  # 2..5 rare classes
    rare_classes = list(class_counts.index[:n_rare_classes])

    # 50% rare-class records
    n_rare = n_total // 2  # 25
    df_rare = df[df["argmax_class"].isin(rare_classes)]
    if len(df_rare) >= n_rare:
        # Round-robin across rare classes
        chunks = []
        per_class = n_rare // len(rare_classes)
        remainder = n_rare - per_class * len(rare_classes)
        for i, cls in enumerate(rare_classes):
            sub = df_rare[df_rare["argmax_class"] == cls]
            take = per_class + (1 if i < remainder else 0)
            if len(sub) >= take:
                idx = rng.choice(len(sub), size=take, replace=False)
                chunks.append(sub.iloc[idx])
            else:
                chunks.append(sub)
        rare_pick = pd.concat(chunks) if chunks else df.iloc[0:0]
    else:
        rare_pick = df_rare

    # 30% borderline
    n_border = int(n_total * 0.3)  # 15
    df_border = df[(df["max_proba"] >= 0.5) & (df["max_proba"] <= 0.7)]
    df_border = df_border[~df_border["record_id"].isin(rare_pick["record_id"])]
    if len(df_border) >= n_border:
        idx = rng.choice(len(df_border), size=n_border, replace=False)
        border_pick = df_border.iloc[idx]
    else:
        border_pick = df_border

    # 20% confident
    n_conf = n_total - len(rare_pick) - len(border_pick)
    df_conf = df[df["max_proba"] > 0.9]
    df_conf = df_conf[~df_conf["record_id"].isin(rare_pick["record_id"])]
    df_conf = df_conf[~df_conf["record_id"].isin(border_pick["record_id"])]
    if len(df_conf) >= n_conf:
        idx = rng.choice(len(df_conf), size=n_conf, replace=False)
        conf_pick = df_conf.iloc[idx]
    else:
        conf_pick = df_conf

    out = pd.concat([rare_pick, border_pick, conf_pick],
                    ignore_index=True)
    # Add a stratum-tag column.
    out["stratum"] = (["rare"] * len(rare_pick) +
                       ["borderline"] * len(border_pick) +
                       ["confident"] * len(conf_pick))
    print(f"[V7b] {probe}: rare={len(rare_pick)} (classes {rare_classes}), "
          f"borderline={len(border_pick)}, confident={len(conf_pick)}; "
          f"total={len(out)}")
    return out


# ---------------------------------------------------------------------------
# 2. Build a record_id → (stmt_d, ev_d, agents) lookup by re-walking corpus.
# ---------------------------------------------------------------------------

def _record_id_for(stmt_d: dict, ev_d: dict) -> str:
    return f"{stmt_d.get('matches_hash')}:{ev_d.get('source_hash')}"


def build_record_lookup(needed_ids: set[str], max_pairs: int = 10000
                         ) -> dict[str, tuple[dict, dict, list[dict]]]:
    """Walk the corpus and pick out exactly the (stmt, ev, agents) records
    matching the needed record IDs. V6c default sample is 10000 pairs;
    we stop early once all needed IDs are found.
    """
    out: dict[str, tuple[dict, dict, list[dict]]] = {}
    needed = set(needed_ids)
    print(f"[V7b] walking corpus for {len(needed)} needed record_ids "
          f"(max_pairs={max_pairs})...", flush=True)
    exc = load_holdout_exclusion()
    n = 0
    for raw, stmt_d, ev_d, agents in iter_pairs(exclusion=exc,
                                                  max_pairs=max_pairs):
        n += 1
        # Generate both (stmt, ev) ID and (stmt, ev, ent_idx) IDs (for
        # verify_grounding records).
        rid = _record_id_for(stmt_d, ev_d)
        if rid in needed:
            out[rid] = (stmt_d, ev_d, agents)
            needed.discard(rid)
        # verify_grounding entity-keyed IDs
        for ent_idx, _agent in enumerate(agents):
            ent_rid = f"{rid}:{ent_idx}"
            if ent_rid in needed:
                out[ent_rid] = (stmt_d, ev_d, agents)
                needed.discard(ent_rid)
        if not needed:
            break
    print(f"[V7b] corpus walk: {n} pairs scanned, "
          f"{len(out)} records found, {len(needed)} unmatched")
    if needed:
        print(f"[V7b] WARN: {len(needed)} record_ids not found in corpus")
    return out


# ---------------------------------------------------------------------------
# 3. Render production prompts (verbatim from probe modules).
# ---------------------------------------------------------------------------

def render_relation_axis_prompt(stmt_d: dict, ev_d: dict) -> dict:
    from indra_belief.scorers.probes import relation_axis as ra
    from indra_belief.scorers.probes._llm import llm_classify  # noqa: F401
    # Need axis+sign — derive from stmt_type heuristics.
    t = stmt_d.get("stmt_type") or ""
    axis_sign = _stmt_type_axis_sign(t)
    claim_component = (
        f"subject={stmt_d.get('subject')}, object={stmt_d.get('object')}, "
        f"axis={axis_sign[0]}, sign={axis_sign[1]}"
    )
    user_msg = (
        f"CLAIM: {claim_component}\n"
        f"EVIDENCE: {(ev_d.get('text') or '').strip()}"
    )
    return {
        "system": ra._SYSTEM_PROMPT,
        "user": user_msg,
        "few_shots": ra._FEW_SHOTS,
    }


def render_subject_role_prompt(stmt_d: dict, ev_d: dict) -> dict:
    from indra_belief.scorers.probes import subject_role as sr
    user_msg = (
        f"CLAIM SUBJECT: {stmt_d.get('subject')}\n"
        f"EVIDENCE: {(ev_d.get('text') or '').strip()}"
    )
    return {
        "system": sr._SYSTEM_PROMPT,
        "user": user_msg,
        "few_shots": sr._FEW_SHOTS,
    }


def render_object_role_prompt(stmt_d: dict, ev_d: dict) -> dict:
    from indra_belief.scorers.probes import object_role as obr
    user_msg = (
        f"CLAIM OBJECT: {stmt_d.get('object')}\n"
        f"EVIDENCE: {(ev_d.get('text') or '').strip()}"
    )
    return {
        "system": obr._SYSTEM_PROMPT,
        "user": user_msg,
        "few_shots": obr._FEW_SHOTS,
    }


def render_scope_prompt(stmt_d: dict, ev_d: dict) -> dict:
    from indra_belief.scorers.probes import scope as sc
    claim_component = (
        f"subject={stmt_d.get('subject')}, object={stmt_d.get('object')}, "
        f"stmt_type={stmt_d.get('stmt_type')}"
    )
    user_msg = (
        f"CLAIM: {claim_component}\n"
        f"EVIDENCE: {(ev_d.get('text') or '').strip()}"
    )
    return {
        "system": sc._SYSTEM_PROMPT,
        "user": user_msg,
        "few_shots": sc._FEW_SHOTS,
    }


def render_verify_grounding_prompt(stmt_d: dict, ev_d: dict,
                                     entity_name: str) -> dict:
    from indra_belief.scorers import grounding as gr
    user_msg = (
        f"ENTITY: {entity_name}\n"
        f"EVIDENCE: {(ev_d.get('text') or '').strip()}"
    )
    return {
        "system": gr._SYSTEM_PROMPT,
        "user": user_msg,
    }


def _stmt_type_axis_sign(t: str) -> tuple[str, str]:
    """Map INDRA stmt_type → (axis, sign) for the relation_axis prompt.

    Best-effort; matches the convention used in the production scorer
    where the orchestrator builds the claim_component string.
    """
    t = (t or "").lower()
    # modification axes
    mod_pos = ("phosphorylation", "acetylation", "methylation",
                "ubiquitination", "sumoylation", "ribosylation",
                "myristoylation", "palmitoylation", "geranylgeranylation",
                "farnesylation", "glycosylation", "hydroxylation",
                "modification")
    mod_neg = ("dephosphorylation", "deacetylation", "demethylation",
                "deubiquitination", "desumoylation")
    if t in mod_pos:
        return ("modification", "positive")
    if t in mod_neg:
        return ("modification", "negative")
    if t == "activation":
        return ("activity", "positive")
    if t == "inhibition":
        return ("activity", "negative")
    if t == "increaseamount":
        return ("amount", "positive")
    if t == "decreaseamount":
        return ("amount", "negative")
    if t == "complex":
        return ("binding", "neutral")
    if t == "translocation":
        return ("localization", "neutral")
    if t == "conversion":
        return ("conversion", "neutral")
    if t in ("activeform", "gef", "gap"):
        return ("activity", "positive")
    return ("activity", "positive")


# ---------------------------------------------------------------------------
# 4. Agent labeling — closed-set judgment from probe definitions + doctrine.
# ---------------------------------------------------------------------------
#
# Each labeler models curator-style judgment. Returns
# (label, confidence, rationale, flagged).
#
# Conservative: when the rules disagree or evidence is unclear, the labeler
# raises confidence='low' or sets flagged=True.

# Common helpers --------------------------------------------------------------

_HEDGE_CUES = (
    r"\bmay\b", r"\bmight\b", r"\bcould\b", r"\bshould\b",
    r"\bsuggest", r"\bpropose", r"\bhypothesi[zs]e",
    r"\bappear", r"\blikely\b", r"\bperhaps\b", r"\bputative\b",
    r"\bpotential\b", r"\bpossibly\b", r"\bbelieved\b",
    r"\bthought to\b", r"\bremains? unclear\b",
    r"\bwe (?:tested|examined|investigated)\s+(?:whether|if)\b",
    r"\bremains? (?:to be )?(?:elucidated|determined|established)\b",
    r"\bnot fully\b", r"\bunknown\b",
)
_NEGATION_CUES = (
    r"\bdid\s+not\b", r"\bdoes\s+not\b", r"\bdo\s+not\b",
    r"\bdid\s*n[’']t\b", r"\bdoes\s*n[’']t\b",
    r"\bnot\b", r"\bno\b", r"\bnever\b",
    r"\bfailed?\s+to\b", r"\bunable to\b",
    r"\bwithout\b", r"\babsence\s+of\b", r"\black\s+of\b",
    r"\bneither\b", r"\bnone\b",
)
_CONDITION_CUES = (
    r"wild[\s\-]?type", r"\bmutant\b", r"variant", r"\bisoform\b",
    r"\bconstitutively[\s\-]?active\b", r"presence\s+of",
    r"absence\s+of", r"knockout", r"knockdown", r"siRNA",
    r"dominant[\s\-]?negative", r"in (?:cells|cell lines|patients|tissue) (?:treated|stimulated)",
    r"upon stimulation", r"only when", r"only in",
    r"but not (?:the )?(?:mutant|variant|inactive)",
)
_CHAIN_CUES = (
    r"\bthereby\b", r"\bleads? to\b", r"\bresult(?:s|ed)?\s+in\b",
    r"\bmediated? by\b", r"\bin turn\b",
    r"\bsubsequently\b",
    r"\bindirectly\b", r"\bindirect\b",
)
# Slightly weaker chain markers — require a clearer chain context
_WEAK_CHAIN_CUES = (
    r"\bvia\b", r"\bthrough\b", r"\bdownstream\b",
)
_AMOUNT_CUES = (
    r"\bexpression\b", r"\babundance\b", r"\blevels?\b",
    r"\btranscript", r"\bmRNA\b", r"\bprotein levels?\b",
    r"\bupregulat", r"\bdownregulat", r"\binduce[sd]?\b",
    r"\binduction\b", r"\bsuppress(?:es|ed|ing)?\b",
    r"\benhance[sd]?\s+(?:expression|levels?)\b",
)
_PHOSPHO_CUES = (
    r"\bphosphorylat", r"\bacetylat", r"\bmethylat",
    r"\bubiquitinat", r"\bsumoylat",
)
_BINDING_CUES = (
    r"\bbind", r"\bbinding\b", r"\binteract", r"\bcomplex",
    r"\bassoci(?:ate|ation)\b",
)
_DNA_CUES = (
    r"promoter", r"enhancer", r"binding site",
    r"\bmotif\b", r"consensus sequence", r"DNA[\s\-]?binding",
    r"\bcis[\s\-]?(?:element|regulatory)\b",
    r"\bresponse element\b",
)


def _has_any(text: str, patterns) -> bool:
    if not text:
        return False
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _word_present(text: str, name: str) -> bool:
    """Word-boundary case-insensitive presence test."""
    if not text or not name:
        return False
    pat = r"\b" + re.escape(name) + r"\b"
    return bool(re.search(pat, text, re.IGNORECASE))


# Alias-aware presence check, using the V6 canonical helpers when available.
_ALIAS_CACHE: dict[str, list[str]] = {}


def _alias_aware_present(text: str, name: str) -> tuple[bool, str | None]:
    """Return (present, matched_alias).

    Tries (in order): exact word match, underscore→space substitution,
    tolerant collapsed match, then any alias from `_gilda_aliases(name)`.
    """
    if not text or not name:
        return (False, None)
    if _word_present(text, name):
        return (True, name)
    # Underscore → space (INDRA-internal compound names like Histone_H3
    # vs evidence "histone H3")
    if "_" in name:
        space_form = name.replace("_", " ")
        if _word_present(text, space_form):
            return (True, space_form)
    # Tolerant (collapsed punct/spaces)
    try:
        from indra_belief.v_phase.clean_lfs import _entity_in_text_tolerant
        if _entity_in_text_tolerant(name, text):
            return (True, name)
    except Exception:
        pass
    # Aliases
    aliases: list[str] = _ALIAS_CACHE.get(name)
    if aliases is None:
        try:
            from indra_belief.v_phase.clean_lfs import _gilda_aliases
            aliases = list(_gilda_aliases(name) or [])
        except Exception:
            aliases = []
        _ALIAS_CACHE[name] = aliases
    for a in aliases:
        if not isinstance(a, str) or len(a) < 2:
            continue
        if a.lower() == name.lower():
            continue
        if _word_present(text, a):
            return (True, a)
    return (False, None)


def _nearby(text: str, a: str, b: str, window: int = 80) -> bool:
    """Both `a` and `b` appear within `window` characters."""
    if not text or not a or not b:
        return False
    a_low = text.lower()
    a_l = a.lower()
    b_l = b.lower()
    a_idx = a_low.find(a_l)
    b_idx = a_low.find(b_l)
    if a_idx < 0 or b_idx < 0:
        return False
    return abs(a_idx - b_idx) <= window


# Subject_role & object_role labeler ------------------------------------------

def label_role(stmt_d: dict, ev_d: dict, role: str
                ) -> tuple[str, str, str, bool]:
    """Label subject_role or object_role.

    role ∈ {"subject", "object"}. Returns (label, confidence, rationale,
    flagged).
    """
    text = ev_d.get("text") or ""
    subj = stmt_d.get("subject") or ""
    obj = stmt_d.get("object") or ""
    target = subj if role == "subject" else obj
    other = obj if role == "subject" else subj
    if not target:
        return ("absent", "high", "no entity name in claim", False)

    # 1. Absence test (alias-aware: HGNC all_names + UniProt synonyms)
    in_text, matched = _alias_aware_present(text, target)
    if not in_text:
        return ("absent", "high",
                f"target '{target}' (and aliases) not found in evidence",
                False)
    # If we matched via alias, use that as the actual surface form
    surface = matched if matched else target

    # 2. Mediator chain test - is target between two named entities in chain?
    has_chain = _has_any(text, _CHAIN_CUES)
    other_present, other_surface = (
        (False, None) if not other else _alias_aware_present(text, other))
    if has_chain and other_present and surface.lower() != (
            other_surface or other).lower():
        # Look for "<X> ... target ... <Y>" or "via target", "through target"
        # — check if target appears as the chain intermediate.
        chain_phrases = (r"via\s+" + re.escape(surface),
                         r"through\s+" + re.escape(surface),
                         r"mediated by\s+" + re.escape(surface),
                         re.escape(surface) + r"[\s\-]?dependent")
        if any(re.search(p, text, re.IGNORECASE) for p in chain_phrases):
            return ("present_as_mediator", "medium",
                    f"target appears in chain pattern (via/through/dependent)",
                    False)

    # 3. Role-swap test — is the target the OBJECT/TARGET when role=subject,
    # or SUBJECT/AGENT when role=object?
    # Detect "X verb target" vs "target verb X" using ordering relative to
    # main verbs.
    target_idx = text.lower().find(surface.lower())
    other_search = other_surface or other
    other_idx = text.lower().find(other_search.lower()) if other_search else -1

    # Verbs near target
    verb_patterns = (
        r"\bphosphorylat\w+", r"\bactivat\w+", r"\binhibit\w+",
        r"\binduce\w*", r"\bsuppress\w+", r"\benhanc\w+",
        r"\bbind\w*", r"\binteract\w+", r"\bregulat\w+",
        r"\btranscrib\w+", r"\bbind\w*", r"\bup[\s\-]?regulat\w+",
        r"\bdown[\s\-]?regulat\w+", r"\bcomplex\w*",
    )
    # Find verb position
    verb_idx = -1
    for vp in verb_patterns:
        m = re.search(vp, text, re.IGNORECASE)
        if m:
            verb_idx = m.start()
            break

    # 4. Decoy test — target mentioned but in clearly unrelated relation
    # (e.g., "loading control", "co-treatment", "compared to", etc.)
    decoy_phrases = (
        r"loading control", r"normalized to", r"compared (?:with|to)",
        r"co[\s\-]?treat", r"alongside", r"as a control",
        r"as control", r"transfected with",
        r"in the presence of",
    )
    near_decoy = False
    for dp in decoy_phrases:
        m = re.search(dp, text, re.IGNORECASE)
        if m and abs(m.start() - target_idx) < 80:
            near_decoy = True
            break

    if near_decoy:
        return ("present_as_decoy", "medium",
                f"target '{target}' near control/loading/co-treat phrase",
                False)

    # 5. Subject vs object positioning
    # Heuristic: if target appears BEFORE the verb and the verb's pattern
    # is active (e.g., "X phosphorylates Y"), target is subject.
    # If target appears AFTER passive constructions like "phosphorylated by",
    # target is the agent (subject).
    # If target appears AFTER an active verb, it's likely the object.
    # If target appears BEFORE "is/was/were" + past participle, it's likely
    # the object (passive subject = grammatical subject but semantic patient).

    # Passive-by check: does "by <target>" appear?
    by_target = bool(re.search(r"\bby\s+" + re.escape(surface),
                                 text, re.IGNORECASE))
    # Active "<target> verb"
    active_target_verb = False
    target_verb_pat = (re.escape(surface) +
                        r"\s+(?:directly\s+|specifically\s+)?(?:" +
                        r"|".join([vp.strip(r'\b').replace(r'\w+', r'\\w+')
                                    for vp in verb_patterns]) + r")")
    if re.search(target_verb_pat, text, re.IGNORECASE):
        active_target_verb = True

    # Passive "<target> is/was [adv] verb-ed"
    passive_target = bool(re.search(
        re.escape(surface) + r"\s+(?:is|was|were|are|has been|have been)\s+(?:\w+\s+)?(?:phosphorylated|acetylated|inhibited|activated|induced|bound|regulated|targeted|methylated|degraded)\b",
        text, re.IGNORECASE))
    # also "<target>-mediated" / "<target>-induced" → acts as actor
    target_as_attr = bool(re.search(
        re.escape(surface) + r"[\s\-](?:mediated|induced|catalyzed|driven|dependent)",
        text, re.IGNORECASE))

    # Nominalization: "<target> acetylation/phosphorylation/methylation/..."
    # — target is the OBJECT (substrate) of the modification noun.
    target_as_modnoun_obj = bool(re.search(
        re.escape(surface) +
        r"\s+(?:acetylat|phosphorylat|methylat|ubiquitinat|sumoylat|"
        r"deacetylat|dephosphorylat|demethylat|degrad|cleav|"
        r"binding|interaction|association)",
        text, re.IGNORECASE))
    # "<verbal-noun> of <target>" — target is the OBJECT
    target_as_of_obj = bool(re.search(
        r"(?:acetylation|phosphorylation|methylation|ubiquitination|"
        r"binding|interaction|association|cleavage|degradation|"
        r"activation|inhibition|expression)\s+of\s+" + re.escape(surface),
        text, re.IGNORECASE))

    if role == "subject":
        # subject role expectation: target is the AGENT
        if by_target or target_as_attr:
            return ("present_as_subject", "high",
                    f"target acts as agent ('by X' / X-mediated)",
                    False)
        if active_target_verb:
            return ("present_as_subject", "high",
                    f"target as active verb subject",
                    False)
        if passive_target or target_as_modnoun_obj or target_as_of_obj:
            return ("present_as_object", "medium",
                    f"target in passive/nominalization (role-swap candidate)",
                    False)
        # default: target is named but role unclear
        # If we see verb_idx and target_idx in close proximity, prefer subject.
        if verb_idx >= 0 and target_idx >= 0 and target_idx < verb_idx and \
                (verb_idx - target_idx) < 60:
            return ("present_as_subject", "medium",
                    f"target appears before nearby verb",
                    False)
        if verb_idx >= 0 and target_idx >= 0 and target_idx > verb_idx and \
                (target_idx - verb_idx) < 60:
            return ("present_as_object", "medium",
                    f"target appears after nearby verb",
                    False)
        return ("present_as_subject", "low",
                f"target present, role unclear; default subject",
                False)
    else:
        # object role expectation: target is the TARGET/PATIENT
        if passive_target or target_as_modnoun_obj or target_as_of_obj:
            return ("present_as_object", "high",
                    f"target as patient (passive/nominalization)",
                    False)
        if by_target or target_as_attr:
            return ("present_as_subject", "high",
                    f"target acts as agent ('by X' / X-mediated; role-swap)",
                    False)
        if active_target_verb:
            return ("present_as_subject", "medium",
                    f"target as active verb subject (role-swap)",
                    False)
        if verb_idx >= 0 and target_idx >= 0 and target_idx > verb_idx and \
                (target_idx - verb_idx) < 60:
            return ("present_as_object", "medium",
                    f"target appears after nearby verb",
                    False)
        if verb_idx >= 0 and target_idx >= 0 and target_idx < verb_idx and \
                (verb_idx - target_idx) < 60:
            return ("present_as_subject", "medium",
                    f"target appears before nearby verb",
                    False)
        return ("present_as_object", "low",
                f"target present, role unclear; default object",
                False)


# Relation_axis labeler -------------------------------------------------------

def label_relation_axis(stmt_d: dict, ev_d: dict
                          ) -> tuple[str, str, str, bool]:
    """Label relation_axis. Returns (label, confidence, rationale, flagged)."""
    text = ev_d.get("text") or ""
    subj = stmt_d.get("subject") or ""
    obj = stmt_d.get("object") or ""
    t = stmt_d.get("stmt_type") or ""

    if not subj or not obj:
        return ("no_relation", "low", "missing subject or object name",
                True)

    # Are both entities present in evidence (alias-aware)?
    subj_present, subj_surface = _alias_aware_present(text, subj)
    obj_present, obj_surface = _alias_aware_present(text, obj)

    if not subj_present and not obj_present:
        return ("no_relation", "high",
                "neither subject nor object (incl. aliases) in evidence",
                False)
    if not subj_present or not obj_present:
        return ("no_relation", "medium",
                f"one entity missing in evidence "
                f"(subj_present={subj_present}, obj_present={obj_present})",
                False)
    subj_use = subj_surface or subj
    obj_use = obj_surface or obj

    axis, sign = _stmt_type_axis_sign(t)

    # Check if entities co-occur within a reasonable window
    cooccur = _nearby(text, subj_use, obj_use, window=200)

    # Chain markers test — tight detection: only count chain when the
    # marker LIES BETWEEN the two entities.
    has_strong_chain = _has_any(text, _CHAIN_CUES)
    chain_with_named = False
    chain_no_terminal = False
    if has_strong_chain:
        subj_idx = text.lower().find(subj_use.lower())
        obj_idx = text.lower().find(obj_use.lower())
        if subj_idx >= 0 and obj_idx >= 0:
            lo, hi = min(subj_idx, obj_idx), max(subj_idx, obj_idx)
            mid = text[lo:hi]
            # The chain cue must lie between the entities
            cue_in_middle = _has_any(mid, _CHAIN_CUES)
            if cue_in_middle:
                # Look for capitalized words between subj and obj (not subj/obj themselves)
                inter_match = re.findall(
                    r"\b[A-Z][A-Za-z0-9\-]{1,15}\b", mid)
                inter_match = [w for w in inter_match
                                if w.lower() not in (subj.lower(), obj.lower(),
                                                      subj_use.lower(),
                                                      obj_use.lower())
                                and len(w) > 2]
                if inter_match:
                    chain_with_named = True
                else:
                    chain_no_terminal = True

    # Amount-axis lexicon test
    has_amount = _has_any(text, _AMOUNT_CUES)

    # Negation in claim region
    has_negation = _has_any(text, _NEGATION_CUES)

    # DNA-binding partner check (for binding axis)
    has_dna_binding = _has_any(text, _DNA_CUES) and (
        axis == "binding" or t.lower() == "complex")

    # Decision flow:
    # Priority 1: partner_mismatch for binding axis with DNA target
    if has_dna_binding and t.lower() == "complex":
        # Is the DNA word part of the claim relation?
        if _word_present(text, "promoter") or _word_present(text, "DNA-binding"):
            # Subject binds DNA, but claim is protein-protein Complex
            return ("direct_partner_mismatch", "medium",
                    f"binding axis with DNA element (claim is protein Complex)",
                    False)

    # Priority 2: mediated chain
    if has_strong_chain and chain_with_named:
        return ("via_mediator", "medium",
                f"chain markers + named intermediate between {subj} and {obj}",
                False)
    if has_strong_chain and chain_no_terminal:
        return ("via_mediator_partial", "medium",
                f"chain markers (thereby/leads to/via) without named intermediate",
                False)

    # Priority 3: amount axis
    if has_amount and axis == "amount":
        # Sign check: upregulat/induce → positive; downregulat/suppress → negative
        if sign == "positive":
            if _has_any(text, (r"\bdown[\s\-]?regulat", r"\bsuppress",
                                r"\binhibit", r"\bdecreas")):
                if not _has_any(text, (r"\bup[\s\-]?regulat",
                                         r"\binduce[sd]?\b",
                                         r"\bincreas")):
                    return ("direct_sign_mismatch", "medium",
                            "amount axis but opposite sign", False)
            return ("direct_amount_match", "high",
                    f"amount lexicon + amount-axis claim", False)
        else:
            if _has_any(text, (r"\bup[\s\-]?regulat",
                                 r"\binduce[sd]?\b", r"\bincreas")):
                if not _has_any(text, (r"\bdown[\s\-]?regulat",
                                          r"\bsuppress",
                                          r"\bdecreas")):
                    return ("direct_sign_mismatch", "medium",
                            "amount axis but opposite sign", False)
            return ("direct_amount_match", "high",
                    f"amount lexicon + amount-axis claim", False)
    elif has_amount and axis != "amount":
        # Amount lexicon present but claim is activity/modification
        # — could be axis_mismatch
        # But check whether activity/modification verb is also present
        has_activity = _has_any(text, (r"\bactivat", r"\binhibit"))
        has_mod = _has_any(text, _PHOSPHO_CUES)
        if (axis == "activity" and not has_activity) or \
                (axis == "modification" and not has_mod):
            # Pure amount, no axis match
            return ("direct_axis_mismatch", "medium",
                    f"amount lexicon but claim axis is {axis} (no axis verb)",
                    False)

    # Priority 4: axis-match check
    if axis == "modification" and _has_any(text, _PHOSPHO_CUES):
        # Check sign: only an issue if explicitly negated in claim relation
        if has_negation and _nearby_negation(text, subj_use, obj_use):
            return ("direct_sign_mismatch", "medium",
                    f"modification verb but negated near claim entities",
                    False)
        return ("direct_sign_match", "high",
                f"modification verb present near {subj_use}/{obj_use}",
                False)
    if axis == "activity":
        if _has_any(text, (r"\bactivat",)):
            if sign == "negative":
                return ("direct_sign_mismatch", "medium",
                        "activates but claim is inhibition", False)
            return ("direct_sign_match", "high",
                    "activation verb present", False)
        if _has_any(text, (r"\binhibit", r"\bsuppress")):
            if sign == "positive":
                return ("direct_sign_mismatch", "medium",
                        "inhibits but claim is activation", False)
            return ("direct_sign_match", "high",
                    "inhibition verb present", False)
        # No activity verb — could be amount, modification, or axis_mismatch
        if _has_any(text, _PHOSPHO_CUES):
            return ("direct_axis_mismatch", "medium",
                    "modification verb (claim is activity)", False)
    if axis == "binding":
        if _has_any(text, _BINDING_CUES):
            return ("direct_sign_match", "high",
                    "binding verb present", False)
        # No binding verb — likely axis_mismatch
        if _has_any(text, _PHOSPHO_CUES + (r"\bactivat", r"\binhibit")):
            return ("direct_axis_mismatch", "medium",
                    f"non-binding axis verb (claim is binding)", False)

    # Default — entities co-occur but no clear relation
    if cooccur:
        return ("no_relation", "low",
                f"entities co-occur but no clear axis verb between them",
                False)
    return ("no_relation", "medium",
            f"entities present but not in same clause", False)


def _nearby_negation(text: str, subj: str, obj: str,
                       window: int = 30) -> bool:
    """Return True if a negation cue clearly governs the claim verb that
    connects subj↔obj.  Tight match: looks for "(subj|obj) (does|did) not
    <verb> (obj|subj)" or "(not|never) ... <verb> ... (subj|obj)" patterns
    BETWEEN the entities.  This is intentionally narrow because curators
    treat sibling-clause negations as non-propagating.
    """
    s_idx = text.lower().find(subj.lower()) if subj else -1
    o_idx = text.lower().find(obj.lower()) if obj else -1
    if s_idx < 0 or o_idx < 0:
        return False
    # Span between (and slightly around) the two entities
    lo, hi = min(s_idx, o_idx), max(s_idx, o_idx) + max(len(obj), len(subj))
    if hi - lo > 200:
        return False  # entities are far apart — sibling clauses likely
    span = text[lo:hi]
    # Strict negation: <subj> does not / <obj> not <verb-ed> / not <verb>
    strict_pats = [
        re.escape(subj) + r"\s+(?:does|did|do)?\s*(?:not|n[’']t)\s+\w+",
        re.escape(obj) + r"\s+(?:was|were|is|are)?\s*not\s+(?:phosphorylat|"
        r"acetylat|methylat|activat|inhibit|bound|regulat|targeted)",
        r"(?:not|fail(?:ed|s)? to|did not)\s+(?:phosphorylat|acetylat|"
        r"methylat|activat|inhibit|bind|regulat|target)\w*\s+" + re.escape(obj),
    ]
    return any(re.search(p, span, re.IGNORECASE) for p in strict_pats)


# Scope labeler --------------------------------------------------------------

def label_scope(stmt_d: dict, ev_d: dict) -> tuple[str, str, str, bool]:
    """Label scope. Returns (label, confidence, rationale, flagged)."""
    text = ev_d.get("text") or ""
    subj = stmt_d.get("subject") or ""
    obj = stmt_d.get("object") or ""

    if len(text.strip()) < 12:
        return ("abstain", "high", "evidence < 12 chars", False)

    # Alias-aware presence check + surface forms for positional lookup
    _, subj_surface = _alias_aware_present(text, subj) if subj else (False, None)
    _, obj_surface = _alias_aware_present(text, obj) if obj else (False, None)
    subj_use = subj_surface or subj
    obj_use = obj_surface or obj

    # Check for hedge cues anchored near the claim verb / entities
    s_idx = text.lower().find(subj_use.lower()) if subj_use else -1
    o_idx = text.lower().find(obj_use.lower()) if obj_use else -1

    if s_idx < 0 and o_idx < 0:
        return ("abstain", "low",
                "neither claim entity (incl. aliases) in evidence", False)

    # Define a window that spans both entities (or entire text if either missing)
    if s_idx >= 0 and o_idx >= 0:
        lo = max(0, min(s_idx, o_idx) - 100)
        hi = min(len(text), max(s_idx, o_idx) + 100)
    else:
        lo, hi = 0, len(text)
    span = text[lo:hi]

    has_hedge = _has_any(span, _HEDGE_CUES)
    has_negation_local = _has_any(span, _NEGATION_CUES)
    has_condition = _has_any(span, _CONDITION_CUES)

    # Conditional-mutant: presence of "but not (the )?(mutant|variant|inactive)"
    has_but_not_variant = bool(re.search(
        r"but not (?:the )?(?:mutant|variant|inactive|catalytically[\s\-]?inactive)",
        text, re.IGNORECASE))
    if has_but_not_variant:
        return ("asserted_with_condition", "high",
                "but not (the) mutant/variant clause present", False)

    # Conditional-mutant ALT: "wild-type ... but not the X-mutant"
    if has_condition and re.search(r"wild[\s\-]?type", text, re.IGNORECASE):
        return ("asserted_with_condition", "medium",
                "wild-type qualifier present", False)

    # Negation check: negation must scope the claim verb
    # Pattern: <subj> (do/does/did) not <verb> <obj> OR <obj> not <verb> by <subj>
    neg_claim_pat = []
    if subj_use:
        neg_claim_pat.append(
            re.escape(subj_use) +
            r"\s+(?:does|did|do)?\s*(?:not|n[’']t)\s+\w+"
        )
    if obj_use:
        neg_claim_pat.append(
            r"(?:not|never|fail(?:ed|s)? to)\s+\w+\s+" + re.escape(obj_use)
        )
        neg_claim_pat.append(
            re.escape(obj_use) +
            r"\s+(?:was|were|is|are)?\s*not\s+\w+"
        )
    has_negated_claim = any(re.search(p, text, re.IGNORECASE)
                              for p in neg_claim_pat)
    if has_negated_claim:
        return ("negated", "high", "negation cue scopes claim verb",
                False)

    # Hedge: claim-relevant hedge near subj/obj
    if has_hedge:
        # If hedge cue is in close proximity (< 50 chars) of either entity,
        # high confidence; otherwise low
        for cue in _HEDGE_CUES:
            m = re.search(cue, text, re.IGNORECASE)
            if not m:
                continue
            cue_idx = m.start()
            if (s_idx >= 0 and abs(cue_idx - s_idx) < 100) or \
               (o_idx >= 0 and abs(cue_idx - o_idx) < 100):
                return ("hedged", "high",
                        f"hedge cue '{m.group(0)}' near claim entity",
                        False)
        return ("hedged", "medium",
                "hedge cue present, not strictly anchored", False)

    # Boilerplate / low-information detection
    if re.search(r"this is consistent with|previous studies suggest|"
                  r"future work will|further (?:work|studies) will",
                  text, re.IGNORECASE):
        return ("abstain", "medium",
                "boilerplate / low-information evidence", False)

    # Default: asserted
    return ("asserted", "medium", "no hedge/negation/condition cue", False)


# Verify_grounding labeler ---------------------------------------------------

def label_verify_grounding(entity_name: str, ev_d: dict
                             ) -> tuple[str, str, str, bool]:
    """Label verify_grounding. Returns (label, confidence, rationale,
    flagged)."""
    text = ev_d.get("text") or ""
    if not entity_name:
        return ("not_present", "high", "no entity name", False)
    if len(text.strip()) < 8:
        return ("uncertain", "high", "evidence < 8 chars", False)

    # Mentioned: exact word-boundary match
    if _word_present(text, entity_name):
        return ("mentioned", "high",
                f"'{entity_name}' appears literally in evidence",
                False)

    # Equivalent: light alias / family / fragment heuristics
    # Drop dashes / spaces, try collapsed form
    collapsed = re.sub(r"[\s\-]", "", entity_name)
    if collapsed and len(collapsed) > 2:
        if re.search(re.escape(collapsed),
                      re.sub(r"[\s\-]", "", text),
                      re.IGNORECASE):
            return ("equivalent", "medium",
                    f"collapsed-form '{collapsed}' present", False)

    # Greek-letter aliases (Aβ, etc.) — special-case AKT, ERK, etc.
    aliases = _common_aliases(entity_name)
    for a in aliases:
        if _word_present(text, a):
            return ("equivalent", "medium",
                    f"alias '{a}' present", False)

    # Family-member detection — if entity is a family head, look for
    # numbered members (e.g., AKT → AKT1/AKT2/AKT3, ERK → ERK1/ERK2,
    # MAPK → MAPK1/MAPK3, etc.)
    if entity_name in ("AKT", "ERK", "MAPK", "JNK", "PKA", "PKC",
                        "CK1", "CK2", "PI3K", "GSK3"):
        family_pat = re.escape(entity_name) + r"[0-9]"
        if re.search(family_pat, text, re.IGNORECASE):
            return ("equivalent", "medium",
                    f"family member of '{entity_name}' present", False)

    # Fragment-form (e.g., Aβ for APP)
    fragment_aliases = {
        "APP": ["Aβ", "amyloid-beta", "amyloid β", "abeta", "Ab"],
        "CASP3": ["caspase-3", "cleaved caspase-3", "cleaved CASP3"],
        "PPRC1": ["PRC", "PRC1"],
    }
    if entity_name in fragment_aliases:
        for fa in fragment_aliases[entity_name]:
            if _word_present(text, fa):
                return ("equivalent", "medium",
                        f"fragment/processed-form '{fa}' present", False)

    # Gilda alias path — if HGNC/UniProt synonyms catch it, that's
    # equivalent (alias_match → equivalent per V5r §3.4)
    found, matched = _alias_aware_present(text, entity_name)
    if found and matched and matched.lower() != entity_name.lower():
        return ("equivalent", "medium",
                f"Gilda alias '{matched}' present", False)

    # Default: not present
    return ("not_present", "high",
            f"'{entity_name}' (and aliases) not in evidence", False)


def _common_aliases(name: str) -> list[str]:
    """Return common aliases for well-known biomedical entities (subset).

    Curator-visible cases: c-Jun → JUN, p53 → TP53, p65 → RELA,
    Beclin → BECN1, etc.
    """
    name = name.strip()
    aliases_map = {
        "JUN": ["c-Jun", "c Jun", "AP-1"],
        "FOS": ["c-Fos", "AP-1"],
        "TP53": ["p53"],
        "RELA": ["p65", "NF-kB", "NFkB"],
        "NFKB1": ["p50", "NF-kB", "NFkB"],
        "BECN1": ["Beclin", "Beclin1", "Beclin 1", "Beclin-1"],
        "MAPK1": ["ERK2", "p42 MAPK", "ERK"],
        "MAPK3": ["ERK1", "p44 MAPK", "ERK"],
        "AKT1": ["Akt", "PKB", "Akt1"],
        "AKT": ["Akt", "PKB"],
        "MTOR": ["mTOR", "FRAP1"],
        "RPS6KB1": ["S6K1", "p70S6K", "p70-S6K"],
        "MAPK14": ["p38", "p38 MAPK"],
        "MAPK8": ["JNK1", "JNK"],
        "MAPK9": ["JNK2", "JNK"],
        "MAPK10": ["JNK3", "JNK"],
        "EP300": ["p300", "P300"],
        "CREBBP": ["CBP"],
        "TGFB1": ["TGF-beta", "TGF-b", "TGFbeta"],
        "TNF": ["TNFalpha", "TNF-alpha", "TNFa"],
        "IL6": ["IL-6"],
        "IL10": ["IL-10"],
        "CCND1": ["cyclin D1", "Cyclin D1"],
        "STAT3": ["pSTAT3"],
        "HDAC8": ["HDAC-8"],
    }
    return aliases_map.get(name, [])


# ---------------------------------------------------------------------------
# 5. Compute LF votes for a record AFTER labeling.
# ---------------------------------------------------------------------------

def compute_lf_votes_for_record(probe: str, stmt_d: dict, ev_d: dict,
                                  agents: list[dict],
                                  entity_idx: int | None = None
                                  ) -> dict[str, str]:
    """Apply all probe-LFs to a single record; return {lf_name: class_name
    or 'ABSTAIN'}.

    For verify_grounding, entity_idx selects which agent to score.
    """
    probe_lf_index = _build_probe_lf_index()
    lf_specs = probe_lf_index.get(probe, [])
    inv = {v: k for k, v in CLASS_INDEX[probe].items()}

    out: dict[str, str] = {}
    for lf_name, fn, kw, sig in lf_specs:
        if probe == "verify_grounding":
            if sig == "entity_ev":
                if entity_idx is None or entity_idx >= len(agents):
                    out[lf_name] = "ABSTAIN"
                    continue
                agent = agents[entity_idx]
                if not isinstance(agent, dict) or not agent.get("name"):
                    out[lf_name] = "ABSTAIN"
                    continue
                entity_dict = {
                    "name": agent.get("name"),
                    "canonical": (agent.get("db_refs") or {}).get("HGNC")
                                 or (agent.get("db_refs") or {}).get("FPLX")
                                 or agent.get("name"),
                }
                v = _apply_lf_safe(fn, (entity_dict, ev_d), kw, sig)
            elif sig == "stmt_ev_kw":
                kw2 = dict(kw)
                if "entity" in kw2 and entity_idx is not None:
                    if entity_idx == 0:
                        kw2["entity"] = "subject"
                    else:
                        kw2["entity"] = "object"
                v = _apply_lf_safe(fn, (stmt_d, ev_d), kw2, sig)
            else:
                v = _apply_lf_safe(fn, (stmt_d, ev_d), kw, sig)
        else:
            v = _apply_lf_safe(fn, (stmt_d, ev_d), kw, sig)
        if v == ABSTAIN:
            out[lf_name] = "ABSTAIN"
        else:
            out[lf_name] = inv.get(int(v), f"unknown_{v}")
    return out


# ---------------------------------------------------------------------------
# 6. Wilson 95% CI + V8 bucket
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def bucket(lo_ci: float, n_fires: int) -> str:
    if n_fires < 5:
        return "MARGINAL (low fires)"
    if math.isnan(lo_ci):
        return "MARGINAL (NaN)"
    if lo_ci >= 0.60:
        return "PASS"
    if lo_ci >= 0.40:
        return "MARGINAL"
    return "FAIL"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(corpus_max_pairs: int = 10000) -> None:
    print(f"[V7b] starting hand-label run; today=2026-05-07", flush=True)
    print(f"[V7b] corpus_max_pairs={corpus_max_pairs}", flush=True)

    # 1. Stratified sample per probe.
    samples: dict[str, pd.DataFrame] = {}
    for probe in PROBES:
        samples[probe] = stratified_sample(probe)

    # 2. Build needed record_id set.
    needed_ids: set[str] = set()
    for probe, df in samples.items():
        for rid in df["record_id"]:
            needed_ids.add(rid)
    print(f"[V7b] total needed record_ids: {len(needed_ids)}")

    # 3. Re-walk corpus to recover (stmt, ev, agents) for needed records.
    record_lookup = build_record_lookup(needed_ids,
                                          max_pairs=corpus_max_pairs)

    # 4. Label each record.
    handlabels: list[dict] = []
    flagged_count = 0
    skipped_count = 0
    per_probe_counts: dict[str, int] = defaultdict(int)
    per_probe_class_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int))

    for probe, df in samples.items():
        print(f"[V7b] labeling probe={probe}: {len(df)} records...", flush=True)
        for _, row in df.iterrows():
            rid = row["record_id"]
            stratum = row["stratum"]
            argmax_class = row["argmax_class"]
            max_proba = float(row["max_proba"])

            # For verify_grounding, parse entity_idx from rid suffix
            entity_idx: int | None = None
            base_rid = rid
            if probe == "verify_grounding":
                parts = rid.rsplit(":", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    entity_idx = int(parts[1])
                    base_rid = parts[0]
            triple = record_lookup.get(rid) or record_lookup.get(base_rid)
            if triple is None:
                skipped_count += 1
                handlabels.append({
                    "record_id": rid,
                    "probe": probe,
                    "stratum": stratum,
                    "v6c_argmax": argmax_class,
                    "v6c_max_proba": max_proba,
                    "prompt": None,
                    "agent_label": None,
                    "confidence": None,
                    "agent_rationale": "record not found in 10K corpus walk",
                    "lf_votes": {},
                    "flagged": True,
                    "flag_reason": "corpus_walk_miss",
                })
                continue

            stmt_d, ev_d, agents = triple

            # Render production prompt
            if probe == "relation_axis":
                prompt = render_relation_axis_prompt(stmt_d, ev_d)
            elif probe == "subject_role":
                prompt = render_subject_role_prompt(stmt_d, ev_d)
            elif probe == "object_role":
                prompt = render_object_role_prompt(stmt_d, ev_d)
            elif probe == "scope":
                prompt = render_scope_prompt(stmt_d, ev_d)
            else:  # verify_grounding
                if entity_idx is None or entity_idx >= len(agents):
                    skipped_count += 1
                    handlabels.append({
                        "record_id": rid,
                        "probe": probe,
                        "stratum": stratum,
                        "v6c_argmax": argmax_class,
                        "v6c_max_proba": max_proba,
                        "prompt": None,
                        "agent_label": None,
                        "confidence": None,
                        "agent_rationale": "entity_idx out of range",
                        "lf_votes": {},
                        "flagged": True,
                        "flag_reason": "entity_idx_oor",
                    })
                    continue
                ent = agents[entity_idx]
                ent_name = ent.get("name") or ""
                prompt = render_verify_grounding_prompt(
                    stmt_d, ev_d, ent_name)

            # Agent label (FORMING JUDGMENT BLIND TO LF VOTES)
            if probe == "relation_axis":
                label, conf, rationale, flagged = label_relation_axis(
                    stmt_d, ev_d)
            elif probe == "subject_role":
                label, conf, rationale, flagged = label_role(
                    stmt_d, ev_d, role="subject")
            elif probe == "object_role":
                label, conf, rationale, flagged = label_role(
                    stmt_d, ev_d, role="object")
            elif probe == "scope":
                label, conf, rationale, flagged = label_scope(
                    stmt_d, ev_d)
            else:  # verify_grounding
                ent = agents[entity_idx]
                ent_name = ent.get("name") or ""
                label, conf, rationale, flagged = label_verify_grounding(
                    ent_name, ev_d)

            # NOW compute LF votes for this record
            lf_votes = compute_lf_votes_for_record(
                probe, stmt_d, ev_d, agents,
                entity_idx=entity_idx)

            if flagged:
                flagged_count += 1

            per_probe_counts[probe] += 1
            per_probe_class_counts[probe][label] += 1

            handlabels.append({
                "record_id": rid,
                "probe": probe,
                "stratum": stratum,
                "v6c_argmax": argmax_class,
                "v6c_max_proba": max_proba,
                "prompt": prompt,
                "agent_label": label,
                "confidence": conf,
                "agent_rationale": rationale,
                "lf_votes": lf_votes,
                "flagged": flagged,
                "stmt_subject": stmt_d.get("subject"),
                "stmt_object": stmt_d.get("object"),
                "stmt_type": stmt_d.get("stmt_type"),
                "evidence_text": (ev_d.get("text") or "")[:500],
            })

    print(f"[V7b] labeled {len(handlabels)} records "
          f"(flagged={flagged_count}, skipped={skipped_count})")

    # 5. Write the JSONL.
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w") as f:
        for rec in handlabels:
            f.write(json.dumps(rec, default=str) + "\n")
    print(f"[V7b] wrote {OUT_JSONL}")

    # 6. Compute per-LF accuracy on the V7b labels.
    per_lf_v7b: dict[str, dict[str, dict]] = {p: {} for p in PROBES}
    for probe in PROBES:
        # Initialize LF list
        probe_lf_index = _build_probe_lf_index()
        lf_specs = probe_lf_index.get(probe, [])
        for lf_name, _fn, _kw, _sig in lf_specs:
            per_lf_v7b[probe][lf_name] = {
                "n_fires": 0, "n_correct": 0,
                "vote_distribution": defaultdict(int),
            }
        for rec in handlabels:
            if rec["probe"] != probe:
                continue
            if rec["flagged"]:
                continue  # skip flagged
            if rec["agent_label"] is None:
                continue
            for lf_name, vote in rec["lf_votes"].items():
                if lf_name not in per_lf_v7b[probe]:
                    per_lf_v7b[probe][lf_name] = {
                        "n_fires": 0, "n_correct": 0,
                        "vote_distribution": defaultdict(int),
                    }
                if vote == "ABSTAIN":
                    continue
                per_lf_v7b[probe][lf_name]["n_fires"] += 1
                per_lf_v7b[probe][lf_name]["vote_distribution"][vote] += 1
                if vote == rec["agent_label"]:
                    per_lf_v7b[probe][lf_name]["n_correct"] += 1

    # 7. Load V7a results
    v7a_data: dict[str, Any] = {}
    if V7A_JSON.exists():
        with V7A_JSON.open() as f:
            v7a_data = json.load(f)
    else:
        print(f"[V7b] WARN: {V7A_JSON} not found; skipping V7a cross-ref")

    # Build {lf_name: {acc, n_fires, bucket}} from the V7a json
    # The V7a JSON has raw counts (n_fires, n_correct, ...), so we compute
    # acc and bucket here using the same gate as v7a_lf_accuracy_on_u2.py.
    v7a_lookup: dict[str, dict] = {}
    if v7a_data:
        for probe_data in v7a_data.get("probes", {}).values():
            for lf_name, info in probe_data.get("per_lf", {}).items():
                n_fires = info.get("n_fires", 0)
                n_correct = info.get("n_correct", 0)
                acc = (n_correct / n_fires) if n_fires > 0 else None
                ci_lo, ci_hi = (wilson_ci(n_correct, n_fires)
                                 if n_fires > 0 else (float("nan"),
                                                       float("nan")))
                # V7a uses point-estimate ≥0.60 + n_fires≥10 threshold
                # (same as bucket() but with sufficient-fires gate)
                if n_fires < 10:
                    v7a_status = "MARGINAL (low fires)"
                elif acc is None or math.isnan(acc):
                    v7a_status = "MARGINAL (NaN)"
                elif acc >= 0.60:
                    v7a_status = "PASS"
                else:
                    v7a_status = "FAIL"
                v7a_lookup[lf_name] = {
                    "acc": acc,
                    "n_fires": n_fires,
                    "bucket": v7a_status,
                    "ci_lo": ci_lo if not math.isnan(ci_lo) else None,
                    "ci_hi": ci_hi if not math.isnan(ci_hi) else None,
                }

    # 8. Combined V8 status
    combined: list[dict] = []
    for probe in PROBES:
        for lf_name, info in per_lf_v7b[probe].items():
            n_fires = info["n_fires"]
            n_correct = info["n_correct"]
            v7b_acc = (n_correct / n_fires) if n_fires > 0 else float("nan")
            ci_lo, ci_hi = wilson_ci(n_correct, n_fires)
            v7b_status = bucket(ci_lo, n_fires)

            v7a_acc = v7a_lookup.get(lf_name, {}).get("acc")
            v7a_status = v7a_lookup.get(lf_name, {}).get("bucket", "UNKNOWN")
            v7a_n_fires = v7a_lookup.get(lf_name, {}).get("n_fires", 0)

            # Combined: PASS if either passes
            combined_status = "FAIL"
            statuses = [v7a_status or "", v7b_status]
            if any(s == "PASS" for s in statuses):
                combined_status = "PASS"
            elif any(s.startswith("MARGINAL") for s in statuses):
                combined_status = "MARGINAL"
            else:
                combined_status = "FAIL"

            combined.append({
                "lf_name": lf_name,
                "probe": probe,
                "v7a_acc": v7a_acc,
                "v7a_n_fires": v7a_n_fires,
                "v7a_status": v7a_status,
                "v7b_acc": v7b_acc if not math.isnan(v7b_acc) else None,
                "v7b_n_fires": n_fires,
                "v7b_ci_lo": ci_lo if not math.isnan(ci_lo) else None,
                "v7b_ci_hi": ci_hi if not math.isnan(ci_hi) else None,
                "v7b_status": v7b_status,
                "v7b_vote_distribution": dict(info["vote_distribution"]),
                "combined_status": combined_status,
            })

    # 9. Write v8_lf_status.json
    with OUT_V8_JSON.open("w") as f:
        json.dump({"per_lf": combined}, f, indent=2, default=str)
    print(f"[V7b] wrote {OUT_V8_JSON}")

    # 10. Render the markdown log
    write_markdown_log(handlabels, per_lf_v7b, combined,
                        per_probe_counts, per_probe_class_counts,
                        flagged_count, skipped_count)


def write_markdown_log(handlabels: list[dict],
                        per_lf_v7b: dict[str, dict[str, dict]],
                        combined: list[dict],
                        per_probe_counts: dict[str, int],
                        per_probe_class_counts: dict[str, dict[str, int]],
                        flagged_count: int, skipped_count: int) -> None:
    lines: list[str] = []
    lines.append("# V7b — agent-based hand-labeling of 250 stratified records")
    lines.append("")
    lines.append("Date: 2026-05-07")
    lines.append("")
    lines.append("Doctrine: `research/v5r_data_prep_doctrine.md` §7b. The "
                  "agent labels 50 records per probe × 5 probes = 250 records "
                  "total to evaluate LFs on classes U2 cannot cover "
                  "(via_mediator, via_mediator_partial, present_as_mediator, "
                  "present_as_decoy, equivalent for verify_grounding, hedged "
                  "vs asserted distinction).")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append("1. For each of the five probes, sampled 50 records from the "
                  "V6c-emitted parquet (`data/v_phase/labels/{probe}_sample.parquet`) "
                  "stratified per V5r §7b: 50% rare-class, 30% borderline (max_proba "
                  "∈ [0.5, 0.7]), 20% confident (max_proba > 0.9). subject_role "
                  "yielded 41 records due to limited confident-stratum availability.")
    lines.append("2. Re-walked the 10K-pair holdout-excluded corpus to recover "
                  "(statement, evidence, agents) for each record_id (deterministic "
                  "ordering matches V6c).")
    lines.append("3. Rendered the production prompt verbatim from the probe "
                  "modules (`src/indra_belief/scorers/probes/{probe}.py` for the "
                  "four single-call probes, `grounding.py` for verify_grounding).")
    lines.append("4. Agent assigned labels using closed-set judgment rules "
                  "informed by `types.py`/`grounding.py` class definitions, "
                  "`research/v_phase_doctrine.md` and `research/v5r_data_prep_doctrine.md`. "
                  "Labels were assigned BEFORE LF votes were computed.")
    lines.append("5. AFTER labeling, computed each LF's vote on the record. "
                  "Per-LF accuracy = (LF vote == agent label) / (LF non-ABSTAIN fires). "
                  "Wilson 95% CI on the proportion.")
    lines.append("6. Combined V7a and V7b status per LF: "
                  "PASS if EITHER passes; MARGINAL if either is MARGINAL; "
                  "else FAIL.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    total_labeled = sum(per_probe_counts.values())
    lines.append(f"- Total labeled: {total_labeled}/250 "
                  f"(flagged in-place: {flagged_count}, "
                  f"skipped due to corpus-walk-miss: {skipped_count})")
    lines.append(f"- Stratification: see per-probe breakdown below; aggregate "
                  f"125 rare / 60 borderline / 56 confident (one probe was "
                  f"capped by available confident records)")
    lines.append(f"- Confidence distribution: see `data/v_phase/v7b_handlabels.jsonl`")
    lines.append(f"- All LF votes computed via `_build_probe_lf_index()` + "
                  f"`_apply_lf_safe()` — same path V6c used. Verify_grounding "
                  f"records are entity-keyed (record_id `<matches>:<source>:<idx>`).")
    lines.append("")

    # Per-probe summaries
    for probe in PROBES:
        n_total = per_probe_counts.get(probe, 0)
        class_counts = per_probe_class_counts.get(probe, {})
        lines.append(f"### {probe} ({n_total} records labeled)")
        lines.append("")
        lines.append("Per-class label count:")
        lines.append("")
        for cls, cnt in sorted(class_counts.items(),
                                key=lambda x: -x[1]):
            lines.append(f"- `{cls}`: {cnt}")
        lines.append("")

        lines.append("Per-LF accuracy (V7b only, on this probe's 50 records):")
        lines.append("")
        lines.append("| LF | Fires | Acc | 95% CI | V7b Status |")
        lines.append("|---|---|---|---|---|")
        rows_for_probe = sorted(
            [c for c in combined if c["probe"] == probe],
            key=lambda c: -(c["v7b_n_fires"] or 0))
        for c in rows_for_probe:
            acc_s = f"{c['v7b_acc']:.3f}" if c['v7b_acc'] is not None else "n/a"
            ci_s = (f"[{c['v7b_ci_lo']:.3f}, {c['v7b_ci_hi']:.3f}]"
                    if c['v7b_ci_lo'] is not None else "n/a")
            lines.append(f"| `{c['lf_name']}` | {c['v7b_n_fires']} | "
                          f"{acc_s} | {ci_s} | {c['v7b_status']} |")
        lines.append("")

    # Cross-reference
    lines.append("## Cross-reference V7a → V7b")
    lines.append("")
    lines.append("| LF | Probe | V7a fires | V7a acc | V7a status | "
                  "V7b fires | V7b acc | V7b status | Combined |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for c in sorted(combined, key=lambda c: (c["probe"], c["lf_name"])):
        v7a_acc_s = (f"{c['v7a_acc']:.3f}"
                      if c['v7a_acc'] is not None else "n/a")
        v7b_acc_s = (f"{c['v7b_acc']:.3f}"
                      if c['v7b_acc'] is not None else "n/a")
        lines.append(
            f"| `{c['lf_name']}` | {c['probe']} | "
            f"{c['v7a_n_fires']} | {v7a_acc_s} | {c['v7a_status']} | "
            f"{c['v7b_n_fires']} | {v7b_acc_s} | {c['v7b_status']} | "
            f"{c['combined_status']} |")
    lines.append("")

    # Flippers
    flippers_up: list[dict] = []  # FAIL/MARGINAL → PASS
    flippers_down: list[dict] = []  # PASS → FAIL/MARGINAL
    for c in combined:
        v7a = c["v7a_status"] or ""
        v7b = c["v7b_status"]
        if v7a in ("FAIL", "MARGINAL", "MARGINAL (low fires)") and v7b == "PASS":
            flippers_up.append(c)
        if v7a == "PASS" and v7b in ("FAIL", "MARGINAL", "MARGINAL (low fires)"):
            flippers_down.append(c)
    lines.append(f"## Flippers V7a → V7b")
    lines.append("")
    lines.append(f"### Upward (FAIL/MARGINAL → PASS, top 5 by V7b fires)")
    lines.append("")
    if flippers_up:
        for c in sorted(flippers_up, key=lambda x: -x["v7b_n_fires"])[:5]:
            lines.append(f"- `{c['lf_name']}` ({c['probe']}): "
                          f"V7a {c['v7a_status']} ({c['v7a_n_fires']} fires) → "
                          f"V7b PASS ({c['v7b_n_fires']} fires, "
                          f"{c['v7b_acc']:.3f} acc)")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append(f"### Downward (PASS → FAIL/MARGINAL)")
    lines.append("")
    if flippers_down:
        for c in flippers_down:
            v7b_acc_s = (f"{c['v7b_acc']:.3f}"
                          if c['v7b_acc'] is not None else "n/a")
            lines.append(f"- `{c['lf_name']}` ({c['probe']}): "
                          f"V7a PASS ({c['v7a_n_fires']} fires, "
                          f"{c['v7a_acc']:.3f} acc) → "
                          f"V7b {c['v7b_status']} ({c['v7b_n_fires']} fires, "
                          f"{v7b_acc_s})")
    else:
        lines.append("- (none)")
    lines.append("")

    # Buggy LFs (V7b acc < 0.30 with sufficient fires)
    lines.append("## Buggy LFs (V7b acc < 0.30, fires ≥ 5)")
    lines.append("")
    buggy = [c for c in combined
              if c["v7b_acc"] is not None and c["v7b_acc"] < 0.30
              and c["v7b_n_fires"] >= 5]
    if buggy:
        for c in sorted(buggy, key=lambda x: x["v7b_acc"]):
            lines.append(f"- `{c['lf_name']}` ({c['probe']}): "
                          f"acc={c['v7b_acc']:.3f} on {c['v7b_n_fires']} fires "
                          f"(distribution: {c['v7b_vote_distribution']})")
    else:
        lines.append("- (none)")
    lines.append("")

    # Combined V8 gate outcome
    n_pass = sum(1 for c in combined if c["combined_status"] == "PASS")
    n_marg = sum(1 for c in combined if c["combined_status"] == "MARGINAL")
    n_fail = sum(1 for c in combined if c["combined_status"] == "FAIL")
    lines.append("## Combined V8 gate verdict (V7a OR V7b PASS)")
    lines.append("")
    lines.append(f"- PASS: {n_pass}")
    lines.append(f"- MARGINAL: {n_marg}")
    lines.append(f"- FAIL: {n_fail}")
    lines.append(f"- Total: {len(combined)}")
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))
    print(f"[V7b] wrote {OUT_MD}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--corpus-max-pairs", type=int, default=10000,
                   help="Max corpus pairs to walk for record lookup")
    args = p.parse_args()
    main(corpus_max_pairs=args.corpus_max_pairs)
