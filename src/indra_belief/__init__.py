"""Evidence-quality scoring for INDRA biomedical text-mining extractions.

Public API:
    score_statement(statement, client) -> list[dict]
        Score every evidence in an INDRA Statement. Mirrors INDRA's
        abstraction (a Statement owns a list of Evidence objects);
        returns one per-sentence verdict per evidence.
    score_evidence(statement, evidence, client) -> dict
        Score a single (Statement, Evidence) pair — the atomic unit.
        Use when you want to judge one evidence sentence without
        iterating the rest of the Statement's evidence list.
    ModelClient(model_name)
        Backend-agnostic transport (OpenAI-compatible, Anthropic).

See README for usage examples.
"""
from indra_belief.model_client import ModelClient, ModelResponse
from indra_belief.scorers.scorer import score_evidence, score_statement

__all__ = [
    "score_statement",
    "score_evidence",
    "ModelClient",
    "ModelResponse",
]
