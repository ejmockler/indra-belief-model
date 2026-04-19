"""Evidence-quality scoring for INDRA biomedical text-mining extractions.

Public API:
    score_statement(statement, evidence, client) -> dict
        Score a single INDRA Statement + Evidence pair.
    ModelClient(model_name)
        Backend-agnostic transport (OpenAI-compatible, Anthropic).

See README for usage examples.
"""
from indra_belief.model_client import ModelClient, ModelResponse
from indra_belief.scorers.scorer import score_statement

__all__ = [
    "score_statement",
    "ModelClient",
    "ModelResponse",
]
