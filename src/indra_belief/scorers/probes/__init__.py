"""S-phase probe pipeline.

Replaces the monolithic parse_evidence with four narrow probes
(subject_role, object_role, relation_axis, scope) routed through a
substrate-first orchestrator. Each probe commits a single decision from
a closed answer set; the adjudicator combines them via a flat decision
table.

See research/s_phase_doctrine.md for full design rationale.
"""
from indra_belief.scorers.probes import (
    object_role,
    relation_axis,
    scope,
    subject_role,
)
from indra_belief.scorers.probes.router import substrate_route
from indra_belief.scorers.probes.types import (
    ObjectRoleAnswer,
    PerturbationMarker,
    ProbeBundle,
    ProbeConfidence,
    ProbeKind,
    ProbeRequest,
    ProbeResponse,
    ProbeSource,
    RelationAxisAnswer,
    ScopeAnswer,
    SubjectRoleAnswer,
)

__all__ = [
    "ObjectRoleAnswer",
    "PerturbationMarker",
    "ProbeBundle",
    "ProbeConfidence",
    "ProbeKind",
    "ProbeRequest",
    "ProbeResponse",
    "ProbeSource",
    "RelationAxisAnswer",
    "ScopeAnswer",
    "SubjectRoleAnswer",
    "object_role",
    "relation_axis",
    "scope",
    "subject_role",
    "substrate_route",
]
