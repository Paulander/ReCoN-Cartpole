from __future__ import annotations

from .actuators import clamp_force
from .scripts import ForceProposal


def proposal_score(proposal: ForceProposal) -> float:
    if proposal.suppressed:
        return 0.0
    if proposal.score > 0.0:
        return proposal.score
    return max(0.01, proposal.confidence) * (1.0 + proposal.urgency)


def arbitrate_force(proposals: list[ForceProposal], force_mag: float = 10.0) -> ForceProposal:
    active = [proposal for proposal in proposals if not proposal.suppressed and proposal_score(proposal) > 0.0]
    if not active:
        return ForceProposal("none", 0.0, 0.0, 0.0, "no proposal", suppressed=True)
    numerator = sum(p.force * proposal_score(p) for p in active)
    denominator = sum(proposal_score(p) for p in active)
    winner = max(active, key=proposal_score)
    return ForceProposal(
        source_node=winner.source_node,
        force=clamp_force(numerator / denominator, force_mag),
        confidence=winner.confidence,
        urgency=winner.urgency,
        reason=winner.reason,
        score=proposal_score(winner),
        raw_confidence=winner.raw_confidence,
        raw_urgency=winner.raw_urgency,
        select_edge_weight=winner.select_edge_weight,
        proposal_edge_weight=winner.proposal_edge_weight,
        bandit_score=winner.bandit_score,
        selection_multiplier=winner.selection_multiplier,
        selected=winner.selected,
        suppressed=winner.suppressed,
        selection_mode=winner.selection_mode,
    )

