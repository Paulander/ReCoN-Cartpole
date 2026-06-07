from __future__ import annotations

from .actuators import clamp_force
from .scripts import ForceProposal


def arbitrate_force(proposals: list[ForceProposal], force_mag: float = 10.0) -> ForceProposal:
    if not proposals:
        return ForceProposal("none", 0.0, 0.0, 0.0, "no proposal")
    numerator = sum(p.force * max(0.01, p.confidence) * (1.0 + p.urgency) for p in proposals)
    denominator = sum(max(0.01, p.confidence) * (1.0 + p.urgency) for p in proposals)
    winner = max(proposals, key=lambda p: p.confidence + p.urgency)
    return ForceProposal(
        source_node=winner.source_node,
        force=clamp_force(numerator / denominator, force_mag),
        confidence=winner.confidence,
        urgency=winner.urgency,
        reason=winner.reason,
    )

