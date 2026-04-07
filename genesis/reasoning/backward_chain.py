# genesis/reasoning/backward_chain.py
from dataclasses import dataclass, field
from typing import List, Optional
from genesis.core.sdr import SDR
from genesis.core.organism import Organism
from genesis.reasoning.confidence import propagate

MAX_DEPTH = 6
THETA_VERIFY = 0.35


@dataclass
class VerificationResult:
    verified: bool = False
    confidence: float = 0.0
    support_chain: List[tuple] = field(default_factory=list)


class BackwardChain:
    """Backward chaining: given a goal, find supporting facts in the colony."""

    def verify(self, goal: SDR, organism: Organism,
               known_facts: SDR, depth: int = 0) -> VerificationResult:

        if depth > MAX_DEPTH:
            return VerificationResult(verified=False)

        # If goal is already a known fact, it's trivially true
        if goal.similarity(known_facts) > THETA_VERIFY:
            return VerificationResult(
                verified=True,
                confidence=goal.similarity(known_facts),
                support_chain=[("known_fact", goal, 1.0)],
            )

        # Find rules whose postcondition matches the goal
        supporting: List[tuple] = []
        for cell in organism.cells.values():
            for rule in cell.rules:
                if rule.postcondition.similarity(goal) > THETA_VERIFY:
                    # Recursively verify precondition
                    sub = self.verify(rule.precondition, organism,
                                      known_facts, depth + 1)
                    if sub.verified:
                        conf = propagate(
                            [rule.confidence, sub.confidence], depth + 1
                        )
                        supporting.append((rule, sub, conf))

        if not supporting:
            return VerificationResult(verified=False)

        supporting.sort(key=lambda x: x[2], reverse=True)
        best_rule, best_sub, best_conf = supporting[0]
        chain = [(best_rule, goal, best_conf)] + best_sub.support_chain

        return VerificationResult(
            verified=True,
            confidence=best_conf,
            support_chain=chain,
        )
