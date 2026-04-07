# genesis/reasoning/forward_chain.py
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from genesis.core.sdr import SDR
from genesis.core.cell import Cell, Rule
from genesis.core.organism import Organism
from genesis.core.working_memory import WorkingMemory
from .confidence import propagate, DECAY

MAX_DEPTH = 8
THETA_ANSWER = 0.40
THETA_CONFIDENCE = 0.25


@dataclass
class ChainStep:
    precondition: SDR
    postcondition: SDR
    rule_confidence: float
    cell_id: str


@dataclass
class ReasoningResult:
    answer: Optional[SDR] = None
    chain: List[ChainStep] = field(default_factory=list)
    confidence: float = 0.0


class ForwardChain:
    """Forward causal chain inference through the cell colony."""

    def reason(self, query: SDR, organism: Organism,
               working_memory: WorkingMemory,
               max_depth: int = MAX_DEPTH) -> ReasoningResult:

        context = working_memory.union()
        facts = query.union(context)
        chain: List[ChainStep] = []
        confidence_path: List[float] = []

        for depth in range(max_depth):
            active_cells = organism.route(facts)
            if not active_cells:
                break

            new_facts: List[Tuple[SDR, float, Cell, Rule]] = []
            for cell in active_cells:
                for rule, score in cell.apply_rules(facts):
                    new_facts.append((rule.postcondition, score, cell, rule))

            if not new_facts:
                break

            # Pick highest-scoring new fact
            new_facts.sort(key=lambda x: x[1], reverse=True)
            best_post, best_score, best_cell, best_rule = new_facts[0]

            chain.append(ChainStep(
                precondition=best_rule.precondition,
                postcondition=best_post,
                rule_confidence=best_rule.confidence,
                cell_id=best_cell.id,
            ))
            confidence_path.append(best_rule.confidence)

            total_conf = propagate(confidence_path, depth + 1)
            if total_conf > THETA_ANSWER:
                return ReasoningResult(
                    answer=best_post,
                    chain=chain,
                    confidence=total_conf,
                )

            facts = facts.union(best_post)

        return ReasoningResult(answer=None, chain=chain, confidence=0.0)
