# genesis/reasoning/beam_chain.py
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from genesis.core.sdr import SDR
from genesis.core.cell import Cell, Rule
from genesis.core.organism import Organism
from genesis.core.working_memory import WorkingMemory
from genesis.reasoning.forward_chain import ChainStep, ReasoningResult

MAX_DEPTH = 8
THETA_EARLY_RETURN = 0.65   # accept depth-0 answer only if very confident
THETA_ANSWER = 0.35         # accept deeper answers with lower bar
BEAM_WIDTH = 3              # max parallel chains explored at each depth
BEAM_DECAY = 0.98           # less aggressive than ForwardChain's 0.95


@dataclass
class _Beam:
    route_sdr: SDR
    facts: SDR
    chain: List[ChainStep] = field(default_factory=list)
    confidence_path: List[float] = field(default_factory=list)


class BeamChain:
    """Width-3 beam search forward inference.

    Difference from ForwardChain:
    - Depth-0 answers are only accepted if confidence >= THETA_EARLY_RETURN (0.65).
    - When depth-0 confidence is medium (0.35-0.65), the beam continues exploring
      2-hop chains that may reach THETA_ANSWER at depth 1.
    - Up to BEAM_WIDTH parallel chains are explored at each depth."""

    def reason(self, query: SDR, organism: Organism,
               working_memory: WorkingMemory,
               max_depth: int = MAX_DEPTH) -> ReasoningResult:

        beams: List[_Beam] = [_Beam(route_sdr=query, facts=query)]

        for depth in range(max_depth):
            next_beams: List[_Beam] = []
            context = working_memory.union()

            for beam in beams:
                active_cells = organism.route(beam.route_sdr)
                if not active_cells:
                    continue

                # Context re-ranking within each beam
                if context.popcount() > 0:
                    active_cells.sort(
                        key=lambda c: c.receptive_field.similarity(context),
                        reverse=True,
                    )

                for cell in active_cells:
                    for rule, score in cell.apply_rules(beam.facts):
                        new_conf_path = beam.confidence_path + [rule.confidence]
                        total_conf = self._propagate(new_conf_path, depth + 1)
                        new_step = ChainStep(
                            precondition=rule.precondition,
                            postcondition=rule.postcondition,
                            rule_confidence=rule.confidence,
                            cell_id=cell.id,
                        )
                        new_chain = beam.chain + [new_step]

                        # Threshold depends on depth
                        threshold = THETA_EARLY_RETURN if depth == 0 else THETA_ANSWER
                        if total_conf > threshold:
                            return ReasoningResult(
                                answer=rule.postcondition,
                                chain=new_chain,
                                confidence=total_conf,
                            )

                        next_beams.append(_Beam(
                            route_sdr=rule.postcondition,
                            facts=beam.facts.union(rule.postcondition),
                            chain=new_chain,
                            confidence_path=new_conf_path,
                        ))

            if not next_beams:
                break

            # Prune to BEAM_WIDTH best beams by current total confidence
            next_beams.sort(
                key=lambda b: self._propagate(b.confidence_path, depth + 1),
                reverse=True,
            )
            beams = next_beams[:BEAM_WIDTH]

        return ReasoningResult(answer=None, chain=[], confidence=0.0)

    @staticmethod
    def _propagate(confidences: List[float], depth: int) -> float:
        if not confidences:
            return 0.0
        result = 1.0
        for c in confidences:
            result *= c
        return result * (BEAM_DECAY ** depth)
