import math
from typing import List
from genesis.core.sdr import SDR
from genesis.core.cell import Cell, MAX_RULES, MIN_AGE, THETA_DEATH
from genesis.core.organism import Organism

THETA_MERGE = 0.70     # merge cells whose RFs are this similar
THETA_SPLIT = 2.0      # Shannon entropy threshold for forced division


def _shannon_entropy(values: List[float]) -> float:
    total = sum(values)
    if total == 0:
        return 0.0
    probs = [v / total for v in values if v > 0]
    return -sum(p * math.log2(p) for p in probs)


def _rule_diversity(cell: Cell) -> float:
    """Entropy of pairwise precondition similarities — high = diverse rules."""
    rules = cell.rules
    if len(rules) < 2:
        return 0.0
    sims = []
    for i in range(len(rules)):
        for j in range(i + 1, len(rules)):
            sims.append(rules[i].precondition.similarity(rules[j].precondition))
    return _shannon_entropy(sims)


class LifecycleManager:

    def should_divide(self, cell: Cell) -> bool:
        # Only apply entropy-based split when cell has enough rules to be meaningful
        entropy_split = (len(cell.rules) >= 20 and _rule_diversity(cell) > THETA_SPLIT)
        return len(cell.rules) > MAX_RULES or entropy_split

    def divide(self, cell: Cell, organism: Organism):
        """Split cell into two daughters by clustering rules."""
        rules = cell.rules
        mid = len(rules) // 2
        # Simple split: first half vs second half
        cluster_a = rules[:mid]
        cluster_b = rules[mid:]

        daughter_a = Cell()
        daughter_a.receptive_field = (
            cluster_a[len(cluster_a) // 2].precondition if cluster_a
            else SDR.random()
        )
        daughter_a.rules = cluster_a

        daughter_b = Cell()
        daughter_b.receptive_field = (
            cluster_b[len(cluster_b) // 2].precondition if cluster_b
            else SDR.random()
        )
        daughter_b.rules = cluster_b

        organism.remove_cell(cell.id)
        organism.add_cell(daughter_a)
        organism.add_cell(daughter_b)

    def should_merge(self, a: Cell, b: Cell) -> bool:
        return a.receptive_field.similarity(b.receptive_field) > THETA_MERGE

    def merge(self, a: Cell, b: Cell, organism: Organism):
        """Fuse two cells into one merged cell."""
        merged = Cell()
        merged.receptive_field = a.receptive_field.compose(b.receptive_field)
        # Combine rules, removing near-duplicates
        seen: list = []
        for rule in a.rules + b.rules:
            if not any(rule.precondition.similarity(s.precondition) > 0.9
                       and rule.postcondition.similarity(s.postcondition) > 0.9
                       for s in seen):
                seen.append(rule)
        merged.rules = seen
        merged.fitness = (a.fitness + b.fitness) / 2

        organism.remove_cell(a.id)
        organism.remove_cell(b.id)
        organism.add_cell(merged)

    def should_die(self, cell: Cell) -> bool:
        return cell.fitness < THETA_DEATH and cell.age > MIN_AGE

    def retire(self, cell: Cell, organism: Organism):
        """Remove a dead cell; its rules are lost (low fitness = low value)."""
        organism.remove_cell(cell.id)

    def run_maintenance(self, organism: Organism):
        """One pass of division, merging, and death across all cells."""
        cells = list(organism.cells.values())
        for cell in cells:
            if cell.id not in organism.cells:
                continue
            if self.should_divide(cell):
                self.divide(cell, organism)
            elif self.should_die(cell):
                self.retire(cell, organism)

        cells = list(organism.cells.values())
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                a, b = cells[i], cells[j]
                if a.id in organism.cells and b.id in organism.cells:
                    if self.should_merge(a, b):
                        self.merge(a, b, organism)
