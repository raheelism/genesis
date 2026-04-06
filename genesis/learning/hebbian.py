from genesis.core.sdr import SDR
from genesis.core.cell import Cell, Rule, THETA_RULE

ETA = 0.05              # learning rate
INITIAL_CONFIDENCE = 0.3
THETA_CREATE = 0.30     # create new rule when max rule overlap below this


class HebbianLearner:
    """Local Hebbian updates — no global gradient, no weight matrix."""

    def strengthen(self, rule: Rule):
        """Fire confirmed: push confidence toward 1."""
        rule.confidence = rule.confidence + ETA * (1.0 - rule.confidence)
        rule.confidence = min(rule.confidence, 1.0)

    def weaken(self, rule: Rule):
        """Fire disconfirmed: push confidence toward 0."""
        rule.confidence = rule.confidence * (1.0 - ETA)
        rule.confidence = max(rule.confidence, 0.0)

    def create_rule(self, cell: Cell, precondition: SDR, postcondition: SDR):
        """Create new rule in cell with initial low confidence."""
        cell.add_rule(precondition, postcondition, INITIAL_CONFIDENCE)

    def update(self, cell: Cell, facts: SDR, observed: SDR):
        """Given what the cell saw (facts) and what actually happened (observed),
        strengthen matching rules and weaken mismatching ones.
        Create a new rule if no rule fired at all."""
        fired_any = False
        for rule in cell.rules:
            overlap = facts.similarity(rule.precondition)
            if overlap > THETA_RULE:
                fired_any = True
                match = observed.similarity(rule.postcondition)
                if match > 0.4:
                    self.strengthen(rule)
                    cell.update_fitness(rule.confidence)
                else:
                    self.weaken(rule)
                    cell.update_fitness(rule.confidence * 0.5)
        if not fired_any:
            max_overlap = max(
                (facts.similarity(r.precondition) for r in cell.rules),
                default=0.0
            )
            if max_overlap < THETA_CREATE:
                self.create_rule(cell, facts, observed)
