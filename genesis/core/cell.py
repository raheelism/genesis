# genesis/core/cell.py
import uuid
from dataclasses import dataclass, field
from typing import List, Tuple
from .sdr import SDR

THETA_FIRE = 0.35       # cell activates when input overlaps RF above this
THETA_RULE = 0.40       # rule fires when facts overlap precondition above this
MIN_CONFIDENCE = 0.25   # rules below this confidence are ignored
MAX_RULES = 200         # cell divides when it exceeds this
MIN_AGE = 100           # cell must survive at least this many activations before dying
THETA_DEATH = 0.10      # cell dies if fitness drops below this
FITNESS_ALPHA = 0.1     # EMA factor for fitness update


@dataclass
class Rule:
    precondition: SDR
    postcondition: SDR
    confidence: float = 0.3

    def __post_init__(self):
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class Cell:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    receptive_field: SDR = field(default_factory=SDR.random)
    rules: List[Rule] = field(default_factory=list)
    fitness: float = 0.5
    age: int = 0

    def activates(self, input_sdr: SDR) -> bool:
        return input_sdr.similarity(self.receptive_field) > THETA_FIRE

    def apply_rules(self, facts: SDR) -> List[Tuple[Rule, float]]:
        """Return (rule, score) for rules whose precondition matches facts."""
        fired = []
        for rule in self.rules:
            overlap = facts.similarity(rule.precondition)
            if overlap > THETA_RULE and rule.confidence > MIN_CONFIDENCE:
                score = rule.confidence * overlap
                fired.append((rule, score))
        return sorted(fired, key=lambda x: x[1], reverse=True)

    def add_rule(self, precondition: SDR, postcondition: SDR,
                 confidence: float = 0.3) -> Rule:
        """Add a causal rule. Caller is responsible for checking len(rules) < MAX_RULES
        before calling — lifecycle.py handles division when the limit is exceeded."""
        rule = Rule(precondition=precondition, postcondition=postcondition,
                    confidence=confidence)
        self.rules.append(rule)
        return rule

    def update_fitness(self, signal: float):
        """Exponential moving average update toward signal."""
        raw = (1 - FITNESS_ALPHA) * self.fitness + FITNESS_ALPHA * signal
        self.fitness = max(0.0, min(1.0, raw))
        self.age += 1
