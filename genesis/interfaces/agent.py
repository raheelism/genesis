# genesis/interfaces/agent.py
from typing import Optional
from genesis.core.organism import Organism
from genesis.core.sdr import SDR
from genesis.core.working_memory import WorkingMemory
from genesis.learning.hebbian import HebbianLearner
from genesis.perception.binder import Binder
from genesis.perception.encoder import Encoder
from genesis.perception.tokenizer import Tokenizer
from genesis.reasoning.backward_chain import BackwardChain
from genesis.reasoning.forward_chain import ForwardChain
from genesis.generation.verbalizer import Verbalizer


class AgentInterface:
    """Autonomous agent mode: goal-directed reasoning with self-monitoring."""

    def __init__(self, tokenizer: Tokenizer, encoder: Encoder, binder: Binder,
                 organism: Organism, working_memory: WorkingMemory,
                 forward: ForwardChain, backward: BackwardChain,
                 learner: HebbianLearner, verbalizer: Verbalizer):
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.binder = binder
        self.organism = organism
        self.working_memory = working_memory
        self.forward = forward
        self.backward = backward
        self.learner = learner
        self.verbalizer = verbalizer
        self.goal_sdr: Optional[SDR] = None

    def set_goal(self, goal_text: str):
        tokens = self.tokenizer.tokenize(goal_text)
        self.encoder.register_vocab(self.tokenizer.vocab)
        sdrs = [self.encoder.encode_token(t) for t in tokens]
        self.goal_sdr = self.binder.bind(sdrs)

    def step(self) -> str:
        if self.goal_sdr is None:
            return "<no goal set>"
        # Try to verify/plan toward goal
        known = self.working_memory.union()
        verification = self.backward.verify(self.goal_sdr, self.organism, known)
        if verification.verified:
            return f"<goal achieved: {self.verbalizer.verbalize(self.goal_sdr, self.encoder, 6)}>"
        # Forward chain to find next action toward goal
        result = self.forward.reason(self.goal_sdr, self.organism,
                                     self.working_memory)
        if result.answer:
            self.working_memory.push(result.answer)
            return self.verbalizer.verbalize(result.answer, self.encoder, 8)
        return "<reasoning — no path found yet>"
