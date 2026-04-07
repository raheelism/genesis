# genesis/interfaces/embed.py
from typing import List
from genesis.core.cell import Cell
from genesis.core.organism import Organism
from genesis.learning.hebbian import HebbianLearner
from genesis.perception.binder import Binder
from genesis.perception.encoder import Encoder
from genesis.perception.tokenizer import Tokenizer


class EmbedInterface:
    """Embedded mode: continuously learn from a stream of text events."""

    def __init__(self, tokenizer: Tokenizer, encoder: Encoder,
                 binder: Binder, organism: Organism, learner: HebbianLearner):
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.binder = binder
        self.organism = organism
        self.learner = learner

    def process(self, events: List[str]) -> List[str]:
        outputs = []
        for event in events:
            tokens = self.tokenizer.tokenize(event)
            self.encoder.register_vocab(self.tokenizer.vocab)
            sdrs = [self.encoder.encode_token(t) for t in tokens]
            if not sdrs:
                continue
            event_sdr = self.binder.bind(sdrs)
            active = self.organism.route(event_sdr)
            if active:
                for cell in active:
                    self.learner.update(cell, event_sdr, event_sdr)
                    outputs.append(f"processed by cell {cell.id[:8]}")
            else:
                new_cell = Cell()
                new_cell.receptive_field = event_sdr
                new_cell.add_rule(event_sdr, event_sdr, confidence=0.3)
                self.organism.add_cell(new_cell)
                outputs.append(f"new cell created")
        return outputs
