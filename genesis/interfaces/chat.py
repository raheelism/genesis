# genesis/interfaces/chat.py
from genesis.core.organism import Organism
from genesis.core.working_memory import WorkingMemory
from genesis.learning.hebbian import HebbianLearner
from genesis.perception.binder import Binder
from genesis.perception.encoder import Encoder
from genesis.perception.tokenizer import Tokenizer
from genesis.reasoning.forward_chain import ForwardChain
from genesis.generation.verbalizer import Verbalizer


class ChatInterface:
    """Conversational mode: text in → encode → reason → verbalize → learn."""

    def __init__(self, tokenizer: Tokenizer, encoder: Encoder,
                 binder: Binder, organism: Organism,
                 working_memory: WorkingMemory, reasoner: ForwardChain,
                 learner: HebbianLearner, verbalizer: Verbalizer):
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.binder = binder
        self.organism = organism
        self.working_memory = working_memory
        self.reasoner = reasoner
        self.learner = learner
        self.verbalizer = verbalizer
        self.turn_count = 0

    def turn(self, user_input: str) -> str:
        # 1. Encode input
        tokens = self.tokenizer.tokenize(user_input)
        self.encoder.register_vocab(self.tokenizer.vocab)
        sdrs = [self.encoder.encode_token(t) for t in tokens]
        if not sdrs:
            return "<empty input>"
        query_sdr = self.binder.bind(sdrs)

        # 2. Push to working memory
        self.working_memory.push(query_sdr)

        # 3. Reason
        result = self.reasoner.reason(query_sdr, self.organism,
                                      self.working_memory)

        # 4. Learn from this interaction
        active_cells = self.organism.route(query_sdr)
        for cell in active_cells:
            self.learner.update(cell, query_sdr,
                                result.answer if result.answer else query_sdr)
            cell.update_fitness(result.confidence if result.answer else 0.1)

        # 5. Push answer to working memory
        if result.answer:
            self.working_memory.push(result.answer)

        # 6. Generate response
        if result.answer and result.confidence > 0.30:
            response = self.verbalizer.verbalize(result.answer, self.encoder,
                                                 max_tokens=12)
        else:
            response = "<I don't know yet — still learning>"

        self.turn_count += 1
        return response
