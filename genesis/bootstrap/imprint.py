from typing import List, Tuple
from genesis.core.cell import Cell
from genesis.core.organism import Organism
from genesis.core.sdr import SDR
from genesis.perception.tokenizer import Tokenizer
from genesis.perception.encoder import Encoder
from genesis.perception.binder import Binder

WINDOW = 3   # sliding window size for phrase extraction


class Imprinter:
    """Runs the 4-phase bootstrap: vocabulary → patterns → concepts → reasoning."""

    def phase0(self, sentences: List[str], tokenizer: Tokenizer,
               encoder: Encoder):
        """Assign random SDRs to every token seen in the corpus."""
        for sent in sentences:
            tokenizer.tokenize(sent)
        encoder.register_vocab(tokenizer.vocab)

    def phase1(self, sentences: List[str], tokenizer: Tokenizer,
               encoder: Encoder, binder: Binder, organism: Organism):
        """Create Pattern Cells: for each sliding window, learn phrase→next_phrase."""
        for sent in sentences:
            tokens = tokenizer.tokenize(sent)
            if len(tokens) < 2:
                continue
            sdrs = [encoder.encode_token(t) for t in tokens]
            for i in range(len(sdrs) - WINDOW):
                window_sdrs = sdrs[i:i + WINDOW]
                phrase_sdr = binder.bind(window_sdrs)
                next_sdrs = sdrs[i + 1:i + WINDOW + 1]
                next_sdr = binder.bind(next_sdrs)

                # Find or create a cell for this phrase
                candidates = organism.route(phrase_sdr)
                if candidates:
                    cell = candidates[0]
                else:
                    cell = Cell()
                    cell.receptive_field = phrase_sdr
                    organism.add_cell(cell)

                cell.add_rule(phrase_sdr, next_sdr, confidence=0.7)

    def phase2(self, sentences: List[str], tokenizer: Tokenizer,
               encoder: Encoder, binder: Binder, organism: Organism):
        """Create Concept Cells: entity + context → relation."""
        for sent in sentences:
            tokens = tokenizer.tokenize(sent)
            if len(tokens) < 3:
                continue
            sdrs = [encoder.encode_token(t) for t in tokens]
            sent_sdr = binder.bind(sdrs)
            # Each token in context of full sentence = concept cell
            for i, (tok, sdr) in enumerate(zip(tokens, sdrs)):
                candidates = organism.route(sdr)
                if not candidates:
                    cell = Cell()
                    cell.receptive_field = sdr
                    organism.add_cell(cell)
                    cell.add_rule(sdr, sent_sdr, confidence=0.5)

    def phase3(self, qa_pairs: List[Tuple[str, str]], tokenizer: Tokenizer,
               encoder: Encoder, binder: Binder, organism: Organism,
               phrase_store=None):
        """Create Reasoning Cells from structured QA pairs.
        If phrase_store is provided, registers the answer SDR -> answer string
        so Verbalizer can emit exact phrases for compound answers."""
        for question, answer in qa_pairs:
            q_tokens = tokenizer.tokenize(question)
            a_tokens = tokenizer.tokenize(answer)
            encoder.register_vocab(tokenizer.vocab)
            q_sdr = binder.bind([encoder.encode_token(t) for t in q_tokens])
            a_sdr = binder.bind([encoder.encode_token(t) for t in a_tokens])
            cell = Cell()
            cell.receptive_field = q_sdr
            cell.add_rule(q_sdr, a_sdr, confidence=1.0)
            organism.add_cell(cell)
            if phrase_store is not None:
                phrase_store.register(a_sdr, answer)
