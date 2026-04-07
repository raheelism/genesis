# genesis/generation/verbalizer.py
from typing import Optional
from genesis.core.sdr import SDR
from genesis.perception.encoder import Encoder
from genesis.generation.phrase_store import PhraseStore

THETA_VERBALIZE = 0.05   # minimum similarity to include a token candidate


class Verbalizer:
    """Converts a conclusion SDR back into a token sequence.

    First checks PhraseStore for an exact registered phrase (covers multi-word
    answers like 'heat and light'). Falls back to greedy bit-coverage
    reconstruction if no registered phrase is close enough."""

    def __init__(self, phrase_store: Optional[PhraseStore] = None):
        self.phrase_store = phrase_store

    def verbalize(self, sdr: SDR, encoder: Encoder,
                  max_tokens: int = 10) -> str:
        if sdr.popcount() == 0:
            return "<unknown>"

        # 1. Check phrase store — exact registered phrase takes priority
        if self.phrase_store is not None:
            phrase = self.phrase_store.lookup(sdr)
            if phrase is not None:
                return phrase

        # 2. Fall back: greedy token coverage
        scored = [
            (token, sdr.similarity(tok_sdr))
            for token, tok_sdr in encoder._vocab_sdrs.items()
            if token not in ("<pad>", "<unk>", "<start>", "<end>")
        ]
        scored = [(tok, s) for tok, s in scored if s > THETA_VERBALIZE]
        scored.sort(key=lambda x: x[1], reverse=True)

        if not scored:
            return "<unknown>"

        covered = SDR.zeros()
        selected = []
        for token, score in scored[:50]:
            tok_sdr = encoder.encode_token(token)
            new_coverage = tok_sdr.similarity(sdr) - covered.similarity(tok_sdr)
            if new_coverage > 0:
                selected.append(token)
                covered = covered.compose(tok_sdr)
                if len(selected) >= max_tokens:
                    break

        return " ".join(selected) if selected else "<unknown>"
