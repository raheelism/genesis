import random
from typing import Dict
from genesis.core.sdr import SDR, SDR_BITS, SDR_ACTIVE

_RNG_SEED_SALT = 0xDEADBEEF


class Encoder:
    """Maps tokens to random but stable SDRs. SDRs co-evolve via Hebbian learning."""

    def __init__(self):
        self._vocab_sdrs: Dict[str, SDR] = {}
        self._unk_sdr: SDR = self._make_sdr("<unk>")

    def _make_sdr(self, token: str) -> SDR:
        rng = random.Random(hash(token) ^ _RNG_SEED_SALT)
        indices = rng.sample(range(SDR_BITS), SDR_ACTIVE)
        return SDR(indices)

    def register(self, token: str):
        if token not in self._vocab_sdrs:
            self._vocab_sdrs[token] = self._make_sdr(token)

    def register_vocab(self, vocab: Dict[str, int]):
        for token in vocab:
            self.register(token)

    def encode_token(self, token: str) -> SDR:
        return self._vocab_sdrs.get(token, self._unk_sdr)

    def vocab_size(self) -> int:
        return len(self._vocab_sdrs)

    def decode_sdr(self, sdr: SDR, top_k: int = 5) -> list:
        """Return top_k tokens whose SDR is most similar to given SDR."""
        scored = [
            (tok, sdr.similarity(tok_sdr))
            for tok, tok_sdr in self._vocab_sdrs.items()
        ]
        return [tok for tok, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]]
