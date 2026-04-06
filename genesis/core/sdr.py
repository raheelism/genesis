# genesis/core/sdr.py
import numpy as np
import random
from typing import Optional

SDR_BITS = 1024
SDR_WORDS = SDR_BITS // 64   # 16 uint64 words
SDR_ACTIVE = 20              # ~2% sparsity
SHIFT_STRIDE = 53            # prime, spreads positions without clustering


class SDR:
    """1024-bit sparse distributed representation with exactly 20 active bits."""

    __slots__ = ("words",)

    def __init__(self, active_indices: Optional[list] = None):
        self.words: np.ndarray = np.zeros(SDR_WORDS, dtype=np.uint64)
        if active_indices is not None and len(active_indices) > 0:
            for idx in active_indices:
                idx = int(idx) % SDR_BITS
                word, bit = divmod(idx, 64)
                self.words[word] |= np.uint64(1 << bit)

    @classmethod
    def random(cls) -> "SDR":
        return cls(random.sample(range(SDR_BITS), SDR_ACTIVE))

    @classmethod
    def zeros(cls) -> "SDR":
        return cls()

    def active_indices(self) -> list:
        bits = np.unpackbits(self.words.view(np.uint8), bitorder='little')
        return list(np.nonzero(bits)[0])

    def popcount(self) -> int:
        return int(np.bitwise_count(self.words).sum())

    def similarity(self, other: "SDR") -> float:
        """Jaccard similarity via bitwise AND/OR."""
        and_bits = np.bitwise_and(self.words, other.words)
        or_bits = np.bitwise_or(self.words, other.words)
        and_count = sum(bin(int(w)).count("1") for w in and_bits)
        or_count = sum(bin(int(w)).count("1") for w in or_bits)
        return and_count / or_count if or_count else 0.0

    def compose(self, other: "SDR") -> "SDR":
        """OR then keep SDR_ACTIVE bits (deterministic: lowest indices)."""
        or_words = np.bitwise_or(self.words, other.words)
        temp = SDR()
        temp.words = or_words
        all_active = temp.active_indices()
        if len(all_active) <= SDR_ACTIVE:
            return temp
        kept = sorted(all_active)[:SDR_ACTIVE]
        return SDR(kept)

    def shift(self, offset: int) -> "SDR":
        """Rotate each active index by offset*STRIDE to encode word position."""
        shifted = [(idx + offset * SHIFT_STRIDE) % SDR_BITS
                   for idx in self.active_indices()]
        return SDR(shifted)

    def union(self, other: "SDR") -> "SDR":
        """OR without capping — used to merge fact sets in reasoning."""
        result = SDR()
        result.words = np.bitwise_or(self.words, other.words)
        return result

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SDR):
            return False
        return bool(np.array_equal(self.words, other.words))

    def __hash__(self):
        return hash(self.words.tobytes())

    def __repr__(self) -> str:
        return f"SDR(active={self.popcount()}, indices={self.active_indices()[:5]}...)"
