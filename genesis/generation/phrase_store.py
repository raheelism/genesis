# genesis/generation/phrase_store.py
from typing import List, Optional, Tuple
from genesis.core.sdr import SDR

THETA_PHRASE = 0.85   # minimum Jaccard similarity to accept a phrase match


class PhraseStore:
    """Registry mapping answer SDRs to their original phrase strings.
    Used by Verbalizer to emit exact multi-word phrases instead of
    reconstructing token-by-token from bit coverage."""

    def __init__(self):
        self._entries: List[Tuple[SDR, str]] = []

    def register(self, sdr: SDR, phrase: str):
        """Register an SDR -> phrase mapping."""
        self._entries.append((sdr, phrase))

    def lookup(self, sdr: SDR) -> Optional[str]:
        """Return the phrase whose registered SDR is most similar to sdr,
        or None if no entry exceeds THETA_PHRASE."""
        best_sim = THETA_PHRASE
        best_phrase = None
        for reg_sdr, phrase in self._entries:
            sim = sdr.similarity(reg_sdr)
            if sim > best_sim:
                best_sim = sim
                best_phrase = phrase
        return best_phrase

    def __len__(self) -> int:
        return len(self._entries)
