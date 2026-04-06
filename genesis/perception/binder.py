from typing import List
from genesis.core.sdr import SDR


class Binder:
    """Combines a sequence of token SDRs into a single sentence SDR using
    positional shifts — preserving word order information."""

    def bind(self, sdrs: List[SDR]) -> SDR:
        if not sdrs:
            return SDR.zeros()
        result = sdrs[0].shift(0)
        for i, sdr in enumerate(sdrs[1:], start=1):
            result = result.compose(sdr.shift(i))
        return result
