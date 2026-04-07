# genesis/core/working_memory.py
from collections import deque
from .sdr import SDR

CAPACITY = 12


class WorkingMemory:
    """12-slot circular buffer of active concept SDRs."""

    def __init__(self, capacity: int = CAPACITY):
        self.capacity = capacity
        self._slots: deque = deque(maxlen=capacity)

    def push(self, sdr: SDR):
        self._slots.append(sdr)

    def union(self) -> SDR:
        """Merge all active SDRs into an uncapped context pattern (OR without bit-capping).
        Returns an SDR that may have >20 active bits — this is intentional.
        Working memory feeds into ForwardChain which already uses uncapped union for facts,
        so a capped 20-bit result would silently drop concepts registered at higher indices."""
        slots = list(self._slots)
        if not slots:
            return SDR.zeros()
        result = slots[0]
        for sdr in slots[1:]:
            result = result.union(sdr)
        return result

    def clear(self):
        self._slots.clear()

    def __len__(self) -> int:
        return len(self._slots)

    def __iter__(self):
        return iter(self._slots)
