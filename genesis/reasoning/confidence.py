# genesis/reasoning/confidence.py
from typing import List

DECAY = 0.95   # per-hop confidence decay


def propagate(confidences: List[float], depth: int) -> float:
    """Multiply rule confidences and apply depth decay."""
    if not confidences:
        return 0.0
    result = 1.0
    for c in confidences:
        result *= c
    return result * (DECAY ** depth)
