import random
from typing import Dict, List
from .cell import Cell
from .sdr import SDR, SDR_BITS, SDR_ACTIVE

NUM_HASH_BITS = 10   # 2^10 = 1024 buckets


class Organism:
    """Cell colony with LSH-based routing for O(1) relevant-cell lookup."""

    def __init__(self):
        self.cells: Dict[str, Cell] = {}
        self._buckets: Dict[tuple, List[str]] = {}
        self._projections: List[List[int]] = self._make_projections()

    def _make_projections(self) -> List[List[int]]:
        rng = random.Random(42)   # fixed seed → deterministic routing
        return [
            [rng.choice((-1, 1)) for _ in range(SDR_BITS)]
            for _ in range(NUM_HASH_BITS)
        ]

    def _lsh_hash(self, sdr: SDR) -> tuple:
        active = set(sdr.active_indices())
        return tuple(
            1 if sum(proj[i] for i in active) >= 0 else 0
            for proj in self._projections
        )

    def add_cell(self, cell: Cell):
        self.cells[cell.id] = cell
        bucket = self._lsh_hash(cell.receptive_field)
        self._buckets.setdefault(bucket, []).append(cell.id)

    def remove_cell(self, cell_id: str):
        cell = self.cells.pop(cell_id, None)
        if cell is None:
            return
        bucket = self._lsh_hash(cell.receptive_field)
        bucket_list = self._buckets.get(bucket, [])
        self._buckets[bucket] = [cid for cid in bucket_list if cid != cell_id]

    def route(self, sdr: SDR) -> List[Cell]:
        """Return cells whose receptive fields overlap sdr above THETA_FIRE."""
        bucket = self._lsh_hash(sdr)
        candidate_ids: set = set(self._buckets.get(bucket, []))
        # Check 1-bit-flip neighbors for recall robustness
        bucket_list = list(bucket)
        for i in range(NUM_HASH_BITS):
            flipped = bucket_list[:]
            flipped[i] ^= 1
            candidate_ids.update(self._buckets.get(tuple(flipped), []))
        return [
            self.cells[cid]
            for cid in candidate_ids
            if cid in self.cells and self.cells[cid].activates(sdr)
        ]

    def cell_count(self) -> int:
        return len(self.cells)
