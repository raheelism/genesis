from dataclasses import dataclass, field
from typing import List
from genesis.core.sdr import SDR
from genesis.core.cell import Cell
from genesis.core.organism import Organism

CONF_THRESHOLD = 0.60
ACCESS_THRESHOLD = 3


@dataclass
class Episode:
    query: SDR
    answer: SDR
    chain: list
    confidence: float
    access_count: int = 0


class Consolidator:
    """Converts high-value episodic memories into permanent cell rules.
    Mirrors biological sleep consolidation."""

    def consolidate(self, episodes: List[Episode], organism: Organism):
        """For each high-value episode, embed its (query→answer) as a permanent rule."""
        for episode in episodes:
            if (episode.confidence >= CONF_THRESHOLD and
                    episode.access_count >= ACCESS_THRESHOLD):
                self._embed(episode, organism)

    def _embed(self, episode: Episode, organism: Organism):
        # Find best existing cell to host this rule, or create new one
        best_cell = None
        best_sim = 0.0
        for cell in organism.cells.values():
            sim = episode.query.similarity(cell.receptive_field)
            if sim > best_sim:
                best_sim = sim
                best_cell = cell

        if best_cell is None or best_sim < 0.20:
            # Create a dedicated Reasoning Cell for this episode
            new_cell = Cell()
            new_cell.receptive_field = episode.query
            new_cell.add_rule(episode.query, episode.answer,
                              confidence=episode.confidence)
            organism.add_cell(new_cell)
        else:
            best_cell.add_rule(episode.query, episode.answer,
                               confidence=episode.confidence)
