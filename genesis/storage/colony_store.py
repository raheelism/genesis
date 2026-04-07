import pickle
import gzip
from genesis.core.organism import Organism


class ColonyStore:
    """Saves and loads the cell colony using compressed pickle.
    Format: gzip-compressed pickle of the Organism."""

    def save(self, organism: Organism, path: str):
        with gzip.open(path, "wb") as f:
            pickle.dump(organism, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str) -> Organism:
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
