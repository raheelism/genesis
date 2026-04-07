import os
import tempfile
from genesis.core.sdr import SDR
from genesis.core.cell import Cell
from genesis.core.organism import Organism
from genesis.storage.colony_store import ColonyStore

def _make_organism() -> Organism:
    org = Organism()
    cell = Cell()
    cell.receptive_field = SDR(list(range(0, 20)))
    cell.add_rule(SDR(list(range(0, 20))), SDR(list(range(100, 120))), 0.8)
    org.add_cell(cell)
    return org

def test_save_creates_file():
    org = _make_organism()
    store = ColonyStore()
    with tempfile.NamedTemporaryFile(suffix=".gen", delete=False) as f:
        path = f.name
    store.save(org, path)
    assert os.path.exists(path)
    os.unlink(path)

def test_load_restores_cell_count():
    org = _make_organism()
    store = ColonyStore()
    with tempfile.NamedTemporaryFile(suffix=".gen", delete=False) as f:
        path = f.name
    store.save(org, path)
    loaded = store.load(path)
    os.unlink(path)
    assert loaded.cell_count() == org.cell_count()

def test_load_restores_rules():
    org = _make_organism()
    store = ColonyStore()
    with tempfile.NamedTemporaryFile(suffix=".gen", delete=False) as f:
        path = f.name
    store.save(org, path)
    loaded = store.load(path)
    os.unlink(path)
    original_cell = list(org.cells.values())[0]
    loaded_cell = list(loaded.cells.values())[0]
    assert len(loaded_cell.rules) == len(original_cell.rules)

def test_load_restores_rule_confidence():
    org = _make_organism()
    store = ColonyStore()
    with tempfile.NamedTemporaryFile(suffix=".gen", delete=False) as f:
        path = f.name
    store.save(org, path)
    loaded = store.load(path)
    os.unlink(path)
    orig_conf = list(org.cells.values())[0].rules[0].confidence
    load_conf = list(loaded.cells.values())[0].rules[0].confidence
    assert abs(orig_conf - load_conf) < 1e-6
