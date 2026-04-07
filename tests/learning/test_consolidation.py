from genesis.core.sdr import SDR
from genesis.core.cell import Cell
from genesis.core.organism import Organism
from genesis.learning.consolidation import Consolidator, Episode

def _make_episode(confidence: float, access_count: int) -> Episode:
    return Episode(
        query=SDR.random(),
        answer=SDR.random(),
        chain=[],
        confidence=confidence,
        access_count=access_count,
    )

def test_episode_creation():
    ep = _make_episode(0.8, 5)
    assert ep.confidence == 0.8
    assert ep.access_count == 5

def test_consolidate_adds_rules_for_high_value_episodes():
    cons = Consolidator()
    org = Organism()
    # One high-value cell to receive rules
    cell = Cell()
    cell.receptive_field = SDR(list(range(0, 20)))
    org.add_cell(cell)

    ep = Episode(
        query=SDR(list(range(0, 20))),
        answer=SDR(list(range(200, 220))),
        chain=[],
        confidence=0.75,
        access_count=4,
    )
    before = len(cell.rules)
    cons.consolidate([ep], org)
    # A Reasoning Cell should have been created with the rule
    total_rules = sum(len(c.rules) for c in org.cells.values())
    assert total_rules > before

def test_consolidate_ignores_low_confidence_episodes():
    cons = Consolidator()
    org = Organism()
    initial_cells = org.cell_count()
    ep = _make_episode(confidence=0.3, access_count=10)
    cons.consolidate([ep], org)
    assert org.cell_count() == initial_cells

def test_consolidate_ignores_low_access_episodes():
    cons = Consolidator()
    org = Organism()
    initial_cells = org.cell_count()
    ep = _make_episode(confidence=0.9, access_count=1)
    cons.consolidate([ep], org)
    assert org.cell_count() == initial_cells
