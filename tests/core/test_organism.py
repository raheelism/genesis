from genesis.core.sdr import SDR
from genesis.core.cell import Cell
from genesis.core.organism import Organism


def test_add_cell_increases_count():
    org = Organism()
    org.add_cell(Cell())
    assert org.cell_count() == 1


def test_remove_cell_decreases_count():
    org = Organism()
    cell = Cell()
    org.add_cell(cell)
    org.remove_cell(cell.id)
    assert org.cell_count() == 0


def test_route_returns_activating_cells():
    org = Organism()
    rf = SDR.random()
    cell = Cell()
    cell.receptive_field = rf
    org.add_cell(cell)
    # query identical to receptive field must activate cell
    result = org.route(rf)
    assert any(c.id == cell.id for c in result)


def test_route_does_not_return_non_activating_cells():
    org = Organism()
    cell = Cell()
    cell.receptive_field = SDR(list(range(0, 20)))
    org.add_cell(cell)
    # completely disjoint query
    result = org.route(SDR(list(range(500, 520))))
    assert not any(c.id == cell.id for c in result)


def test_route_handles_empty_organism():
    org = Organism()
    result = org.route(SDR.random())
    assert result == []


def test_cell_accessible_by_id():
    org = Organism()
    cell = Cell()
    org.add_cell(cell)
    assert cell.id in org.cells
