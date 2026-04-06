import pytest
from genesis.core.sdr import SDR
from genesis.core.cell import Cell, MAX_RULES, MIN_AGE, THETA_DEATH
from genesis.core.organism import Organism
from genesis.learning.lifecycle import LifecycleManager


def _make_loaded_cell(num_rules: int) -> Cell:
    cell = Cell()
    for i in range(num_rules):
        cell.add_rule(SDR.random(), SDR.random(), 0.5)
    return cell


def test_should_divide_when_too_many_rules():
    mgr = LifecycleManager()
    cell = _make_loaded_cell(MAX_RULES + 1)
    assert mgr.should_divide(cell)


def test_should_not_divide_when_few_rules():
    mgr = LifecycleManager()
    cell = _make_loaded_cell(5)
    assert not mgr.should_divide(cell)


def test_divide_removes_parent_and_adds_two_daughters():
    mgr = LifecycleManager()
    org = Organism()
    cell = _make_loaded_cell(MAX_RULES + 1)
    org.add_cell(cell)
    mgr.divide(cell, org)
    assert cell.id not in org.cells
    assert org.cell_count() == 2


def test_divide_daughters_have_fewer_rules_than_parent():
    mgr = LifecycleManager()
    org = Organism()
    cell = _make_loaded_cell(MAX_RULES + 1)
    parent_rule_count = len(cell.rules)
    org.add_cell(cell)
    mgr.divide(cell, org)
    for daughter in org.cells.values():
        assert len(daughter.rules) < parent_rule_count


def test_should_merge_similar_cells():
    mgr = LifecycleManager()
    rf = SDR(list(range(0, 20)))
    a = Cell(); a.receptive_field = rf
    b = Cell(); b.receptive_field = rf  # identical RF
    assert mgr.should_merge(a, b)


def test_should_not_merge_dissimilar_cells():
    mgr = LifecycleManager()
    a = Cell(); a.receptive_field = SDR(list(range(0, 20)))
    b = Cell(); b.receptive_field = SDR(list(range(500, 520)))
    assert not mgr.should_merge(a, b)


def test_merge_removes_both_parents_adds_one_child():
    mgr = LifecycleManager()
    org = Organism()
    rf = SDR(list(range(0, 20)))
    a = Cell(); a.receptive_field = rf
    b = Cell(); b.receptive_field = rf
    org.add_cell(a); org.add_cell(b)
    mgr.merge(a, b, org)
    assert a.id not in org.cells
    assert b.id not in org.cells
    assert org.cell_count() == 1


def test_should_die_when_low_fitness_and_old():
    mgr = LifecycleManager()
    cell = Cell()
    cell.fitness = THETA_DEATH - 0.01
    cell.age = MIN_AGE + 1
    assert mgr.should_die(cell)


def test_should_not_die_when_young():
    mgr = LifecycleManager()
    cell = Cell()
    cell.fitness = 0.0
    cell.age = MIN_AGE - 1
    assert not mgr.should_die(cell)


def test_retire_removes_cell_from_organism():
    mgr = LifecycleManager()
    org = Organism()
    cell = Cell()
    org.add_cell(cell)
    mgr.retire(cell, org)
    assert cell.id not in org.cells
