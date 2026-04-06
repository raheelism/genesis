# tests/core/test_cell.py
import pytest
from genesis.core.sdr import SDR
from genesis.core.cell import Rule, Cell, THETA_FIRE, THETA_RULE

def test_rule_stores_precondition_and_postcondition():
    pre = SDR.random()
    post = SDR.random()
    rule = Rule(precondition=pre, postcondition=post, confidence=0.5)
    assert rule.precondition == pre
    assert rule.postcondition == post
    assert rule.confidence == 0.5

def test_cell_activates_when_overlap_above_threshold():
    cell = Cell()
    cell.receptive_field = SDR(list(range(0, 20)))
    # Input shares 10/20 bits with receptive field = Jaccard 10/30 ≈ 0.33
    # Below THETA_FIRE=0.35 → should NOT activate
    input_sdr = SDR(list(range(10, 30)))
    assert not cell.activates(input_sdr)

def test_cell_activates_when_identical():
    cell = Cell()
    rf = SDR.random()
    cell.receptive_field = rf
    assert cell.activates(rf)

def test_cell_does_not_activate_disjoint_input():
    cell = Cell()
    cell.receptive_field = SDR(list(range(0, 20)))
    input_sdr = SDR(list(range(100, 120)))
    assert not cell.activates(input_sdr)

def test_apply_rules_returns_matching_rules():
    cell = Cell()
    facts = SDR(list(range(0, 20)))
    pre = SDR(list(range(0, 20)))   # identical to facts
    post = SDR.random()
    cell.add_rule(pre, post, confidence=0.8)
    fired = cell.apply_rules(facts)
    assert len(fired) == 1
    rule, score = fired[0]
    assert rule.postcondition == post
    assert score > 0

def test_apply_rules_skips_low_confidence():
    cell = Cell()
    facts = SDR(list(range(0, 20)))
    pre = SDR(list(range(0, 20)))
    post = SDR.random()
    cell.add_rule(pre, post, confidence=0.1)  # below MIN_CONFIDENCE=0.25
    fired = cell.apply_rules(facts)
    assert len(fired) == 0

def test_apply_rules_sorted_by_score_descending():
    cell = Cell()
    facts = SDR(list(range(0, 20)))
    pre = SDR(list(range(0, 20)))
    cell.add_rule(pre, SDR.random(), confidence=0.9)
    cell.add_rule(pre, SDR.random(), confidence=0.5)
    fired = cell.apply_rules(facts)
    assert fired[0][1] >= fired[1][1]

def test_update_fitness_moves_toward_signal():
    cell = Cell()
    cell.fitness = 0.5
    cell.update_fitness(1.0)
    assert cell.fitness > 0.5
    cell2 = Cell()
    cell2.fitness = 0.5
    cell2.update_fitness(0.0)
    assert cell2.fitness < 0.5
