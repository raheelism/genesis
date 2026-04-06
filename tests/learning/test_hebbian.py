import pytest
from genesis.core.sdr import SDR
from genesis.core.cell import Cell, Rule
from genesis.learning.hebbian import HebbianLearner


def test_strengthen_increases_confidence():
    learner = HebbianLearner()
    rule = Rule(SDR.random(), SDR.random(), confidence=0.5)
    learner.strengthen(rule)
    assert rule.confidence > 0.5


def test_strengthen_never_exceeds_one():
    learner = HebbianLearner()
    rule = Rule(SDR.random(), SDR.random(), confidence=0.99)
    for _ in range(100):
        learner.strengthen(rule)
    assert rule.confidence <= 1.0


def test_weaken_decreases_confidence():
    learner = HebbianLearner()
    rule = Rule(SDR.random(), SDR.random(), confidence=0.5)
    learner.weaken(rule)
    assert rule.confidence < 0.5


def test_weaken_never_goes_below_zero():
    learner = HebbianLearner()
    rule = Rule(SDR.random(), SDR.random(), confidence=0.01)
    for _ in range(100):
        learner.weaken(rule)
    assert rule.confidence >= 0.0


def test_create_rule_adds_to_cell():
    learner = HebbianLearner()
    cell = Cell()
    pre = SDR.random()
    post = SDR.random()
    learner.create_rule(cell, pre, post)
    assert len(cell.rules) == 1
    assert cell.rules[0].precondition == pre
    assert cell.rules[0].postcondition == post


def test_create_rule_starts_at_initial_confidence():
    learner = HebbianLearner()
    cell = Cell()
    learner.create_rule(cell, SDR.random(), SDR.random())
    assert cell.rules[0].confidence == pytest.approx(0.3)


def test_update_processes_correct_prediction():
    learner = HebbianLearner()
    cell = Cell()
    pre = SDR(list(range(0, 20)))
    post = SDR(list(range(100, 120)))
    rule = cell.add_rule(pre, post, confidence=0.5)
    facts = SDR(list(range(0, 20)))
    observed = SDR(list(range(100, 120)))  # matches postcondition
    learner.update(cell, facts, observed)
    assert rule.confidence > 0.5


def test_update_processes_wrong_prediction():
    learner = HebbianLearner()
    cell = Cell()
    pre = SDR(list(range(0, 20)))
    post = SDR(list(range(100, 120)))
    rule = cell.add_rule(pre, post, confidence=0.5)
    facts = SDR(list(range(0, 20)))
    observed = SDR(list(range(500, 520)))  # doesn't match post
    learner.update(cell, facts, observed)
    assert rule.confidence < 0.5
