# tests/reasoning/test_confidence.py
import pytest
from genesis.reasoning.confidence import propagate, DECAY

def test_single_hop_returns_confidence():
    assert propagate([0.8], depth=1) == pytest.approx(0.8 * DECAY)

def test_two_hops_multiplies_and_decays():
    result = propagate([0.8, 0.6], depth=2)
    expected = 0.8 * 0.6 * (DECAY ** 2)
    assert result == pytest.approx(expected)

def test_empty_chain_returns_zero():
    assert propagate([], depth=0) == 0.0

def test_deeper_chain_has_lower_confidence():
    shallow = propagate([0.9, 0.9], depth=2)
    deep = propagate([0.9, 0.9, 0.9, 0.9], depth=4)
    assert deep < shallow
