# tests/core/test_working_memory.py
import pytest
from genesis.core.sdr import SDR, SDR_ACTIVE
from genesis.core.working_memory import WorkingMemory

def test_empty_memory_has_zero_length():
    wm = WorkingMemory()
    assert len(wm) == 0

def test_push_increases_length():
    wm = WorkingMemory()
    wm.push(SDR.random())
    assert len(wm) == 1

def test_capacity_not_exceeded():
    wm = WorkingMemory(capacity=3)
    for _ in range(10):
        wm.push(SDR.random())
    assert len(wm) == 3

def test_union_of_empty_is_zeros():
    wm = WorkingMemory()
    result = wm.union()
    assert result.popcount() == 0

def test_union_of_single_item():
    wm = WorkingMemory()
    sdr = SDR.random()
    wm.push(sdr)
    result = wm.union()
    assert result.similarity(sdr) == pytest.approx(1.0)

def test_union_contains_bits_from_both():
    wm = WorkingMemory()
    a = SDR(list(range(0, 20)))
    b = SDR(list(range(100, 120)))
    wm.push(a)
    wm.push(b)
    result = wm.union()
    # result should overlap with both a and b
    assert result.similarity(a) > 0
    assert result.similarity(b) > 0

def test_clear_empties_memory():
    wm = WorkingMemory()
    wm.push(SDR.random())
    wm.clear()
    assert len(wm) == 0
