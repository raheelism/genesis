# tests/core/test_sdr.py
import pytest
from genesis.core.sdr import SDR, SDR_BITS, SDR_ACTIVE

def test_random_sdr_has_correct_popcount():
    sdr = SDR.random()
    assert sdr.popcount() == SDR_ACTIVE

def test_zeros_sdr_has_zero_popcount():
    sdr = SDR.zeros()
    assert sdr.popcount() == 0

def test_similarity_identical_is_one():
    sdr = SDR.random()
    assert sdr.similarity(sdr) == pytest.approx(1.0)

def test_similarity_disjoint_is_zero():
    a = SDR(list(range(0, 20)))
    b = SDR(list(range(20, 40)))
    assert a.similarity(b) == pytest.approx(0.0)

def test_similarity_partial_overlap():
    a = SDR(list(range(0, 20)))
    b = SDR(list(range(10, 30)))  # 10 shared
    # intersection=10, union=30 → Jaccard = 10/30 ≈ 0.333
    assert 0.30 < a.similarity(b) < 0.36

def test_compose_maintains_sparsity():
    a = SDR.random()
    b = SDR.random()
    c = a.compose(b)
    assert c.popcount() == SDR_ACTIVE

def test_shift_changes_pattern():
    a = SDR.random()
    b = a.shift(1)
    assert a != b

def test_shift_preserves_sparsity():
    a = SDR.random()
    b = a.shift(3)
    assert b.popcount() == SDR_ACTIVE

def test_order_encoding_differs():
    # "dog bites" vs "bites dog" should produce different SDRs
    dog = SDR(list(range(0, 20)))
    bites = SDR(list(range(100, 120)))
    phrase1 = dog.shift(0).compose(bites.shift(1))
    phrase2 = bites.shift(0).compose(dog.shift(1))
    assert phrase1 != phrase2

def test_active_indices_roundtrip():
    indices = [0, 63, 64, 127, 500, 1000, 1023, 7, 200, 300,
               400, 500, 600, 700, 800, 900, 100, 150, 250, 350]
    indices = sorted(set(indices))[:20]
    sdr = SDR(indices)
    recovered = sorted(sdr.active_indices())
    assert recovered == sorted(indices)
