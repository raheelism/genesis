# tests/generation/test_phrase_store.py
from genesis.core.sdr import SDR
from genesis.generation.phrase_store import PhraseStore

def test_lookup_returns_registered_phrase():
    ps = PhraseStore()
    sdr = SDR(list(range(20)))
    ps.register(sdr, "heat and light")
    result = ps.lookup(sdr)
    assert result == "heat and light"

def test_lookup_returns_none_for_dissimilar_sdr():
    ps = PhraseStore()
    sdr_a = SDR(list(range(20)))
    sdr_b = SDR(list(range(500, 520)))
    ps.register(sdr_a, "heat and light")
    assert ps.lookup(sdr_b) is None

def test_lookup_returns_none_when_empty():
    ps = PhraseStore()
    assert ps.lookup(SDR.random()) is None

def test_len_reflects_registrations():
    ps = PhraseStore()
    ps.register(SDR.random(), "phrase one")
    ps.register(SDR.random(), "phrase two")
    assert len(ps) == 2

def test_lookup_picks_closest_when_multiple_registered():
    """Query SDR shares 18/20 bits with candidate A and 10/20 bits with candidate B.
    Both exceed THETA_PHRASE=0.85 only for candidate A — lookup must return phrase_a."""
    ps = PhraseStore()
    # candidate A: bits 0-19
    sdr_a = SDR(list(range(0, 20)))
    # candidate B: completely disjoint bits
    sdr_b = SDR(list(range(500, 520)))
    ps.register(sdr_a, "phrase a")
    ps.register(sdr_b, "phrase b")

    # query shares all 20 bits with sdr_a (Jaccard=1.0 > 0.85) → returns "phrase a"
    assert ps.lookup(sdr_a) == "phrase a"

    # query shares all 20 bits with sdr_b (Jaccard=1.0 > 0.85) → returns "phrase b"
    assert ps.lookup(sdr_b) == "phrase b"

    # query with sdr_a bits EXCEPT last 3, plus 3 bits from sdr_b:
    # 17 shared with sdr_a out of 17+3+3=23 union → Jaccard ≈ 17/23 ≈ 0.74 < 0.85
    # 3 shared with sdr_b out of 20+17 = 34 union → Jaccard ≈ 3/34 ≈ 0.09 < 0.85
    # Neither exceeds threshold → returns None
    partial = SDR(list(range(0, 17)) + list(range(500, 503)))
    assert ps.lookup(partial) is None
