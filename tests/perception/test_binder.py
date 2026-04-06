from genesis.core.sdr import SDR, SDR_ACTIVE
from genesis.perception.binder import Binder


def test_bind_single_token_returns_sdr():
    b = Binder()
    sdr = SDR.random()
    result = b.bind([sdr])
    assert isinstance(result, SDR)
    assert result.popcount() == SDR_ACTIVE


def test_bind_empty_returns_zeros():
    b = Binder()
    result = b.bind([])
    assert result.popcount() == 0


def test_bind_order_matters():
    b = Binder()
    a = SDR(list(range(0, 20)))
    c = SDR(list(range(100, 120)))
    phrase1 = b.bind([a, c])
    phrase2 = b.bind([c, a])
    assert phrase1 != phrase2


def test_bind_produces_stable_output():
    b = Binder()
    sdrs = [SDR.random() for _ in range(4)]
    r1 = b.bind(sdrs)
    r2 = b.bind(sdrs)
    assert r1 == r2
