from genesis.core.sdr import SDR
from genesis.perception.encoder import Encoder
from genesis.generation.verbalizer import Verbalizer

def _setup():
    enc = Encoder()
    for word in ["the", "cat", "sat", "on", "mat", "dog", "ran", "fast"]:
        enc.register(word)
    return enc

def test_verbalize_returns_string():
    enc = _setup()
    v = Verbalizer()
    sdr = enc.encode_token("cat")
    result = v.verbalize(sdr, enc, max_tokens=3)
    assert isinstance(result, str)

def test_verbalize_returns_nonempty_for_known_sdr():
    enc = _setup()
    v = Verbalizer()
    sdr = enc.encode_token("dog")
    result = v.verbalize(sdr, enc, max_tokens=3)
    assert len(result.strip()) > 0

def test_verbalize_top_match_is_registered_token():
    enc = _setup()
    v = Verbalizer()
    sdr = enc.encode_token("cat")
    result = v.verbalize(sdr, enc, max_tokens=1)
    assert "cat" in result

def test_verbalize_empty_sdr_returns_unknown():
    enc = _setup()
    v = Verbalizer()
    result = v.verbalize(SDR.zeros(), enc, max_tokens=3)
    assert result == "<unknown>"
