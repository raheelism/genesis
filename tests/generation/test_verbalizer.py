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

def test_verbalize_uses_phrase_store_when_sdr_matches():
    from genesis.generation.phrase_store import PhraseStore
    enc = _setup()
    ps = PhraseStore()
    v = Verbalizer(phrase_store=ps)
    sdr = enc.encode_token("cat")
    ps.register(sdr, "the cat sat on the mat")
    result = v.verbalize(sdr, enc, max_tokens=5)
    assert result == "the cat sat on the mat"

def test_verbalize_falls_back_to_greedy_when_no_phrase_match():
    from genesis.generation.phrase_store import PhraseStore
    enc = _setup()
    ps = PhraseStore()   # empty — no registrations
    v = Verbalizer(phrase_store=ps)
    sdr = enc.encode_token("dog")
    result = v.verbalize(sdr, enc, max_tokens=3)
    assert "dog" in result

def test_verbalize_no_phrase_store_still_works():
    enc = _setup()
    v = Verbalizer()   # no phrase_store
    sdr = enc.encode_token("cat")
    result = v.verbalize(sdr, enc, max_tokens=3)
    assert isinstance(result, str)
    assert len(result) > 0

def test_verbalize_falls_back_to_greedy_with_populated_store_no_match():
    """PhraseStore is populated, but the query SDR doesn't match any entry.
    Verbalizer must fall back to greedy token coverage (not return None or crash)."""
    from genesis.generation.phrase_store import PhraseStore
    enc = _setup()
    ps = PhraseStore()
    # Register a phrase for "cat" SDR
    cat_sdr = enc.encode_token("cat")
    ps.register(cat_sdr, "the cat sat")
    v = Verbalizer(phrase_store=ps)
    # Query with "dog" SDR — disjoint from "cat" SDR, similarity << 0.85
    dog_sdr = enc.encode_token("dog")
    result = v.verbalize(dog_sdr, enc, max_tokens=3)
    # Must fall back to greedy, not return None
    assert isinstance(result, str)
    assert len(result) > 0
    assert result != "the cat sat"   # must NOT return the wrong phrase
    assert "dog" in result           # greedy should find "dog" as top token
