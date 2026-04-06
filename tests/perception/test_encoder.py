from genesis.core.sdr import SDR, SDR_ACTIVE
from genesis.perception.tokenizer import Tokenizer
from genesis.perception.encoder import Encoder


def test_encode_token_returns_sdr():
    enc = Encoder()
    enc.register("cat")
    sdr = enc.encode_token("cat")
    assert isinstance(sdr, SDR)


def test_encode_token_has_correct_sparsity():
    enc = Encoder()
    enc.register("dog")
    sdr = enc.encode_token("dog")
    assert sdr.popcount() == SDR_ACTIVE


def test_same_token_returns_same_sdr():
    enc = Encoder()
    enc.register("fish")
    a = enc.encode_token("fish")
    b = enc.encode_token("fish")
    assert a == b


def test_different_tokens_return_different_sdrs():
    enc = Encoder()
    enc.register("cat")
    enc.register("dog")
    a = enc.encode_token("cat")
    b = enc.encode_token("dog")
    assert a != b


def test_unknown_token_returns_unk_sdr():
    enc = Encoder()
    sdr = enc.encode_token("zzzunknownzzz")
    assert isinstance(sdr, SDR)
    assert sdr.popcount() == SDR_ACTIVE


def test_register_from_vocab():
    tok = Tokenizer()
    tok.tokenize("the quick brown fox")
    enc = Encoder()
    enc.register_vocab(tok.vocab)
    sdr = enc.encode_token("quick")
    assert isinstance(sdr, SDR)


def test_vocab_size_returns_count():
    enc = Encoder()
    enc.register("alpha")
    enc.register("beta")
    assert enc.vocab_size() == 2
