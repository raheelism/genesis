from genesis.perception.tokenizer import Tokenizer


def test_tokenize_returns_list_of_strings():
    t = Tokenizer()
    result = t.tokenize("hello world")
    assert isinstance(result, list)
    assert all(isinstance(tok, str) for tok in result)


def test_tokenize_simple_sentence():
    t = Tokenizer()
    result = t.tokenize("the cat sat")
    assert "the" in result
    assert "cat" in result
    assert "sat" in result


def test_tokenize_lowercases():
    t = Tokenizer()
    result = t.tokenize("Hello WORLD")
    assert "hello" in result
    assert "world" in result


def test_tokenize_strips_punctuation():
    t = Tokenizer()
    result = t.tokenize("hello, world!")
    assert "hello" in result
    assert "world" in result
    assert "," not in result


def test_add_to_vocab_increments():
    t = Tokenizer()
    t.tokenize("cat dog bird")
    assert "cat" in t.vocab
    assert "dog" in t.vocab


def test_unknown_token_returns_unk():
    t = Tokenizer()
    t.tokenize("known")
    tokens = t.tokenize("unknown_xyz")
    assert "<unk>" in t.vocab


def test_vocab_contains_special_tokens():
    t = Tokenizer()
    assert "<unk>" in t.vocab
    assert "<pad>" in t.vocab
