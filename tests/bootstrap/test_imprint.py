import os
import tempfile
from genesis.core.organism import Organism
from genesis.perception.tokenizer import Tokenizer
from genesis.perception.encoder import Encoder
from genesis.perception.binder import Binder
from genesis.bootstrap.seed_loader import SeedLoader
from genesis.bootstrap.imprint import Imprinter

SAMPLE_TEXT = """The cat sat on the mat.
The dog ran fast.
Birds fly high in the sky.
Water flows downhill.
Fire is hot and bright.
The sun rises in the east.
"""

def _write_corpus(text: str) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    f.write(text)
    f.close()
    return f.name

def test_seed_loader_yields_sentences():
    path = _write_corpus(SAMPLE_TEXT)
    loader = SeedLoader()
    sentences = list(loader.load(path))
    os.unlink(path)
    assert len(sentences) > 0
    assert all(isinstance(s, str) for s in sentences)

def test_seed_loader_strips_empty_lines():
    path = _write_corpus(SAMPLE_TEXT)
    loader = SeedLoader()
    sentences = list(loader.load(path))
    os.unlink(path)
    assert all(len(s.strip()) > 0 for s in sentences)

def test_phase0_populates_encoder_vocab():
    path = _write_corpus(SAMPLE_TEXT)
    loader = SeedLoader()
    sentences = list(loader.load(path))
    os.unlink(path)
    tok = Tokenizer()
    enc = Encoder()
    imp = Imprinter()
    imp.phase0(sentences, tok, enc)
    assert enc.vocab_size() > 5  # at least the content words

def test_phase1_adds_pattern_cells():
    path = _write_corpus(SAMPLE_TEXT)
    loader = SeedLoader()
    sentences = list(loader.load(path))
    os.unlink(path)
    tok = Tokenizer()
    enc = Encoder()
    binder = Binder()
    org = Organism()
    imp = Imprinter()
    imp.phase0(sentences, tok, enc)
    imp.phase1(sentences, tok, enc, binder, org)
    assert org.cell_count() > 0

def test_phase3_adds_reasoning_cells():
    org = Organism()
    enc = Encoder()
    tok = Tokenizer()
    binder = Binder()
    qa_pairs = [
        ("what is water", "liquid"),
        ("what color is sky", "blue"),
    ]
    imp = Imprinter()
    # Register tokens first
    for q, a in qa_pairs:
        tok.tokenize(q)
        tok.tokenize(a)
    enc.register_vocab(tok.vocab)
    imp.phase3(qa_pairs, tok, enc, binder, org)
    assert org.cell_count() > 0

def test_phase3_registers_phrases_in_store():
    from genesis.generation.phrase_store import PhraseStore
    org = Organism()
    enc = Encoder()
    tok = Tokenizer()
    binder = Binder()
    qa = [("what does fire produce", "heat and light")]
    for q, a in qa:
        tok.tokenize(q)
        tok.tokenize(a)
    enc.register_vocab(tok.vocab)
    ps = PhraseStore()
    imp = Imprinter()
    imp.phase3(qa, tok, enc, binder, org, phrase_store=ps)
    assert len(ps) == 1

def test_phase3_without_phrase_store_still_works():
    org = Organism()
    enc = Encoder()
    tok = Tokenizer()
    binder = Binder()
    qa = [("what is water", "liquid")]
    for q, a in qa:
        tok.tokenize(q)
        tok.tokenize(a)
    enc.register_vocab(tok.vocab)
    imp = Imprinter()
    imp.phase3(qa, tok, enc, binder, org)   # no phrase_store arg
    assert org.cell_count() > 0
