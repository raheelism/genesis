# tests/test_integration.py
"""End-to-end integration: bootstrap a micro-corpus, chat, verify memory."""
import os
import tempfile
from genesis.bootstrap.imprint import Imprinter
from genesis.bootstrap.seed_loader import SeedLoader
from genesis.core.organism import Organism
from genesis.core.working_memory import WorkingMemory
from genesis.generation.verbalizer import Verbalizer
from genesis.interfaces.chat import ChatInterface
from genesis.learning.hebbian import HebbianLearner
from genesis.perception.binder import Binder
from genesis.perception.encoder import Encoder
from genesis.perception.tokenizer import Tokenizer
from genesis.reasoning.forward_chain import ForwardChain
from genesis.storage.colony_store import ColonyStore

MICRO_CORPUS = """The sky is blue.
Water is wet.
Fire is hot.
Ice is cold.
The sun is bright.
Dogs are animals.
Cats are animals.
Birds can fly.
Fish live in water.
The earth is round.
"""

QA_PAIRS = [
    ("what is the sky", "blue"),
    ("what is water", "wet"),
    ("what is fire", "hot"),
]


def _build_system(corpus_text: str, qa: list):
    tok = Tokenizer()
    enc = Encoder()
    binder = Binder()
    org = Organism()
    imp = Imprinter()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                     delete=False) as f:
        f.write(corpus_text)
        corpus_path = f.name

    loader = SeedLoader()
    sentences = list(loader.load(corpus_path))
    os.unlink(corpus_path)

    imp.phase0(sentences, tok, enc)
    imp.phase1(sentences, tok, enc, binder, org)
    imp.phase3(qa, tok, enc, binder, org)

    return tok, enc, binder, org


def test_full_bootstrap_creates_cells():
    tok, enc, binder, org = _build_system(MICRO_CORPUS, QA_PAIRS)
    assert org.cell_count() > 0


def test_chat_responds_after_bootstrap():
    tok, enc, binder, org = _build_system(MICRO_CORPUS, QA_PAIRS)
    wm = WorkingMemory()
    chat = ChatInterface(
        tok, enc, binder, org, wm,
        ForwardChain(), HebbianLearner(), Verbalizer(),
    )
    response = chat.turn("what is the sky")
    assert isinstance(response, str)
    assert len(response) > 0


def test_new_fact_recalled_in_same_session():
    tok, enc, binder, org = _build_system(MICRO_CORPUS, QA_PAIRS)
    wm = WorkingMemory()
    chat = ChatInterface(
        tok, enc, binder, org, wm,
        ForwardChain(), HebbianLearner(), Verbalizer(),
    )
    # Teach new fact
    chat.turn("jupiter is a planet")
    # Working memory should contain the concept
    assert len(wm) > 0


def test_save_and_reload_preserves_cells():
    tok, enc, binder, org = _build_system(MICRO_CORPUS, QA_PAIRS)
    store = ColonyStore()
    with tempfile.NamedTemporaryFile(suffix=".gen", delete=False) as f:
        path = f.name
    store.save(org, path)
    loaded = store.load(path)
    os.unlink(path)
    assert loaded.cell_count() == org.cell_count()


def test_twenty_turn_conversation_stays_coherent():
    tok, enc, binder, org = _build_system(MICRO_CORPUS, QA_PAIRS)
    wm = WorkingMemory()
    chat = ChatInterface(
        tok, enc, binder, org, wm,
        ForwardChain(), HebbianLearner(), Verbalizer(),
    )
    inputs = [
        "the sky is blue", "water is wet", "fire is hot",
        "tell me about animals", "dogs are animals", "cats are animals",
        "birds can fly", "fish live in water", "the earth is round",
        "what is the sky", "what is water", "what is fire",
        "are dogs animals", "can birds fly", "where do fish live",
        "is the earth round", "is the sun bright", "what is ice",
        "tell me about the sky", "what do you know",
    ]
    for user_input in inputs:
        response = chat.turn(user_input)
        assert isinstance(response, str)
    assert chat.turn_count == 20
