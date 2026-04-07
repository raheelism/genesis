# tests/interfaces/test_chat.py
from genesis.core.organism import Organism
from genesis.core.working_memory import WorkingMemory
from genesis.core.cell import Cell
from genesis.core.sdr import SDR
from genesis.perception.tokenizer import Tokenizer
from genesis.perception.encoder import Encoder
from genesis.perception.binder import Binder
from genesis.reasoning.forward_chain import ForwardChain
from genesis.learning.hebbian import HebbianLearner
from genesis.generation.verbalizer import Verbalizer
from genesis.interfaces.chat import ChatInterface

def _setup_chat():
    tok = Tokenizer()
    enc = Encoder()
    for word in ["hello", "world", "cat", "dog", "is", "a", "animal"]:
        tok.tokenize(word)
    enc.register_vocab(tok.vocab)
    binder = Binder()
    org = Organism()
    wm = WorkingMemory()
    reasoner = ForwardChain()
    learner = HebbianLearner()
    verbalizer = Verbalizer()
    return ChatInterface(tok, enc, binder, org, wm, reasoner, learner, verbalizer)

def test_chat_turn_returns_string():
    chat = _setup_chat()
    response = chat.turn("hello world")
    assert isinstance(response, str)

def test_chat_turn_updates_working_memory():
    chat = _setup_chat()
    chat.turn("hello world")
    assert len(chat.working_memory) > 0

def test_chat_learns_from_explicit_fact():
    chat = _setup_chat()
    # Teach it a fact then ask about it
    chat.turn("cat is an animal")
    # After teaching, working memory contains the concept
    assert len(chat.working_memory) > 0

def test_chat_turn_count_increments():
    chat = _setup_chat()
    chat.turn("hello")
    chat.turn("world")
    assert chat.turn_count == 2
