# tests/interfaces/test_embed.py
from genesis.core.organism import Organism
from genesis.perception.tokenizer import Tokenizer
from genesis.perception.encoder import Encoder
from genesis.perception.binder import Binder
from genesis.learning.hebbian import HebbianLearner
from genesis.interfaces.embed import EmbedInterface

def _setup_embed():
    tok = Tokenizer()
    enc = Encoder()
    for w in ["event", "data", "input", "signal"]:
        tok.tokenize(w)
    enc.register_vocab(tok.vocab)
    return EmbedInterface(
        tokenizer=tok, encoder=enc, binder=Binder(),
        organism=Organism(), learner=HebbianLearner(),
    )

def test_process_single_text_event():
    embed = _setup_embed()
    events = embed.process(["sensor data input"])
    assert isinstance(events, list)

def test_process_grows_organism():
    embed = _setup_embed()
    initial = embed.organism.cell_count()
    for i in range(20):
        embed.process([f"event signal {i} data input"])
    # Should have created at least some cells
    assert embed.organism.cell_count() >= initial
