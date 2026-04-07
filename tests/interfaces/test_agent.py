# tests/interfaces/test_agent.py
from genesis.core.organism import Organism
from genesis.core.working_memory import WorkingMemory
from genesis.perception.tokenizer import Tokenizer
from genesis.perception.encoder import Encoder
from genesis.perception.binder import Binder
from genesis.reasoning.backward_chain import BackwardChain
from genesis.reasoning.forward_chain import ForwardChain
from genesis.learning.hebbian import HebbianLearner
from genesis.generation.verbalizer import Verbalizer
from genesis.interfaces.agent import AgentInterface

def _setup_agent():
    tok = Tokenizer()
    enc = Encoder()
    for w in ["find", "answer", "goal", "complete", "task"]:
        tok.tokenize(w)
    enc.register_vocab(tok.vocab)
    return AgentInterface(
        tokenizer=tok, encoder=enc, binder=Binder(),
        organism=Organism(), working_memory=WorkingMemory(),
        forward=ForwardChain(), backward=BackwardChain(),
        learner=HebbianLearner(), verbalizer=Verbalizer(),
    )

def test_set_goal_stores_goal_sdr():
    agent = _setup_agent()
    agent.set_goal("find answer")
    assert agent.goal_sdr is not None

def test_step_returns_action_string():
    agent = _setup_agent()
    agent.set_goal("complete task")
    action = agent.step()
    assert isinstance(action, str)

def test_agent_is_idle_without_goal():
    agent = _setup_agent()
    action = agent.step()
    assert action == "<no goal set>"
