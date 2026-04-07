# tests/reasoning/test_backward_chain.py
from genesis.core.sdr import SDR
from genesis.core.cell import Cell
from genesis.core.organism import Organism
from genesis.reasoning.backward_chain import BackwardChain, VerificationResult

def _setup_provable_goal():
    """Rule: fact_A → goal_B. Known fact: fact_A. Goal: goal_B."""
    pre = SDR(list(range(0, 20)))
    post = SDR(list(range(200, 220)))
    cell = Cell()
    cell.receptive_field = pre
    cell.add_rule(pre, post, confidence=0.9)
    org = Organism()
    org.add_cell(cell)
    known_facts = pre
    goal = post
    return org, known_facts, goal

def test_verify_provable_goal_returns_true():
    org, known, goal = _setup_provable_goal()
    planner = BackwardChain()
    result = planner.verify(goal, org, known)
    assert isinstance(result, VerificationResult)
    assert result.verified

def test_verify_unprovable_goal_returns_false():
    org = Organism()
    planner = BackwardChain()
    goal = SDR(list(range(0, 20)))
    known = SDR(list(range(500, 520)))
    result = planner.verify(goal, org, known)
    assert not result.verified

def test_verify_returns_confidence():
    org, known, goal = _setup_provable_goal()
    planner = BackwardChain()
    result = planner.verify(goal, org, known)
    assert result.confidence > 0.0

def test_verify_returns_chain():
    org, known, goal = _setup_provable_goal()
    planner = BackwardChain()
    result = planner.verify(goal, org, known)
    assert len(result.support_chain) > 0
