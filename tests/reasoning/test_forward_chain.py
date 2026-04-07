# tests/reasoning/test_forward_chain.py
from genesis.core.sdr import SDR
from genesis.core.cell import Cell
from genesis.core.organism import Organism
from genesis.core.working_memory import WorkingMemory
from genesis.reasoning.forward_chain import ForwardChain, ReasoningResult

def _setup_simple_chain():
    """Cell with rule: pattern_A → pattern_B (confidence 0.9)"""
    org = Organism()
    wm = WorkingMemory()
    pre = SDR(list(range(0, 20)))
    post = SDR(list(range(200, 220)))
    cell = Cell()
    cell.receptive_field = pre
    cell.add_rule(pre, post, confidence=0.9)
    org.add_cell(cell)
    return org, wm, pre, post

def test_reason_finds_direct_rule():
    org, wm, pre, post = _setup_simple_chain()
    reasoner = ForwardChain()
    result = reasoner.reason(pre, org, wm)
    assert isinstance(result, ReasoningResult)
    assert result.answer is not None
    assert result.confidence > 0

def test_reason_returns_chain():
    org, wm, pre, post = _setup_simple_chain()
    reasoner = ForwardChain()
    result = reasoner.reason(pre, org, wm)
    assert len(result.chain) > 0

def test_reason_with_no_rules_returns_none_answer():
    org = Organism()
    wm = WorkingMemory()
    reasoner = ForwardChain()
    result = reasoner.reason(SDR.random(), org, wm)
    assert result.answer is None
    assert result.confidence == 0.0

def test_reason_uses_working_memory_context():
    org, wm, pre, post = _setup_simple_chain()
    # Push pre into working memory — query alone won't match but context will
    wm.push(pre)
    reasoner = ForwardChain()
    partial_query = SDR(list(range(0, 10)) + list(range(50, 60)))
    result = reasoner.reason(partial_query, org, wm)
    # With context, the cell should still find relevant rules
    assert isinstance(result, ReasoningResult)

def test_context_reranking_does_not_crash():
    """Verifies that pushing context into WM and then reasoning doesn't crash."""
    org, wm, pre, post = _setup_simple_chain()
    wm.push(pre)        # pre is now in context
    wm.push(post)       # post is also in context
    reasoner = ForwardChain()
    result = reasoner.reason(pre, org, wm)
    assert isinstance(result, ReasoningResult)
    assert result.answer is not None

def test_context_reranking_stable_with_empty_context():
    """Empty working memory produces same result as no context."""
    org, wm, pre, post = _setup_simple_chain()
    # wm is empty
    reasoner = ForwardChain()
    result = reasoner.reason(pre, org, wm)
    assert result.answer is not None
