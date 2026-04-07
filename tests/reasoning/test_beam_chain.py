# tests/reasoning/test_beam_chain.py
from genesis.core.sdr import SDR
from genesis.core.cell import Cell
from genesis.core.organism import Organism
from genesis.core.working_memory import WorkingMemory
from genesis.reasoning.beam_chain import BeamChain
from genesis.reasoning.forward_chain import ReasoningResult


def _make_cell(rf_bits, pre_bits, post_bits, confidence):
    cell = Cell()
    cell.receptive_field = SDR(rf_bits)
    pre = SDR(pre_bits)
    post = SDR(post_bits)
    cell.add_rule(pre, post, confidence=confidence)
    return cell, pre, post


def test_beam_chain_finds_direct_rule():
    """Single-hop answer with high confidence is returned immediately."""
    org = Organism()
    wm = WorkingMemory()
    cell, pre, post = _make_cell(
        list(range(0, 20)), list(range(0, 20)), list(range(200, 220)),
        confidence=0.9,
    )
    org.add_cell(cell)
    result = BeamChain().reason(pre, org, wm)
    assert result.answer is not None
    assert result.confidence > 0


def test_beam_chain_finds_two_hop_chain():
    """2-hop chain: pre -> mid -> post.
    Depth-0 answer mid has confidence 0.50 * 0.98 = 0.49 < THETA_EARLY_RETURN=0.65.
    Depth-1 answer post has confidence 0.50 * 0.90 * 0.98^2 = 0.432 > THETA_ANSWER=0.35."""
    org = Organism()
    wm = WorkingMemory()

    pre = SDR(list(range(0, 20)))
    mid = SDR(list(range(200, 220)))
    post = SDR(list(range(400, 420)))

    cell_a = Cell()
    cell_a.receptive_field = pre
    cell_a.add_rule(pre, mid, confidence=0.50)
    org.add_cell(cell_a)

    cell_b = Cell()
    cell_b.receptive_field = mid
    cell_b.add_rule(mid, post, confidence=0.90)
    org.add_cell(cell_b)

    result = BeamChain().reason(pre, org, wm)
    assert result.answer is not None
    # Should reach post (the 2-hop answer), not stop at mid
    assert result.answer.similarity(post) > result.answer.similarity(mid)


def test_beam_chain_returns_none_when_no_rules():
    org = Organism()
    wm = WorkingMemory()
    result = BeamChain().reason(SDR.random(), org, wm)
    assert result.answer is None
    assert result.confidence == 0.0


def test_beam_chain_returns_reasoning_result():
    org = Organism()
    wm = WorkingMemory()
    cell, pre, post = _make_cell(
        list(range(0, 20)), list(range(0, 20)), list(range(200, 220)),
        confidence=0.9,
    )
    org.add_cell(cell)
    result = BeamChain().reason(pre, org, wm)
    assert isinstance(result, ReasoningResult)
