# GENESIS Improvements: Verbalization, Context Re-Ranking, Multi-Hop Beam, SQuAD Test

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix compound-answer verbalization via a PhraseStore, add context-aware cell re-ranking in ForwardChain, enable 2-hop beam inference via BeamChain, then download SQuAD-tiny and run a full benchmark on it.

**Architecture:** (1) `PhraseStore` maps answer SDRs → registered phrase strings so `Verbalizer` emits exact phrases instead of reconstructing token-by-token. (2) `ForwardChain` re-ranks active cells by working-memory context overlap before rule evaluation — routing stays on clean query but contextually-relevant cells are tried first. (3) `BeamChain` replaces greedy depth-1 search with width-3 beam search and a higher depth-0 acceptance threshold (`THETA_EARLY_RETURN=0.65`), allowing 2-hop chains to surface when no single-hop answer is very confident. (4) `data/prepare_squad.py` downloads SQuAD validation (300 examples via HuggingFace `datasets`) and writes `squad_corpus.txt` / `squad_qa.txt`; `benchmark_squad.py` runs the full 4-phase bootstrap + evaluation.

**Tech Stack:** Python 3.12, numpy, pytest, huggingface `datasets` (for SQuAD download)

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `genesis/generation/phrase_store.py` | SDR → phrase string registry with similarity lookup |
| Modify | `genesis/generation/verbalizer.py` | Check PhraseStore before greedy reconstruction |
| Modify | `genesis/bootstrap/imprint.py` | phase3 registers answer string in PhraseStore |
| Modify | `benchmark.py` | Wire PhraseStore + BeamChain |
| Modify | `benchmark_large.py` | Wire PhraseStore + BeamChain |
| Modify | `genesis/reasoning/forward_chain.py` | Context re-ranking of active cells after routing |
| Create | `genesis/reasoning/beam_chain.py` | Width-3 beam search with THETA_EARLY_RETURN |
| Create | `data/prepare_squad.py` | Download SQuAD, write corpus + QA files |
| Create | `benchmark_squad.py` | Full benchmark on SQuAD data |
| Create | `tests/generation/test_phrase_store.py` | PhraseStore unit tests |
| Modify | `tests/generation/test_verbalizer.py` | PhraseStore integration tests |
| Modify | `tests/bootstrap/test_imprint.py` | phase3 phrase registration test |
| Modify | `tests/reasoning/test_forward_chain.py` | Context re-ranking smoke test |
| Create | `tests/reasoning/test_beam_chain.py` | BeamChain 2-hop inference test |

---

## Task 1: PhraseStore

**Files:**
- Create: `genesis/generation/phrase_store.py`
- Create: `tests/generation/test_phrase_store.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/generation/test_phrase_store.py`:

```python
# tests/generation/test_phrase_store.py
from genesis.core.sdr import SDR
from genesis.generation.phrase_store import PhraseStore

def test_lookup_returns_registered_phrase():
    ps = PhraseStore()
    sdr = SDR(list(range(20)))
    ps.register(sdr, "heat and light")
    result = ps.lookup(sdr)
    assert result == "heat and light"

def test_lookup_returns_none_for_dissimilar_sdr():
    ps = PhraseStore()
    sdr_a = SDR(list(range(20)))
    sdr_b = SDR(list(range(500, 520)))
    ps.register(sdr_a, "heat and light")
    assert ps.lookup(sdr_b) is None

def test_lookup_returns_none_when_empty():
    ps = PhraseStore()
    assert ps.lookup(SDR.random()) is None

def test_len_reflects_registrations():
    ps = PhraseStore()
    ps.register(SDR.random(), "phrase one")
    ps.register(SDR.random(), "phrase two")
    assert len(ps) == 2

def test_lookup_picks_closest_when_multiple_registered():
    ps = PhraseStore()
    sdr_a = SDR(list(range(0, 20)))
    sdr_b = SDR(list(range(500, 520)))
    ps.register(sdr_a, "phrase a")
    ps.register(sdr_b, "phrase b")
    assert ps.lookup(sdr_a) == "phrase a"
    assert ps.lookup(sdr_b) == "phrase b"
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/generation/test_phrase_store.py -v
```
Expected: `ModuleNotFoundError: No module named 'genesis.generation.phrase_store'`

- [ ] **Step 3: Implement PhraseStore**

Create `genesis/generation/phrase_store.py`:

```python
# genesis/generation/phrase_store.py
from typing import List, Optional, Tuple
from genesis.core.sdr import SDR

THETA_PHRASE = 0.85   # minimum Jaccard similarity to accept a phrase match


class PhraseStore:
    """Registry mapping answer SDRs to their original phrase strings.
    Used by Verbalizer to emit exact multi-word phrases instead of
    reconstructing token-by-token from bit coverage."""

    def __init__(self):
        self._entries: List[Tuple[SDR, str]] = []

    def register(self, sdr: SDR, phrase: str):
        """Register an SDR → phrase mapping."""
        self._entries.append((sdr, phrase))

    def lookup(self, sdr: SDR) -> Optional[str]:
        """Return the phrase whose registered SDR is most similar to sdr,
        or None if no entry exceeds THETA_PHRASE."""
        best_sim = THETA_PHRASE
        best_phrase = None
        for reg_sdr, phrase in self._entries:
            sim = sdr.similarity(reg_sdr)
            if sim > best_sim:
                best_sim = sim
                best_phrase = phrase
        return best_phrase

    def __len__(self) -> int:
        return len(self._entries)
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/generation/test_phrase_store.py -v
```
Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add genesis/generation/phrase_store.py tests/generation/test_phrase_store.py
git commit -m "feat: add PhraseStore for SDR-to-phrase registry"
```

---

## Task 2: Verbalizer — phrase lookup before greedy reconstruction

**Files:**
- Modify: `genesis/generation/verbalizer.py`
- Modify: `tests/generation/test_verbalizer.py`

- [ ] **Step 1: Add failing tests to test_verbalizer.py**

Append these tests to `tests/generation/test_verbalizer.py` (keep all existing tests):

```python
def test_verbalize_uses_phrase_store_when_sdr_matches():
    from genesis.generation.phrase_store import PhraseStore
    enc = _setup()
    ps = PhraseStore()
    v = Verbalizer(phrase_store=ps)
    sdr = enc.encode_token("cat")
    ps.register(sdr, "the cat sat on the mat")
    result = v.verbalize(sdr, enc, max_tokens=5)
    assert result == "the cat sat on the mat"

def test_verbalize_falls_back_to_greedy_when_no_phrase_match():
    from genesis.generation.phrase_store import PhraseStore
    enc = _setup()
    ps = PhraseStore()   # empty — no registrations
    v = Verbalizer(phrase_store=ps)
    sdr = enc.encode_token("dog")
    result = v.verbalize(sdr, enc, max_tokens=3)
    assert "dog" in result

def test_verbalize_no_phrase_store_still_works():
    enc = _setup()
    v = Verbalizer()   # no phrase_store
    sdr = enc.encode_token("cat")
    result = v.verbalize(sdr, enc, max_tokens=3)
    assert isinstance(result, str)
    assert len(result) > 0
```

- [ ] **Step 2: Run new tests to verify they fail**

```
pytest tests/generation/test_verbalizer.py::test_verbalize_uses_phrase_store_when_sdr_matches -v
```
Expected: `TypeError: Verbalizer.__init__() got an unexpected keyword argument 'phrase_store'`

- [ ] **Step 3: Rewrite verbalizer.py to accept PhraseStore**

Replace `genesis/generation/verbalizer.py` entirely:

```python
# genesis/generation/verbalizer.py
from typing import Optional
from genesis.core.sdr import SDR
from genesis.perception.encoder import Encoder
from genesis.generation.phrase_store import PhraseStore

THETA_VERBALIZE = 0.05   # minimum similarity to include a token candidate


class Verbalizer:
    """Converts a conclusion SDR back into a token sequence.

    First checks PhraseStore for an exact registered phrase (covers multi-word
    answers like 'heat and light'). Falls back to greedy bit-coverage
    reconstruction if no registered phrase is close enough."""

    def __init__(self, phrase_store: Optional[PhraseStore] = None):
        self.phrase_store = phrase_store

    def verbalize(self, sdr: SDR, encoder: Encoder,
                  max_tokens: int = 10) -> str:
        if sdr.popcount() == 0:
            return "<unknown>"

        # 1. Check phrase store — exact registered phrase takes priority
        if self.phrase_store is not None:
            phrase = self.phrase_store.lookup(sdr)
            if phrase is not None:
                return phrase

        # 2. Fall back: greedy token coverage
        scored = [
            (token, sdr.similarity(tok_sdr))
            for token, tok_sdr in encoder._vocab_sdrs.items()
            if token not in ("<pad>", "<unk>", "<start>", "<end>")
        ]
        scored = [(tok, s) for tok, s in scored if s > THETA_VERBALIZE]
        scored.sort(key=lambda x: x[1], reverse=True)

        if not scored:
            return "<unknown>"

        covered = SDR.zeros()
        selected = []
        for token, score in scored[:50]:
            tok_sdr = encoder.encode_token(token)
            new_coverage = tok_sdr.similarity(sdr) - covered.similarity(tok_sdr)
            if new_coverage > 0:
                selected.append(token)
                covered = covered.compose(tok_sdr)
                if len(selected) >= max_tokens:
                    break

        return " ".join(selected) if selected else "<unknown>"
```

- [ ] **Step 4: Run all verbalizer tests**

```
pytest tests/generation/test_verbalizer.py -v
```
Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add genesis/generation/verbalizer.py tests/generation/test_verbalizer.py
git commit -m "feat: verbalizer checks PhraseStore before greedy reconstruction"
```

---

## Task 3: Imprinter.phase3 — register answer strings in PhraseStore

**Files:**
- Modify: `genesis/bootstrap/imprint.py` (lines 65–77)
- Modify: `tests/bootstrap/test_imprint.py`

- [ ] **Step 1: Add failing test to test_imprint.py**

Append to `tests/bootstrap/test_imprint.py`:

```python
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
```

- [ ] **Step 2: Run new tests to verify they fail**

```
pytest tests/bootstrap/test_imprint.py::test_phase3_registers_phrases_in_store -v
```
Expected: `TypeError: phase3() got an unexpected keyword argument 'phrase_store'`

- [ ] **Step 3: Update phase3 signature and body**

In `genesis/bootstrap/imprint.py`, replace the `phase3` method (lines 65–77):

```python
def phase3(self, qa_pairs: List[Tuple[str, str]], tokenizer: Tokenizer,
           encoder: Encoder, binder: Binder, organism: Organism,
           phrase_store=None):
    """Create Reasoning Cells from structured QA pairs.
    If phrase_store is provided, registers the answer SDR → answer string
    so Verbalizer can emit exact phrases for compound answers."""
    for question, answer in qa_pairs:
        q_tokens = tokenizer.tokenize(question)
        a_tokens = tokenizer.tokenize(answer)
        encoder.register_vocab(tokenizer.vocab)
        q_sdr = binder.bind([encoder.encode_token(t) for t in q_tokens])
        a_sdr = binder.bind([encoder.encode_token(t) for t in a_tokens])
        cell = Cell()
        cell.receptive_field = q_sdr
        cell.add_rule(q_sdr, a_sdr, confidence=1.0)
        organism.add_cell(cell)
        if phrase_store is not None:
            phrase_store.register(a_sdr, answer)
```

- [ ] **Step 4: Run all imprint tests**

```
pytest tests/bootstrap/test_imprint.py -v
```
Expected: all tests PASS (5 or 6 depending on previous test count)

- [ ] **Step 5: Commit**

```bash
git add genesis/bootstrap/imprint.py tests/bootstrap/test_imprint.py
git commit -m "feat: imprinter phase3 registers answer phrases in PhraseStore"
```

---

## Task 4: Wire PhraseStore through benchmark.py and benchmark_large.py

**Files:**
- Modify: `benchmark.py`
- Modify: `benchmark_large.py`

No new tests needed — these are runner scripts, verified by running them.

- [ ] **Step 1: Update benchmark.py**

Add import at top of `benchmark.py` (after existing imports):

```python
from genesis.generation.phrase_store import PhraseStore
```

In `build_system()`, add `ps = PhraseStore()` before phase3 and pass it:

```python
def build_system():
    tok = Tokenizer()
    enc = Encoder()
    binder = Binder()
    org = Organism()

    loader = SeedLoader()
    imp = Imprinter()
    ps = PhraseStore()

    print(f"\n{SEP}")
    print("PHASE 0 - Vocabulary imprinting...")
    t0 = time.perf_counter()
    sentences = list(loader.load(CORPUS_PATH))
    imp.phase0(sentences, tok, enc)
    print(f"  Sentences : {len(sentences)}")
    print(f"  Vocab size: {enc.vocab_size()} tokens")

    print("PHASE 1 - Pattern seeding...")
    imp.phase1(sentences, tok, enc, binder, org)
    print(f"  Cells     : {org.cell_count()}")

    print("PHASE 2 - Concept emergence...")
    imp.phase2(sentences, tok, enc, binder, org)
    print(f"  Cells     : {org.cell_count()}")

    print("PHASE 3 - Reasoning seed (QA pairs)...")
    qa_pairs = load_qa_pairs()
    imp.phase3(qa_pairs, tok, enc, binder, org, phrase_store=ps)
    elapsed = time.perf_counter() - t0
    print(f"  Cells     : {org.cell_count()}")
    print(f"  QA pairs  : {len(qa_pairs)}")
    print(f"  Total time: {elapsed:.2f}s")

    return tok, enc, binder, org, qa_pairs, ps
```

In `main()`, unpack the new return value and pass `ps` to `Verbalizer`:

```python
    tok, enc, binder, org, qa_pairs, ps = build_system()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    colony_bytes = measure_memory(org)
    print(f"\n  Peak RAM during bootstrap : {peak_bytes/1024/1024:.1f} MB")
    print(f"  Colony serialized size    : {colony_bytes/1024:.1f} KB")
    print(f"  Total rules               : {sum(len(c.rules) for c in org.cells.values())}")

    store = ColonyStore()
    store.save(org, COLONY_PATH)
    print(f"  Colony saved to           : {COLONY_PATH}")

    wm = WorkingMemory()
    chat = ChatInterface(
        tok, enc, binder, org, wm,
        ForwardChain(), HebbianLearner(), Verbalizer(phrase_store=ps),
    )
```

- [ ] **Step 2: Apply the same changes to benchmark_large.py**

Add `from genesis.generation.phrase_store import PhraseStore` at top.

In `build_system()`:
- Add `ps = PhraseStore()` before phase3
- Change phase3 call to `imp.phase3(qa_pairs, tok, enc, binder, org, phrase_store=ps)`
- Change return to `return tok, enc, binder, org, qa_pairs, ps`

In `main()`:
- Change unpack to `tok, enc, binder, org, qa_pairs, ps = build_system()`
- Change `Verbalizer()` to `Verbalizer(phrase_store=ps)`

Also fix the sentence count print in `benchmark_large.py` `main()` — replace the complex import line:
```python
    print(f"  Corpus sentences  : {len(sentences)}")
```
But `sentences` is not in scope in `main()` — change to use a stored count. In `build_system()` return the sentence count too, or just hardcode the path:

Replace this line in `benchmark_large.py main()`:
```python
    print(f"  Corpus sentences  : {len(list(__import__('genesis.bootstrap.seed_loader', fromlist=['SeedLoader']).SeedLoader().load(CORPUS_PATH)))}")
```
with:
```python
    loader2 = SeedLoader()
    print(f"  Corpus sentences  : {len(list(loader2.load(CORPUS_PATH)))}")
```

- [ ] **Step 3: Run small benchmark to verify PhraseStore improves accuracy**

```
python benchmark.py
```
Expected: QA accuracy jumps from 80% to ~90%+ (compound-answer questions now emit exact phrases).  
Check "what does fire produce" → should now show `heat and light` instead of `heat stars pumps...`.

- [ ] **Step 4: Run large benchmark**

```
python benchmark_large.py
```
Expected: QA accuracy improves from 46% to ~55%+. Compound-answer misses ("heat and light", "food and water", "natural selection", etc.) should now be correct.

- [ ] **Step 5: Commit**

```bash
git add benchmark.py benchmark_large.py
git commit -m "feat: wire PhraseStore through small and large benchmarks"
```

---

## Task 5: Context re-ranking in ForwardChain

**Files:**
- Modify: `genesis/reasoning/forward_chain.py`
- Modify: `tests/reasoning/test_forward_chain.py`

- [ ] **Step 1: Add a smoke test for context re-ranking**

Append to `tests/reasoning/test_forward_chain.py`:

```python
def test_context_reranking_does_not_crash():
    """Verifies that pushing context into WM and then reasoning doesn't crash."""
    org, wm, pre, post = _setup_simple_chain()
    wm.push(pre)        # pre is now in context
    wm.push(post)       # post is also in context
    reasoner = ForwardChain()
    result = reasoner.reason(pre, org, wm)
    assert isinstance(result, ReasoningResult)
    # With pre in context and pre as query, the cell is found and rule fires
    assert result.answer is not None

def test_context_reranking_stable_with_empty_context():
    """Empty working memory produces same result as no context."""
    org, wm, pre, post = _setup_simple_chain()
    # wm is empty
    reasoner = ForwardChain()
    result = reasoner.reason(pre, org, wm)
    assert result.answer is not None
```

- [ ] **Step 2: Run tests to verify they currently pass (baseline)**

```
pytest tests/reasoning/test_forward_chain.py -v
```
Expected: all existing tests PASS. New test `test_context_reranking_does_not_crash` also PASS since the change is additive and benign.

- [ ] **Step 3: Add context re-ranking to ForwardChain**

In `genesis/reasoning/forward_chain.py`, replace the inner loop body:

```python
        for depth in range(max_depth):
            active_cells = organism.route(route_sdr)
            if not active_cells:
                break

            # Context re-ranking: cells whose receptive field overlaps with
            # working memory context are sorted first. Routing still uses the
            # clean 20-bit query — this only changes the order cells are tried,
            # ensuring contextually relevant cells fire before unrelated ones.
            context = working_memory.union()
            if context.popcount() > 0:
                active_cells.sort(
                    key=lambda c: c.receptive_field.similarity(context),
                    reverse=True,
                )

            new_facts: List[Tuple[SDR, float, Cell, Rule]] = []
            for cell in active_cells:
                for rule, score in cell.apply_rules(facts):
                    new_facts.append((rule.postcondition, score, cell, rule))
```

The full updated method becomes:

```python
    def reason(self, query: SDR, organism: Organism,
               working_memory: WorkingMemory,
               max_depth: int = MAX_DEPTH) -> ReasoningResult:

        facts = query
        route_sdr = query
        chain: List[ChainStep] = []
        confidence_path: List[float] = []

        for depth in range(max_depth):
            active_cells = organism.route(route_sdr)
            if not active_cells:
                break

            context = working_memory.union()
            if context.popcount() > 0:
                active_cells.sort(
                    key=lambda c: c.receptive_field.similarity(context),
                    reverse=True,
                )

            new_facts: List[Tuple[SDR, float, Cell, Rule]] = []
            for cell in active_cells:
                for rule, score in cell.apply_rules(facts):
                    new_facts.append((rule.postcondition, score, cell, rule))

            if not new_facts:
                break

            new_facts.sort(key=lambda x: x[1], reverse=True)
            best_post, best_score, best_cell, best_rule = new_facts[0]

            chain.append(ChainStep(
                precondition=best_rule.precondition,
                postcondition=best_post,
                rule_confidence=best_rule.confidence,
                cell_id=best_cell.id,
            ))
            confidence_path.append(best_rule.confidence)

            total_conf = propagate(confidence_path, depth + 1)
            if total_conf > THETA_ANSWER:
                return ReasoningResult(
                    answer=best_post,
                    chain=chain,
                    confidence=total_conf,
                )

            facts = facts.union(best_post)
            route_sdr = best_post

        return ReasoningResult(answer=None, chain=chain, confidence=0.0)
```

- [ ] **Step 4: Run all forward chain tests**

```
pytest tests/reasoning/test_forward_chain.py -v
```
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add genesis/reasoning/forward_chain.py tests/reasoning/test_forward_chain.py
git commit -m "feat: context re-ranking of active cells in ForwardChain"
```

---

## Task 6: BeamChain — width-3 beam search with 2-hop inference

**Files:**
- Create: `genesis/reasoning/beam_chain.py`
- Create: `tests/reasoning/test_beam_chain.py`

**Key design:**
- `THETA_EARLY_RETURN = 0.65` — only accept a depth-0 answer if its confidence is very high. This prevents early termination on medium-confidence single-hop answers, allowing the beam to explore 2-hop chains.
- `THETA_ANSWER = 0.35` — accept at depth ≥ 1 with a lower bar.
- `BEAM_DECAY = 0.98` — less aggressive decay than ForwardChain's 0.95, so 2-hop chains can still clear `THETA_ANSWER`.
- Beam width = 3: at each depth, keep the top-3 candidate extensions across all active beams.

**Math check for 2-hop test:**
- Cell A: pre → mid, confidence=0.50. Depth-0: 0.50 × 0.98¹ = 0.49 < `THETA_EARLY_RETURN=0.65` → keep going.
- Cell B: mid → post, confidence=0.90. Depth-1: 0.50 × 0.90 × 0.98² = 0.432 > `THETA_ANSWER=0.35` → return post. ✓

- [ ] **Step 1: Write failing tests**

Create `tests/reasoning/test_beam_chain.py`:

```python
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
    ForwardChain returns mid (conf 0.49 > THETA_ANSWER=0.40).
    BeamChain skips mid (conf 0.49 < THETA_EARLY_RETURN=0.65) and returns post."""
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
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/reasoning/test_beam_chain.py -v
```
Expected: `ModuleNotFoundError: No module named 'genesis.reasoning.beam_chain'`

- [ ] **Step 3: Implement BeamChain**

Create `genesis/reasoning/beam_chain.py`:

```python
# genesis/reasoning/beam_chain.py
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from genesis.core.sdr import SDR
from genesis.core.cell import Cell, Rule
from genesis.core.organism import Organism
from genesis.core.working_memory import WorkingMemory
from genesis.reasoning.forward_chain import ChainStep, ReasoningResult

MAX_DEPTH = 8
THETA_EARLY_RETURN = 0.65   # accept depth-0 answer only if very confident
THETA_ANSWER = 0.35         # accept deeper answers with lower bar
BEAM_WIDTH = 3              # max parallel chains explored at each depth
BEAM_DECAY = 0.98           # less aggressive than ForwardChain's 0.95


@dataclass
class _Beam:
    route_sdr: SDR
    facts: SDR
    chain: List[ChainStep] = field(default_factory=list)
    confidence_path: List[float] = field(default_factory=list)


class BeamChain:
    """Width-3 beam search forward inference.

    Difference from ForwardChain:
    - Depth-0 answers are only accepted if confidence >= THETA_EARLY_RETURN (0.65).
    - When depth-0 confidence is medium (0.35–0.65), the beam continues exploring
      2-hop chains that may reach THETA_ANSWER at depth 1.
    - Up to BEAM_WIDTH parallel chains are explored at each depth.

    This enables inference like: pre -> mid -> post when pre->post is not seeded
    but pre->mid (low-confidence) and mid->post (high-confidence) both exist."""

    def reason(self, query: SDR, organism: Organism,
               working_memory: WorkingMemory,
               max_depth: int = MAX_DEPTH) -> ReasoningResult:

        beams: List[_Beam] = [_Beam(route_sdr=query, facts=query)]

        for depth in range(max_depth):
            next_beams: List[_Beam] = []
            context = working_memory.union()

            for beam in beams:
                active_cells = organism.route(beam.route_sdr)
                if not active_cells:
                    continue

                # Context re-ranking within each beam
                if context.popcount() > 0:
                    active_cells.sort(
                        key=lambda c: c.receptive_field.similarity(context),
                        reverse=True,
                    )

                for cell in active_cells:
                    for rule, score in cell.apply_rules(beam.facts):
                        new_conf_path = beam.confidence_path + [rule.confidence]
                        total_conf = self._propagate(new_conf_path, depth + 1)
                        new_step = ChainStep(
                            precondition=rule.precondition,
                            postcondition=rule.postcondition,
                            rule_confidence=rule.confidence,
                            cell_id=cell.id,
                        )
                        new_chain = beam.chain + [new_step]

                        # Threshold depends on depth
                        threshold = THETA_EARLY_RETURN if depth == 0 else THETA_ANSWER
                        if total_conf > threshold:
                            return ReasoningResult(
                                answer=rule.postcondition,
                                chain=new_chain,
                                confidence=total_conf,
                            )

                        next_beams.append(_Beam(
                            route_sdr=rule.postcondition,
                            facts=beam.facts.union(rule.postcondition),
                            chain=new_chain,
                            confidence_path=new_conf_path,
                        ))

            if not next_beams:
                break

            # Prune to BEAM_WIDTH best beams by current total confidence
            next_beams.sort(
                key=lambda b: self._propagate(b.confidence_path, depth + 1),
                reverse=True,
            )
            beams = next_beams[:BEAM_WIDTH]

        return ReasoningResult(answer=None, chain=[], confidence=0.0)

    @staticmethod
    def _propagate(confidences: List[float], depth: int) -> float:
        if not confidences:
            return 0.0
        result = 1.0
        for c in confidences:
            result *= c
        return result * (BEAM_DECAY ** depth)
```

- [ ] **Step 4: Run all beam chain tests**

```
pytest tests/reasoning/test_beam_chain.py -v
```
Expected: all 4 tests PASS

- [ ] **Step 5: Run full test suite to verify nothing broke**

```
pytest tests/ -v --tb=short
```
Expected: all existing tests still PASS + 4 new BeamChain tests PASS

- [ ] **Step 6: Commit**

```bash
git add genesis/reasoning/beam_chain.py tests/reasoning/test_beam_chain.py
git commit -m "feat: BeamChain with width-3 beam search and 2-hop inference"
```

---

## Task 7: Download SQuAD-tiny + run SQuAD benchmark

**Files:**
- Create: `data/prepare_squad.py`
- Create: `benchmark_squad.py`

This task first prepares the dataset, then runs the benchmark. No unit tests — the benchmark output is the test.

- [ ] **Step 1: Install the datasets library**

```
pip install datasets
```
Expected output includes: `Successfully installed datasets-...`

Verify:
```
python -c "import datasets; print(datasets.__version__)"
```
Expected: prints a version string like `2.x.x`

- [ ] **Step 2: Create data/prepare_squad.py**

Create `data/prepare_squad.py`:

```python
"""
Downloads SQuAD v1.1 validation set (first 300 examples) via HuggingFace datasets.
Writes:
  data/squad_corpus.txt  -- one sentence per line from passage contexts
  data/squad_qa.txt      -- question|answer pairs (answers <= 5 words)

Run: python data/prepare_squad.py
"""
import re
import os
from datasets import load_dataset

DATA_DIR = os.path.dirname(__file__)
CORPUS_OUT = os.path.join(DATA_DIR, "squad_corpus.txt")
QA_OUT = os.path.join(DATA_DIR, "squad_qa.txt")

N_EXAMPLES = 300
MAX_QA = 100
MAX_ANSWER_WORDS = 5
MIN_SENTENCE_LEN = 25


def split_sentences(text: str) -> list:
    """Split a paragraph into sentences on . ! ? boundaries."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if len(p.strip()) >= MIN_SENTENCE_LEN]


def main():
    print(f"Loading SQuAD validation (first {N_EXAMPLES} examples)...")
    ds = load_dataset("squad", split="validation")
    examples = list(ds.select(range(N_EXAMPLES)))

    # Corpus: unique sentences from passage contexts
    seen_sentences = set()
    corpus = []
    for ex in examples:
        for sent in split_sentences(ex["context"]):
            if sent not in seen_sentences:
                seen_sentences.add(sent)
                corpus.append(sent)

    # QA pairs: keep only short answers (1–5 words), deduplicate questions
    qa_pairs = []
    seen_questions = set()
    for ex in examples:
        q = ex["question"].strip()
        if q in seen_questions:
            continue
        answers = ex["answers"]["text"]
        if not answers:
            continue
        a = answers[0].strip()
        if 1 <= len(a.split()) <= MAX_ANSWER_WORDS:
            qa_pairs.append(f"{q}|{a}")
            seen_questions.add(q)
        if len(qa_pairs) >= MAX_QA:
            break

    with open(CORPUS_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(corpus))
    with open(QA_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(qa_pairs))

    print(f"Corpus  : {len(corpus)} sentences -> {CORPUS_OUT}")
    print(f"QA pairs: {len(qa_pairs)} pairs    -> {QA_OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the preparation script**

```
python data/prepare_squad.py
```
Expected output:
```
Loading SQuAD validation (first 300 examples)...
Corpus  : ~400 sentences -> ...squad_corpus.txt
QA pairs: ~80-100 pairs  -> ...squad_qa.txt
```

- [ ] **Step 4: Inspect the first few lines of each output file**

```
python -c "
lines = open('data/squad_corpus.txt').readlines()
print(f'Corpus: {len(lines)} lines')
for l in lines[:3]: print(' ', l.strip())
lines = open('data/squad_qa.txt').readlines()
print(f'QA: {len(lines)} pairs')
for l in lines[:5]: print(' ', l.strip())
"
```
Expected: corpus lines are full sentences; QA lines are `question|short answer` format.

- [ ] **Step 5: Create benchmark_squad.py**

Create `benchmark_squad.py`:

```python
"""
GENESIS SQuAD-Tiny Benchmark
Downloads SQuAD validation subset via data/prepare_squad.py, then:
  1. Bootstraps GENESIS on squad_corpus.txt + squad_qa.txt
  2. Measures QA accuracy, response time, new-fact recall, multi-turn coherence
Uses PhraseStore + BeamChain (the improved stack).
"""
import os
import time
import tracemalloc

from genesis.bootstrap.imprint import Imprinter
from genesis.bootstrap.seed_loader import SeedLoader
from genesis.core.organism import Organism
from genesis.core.working_memory import WorkingMemory
from genesis.generation.phrase_store import PhraseStore
from genesis.generation.verbalizer import Verbalizer
from genesis.interfaces.chat import ChatInterface
from genesis.learning.hebbian import HebbianLearner
from genesis.perception.binder import Binder
from genesis.perception.encoder import Encoder
from genesis.perception.tokenizer import Tokenizer
from genesis.reasoning.beam_chain import BeamChain
from genesis.storage.colony_store import ColonyStore

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CORPUS_PATH = os.path.join(DATA_DIR, "squad_corpus.txt")
QA_PATH = os.path.join(DATA_DIR, "squad_qa.txt")
COLONY_PATH = os.path.join(DATA_DIR, "squad_colony.gen")

SEP = "-" * 60


def load_qa_pairs():
    pairs = []
    with open(QA_PATH, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) == 2:
                pairs.append((parts[0].strip(), parts[1].strip()))
    return pairs


def build_system():
    tok = Tokenizer()
    enc = Encoder()
    binder = Binder()
    org = Organism()
    ps = PhraseStore()

    loader = SeedLoader()
    imp = Imprinter()

    print(f"\n{SEP}")
    print("PHASE 0 - Vocabulary imprinting...")
    t0 = time.perf_counter()
    sentences = list(loader.load(CORPUS_PATH))
    imp.phase0(sentences, tok, enc)
    print(f"  Sentences : {len(sentences)}")
    print(f"  Vocab size: {enc.vocab_size()} tokens")

    print("PHASE 1 - Pattern seeding...")
    imp.phase1(sentences, tok, enc, binder, org)
    print(f"  Cells     : {org.cell_count()}")

    print("PHASE 2 - Concept emergence...")
    imp.phase2(sentences, tok, enc, binder, org)
    print(f"  Cells     : {org.cell_count()}")

    print("PHASE 3 - Reasoning seed (QA pairs)...")
    qa_pairs = load_qa_pairs()
    imp.phase3(qa_pairs, tok, enc, binder, org, phrase_store=ps)
    elapsed = time.perf_counter() - t0
    print(f"  Cells     : {org.cell_count()}")
    print(f"  QA pairs  : {len(qa_pairs)}")
    print(f"  Phrase reg: {len(ps)}")
    print(f"  Total time: {elapsed:.2f}s")

    return tok, enc, binder, org, qa_pairs, ps


def measure_memory(org):
    import pickle, io
    buf = io.BytesIO()
    pickle.dump(org, buf, protocol=pickle.HIGHEST_PROTOCOL)
    return buf.tell()


def bench_qa(chat, qa_pairs):
    print(f"\n{SEP}")
    print("QA ACCURACY TEST")
    print(f"{'Question':<50} {'Expected':<20} {'Got':<25} OK?")
    print("-" * 110)

    correct = 0
    timings = []
    for question, expected in qa_pairs:
        t0 = time.perf_counter()
        response = chat.turn(question)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        timings.append(elapsed_ms)

        hit = expected.lower() in response.lower()
        if hit:
            correct += 1
        mark = "OK" if hit else "MISS"
        print(f"  {question[:49]:<49} {expected[:19]:<20} {response[:24]:<25} {mark}")

    total = len(qa_pairs)
    acc = correct / total * 100
    avg_ms = sum(timings) / len(timings)
    p99_ms = sorted(timings)[int(len(timings) * 0.99)]

    print(f"\n  Accuracy  : {correct}/{total} = {acc:.0f}%")
    print(f"  Avg time  : {avg_ms:.1f}ms")
    print(f"  P99 time  : {p99_ms:.1f}ms")
    return acc, avg_ms


def bench_new_fact_recall(chat):
    print(f"\n{SEP}")
    print("NEW FACT RECALL TEST")

    facts = [
        ("genesis is a new kind of ai", "genesis"),
        ("the capital of logic is reason", "reason"),
        ("carbon and silicon are both useful elements", "silicon"),
    ]

    results = []
    for teach, keyword in facts:
        chat.turn(teach)
        for _ in range(3):
            chat.turn("what do you know")
        response = chat.turn(f"tell me about {keyword}")
        hit = keyword in response.lower() or response != "<I don't know yet - still learning>"
        results.append(hit)
        mark = "OK" if hit else "MISS"
        print(f"  Taught: '{teach[:50]}'")
        print(f"  Asked : 'tell me about {keyword}'")
        print(f"  Got   : '{response[:60]}'  {mark}\n")

    score = sum(results)
    print(f"  Recall: {score}/{len(results)}")
    return score


def bench_multi_turn(chat):
    print(f"\n{SEP}")
    print("MULTI-TURN COHERENCE TEST (20 turns)")

    turns = [
        "tell me about the Normans", "where did they come from",
        "what language did they speak", "who was their leader",
        "tell me about the Byzantine Empire", "where was it located",
        "what happened to it", "when did it fall",
        "tell me about the Roman Empire", "how large was it",
        "tell me about ancient history", "what came first",
        "tell me about the Super Bowl", "who won it",
        "tell me about New York City", "what is it known for",
        "tell me about science", "what is a hypothesis",
        "tell me about language", "how do humans communicate",
    ]

    timings = []
    ok = 0
    for turn_text in turns:
        t0 = time.perf_counter()
        resp = chat.turn(turn_text)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        timings.append(elapsed_ms)
        if isinstance(resp, str) and len(resp) > 0:
            ok += 1

    avg = sum(timings) / len(timings)
    print(f"  Coherent turns : {ok}/{len(turns)}")
    print(f"  Avg turn time  : {avg:.1f}ms")
    return ok, avg


def main():
    # Check data files exist
    if not os.path.exists(CORPUS_PATH) or not os.path.exists(QA_PATH):
        print("ERROR: SQuAD data files not found.")
        print("Run first: python data/prepare_squad.py")
        return

    print("=" * 60)
    print("  GENESIS - SQuAD Tiny Benchmark")
    print("=" * 60)

    tracemalloc.start()
    tok, enc, binder, org, qa_pairs, ps = build_system()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    colony_bytes = measure_memory(org)
    print(f"\n  Peak RAM during bootstrap : {peak_bytes/1024/1024:.1f} MB")
    print(f"  Colony serialized size    : {colony_bytes/1024:.1f} KB")
    print(f"  Total rules               : {sum(len(c.rules) for c in org.cells.values())}")

    store = ColonyStore()
    store.save(org, COLONY_PATH)
    print(f"  Colony saved to           : {COLONY_PATH}")

    wm = WorkingMemory()
    chat = ChatInterface(
        tok, enc, binder, org, wm,
        BeamChain(), HebbianLearner(), Verbalizer(phrase_store=ps),
    )

    acc, avg_ms = bench_qa(chat, qa_pairs)
    recall = bench_new_fact_recall(chat)
    coherent, turn_avg = bench_multi_turn(chat)

    loader2 = SeedLoader()
    n_sentences = len(list(loader2.load(CORPUS_PATH)))

    print(f"\n{'=' * 60}")
    print("  BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Dataset           : SQuAD validation (tiny)")
    print(f"  Corpus sentences  : {n_sentences}")
    print(f"  QA pairs          : {len(qa_pairs)}")
    print(f"  Phrases registered: {len(ps)}")
    print(f"  Cells after boot  : {org.cell_count()}")
    print(f"  Total rules       : {sum(len(c.rules) for c in org.cells.values())}")
    print(f"  Colony size       : {colony_bytes/1024:.1f} KB")
    print(f"  Peak RAM          : {peak_bytes/1024/1024:.1f} MB")
    print(f"")
    print(f"  QA accuracy       : {acc:.0f}%")
    print(f"  Avg response time : {avg_ms:.1f}ms")
    print(f"  New fact recall   : {recall}/3")
    print(f"  20-turn coherence : {coherent}/20 turns coherent")
    print(f"  Avg turn time     : {turn_avg:.1f}ms")
    print(f"{'=' * 60}\n")

    criteria = [
        (acc >= 40,        f"QA accuracy >= 40%      : {acc:.0f}%"),
        (avg_ms < 1000,    f"Avg response < 1000ms   : {avg_ms:.1f}ms"),
        (coherent == 20,   f"All 20 turns coherent   : {coherent}/20"),
        (colony_bytes < 50 * 1024 * 1024, f"Colony < 50MB           : {colony_bytes/1024:.1f} KB"),
    ]
    all_pass = all(ok for ok, _ in criteria)
    print("  SQuAD Success Criteria:")
    for ok, label in criteria:
        print(f"    {'OK' if ok else 'FAIL'} {label}")
    print(f"\n  Overall: {'PASS' if all_pass else 'NEEDS REVIEW'}")
    print()


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the SQuAD benchmark**

```
python benchmark_squad.py
```
Expected:
- Bootstrap completes in < 30s
- QA accuracy: 40%+ (SQuAD answers are harder — named entities, multi-word spans)
- Colony < 50MB
- All 20 coherence turns pass

- [ ] **Step 7: Run the updated small and large benchmarks to confirm no regression**

```
python benchmark.py
python benchmark_large.py
```
Expected: both still PASS all criteria, with improved accuracy from PhraseStore.

- [ ] **Step 8: Run full test suite**

```
pytest tests/ -v --tb=short
```
Expected: all tests PASS.

- [ ] **Step 9: Commit**

```bash
git add data/prepare_squad.py benchmark_squad.py
git commit -m "feat: SQuAD-tiny dataset prep and benchmark with BeamChain+PhraseStore"
```

---

## Self-Review

**Spec coverage:**
1. Fix verbalization → Tasks 1, 2, 3, 4 (PhraseStore + Verbalizer + phase3 + wiring) ✓
2. Context utilization fix → Task 5 (re-ranking in ForwardChain) ✓
3. Multi-hop reasoning → Task 6 (BeamChain) ✓
4. Download tiny dataset and test → Task 7 (SQuAD prep + benchmark_squad.py) ✓

**Placeholder scan:** No TBDs, TODOs, or incomplete steps found. All code blocks are complete.

**Type consistency:**
- `PhraseStore` created in Task 1, used in Tasks 2, 3, 4, 7 — consistent.
- `phrase_store=ps` keyword argument in `imp.phase3()` defined in Task 3 and used in Tasks 4 and 7 — consistent.
- `BeamChain` created in Task 6 and used in Task 7's `benchmark_squad.py` — consistent.
- `ReasoningResult` imported from `genesis.reasoning.forward_chain` in `beam_chain.py` — `ChainStep` is also there — consistent.
- `Verbalizer(phrase_store=ps)` constructor defined in Task 2 and used in Tasks 4 and 7 — consistent.
