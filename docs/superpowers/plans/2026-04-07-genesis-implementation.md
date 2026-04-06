# GENESIS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build GENESIS — a sparse causal cell network that learns, reasons, and grows continuously on CPU with no GPU and under 300MB RAM.

**Architecture:** Knowledge is stored in Genesis Cells — small reasoning units holding causal rules represented as Sparse Distributed Representations (SDR). Cells self-organize into a layered organism that routes queries via LSH, learns via local Hebbian updates, and grows via cell division/merge/death. Reasoning is explicit forward/backward causal chain traversal, not token prediction.

**Tech Stack:** Python 3.10+, NumPy (bitwise SDR ops), pytest, standard library only beyond NumPy.

---

## File Map

```
genesis/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── sdr.py              # SDR: 1024-bit sparse vectors, all bitwise ops
│   ├── cell.py             # Rule + Cell: causal rules, activation, fitness
│   ├── working_memory.py   # 12-slot circular SDR buffer
│   └── organism.py         # Cell colony: LSH routing, cell registry
├── learning/
│   ├── __init__.py
│   ├── hebbian.py          # Strengthen, weaken, create rules
│   ├── lifecycle.py        # Cell division, merging, death
│   └── consolidation.py    # Episodic → semantic consolidation
├── perception/
│   ├── __init__.py
│   ├── tokenizer.py        # Text → token list + vocab
│   ├── encoder.py          # Token → SDR assignment + lookup
│   └── binder.py           # Token SDRs → sentence SDR via shift+compose
├── reasoning/
│   ├── __init__.py
│   ├── confidence.py       # Confidence propagation along chains
│   ├── forward_chain.py    # Forward chaining inference
│   └── backward_chain.py   # Backward chaining / verification
├── generation/
│   ├── __init__.py
│   └── verbalizer.py       # Conclusion SDR → output token sequence
├── bootstrap/
│   ├── __init__.py
│   ├── seed_loader.py      # Load + chunk seed corpus into sentences
│   └── imprint.py          # Phase 0-3 bootstrap pipeline
├── storage/
│   ├── __init__.py
│   └── colony_store.py     # Binary save/load of cell colony
├── interfaces/
│   ├── __init__.py
│   ├── chat.py             # Conversational mode
│   ├── agent.py            # Autonomous agent mode
│   └── embed.py            # Embedded stream mode
└── main.py                 # CLI entry point

tests/
├── core/
│   ├── test_sdr.py
│   ├── test_cell.py
│   ├── test_working_memory.py
│   └── test_organism.py
├── learning/
│   ├── test_hebbian.py
│   ├── test_lifecycle.py
│   └── test_consolidation.py
├── perception/
│   ├── test_tokenizer.py
│   ├── test_encoder.py
│   └── test_binder.py
├── reasoning/
│   ├── test_confidence.py
│   ├── test_forward_chain.py
│   └── test_backward_chain.py
├── generation/
│   └── test_verbalizer.py
├── bootstrap/
│   └── test_imprint.py
├── storage/
│   └── test_colony_store.py
└── test_integration.py
```

---

### Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `genesis/__init__.py`
- Create: `genesis/core/__init__.py`
- Create: `genesis/learning/__init__.py`
- Create: `genesis/perception/__init__.py`
- Create: `genesis/reasoning/__init__.py`
- Create: `genesis/generation/__init__.py`
- Create: `genesis/bootstrap/__init__.py`
- Create: `genesis/storage/__init__.py`
- Create: `genesis/interfaces/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "genesis"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["numpy>=1.24"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create all package __init__.py files**

Run:
```bash
mkdir -p genesis/core genesis/learning genesis/perception genesis/reasoning genesis/generation genesis/bootstrap genesis/storage genesis/interfaces
mkdir -p tests/core tests/learning tests/perception tests/reasoning tests/generation tests/bootstrap tests/storage
touch genesis/__init__.py genesis/core/__init__.py genesis/learning/__init__.py
touch genesis/perception/__init__.py genesis/reasoning/__init__.py genesis/generation/__init__.py
touch genesis/bootstrap/__init__.py genesis/storage/__init__.py genesis/interfaces/__init__.py
touch tests/__init__.py tests/core/__init__.py tests/learning/__init__.py
touch tests/perception/__init__.py tests/reasoning/__init__.py tests/generation/__init__.py
touch tests/bootstrap/__init__.py tests/storage/__init__.py tests/interfaces/__init__.py
```

- [ ] **Step 3: Install in dev mode**

Run:
```bash
pip install -e ".[dev]" 2>/dev/null || pip install -e . && pip install pytest numpy
```

Expected: No errors. `python -c "import genesis"` prints nothing.

- [ ] **Step 4: Commit**

```bash
git init
git add pyproject.toml genesis/ tests/
git commit -m "feat: project scaffold — package structure and pyproject.toml"
```

---

### Task 2: SDR — Sparse Distributed Representations

**Files:**
- Create: `genesis/core/sdr.py`
- Create: `tests/core/test_sdr.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/core/test_sdr.py
import pytest
from genesis.core.sdr import SDR, SDR_BITS, SDR_ACTIVE

def test_random_sdr_has_correct_popcount():
    sdr = SDR.random()
    assert sdr.popcount() == SDR_ACTIVE

def test_zeros_sdr_has_zero_popcount():
    sdr = SDR.zeros()
    assert sdr.popcount() == 0

def test_similarity_identical_is_one():
    sdr = SDR.random()
    assert sdr.similarity(sdr) == pytest.approx(1.0)

def test_similarity_disjoint_is_zero():
    a = SDR(list(range(0, 20)))
    b = SDR(list(range(20, 40)))
    assert a.similarity(b) == pytest.approx(0.0)

def test_similarity_partial_overlap():
    a = SDR(list(range(0, 20)))
    b = SDR(list(range(10, 30)))  # 10 shared
    # intersection=10, union=30 → Jaccard = 10/30 ≈ 0.333
    assert 0.30 < a.similarity(b) < 0.36

def test_compose_maintains_sparsity():
    a = SDR.random()
    b = SDR.random()
    c = a.compose(b)
    assert c.popcount() == SDR_ACTIVE

def test_shift_changes_pattern():
    a = SDR.random()
    b = a.shift(1)
    assert a != b

def test_shift_preserves_sparsity():
    a = SDR.random()
    b = a.shift(3)
    assert b.popcount() == SDR_ACTIVE

def test_order_encoding_differs():
    # "dog bites" vs "bites dog" should produce different SDRs
    dog = SDR(list(range(0, 20)))
    bites = SDR(list(range(100, 120)))
    phrase1 = dog.shift(0).compose(bites.shift(1))
    phrase2 = bites.shift(0).compose(dog.shift(1))
    assert phrase1 != phrase2

def test_active_indices_roundtrip():
    indices = [0, 63, 64, 127, 500, 1000, 1023, 7, 200, 300,
               400, 500, 600, 700, 800, 900, 100, 150, 250, 350]
    indices = sorted(set(indices))[:20]
    sdr = SDR(indices)
    recovered = sorted(sdr.active_indices())
    assert recovered == sorted(indices)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/core/test_sdr.py -v
```
Expected: `ModuleNotFoundError: No module named 'genesis.core.sdr'`

- [ ] **Step 3: Implement SDR**

```python
# genesis/core/sdr.py
import numpy as np
import random
from typing import Optional

SDR_BITS = 1024
SDR_WORDS = SDR_BITS // 64   # 16 uint64 words
SDR_ACTIVE = 20              # ~2% sparsity
SHIFT_STRIDE = 53            # prime, spreads positions without clustering


class SDR:
    """1024-bit sparse distributed representation with exactly 20 active bits."""

    __slots__ = ("words",)

    def __init__(self, active_indices: Optional[list] = None):
        self.words: np.ndarray = np.zeros(SDR_WORDS, dtype=np.uint64)
        if active_indices:
            for idx in active_indices:
                idx = int(idx) % SDR_BITS
                word, bit = divmod(idx, 64)
                self.words[word] |= np.uint64(1 << bit)

    @classmethod
    def random(cls) -> "SDR":
        return cls(random.sample(range(SDR_BITS), SDR_ACTIVE))

    @classmethod
    def zeros(cls) -> "SDR":
        return cls()

    def active_indices(self) -> list:
        result = []
        for w in range(SDR_WORDS):
            word = int(self.words[w])
            if word == 0:
                continue
            for b in range(64):
                if word & (1 << b):
                    result.append(w * 64 + b)
        return result

    def popcount(self) -> int:
        return sum(bin(int(w)).count("1") for w in self.words)

    def similarity(self, other: "SDR") -> float:
        """Jaccard similarity via bitwise AND/OR."""
        and_bits = np.bitwise_and(self.words, other.words)
        or_bits = np.bitwise_or(self.words, other.words)
        and_count = sum(bin(int(w)).count("1") for w in and_bits)
        or_count = sum(bin(int(w)).count("1") for w in or_bits)
        return and_count / or_count if or_count else 0.0

    def compose(self, other: "SDR") -> "SDR":
        """OR then keep SDR_ACTIVE bits (deterministic: lowest indices)."""
        or_words = np.bitwise_or(self.words, other.words)
        temp = SDR()
        temp.words = or_words
        all_active = temp.active_indices()
        if len(all_active) <= SDR_ACTIVE:
            return temp
        kept = sorted(all_active)[:SDR_ACTIVE]
        return SDR(kept)

    def shift(self, offset: int) -> "SDR":
        """Rotate each active index by offset*STRIDE to encode word position."""
        shifted = [(idx + offset * SHIFT_STRIDE) % SDR_BITS
                   for idx in self.active_indices()]
        return SDR(shifted)

    def union(self, other: "SDR") -> "SDR":
        """OR without capping — used to merge fact sets in reasoning."""
        result = SDR()
        result.words = np.bitwise_or(self.words, other.words)
        return result

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SDR):
            return False
        return bool(np.array_equal(self.words, other.words))

    def __hash__(self):
        return hash(self.words.tobytes())

    def __repr__(self) -> str:
        return f"SDR(active={self.popcount()}, indices={self.active_indices()[:5]}...)"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/core/test_sdr.py -v
```
Expected: All 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add genesis/core/sdr.py tests/core/test_sdr.py
git commit -m "feat: SDR — 1024-bit sparse distributed representations with similarity, compose, shift"
```

---

### Task 3: Rule and Genesis Cell

**Files:**
- Create: `genesis/core/cell.py`
- Create: `tests/core/test_cell.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/core/test_cell.py
import pytest
from genesis.core.sdr import SDR
from genesis.core.cell import Rule, Cell, THETA_FIRE, THETA_RULE

def test_rule_stores_precondition_and_postcondition():
    pre = SDR.random()
    post = SDR.random()
    rule = Rule(precondition=pre, postcondition=post, confidence=0.5)
    assert rule.precondition == pre
    assert rule.postcondition == post
    assert rule.confidence == 0.5

def test_cell_activates_when_overlap_above_threshold():
    cell = Cell()
    cell.receptive_field = SDR(list(range(0, 20)))
    # Input shares 10/20 bits with receptive field = Jaccard 10/30 ≈ 0.33
    # Below THETA_FIRE=0.35 → should NOT activate
    input_sdr = SDR(list(range(10, 30)))
    assert not cell.activates(input_sdr)

def test_cell_activates_when_identical():
    cell = Cell()
    rf = SDR.random()
    cell.receptive_field = rf
    assert cell.activates(rf)

def test_cell_does_not_activate_disjoint_input():
    cell = Cell()
    cell.receptive_field = SDR(list(range(0, 20)))
    input_sdr = SDR(list(range(100, 120)))
    assert not cell.activates(input_sdr)

def test_apply_rules_returns_matching_rules():
    cell = Cell()
    facts = SDR(list(range(0, 20)))
    pre = SDR(list(range(0, 20)))   # identical to facts
    post = SDR.random()
    cell.add_rule(pre, post, confidence=0.8)
    fired = cell.apply_rules(facts)
    assert len(fired) == 1
    rule, score = fired[0]
    assert rule.postcondition == post
    assert score > 0

def test_apply_rules_skips_low_confidence():
    cell = Cell()
    facts = SDR(list(range(0, 20)))
    pre = SDR(list(range(0, 20)))
    post = SDR.random()
    cell.add_rule(pre, post, confidence=0.1)  # below MIN_CONFIDENCE=0.25
    fired = cell.apply_rules(facts)
    assert len(fired) == 0

def test_apply_rules_sorted_by_score_descending():
    cell = Cell()
    facts = SDR(list(range(0, 20)))
    pre = SDR(list(range(0, 20)))
    cell.add_rule(pre, SDR.random(), confidence=0.9)
    cell.add_rule(pre, SDR.random(), confidence=0.5)
    fired = cell.apply_rules(facts)
    assert fired[0][1] >= fired[1][1]

def test_update_fitness_moves_toward_signal():
    cell = Cell()
    cell.fitness = 0.5
    cell.update_fitness(1.0)
    assert cell.fitness > 0.5
    cell2 = Cell()
    cell2.fitness = 0.5
    cell2.update_fitness(0.0)
    assert cell2.fitness < 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/core/test_cell.py -v
```
Expected: `ModuleNotFoundError: No module named 'genesis.core.cell'`

- [ ] **Step 3: Implement Rule and Cell**

```python
# genesis/core/cell.py
import uuid
from dataclasses import dataclass, field
from typing import List, Tuple
from .sdr import SDR

THETA_FIRE = 0.35       # cell activates when input overlaps RF above this
THETA_RULE = 0.40       # rule fires when facts overlap precondition above this
MIN_CONFIDENCE = 0.25   # rules below this confidence are ignored
MAX_RULES = 200         # cell divides when it exceeds this
MIN_AGE = 100           # cell must survive at least this many activations before dying
THETA_DEATH = 0.10      # cell dies if fitness drops below this
FITNESS_ALPHA = 0.1     # EMA factor for fitness update


@dataclass
class Rule:
    precondition: SDR
    postcondition: SDR
    confidence: float = 0.3


@dataclass
class Cell:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    receptive_field: SDR = field(default_factory=SDR.random)
    rules: List[Rule] = field(default_factory=list)
    fitness: float = 0.5
    age: int = 0

    def activates(self, input_sdr: SDR) -> bool:
        return input_sdr.similarity(self.receptive_field) > THETA_FIRE

    def apply_rules(self, facts: SDR) -> List[Tuple[Rule, float]]:
        """Return (rule, score) for rules whose precondition matches facts."""
        fired = []
        for rule in self.rules:
            overlap = facts.similarity(rule.precondition)
            if overlap > THETA_RULE and rule.confidence > MIN_CONFIDENCE:
                score = rule.confidence * overlap
                fired.append((rule, score))
        return sorted(fired, key=lambda x: x[1], reverse=True)

    def add_rule(self, precondition: SDR, postcondition: SDR,
                 confidence: float = 0.3) -> Rule:
        rule = Rule(precondition=precondition, postcondition=postcondition,
                    confidence=confidence)
        self.rules.append(rule)
        return rule

    def update_fitness(self, signal: float):
        """Exponential moving average update toward signal."""
        self.fitness = (1 - FITNESS_ALPHA) * self.fitness + FITNESS_ALPHA * signal
        self.age += 1
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/core/test_cell.py -v
```
Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add genesis/core/cell.py tests/core/test_cell.py
git commit -m "feat: Rule + Cell — causal rule units with activation, rule firing, fitness tracking"
```

---

### Task 4: Working Memory

**Files:**
- Create: `genesis/core/working_memory.py`
- Create: `tests/core/test_working_memory.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/core/test_working_memory.py
import pytest
from genesis.core.sdr import SDR, SDR_ACTIVE
from genesis.core.working_memory import WorkingMemory

def test_empty_memory_has_zero_length():
    wm = WorkingMemory()
    assert len(wm) == 0

def test_push_increases_length():
    wm = WorkingMemory()
    wm.push(SDR.random())
    assert len(wm) == 1

def test_capacity_not_exceeded():
    wm = WorkingMemory(capacity=3)
    for _ in range(10):
        wm.push(SDR.random())
    assert len(wm) == 3

def test_union_of_empty_is_zeros():
    wm = WorkingMemory()
    result = wm.union()
    assert result.popcount() == 0

def test_union_of_single_item():
    wm = WorkingMemory()
    sdr = SDR.random()
    wm.push(sdr)
    result = wm.union()
    assert result.similarity(sdr) == pytest.approx(1.0)

def test_union_contains_bits_from_both():
    wm = WorkingMemory()
    a = SDR(list(range(0, 20)))
    b = SDR(list(range(100, 120)))
    wm.push(a)
    wm.push(b)
    result = wm.union()
    # result should overlap with both a and b
    assert result.similarity(a) > 0
    assert result.similarity(b) > 0

def test_clear_empties_memory():
    wm = WorkingMemory()
    wm.push(SDR.random())
    wm.clear()
    assert len(wm) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/core/test_working_memory.py -v
```
Expected: `ModuleNotFoundError: No module named 'genesis.core.working_memory'`

- [ ] **Step 3: Implement WorkingMemory**

```python
# genesis/core/working_memory.py
from collections import deque
from .sdr import SDR

CAPACITY = 12


class WorkingMemory:
    """12-slot circular buffer of active concept SDRs."""

    def __init__(self, capacity: int = CAPACITY):
        self.capacity = capacity
        self._slots: deque = deque(maxlen=capacity)

    def push(self, sdr: SDR):
        self._slots.append(sdr)

    def union(self) -> SDR:
        """Compose all active SDRs into a single context pattern."""
        slots = list(self._slots)
        if not slots:
            return SDR.zeros()
        result = slots[0]
        for sdr in slots[1:]:
            result = result.compose(sdr)
        return result

    def clear(self):
        self._slots.clear()

    def __len__(self) -> int:
        return len(self._slots)

    def __iter__(self):
        return iter(self._slots)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/core/test_working_memory.py -v
```
Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add genesis/core/working_memory.py tests/core/test_working_memory.py
git commit -m "feat: WorkingMemory — 12-slot circular SDR buffer for conversation context"
```

---

### Task 5: Organism — Cell Colony with LSH Routing

**Files:**
- Create: `genesis/core/organism.py`
- Create: `tests/core/test_organism.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/core/test_organism.py
from genesis.core.sdr import SDR
from genesis.core.cell import Cell
from genesis.core.organism import Organism

def test_add_cell_increases_count():
    org = Organism()
    org.add_cell(Cell())
    assert org.cell_count() == 1

def test_remove_cell_decreases_count():
    org = Organism()
    cell = Cell()
    org.add_cell(cell)
    org.remove_cell(cell.id)
    assert org.cell_count() == 0

def test_route_returns_activating_cells():
    org = Organism()
    rf = SDR.random()
    cell = Cell()
    cell.receptive_field = rf
    org.add_cell(cell)
    # query identical to receptive field must activate cell
    result = org.route(rf)
    assert any(c.id == cell.id for c in result)

def test_route_does_not_return_non_activating_cells():
    org = Organism()
    cell = Cell()
    cell.receptive_field = SDR(list(range(0, 20)))
    org.add_cell(cell)
    # completely disjoint query
    result = org.route(SDR(list(range(500, 520))))
    assert not any(c.id == cell.id for c in result)

def test_route_handles_empty_organism():
    org = Organism()
    result = org.route(SDR.random())
    assert result == []

def test_cell_accessible_by_id():
    org = Organism()
    cell = Cell()
    org.add_cell(cell)
    assert cell.id in org.cells
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/core/test_organism.py -v
```
Expected: `ModuleNotFoundError: No module named 'genesis.core.organism'`

- [ ] **Step 3: Implement Organism**

```python
# genesis/core/organism.py
import random
from typing import Dict, List
from .cell import Cell
from .sdr import SDR, SDR_BITS, SDR_ACTIVE

NUM_HASH_BITS = 10   # 2^10 = 1024 buckets


class Organism:
    """Cell colony with LSH-based routing for O(1) relevant-cell lookup."""

    def __init__(self):
        self.cells: Dict[str, Cell] = {}
        self._buckets: Dict[tuple, List[str]] = {}
        self._projections: List[List[int]] = self._make_projections()

    def _make_projections(self) -> List[List[int]]:
        rng = random.Random(42)   # fixed seed → deterministic routing
        return [
            [rng.choice((-1, 1)) for _ in range(SDR_BITS)]
            for _ in range(NUM_HASH_BITS)
        ]

    def _lsh_hash(self, sdr: SDR) -> tuple:
        active = set(sdr.active_indices())
        return tuple(
            1 if sum(proj[i] for i in active) >= 0 else 0
            for proj in self._projections
        )

    def add_cell(self, cell: Cell):
        self.cells[cell.id] = cell
        bucket = self._lsh_hash(cell.receptive_field)
        self._buckets.setdefault(bucket, []).append(cell.id)

    def remove_cell(self, cell_id: str):
        cell = self.cells.pop(cell_id, None)
        if cell is None:
            return
        bucket = self._lsh_hash(cell.receptive_field)
        bucket_list = self._buckets.get(bucket, [])
        self._buckets[bucket] = [cid for cid in bucket_list if cid != cell_id]

    def route(self, sdr: SDR) -> List[Cell]:
        """Return cells whose receptive fields overlap sdr above THETA_FIRE."""
        bucket = self._lsh_hash(sdr)
        candidate_ids: set = set(self._buckets.get(bucket, []))
        # Check 1-bit-flip neighbors for recall robustness
        bucket_list = list(bucket)
        for i in range(NUM_HASH_BITS):
            flipped = bucket_list[:]
            flipped[i] ^= 1
            candidate_ids.update(self._buckets.get(tuple(flipped), []))
        return [
            self.cells[cid]
            for cid in candidate_ids
            if cid in self.cells and self.cells[cid].activates(sdr)
        ]

    def cell_count(self) -> int:
        return len(self.cells)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/core/test_organism.py -v
```
Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add genesis/core/organism.py tests/core/test_organism.py
git commit -m "feat: Organism — LSH-routed cell colony, O(1) relevant-cell lookup"
```

---

### Task 6: Tokenizer

**Files:**
- Create: `genesis/perception/tokenizer.py`
- Create: `tests/perception/test_tokenizer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/perception/test_tokenizer.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/perception/test_tokenizer.py -v
```
Expected: `ModuleNotFoundError: No module named 'genesis.perception.tokenizer'`

- [ ] **Step 3: Implement Tokenizer**

```python
# genesis/perception/tokenizer.py
import re
from typing import Dict, List

SPECIAL_TOKENS = ["<pad>", "<unk>", "<start>", "<end>"]


class Tokenizer:
    """Simple whitespace + punctuation tokenizer with vocab tracking."""

    def __init__(self):
        self.vocab: Dict[str, int] = {}
        for tok in SPECIAL_TOKENS:
            self._add(tok)

    def _add(self, token: str) -> int:
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab)
        return self.vocab[token]

    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s'\-]", " ", text)
        tokens = text.split()
        result = []
        for tok in tokens:
            if tok:
                self._add(tok)
                result.append(tok)
        return result

    def encode(self, text: str) -> List[int]:
        return [self.vocab.get(tok, self.vocab["<unk>"])
                for tok in self.tokenize(text)]

    def vocab_size(self) -> int:
        return len(self.vocab)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/perception/test_tokenizer.py -v
```
Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add genesis/perception/tokenizer.py tests/perception/test_tokenizer.py
git commit -m "feat: Tokenizer — lowercase + punctuation-stripped tokenizer with vocab"
```

---

### Task 7: Encoder and Binder

**Files:**
- Create: `genesis/perception/encoder.py`
- Create: `genesis/perception/binder.py`
- Create: `tests/perception/test_encoder.py`
- Create: `tests/perception/test_binder.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/perception/test_encoder.py
from genesis.core.sdr import SDR, SDR_ACTIVE
from genesis.perception.tokenizer import Tokenizer
from genesis.perception.encoder import Encoder

def test_encode_token_returns_sdr():
    enc = Encoder()
    enc.register("cat")
    sdr = enc.encode_token("cat")
    assert isinstance(sdr, SDR)

def test_encode_token_has_correct_sparsity():
    enc = Encoder()
    enc.register("dog")
    sdr = enc.encode_token("dog")
    assert sdr.popcount() == SDR_ACTIVE

def test_same_token_returns_same_sdr():
    enc = Encoder()
    enc.register("fish")
    a = enc.encode_token("fish")
    b = enc.encode_token("fish")
    assert a == b

def test_different_tokens_return_different_sdrs():
    enc = Encoder()
    enc.register("cat")
    enc.register("dog")
    a = enc.encode_token("cat")
    b = enc.encode_token("dog")
    assert a != b

def test_unknown_token_returns_unk_sdr():
    enc = Encoder()
    sdr = enc.encode_token("zzzunknownzzz")
    assert isinstance(sdr, SDR)
    assert sdr.popcount() == SDR_ACTIVE

def test_register_from_vocab():
    tok = Tokenizer()
    tok.tokenize("the quick brown fox")
    enc = Encoder()
    enc.register_vocab(tok.vocab)
    sdr = enc.encode_token("quick")
    assert isinstance(sdr, SDR)
```

```python
# tests/perception/test_binder.py
from genesis.core.sdr import SDR, SDR_ACTIVE
from genesis.perception.binder import Binder

def test_bind_single_token_returns_sdr():
    b = Binder()
    sdr = SDR.random()
    result = b.bind([sdr])
    assert isinstance(result, SDR)
    assert result.popcount() == SDR_ACTIVE

def test_bind_empty_returns_zeros():
    b = Binder()
    result = b.bind([])
    assert result.popcount() == 0

def test_bind_order_matters():
    b = Binder()
    a = SDR(list(range(0, 20)))
    c = SDR(list(range(100, 120)))
    phrase1 = b.bind([a, c])
    phrase2 = b.bind([c, a])
    assert phrase1 != phrase2

def test_bind_produces_stable_output():
    b = Binder()
    sdrs = [SDR.random() for _ in range(4)]
    r1 = b.bind(sdrs)
    r2 = b.bind(sdrs)
    assert r1 == r2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/perception/test_encoder.py tests/perception/test_binder.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement Encoder and Binder**

```python
# genesis/perception/encoder.py
import random
from typing import Dict
from genesis.core.sdr import SDR, SDR_BITS, SDR_ACTIVE

_RNG_SEED_SALT = 0xDEADBEEF


class Encoder:
    """Maps tokens to random but stable SDRs. SDRs co-evolve via Hebbian learning."""

    def __init__(self):
        self._vocab_sdrs: Dict[str, SDR] = {}
        self._unk_sdr: SDR = self._make_sdr("<unk>")

    def _make_sdr(self, token: str) -> SDR:
        rng = random.Random(hash(token) ^ _RNG_SEED_SALT)
        indices = rng.sample(range(SDR_BITS), SDR_ACTIVE)
        return SDR(indices)

    def register(self, token: str):
        if token not in self._vocab_sdrs:
            self._vocab_sdrs[token] = self._make_sdr(token)

    def register_vocab(self, vocab: Dict[str, int]):
        for token in vocab:
            self.register(token)

    def encode_token(self, token: str) -> SDR:
        return self._vocab_sdrs.get(token, self._unk_sdr)

    def vocab_size(self) -> int:
        return len(self._vocab_sdrs)

    def decode_sdr(self, sdr: SDR, top_k: int = 5) -> list:
        """Return top_k tokens whose SDR is most similar to given SDR."""
        scored = [
            (tok, sdr.similarity(tok_sdr))
            for tok, tok_sdr in self._vocab_sdrs.items()
        ]
        return [tok for tok, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]]
```

```python
# genesis/perception/binder.py
from typing import List
from genesis.core.sdr import SDR


class Binder:
    """Combines a sequence of token SDRs into a single sentence SDR using
    positional shifts — preserving word order information."""

    def bind(self, sdrs: List[SDR]) -> SDR:
        if not sdrs:
            return SDR.zeros()
        result = sdrs[0].shift(0)
        for i, sdr in enumerate(sdrs[1:], start=1):
            result = result.compose(sdr.shift(i))
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/perception/test_encoder.py tests/perception/test_binder.py -v
```
Expected: All 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add genesis/perception/encoder.py genesis/perception/binder.py
git add tests/perception/test_encoder.py tests/perception/test_binder.py
git commit -m "feat: Encoder + Binder — token→SDR assignment and order-preserving sentence composition"
```

---

### Task 8: Hebbian Learning

**Files:**
- Create: `genesis/learning/hebbian.py`
- Create: `tests/learning/test_hebbian.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/learning/test_hebbian.py
import pytest
from genesis.core.sdr import SDR
from genesis.core.cell import Cell, Rule
from genesis.learning.hebbian import HebbianLearner

def test_strengthen_increases_confidence():
    learner = HebbianLearner()
    rule = Rule(SDR.random(), SDR.random(), confidence=0.5)
    learner.strengthen(rule)
    assert rule.confidence > 0.5

def test_strengthen_never_exceeds_one():
    learner = HebbianLearner()
    rule = Rule(SDR.random(), SDR.random(), confidence=0.99)
    for _ in range(100):
        learner.strengthen(rule)
    assert rule.confidence <= 1.0

def test_weaken_decreases_confidence():
    learner = HebbianLearner()
    rule = Rule(SDR.random(), SDR.random(), confidence=0.5)
    learner.weaken(rule)
    assert rule.confidence < 0.5

def test_weaken_never_goes_below_zero():
    learner = HebbianLearner()
    rule = Rule(SDR.random(), SDR.random(), confidence=0.01)
    for _ in range(100):
        learner.weaken(rule)
    assert rule.confidence >= 0.0

def test_create_rule_adds_to_cell():
    learner = HebbianLearner()
    cell = Cell()
    pre = SDR.random()
    post = SDR.random()
    learner.create_rule(cell, pre, post)
    assert len(cell.rules) == 1
    assert cell.rules[0].precondition == pre
    assert cell.rules[0].postcondition == post

def test_create_rule_starts_at_initial_confidence():
    learner = HebbianLearner()
    cell = Cell()
    learner.create_rule(cell, SDR.random(), SDR.random())
    assert cell.rules[0].confidence == pytest.approx(0.3)

def test_update_processes_correct_prediction():
    learner = HebbianLearner()
    cell = Cell()
    pre = SDR(list(range(0, 20)))
    post = SDR(list(range(100, 120)))
    rule = cell.add_rule(pre, post, confidence=0.5)
    facts = SDR(list(range(0, 20)))
    observed = SDR(list(range(100, 120)))  # matches postcondition
    learner.update(cell, facts, observed)
    assert rule.confidence > 0.5

def test_update_processes_wrong_prediction():
    learner = HebbianLearner()
    cell = Cell()
    pre = SDR(list(range(0, 20)))
    post = SDR(list(range(100, 120)))
    rule = cell.add_rule(pre, post, confidence=0.5)
    facts = SDR(list(range(0, 20)))
    observed = SDR(list(range(500, 520)))  # doesn't match post
    learner.update(cell, facts, observed)
    assert rule.confidence < 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/learning/test_hebbian.py -v
```
Expected: `ModuleNotFoundError: No module named 'genesis.learning.hebbian'`

- [ ] **Step 3: Implement HebbianLearner**

```python
# genesis/learning/hebbian.py
from genesis.core.sdr import SDR
from genesis.core.cell import Cell, Rule, THETA_RULE

ETA = 0.05              # learning rate
INITIAL_CONFIDENCE = 0.3
THETA_CREATE = 0.30     # create new rule when max rule overlap below this


class HebbianLearner:
    """Local Hebbian updates — no global gradient, no weight matrix."""

    def strengthen(self, rule: Rule):
        """Fire confirmed: push confidence toward 1."""
        rule.confidence = rule.confidence + ETA * (1.0 - rule.confidence)
        rule.confidence = min(rule.confidence, 1.0)

    def weaken(self, rule: Rule):
        """Fire disconfirmed: push confidence toward 0."""
        rule.confidence = rule.confidence * (1.0 - ETA)
        rule.confidence = max(rule.confidence, 0.0)

    def create_rule(self, cell: Cell, precondition: SDR, postcondition: SDR):
        """Create new rule in cell with initial low confidence."""
        cell.add_rule(precondition, postcondition, INITIAL_CONFIDENCE)

    def update(self, cell: Cell, facts: SDR, observed: SDR):
        """Given what the cell saw (facts) and what actually happened (observed),
        strengthen matching rules and weaken mismatching ones.
        Create a new rule if no rule fired at all."""
        fired_any = False
        for rule in cell.rules:
            overlap = facts.similarity(rule.precondition)
            if overlap > THETA_RULE:
                fired_any = True
                match = observed.similarity(rule.postcondition)
                if match > 0.4:
                    self.strengthen(rule)
                    cell.update_fitness(rule.confidence)
                else:
                    self.weaken(rule)
                    cell.update_fitness(rule.confidence * 0.5)
        if not fired_any:
            max_overlap = max(
                (facts.similarity(r.precondition) for r in cell.rules),
                default=0.0
            )
            if max_overlap < THETA_CREATE:
                self.create_rule(cell, facts, observed)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/learning/test_hebbian.py -v
```
Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add genesis/learning/hebbian.py tests/learning/test_hebbian.py
git commit -m "feat: HebbianLearner — local strengthen/weaken/create rule updates, no backprop"
```

---

### Task 9: Cell Lifecycle — Division, Merging, Death

**Files:**
- Create: `genesis/learning/lifecycle.py`
- Create: `tests/learning/test_lifecycle.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/learning/test_lifecycle.py
import pytest
from genesis.core.sdr import SDR
from genesis.core.cell import Cell, MAX_RULES, MIN_AGE, THETA_DEATH
from genesis.core.organism import Organism
from genesis.learning.lifecycle import LifecycleManager

def _make_loaded_cell(num_rules: int) -> Cell:
    cell = Cell()
    for i in range(num_rules):
        cell.add_rule(SDR.random(), SDR.random(), 0.5)
    return cell

def test_should_divide_when_too_many_rules():
    mgr = LifecycleManager()
    cell = _make_loaded_cell(MAX_RULES + 1)
    assert mgr.should_divide(cell)

def test_should_not_divide_when_few_rules():
    mgr = LifecycleManager()
    cell = _make_loaded_cell(5)
    assert not mgr.should_divide(cell)

def test_divide_removes_parent_and_adds_two_daughters():
    mgr = LifecycleManager()
    org = Organism()
    cell = _make_loaded_cell(MAX_RULES + 1)
    org.add_cell(cell)
    mgr.divide(cell, org)
    assert cell.id not in org.cells
    assert org.cell_count() == 2

def test_divide_daughters_have_fewer_rules_than_parent():
    mgr = LifecycleManager()
    org = Organism()
    cell = _make_loaded_cell(MAX_RULES + 1)
    parent_rule_count = len(cell.rules)
    org.add_cell(cell)
    mgr.divide(cell, org)
    for daughter in org.cells.values():
        assert len(daughter.rules) < parent_rule_count

def test_should_merge_similar_cells():
    mgr = LifecycleManager()
    rf = SDR(list(range(0, 20)))
    a = Cell(); a.receptive_field = rf
    b = Cell(); b.receptive_field = rf  # identical RF
    assert mgr.should_merge(a, b)

def test_should_not_merge_dissimilar_cells():
    mgr = LifecycleManager()
    a = Cell(); a.receptive_field = SDR(list(range(0, 20)))
    b = Cell(); b.receptive_field = SDR(list(range(500, 520)))
    assert not mgr.should_merge(a, b)

def test_merge_removes_both_parents_adds_one_child():
    mgr = LifecycleManager()
    org = Organism()
    rf = SDR(list(range(0, 20)))
    a = Cell(); a.receptive_field = rf
    b = Cell(); b.receptive_field = rf
    org.add_cell(a); org.add_cell(b)
    mgr.merge(a, b, org)
    assert a.id not in org.cells
    assert b.id not in org.cells
    assert org.cell_count() == 1

def test_should_die_when_low_fitness_and_old():
    mgr = LifecycleManager()
    cell = Cell()
    cell.fitness = THETA_DEATH - 0.01
    cell.age = MIN_AGE + 1
    assert mgr.should_die(cell)

def test_should_not_die_when_young():
    mgr = LifecycleManager()
    cell = Cell()
    cell.fitness = 0.0
    cell.age = MIN_AGE - 1
    assert not mgr.should_die(cell)

def test_retire_removes_cell_from_organism():
    mgr = LifecycleManager()
    org = Organism()
    cell = Cell()
    org.add_cell(cell)
    mgr.retire(cell, org)
    assert cell.id not in org.cells
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/learning/test_lifecycle.py -v
```
Expected: `ModuleNotFoundError: No module named 'genesis.learning.lifecycle'`

- [ ] **Step 3: Implement LifecycleManager**

```python
# genesis/learning/lifecycle.py
import math
from typing import List
from genesis.core.sdr import SDR
from genesis.core.cell import Cell, MAX_RULES, MIN_AGE, THETA_DEATH
from genesis.core.organism import Organism

THETA_MERGE = 0.70     # merge cells whose RFs are this similar
THETA_SPLIT = 2.0      # Shannon entropy threshold for forced division


def _shannon_entropy(values: List[float]) -> float:
    total = sum(values)
    if total == 0:
        return 0.0
    probs = [v / total for v in values if v > 0]
    return -sum(p * math.log2(p) for p in probs)


def _rule_diversity(cell: Cell) -> float:
    """Entropy of pairwise precondition similarities — high = diverse rules."""
    rules = cell.rules
    if len(rules) < 2:
        return 0.0
    sims = []
    for i in range(len(rules)):
        for j in range(i + 1, len(rules)):
            sims.append(rules[i].precondition.similarity(rules[j].precondition))
    return _shannon_entropy(sims)


class LifecycleManager:

    def should_divide(self, cell: Cell) -> bool:
        return (len(cell.rules) > MAX_RULES or
                _rule_diversity(cell) > THETA_SPLIT)

    def divide(self, cell: Cell, organism: Organism):
        """Split cell into two daughters by clustering rules."""
        rules = cell.rules
        mid = len(rules) // 2
        # Simple split: first half vs second half (k-means-lite)
        cluster_a = rules[:mid]
        cluster_b = rules[mid:]

        daughter_a = Cell()
        daughter_a.receptive_field = (
            cluster_a[len(cluster_a) // 2].precondition if cluster_a
            else SDR.random()
        )
        daughter_a.rules = cluster_a

        daughter_b = Cell()
        daughter_b.receptive_field = (
            cluster_b[len(cluster_b) // 2].precondition if cluster_b
            else SDR.random()
        )
        daughter_b.rules = cluster_b

        organism.remove_cell(cell.id)
        organism.add_cell(daughter_a)
        organism.add_cell(daughter_b)

    def should_merge(self, a: Cell, b: Cell) -> bool:
        return a.receptive_field.similarity(b.receptive_field) > THETA_MERGE

    def merge(self, a: Cell, b: Cell, organism: Organism):
        """Fuse two cells into one merged cell."""
        merged = Cell()
        merged.receptive_field = a.receptive_field.compose(b.receptive_field)
        # Combine rules, removing near-duplicates
        seen: list = []
        for rule in a.rules + b.rules:
            if not any(rule.precondition.similarity(s.precondition) > 0.9
                       and rule.postcondition.similarity(s.postcondition) > 0.9
                       for s in seen):
                seen.append(rule)
        merged.rules = seen
        merged.fitness = (a.fitness + b.fitness) / 2

        organism.remove_cell(a.id)
        organism.remove_cell(b.id)
        organism.add_cell(merged)

    def should_die(self, cell: Cell) -> bool:
        return cell.fitness < THETA_DEATH and cell.age > MIN_AGE

    def retire(self, cell: Cell, organism: Organism):
        """Remove a dead cell; its rules are lost (low fitness = low value)."""
        organism.remove_cell(cell.id)

    def run_maintenance(self, organism: Organism):
        """One pass of division, merging, and death across all cells."""
        cells = list(organism.cells.values())
        for cell in cells:
            if cell.id not in organism.cells:
                continue
            if self.should_divide(cell):
                self.divide(cell, organism)
            elif self.should_die(cell):
                self.retire(cell, organism)

        cells = list(organism.cells.values())
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                a, b = cells[i], cells[j]
                if a.id in organism.cells and b.id in organism.cells:
                    if self.should_merge(a, b):
                        self.merge(a, b, organism)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/learning/test_lifecycle.py -v
```
Expected: All 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add genesis/learning/lifecycle.py tests/learning/test_lifecycle.py
git commit -m "feat: LifecycleManager — cell division, merging, death; organism self-maintenance"
```

---

### Task 10: Confidence Propagation + Forward Chain Reasoning

**Files:**
- Create: `genesis/reasoning/confidence.py`
- Create: `genesis/reasoning/forward_chain.py`
- Create: `tests/reasoning/test_confidence.py`
- Create: `tests/reasoning/test_forward_chain.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/reasoning/test_confidence.py
import pytest
from genesis.reasoning.confidence import propagate, DECAY

def test_single_hop_returns_confidence():
    assert propagate([0.8], depth=1) == pytest.approx(0.8 * DECAY)

def test_two_hops_multiplies_and_decays():
    result = propagate([0.8, 0.6], depth=2)
    expected = 0.8 * 0.6 * (DECAY ** 2)
    assert result == pytest.approx(expected)

def test_empty_chain_returns_zero():
    assert propagate([], depth=0) == 0.0

def test_deeper_chain_has_lower_confidence():
    shallow = propagate([0.9, 0.9], depth=2)
    deep = propagate([0.9, 0.9, 0.9, 0.9], depth=4)
    assert deep < shallow
```

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/reasoning/test_confidence.py tests/reasoning/test_forward_chain.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement confidence and forward chain**

```python
# genesis/reasoning/confidence.py
from typing import List

DECAY = 0.95   # per-hop confidence decay


def propagate(confidences: List[float], depth: int) -> float:
    """Multiply rule confidences and apply depth decay."""
    if not confidences:
        return 0.0
    result = 1.0
    for c in confidences:
        result *= c
    return result * (DECAY ** depth)
```

```python
# genesis/reasoning/forward_chain.py
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from genesis.core.sdr import SDR
from genesis.core.cell import Cell, Rule
from genesis.core.organism import Organism
from genesis.core.working_memory import WorkingMemory
from .confidence import propagate, DECAY

MAX_DEPTH = 8
THETA_ANSWER = 0.40
THETA_CONFIDENCE = 0.25


@dataclass
class ChainStep:
    precondition: SDR
    postcondition: SDR
    rule_confidence: float
    cell_id: str


@dataclass
class ReasoningResult:
    answer: Optional[SDR] = None
    chain: List[ChainStep] = field(default_factory=list)
    confidence: float = 0.0


class ForwardChain:
    """Forward causal chain inference through the cell colony."""

    def reason(self, query: SDR, organism: Organism,
               working_memory: WorkingMemory,
               max_depth: int = MAX_DEPTH) -> ReasoningResult:

        context = working_memory.union()
        facts = query.union(context)
        chain: List[ChainStep] = []
        confidence_path: List[float] = []

        for depth in range(max_depth):
            active_cells = organism.route(facts)
            if not active_cells:
                break

            new_facts: List[Tuple[SDR, float, Cell, Rule]] = []
            for cell in active_cells:
                for rule, score in cell.apply_rules(facts):
                    new_facts.append((rule.postcondition, score, cell, rule))

            if not new_facts:
                break

            # Pick highest-scoring new fact
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

        return ReasoningResult(answer=None, chain=chain, confidence=0.0)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/reasoning/test_confidence.py tests/reasoning/test_forward_chain.py -v
```
Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add genesis/reasoning/confidence.py genesis/reasoning/forward_chain.py
git add tests/reasoning/test_confidence.py tests/reasoning/test_forward_chain.py
git commit -m "feat: ForwardChain + confidence propagation — sparse causal chain inference"
```

---

### Task 11: Backward Chain Reasoning

**Files:**
- Create: `genesis/reasoning/backward_chain.py`
- Create: `tests/reasoning/test_backward_chain.py`

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/reasoning/test_backward_chain.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement BackwardChain**

```python
# genesis/reasoning/backward_chain.py
from dataclasses import dataclass, field
from typing import List, Optional
from genesis.core.sdr import SDR
from genesis.core.organism import Organism
from genesis.reasoning.confidence import propagate

MAX_DEPTH = 6
THETA_VERIFY = 0.35


@dataclass
class VerificationResult:
    verified: bool = False
    confidence: float = 0.0
    support_chain: List[tuple] = field(default_factory=list)


class BackwardChain:
    """Backward chaining: given a goal, find supporting facts in the colony."""

    def verify(self, goal: SDR, organism: Organism,
               known_facts: SDR, depth: int = 0) -> VerificationResult:

        if depth > MAX_DEPTH:
            return VerificationResult(verified=False)

        # If goal is already a known fact, it's trivially true
        if goal.similarity(known_facts) > THETA_VERIFY:
            return VerificationResult(
                verified=True,
                confidence=goal.similarity(known_facts),
                support_chain=[("known_fact", goal, 1.0)],
            )

        # Find rules whose postcondition matches the goal
        supporting: List[tuple] = []
        for cell in organism.cells.values():
            for rule in cell.rules:
                if rule.postcondition.similarity(goal) > THETA_VERIFY:
                    # Recursively verify precondition
                    sub = self.verify(rule.precondition, organism,
                                      known_facts, depth + 1)
                    if sub.verified:
                        conf = propagate(
                            [rule.confidence, sub.confidence], depth + 1
                        )
                        supporting.append((rule, sub, conf))

        if not supporting:
            return VerificationResult(verified=False)

        supporting.sort(key=lambda x: x[2], reverse=True)
        best_rule, best_sub, best_conf = supporting[0]
        chain = [(best_rule, goal, best_conf)] + best_sub.support_chain

        return VerificationResult(
            verified=True,
            confidence=best_conf,
            support_chain=chain,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/reasoning/test_backward_chain.py -v
```
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add genesis/reasoning/backward_chain.py tests/reasoning/test_backward_chain.py
git commit -m "feat: BackwardChain — goal-directed verification via recursive precondition support"
```

---

### Task 12: Verbalizer — SDR to Text

**Files:**
- Create: `genesis/generation/verbalizer.py`
- Create: `tests/generation/test_verbalizer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/generation/test_verbalizer.py
from genesis.core.sdr import SDR
from genesis.perception.encoder import Encoder
from genesis.generation.verbalizer import Verbalizer

def _setup():
    enc = Encoder()
    for word in ["the", "cat", "sat", "on", "mat", "dog", "ran", "fast"]:
        enc.register(word)
    return enc

def test_verbalize_returns_string():
    enc = _setup()
    v = Verbalizer()
    sdr = enc.encode_token("cat")
    result = v.verbalize(sdr, enc, max_tokens=3)
    assert isinstance(result, str)

def test_verbalize_returns_nonempty_for_known_sdr():
    enc = _setup()
    v = Verbalizer()
    sdr = enc.encode_token("dog")
    result = v.verbalize(sdr, enc, max_tokens=3)
    assert len(result.strip()) > 0

def test_verbalize_top_match_is_registered_token():
    enc = _setup()
    v = Verbalizer()
    sdr = enc.encode_token("cat")
    result = v.verbalize(sdr, enc, max_tokens=1)
    assert "cat" in result

def test_verbalize_empty_sdr_returns_unknown():
    enc = _setup()
    v = Verbalizer()
    result = v.verbalize(SDR.zeros(), enc, max_tokens=3)
    assert result == "<unknown>"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/generation/test_verbalizer.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement Verbalizer**

```python
# genesis/generation/verbalizer.py
from genesis.core.sdr import SDR
from genesis.perception.encoder import Encoder

THETA_VERBALIZE = 0.05   # minimum similarity to include a token candidate


class Verbalizer:
    """Converts a conclusion SDR back into a token sequence.
    Generation is concept-first: find tokens that best cover the SDR bits."""

    def verbalize(self, sdr: SDR, encoder: Encoder,
                  max_tokens: int = 10) -> str:
        if sdr.popcount() == 0:
            return "<unknown>"

        # Score all registered tokens by SDR similarity
        scored = [
            (token, sdr.similarity(tok_sdr))
            for token, tok_sdr in encoder._vocab_sdrs.items()
            if token not in ("<pad>", "<unk>", "<start>", "<end>")
        ]
        scored = [(tok, s) for tok, s in scored if s > THETA_VERBALIZE]
        scored.sort(key=lambda x: x[1], reverse=True)

        if not scored:
            return "<unknown>"

        # Greedily pick tokens that cover the most uncovered SDR bits
        covered = SDR.zeros()
        selected = []
        for token, score in scored[:50]:   # search top-50 candidates
            tok_sdr = encoder.encode_token(token)
            new_coverage = tok_sdr.similarity(sdr) - covered.similarity(tok_sdr)
            if new_coverage > 0:
                selected.append(token)
                covered = covered.compose(tok_sdr)
                if len(selected) >= max_tokens:
                    break

        return " ".join(selected) if selected else "<unknown>"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/generation/test_verbalizer.py -v
```
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add genesis/generation/verbalizer.py tests/generation/test_verbalizer.py
git commit -m "feat: Verbalizer — concept-first SDR→text: greedy coverage, not next-token prediction"
```

---

### Task 13: Memory Consolidation

**Files:**
- Create: `genesis/learning/consolidation.py`
- Create: `tests/learning/test_consolidation.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/learning/test_consolidation.py
from genesis.core.sdr import SDR
from genesis.core.cell import Cell
from genesis.core.organism import Organism
from genesis.learning.consolidation import Consolidator, Episode

def _make_episode(confidence: float, access_count: int) -> Episode:
    return Episode(
        query=SDR.random(),
        answer=SDR.random(),
        chain=[],
        confidence=confidence,
        access_count=access_count,
    )

def test_episode_creation():
    ep = _make_episode(0.8, 5)
    assert ep.confidence == 0.8
    assert ep.access_count == 5

def test_consolidate_adds_rules_for_high_value_episodes():
    cons = Consolidator()
    org = Organism()
    # One high-value cell to receive rules
    cell = Cell()
    cell.receptive_field = SDR(list(range(0, 20)))
    org.add_cell(cell)

    ep = Episode(
        query=SDR(list(range(0, 20))),
        answer=SDR(list(range(200, 220))),
        chain=[],
        confidence=0.75,
        access_count=4,
    )
    before = len(cell.rules)
    cons.consolidate([ep], org)
    # A Reasoning Cell should have been created with the rule
    total_rules = sum(len(c.rules) for c in org.cells.values())
    assert total_rules > before

def test_consolidate_ignores_low_confidence_episodes():
    cons = Consolidator()
    org = Organism()
    initial_cells = org.cell_count()
    ep = _make_episode(confidence=0.3, access_count=10)
    cons.consolidate([ep], org)
    assert org.cell_count() == initial_cells

def test_consolidate_ignores_low_access_episodes():
    cons = Consolidator()
    org = Organism()
    initial_cells = org.cell_count()
    ep = _make_episode(confidence=0.9, access_count=1)
    cons.consolidate([ep], org)
    assert org.cell_count() == initial_cells
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/learning/test_consolidation.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement Consolidator**

```python
# genesis/learning/consolidation.py
from dataclasses import dataclass, field
from typing import List
from genesis.core.sdr import SDR
from genesis.core.cell import Cell
from genesis.core.organism import Organism

CONF_THRESHOLD = 0.60
ACCESS_THRESHOLD = 3


@dataclass
class Episode:
    query: SDR
    answer: SDR
    chain: list
    confidence: float
    access_count: int = 0


class Consolidator:
    """Converts high-value episodic memories into permanent cell rules.
    Mirrors biological sleep consolidation."""

    def consolidate(self, episodes: List[Episode], organism: Organism):
        """For each high-value episode, embed its (query→answer) as a permanent rule."""
        for episode in episodes:
            if (episode.confidence >= CONF_THRESHOLD and
                    episode.access_count >= ACCESS_THRESHOLD):
                self._embed(episode, organism)

    def _embed(self, episode: Episode, organism: Organism):
        # Find best existing cell to host this rule, or create new one
        best_cell = None
        best_sim = 0.0
        for cell in organism.cells.values():
            sim = episode.query.similarity(cell.receptive_field)
            if sim > best_sim:
                best_sim = sim
                best_cell = cell

        if best_cell is None or best_sim < 0.20:
            # Create a dedicated Reasoning Cell for this episode
            new_cell = Cell()
            new_cell.receptive_field = episode.query
            new_cell.add_rule(episode.query, episode.answer,
                              confidence=episode.confidence)
            organism.add_cell(new_cell)
        else:
            best_cell.add_rule(episode.query, episode.answer,
                               confidence=episode.confidence)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/learning/test_consolidation.py -v
```
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add genesis/learning/consolidation.py tests/learning/test_consolidation.py
git commit -m "feat: Consolidator — episodic→semantic memory consolidation, permanent rule embedding"
```

---

### Task 14: Bootstrap — Seed Loader and Imprinter

**Files:**
- Create: `genesis/bootstrap/seed_loader.py`
- Create: `genesis/bootstrap/imprint.py`
- Create: `tests/bootstrap/test_imprint.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/bootstrap/test_imprint.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/bootstrap/test_imprint.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement SeedLoader and Imprinter**

```python
# genesis/bootstrap/seed_loader.py
import re
from typing import Iterator

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


class SeedLoader:
    """Loads a text corpus and yields cleaned sentences."""

    def load(self, path: str) -> Iterator[str]:
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            for sent in SENT_SPLIT.split(line):
                sent = sent.strip()
                if sent:
                    yield sent
```

```python
# genesis/bootstrap/imprint.py
from typing import List, Tuple
from genesis.core.cell import Cell
from genesis.core.organism import Organism
from genesis.core.sdr import SDR
from genesis.perception.tokenizer import Tokenizer
from genesis.perception.encoder import Encoder
from genesis.perception.binder import Binder

WINDOW = 3   # sliding window size for phrase extraction


class Imprinter:
    """Runs the 4-phase bootstrap: vocabulary → patterns → concepts → reasoning."""

    def phase0(self, sentences: List[str], tokenizer: Tokenizer,
               encoder: Encoder):
        """Assign random SDRs to every token seen in the corpus."""
        for sent in sentences:
            tokens = tokenizer.tokenize(sent)
            encoder.register_vocab(tokenizer.vocab)

    def phase1(self, sentences: List[str], tokenizer: Tokenizer,
               encoder: Encoder, binder: Binder, organism: Organism):
        """Create Pattern Cells: for each sliding window, learn phrase→next_phrase."""
        for sent in sentences:
            tokens = tokenizer.tokenize(sent)
            if len(tokens) < 2:
                continue
            sdrs = [encoder.encode_token(t) for t in tokens]
            for i in range(len(sdrs) - WINDOW):
                window_sdrs = sdrs[i:i + WINDOW]
                phrase_sdr = binder.bind(window_sdrs)
                next_sdrs = sdrs[i + 1:i + WINDOW + 1]
                next_sdr = binder.bind(next_sdrs)

                # Find or create a cell for this phrase
                candidates = organism.route(phrase_sdr)
                if candidates:
                    cell = candidates[0]
                else:
                    cell = Cell()
                    cell.receptive_field = phrase_sdr
                    organism.add_cell(cell)

                cell.add_rule(phrase_sdr, next_sdr, confidence=0.7)

    def phase2(self, sentences: List[str], tokenizer: Tokenizer,
               encoder: Encoder, binder: Binder, organism: Organism):
        """Create Concept Cells: entity + context → relation."""
        for sent in sentences:
            tokens = tokenizer.tokenize(sent)
            if len(tokens) < 3:
                continue
            sdrs = [encoder.encode_token(t) for t in tokens]
            sent_sdr = binder.bind(sdrs)
            # Each token in context of full sentence = concept cell
            for i, (tok, sdr) in enumerate(zip(tokens, sdrs)):
                candidates = organism.route(sdr)
                if not candidates:
                    cell = Cell()
                    cell.receptive_field = sdr
                    organism.add_cell(cell)
                    cell.add_rule(sdr, sent_sdr, confidence=0.5)

    def phase3(self, qa_pairs: List[Tuple[str, str]], tokenizer: Tokenizer,
               encoder: Encoder, binder: Binder, organism: Organism):
        """Create Reasoning Cells from structured QA pairs."""
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/bootstrap/test_imprint.py -v
```
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add genesis/bootstrap/seed_loader.py genesis/bootstrap/imprint.py
git add tests/bootstrap/test_imprint.py
git commit -m "feat: Bootstrap — 4-phase imprinting: vocab, patterns, concepts, reasoning seed"
```

---

### Task 15: Storage — Binary Colony Save/Load

**Files:**
- Create: `genesis/storage/colony_store.py`
- Create: `tests/storage/test_colony_store.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/storage/test_colony_store.py
import os
import tempfile
from genesis.core.sdr import SDR
from genesis.core.cell import Cell
from genesis.core.organism import Organism
from genesis.storage.colony_store import ColonyStore

def _make_organism() -> Organism:
    org = Organism()
    cell = Cell()
    cell.receptive_field = SDR(list(range(0, 20)))
    cell.add_rule(SDR(list(range(0, 20))), SDR(list(range(100, 120))), 0.8)
    org.add_cell(cell)
    return org

def test_save_creates_file():
    org = _make_organism()
    store = ColonyStore()
    with tempfile.NamedTemporaryFile(suffix=".gen", delete=False) as f:
        path = f.name
    store.save(org, path)
    assert os.path.exists(path)
    os.unlink(path)

def test_load_restores_cell_count():
    org = _make_organism()
    store = ColonyStore()
    with tempfile.NamedTemporaryFile(suffix=".gen", delete=False) as f:
        path = f.name
    store.save(org, path)
    loaded = store.load(path)
    os.unlink(path)
    assert loaded.cell_count() == org.cell_count()

def test_load_restores_rules():
    org = _make_organism()
    store = ColonyStore()
    with tempfile.NamedTemporaryFile(suffix=".gen", delete=False) as f:
        path = f.name
    store.save(org, path)
    loaded = store.load(path)
    os.unlink(path)
    original_cell = list(org.cells.values())[0]
    loaded_cell = list(loaded.cells.values())[0]
    assert len(loaded_cell.rules) == len(original_cell.rules)

def test_load_restores_rule_confidence():
    org = _make_organism()
    store = ColonyStore()
    with tempfile.NamedTemporaryFile(suffix=".gen", delete=False) as f:
        path = f.name
    store.save(org, path)
    loaded = store.load(path)
    os.unlink(path)
    orig_conf = list(org.cells.values())[0].rules[0].confidence
    load_conf = list(loaded.cells.values())[0].rules[0].confidence
    assert abs(orig_conf - load_conf) < 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/storage/test_colony_store.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement ColonyStore**

```python
# genesis/storage/colony_store.py
import pickle
import gzip
from genesis.core.organism import Organism


class ColonyStore:
    """Saves and loads the cell colony using compressed pickle.
    Format: gzip-compressed pickle of the Organism."""

    def save(self, organism: Organism, path: str):
        with gzip.open(path, "wb") as f:
            pickle.dump(organism, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str) -> Organism:
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/storage/test_colony_store.py -v
```
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add genesis/storage/colony_store.py tests/storage/test_colony_store.py
git commit -m "feat: ColonyStore — gzip+pickle binary save/load for cell colony"
```

---

### Task 16: Chat Interface

**Files:**
- Create: `genesis/interfaces/chat.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/interfaces/test_chat.py  (create this file)
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/interfaces/test_chat.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement ChatInterface**

```python
# genesis/interfaces/chat.py
from genesis.core.organism import Organism
from genesis.core.working_memory import WorkingMemory
from genesis.learning.hebbian import HebbianLearner
from genesis.perception.binder import Binder
from genesis.perception.encoder import Encoder
from genesis.perception.tokenizer import Tokenizer
from genesis.reasoning.forward_chain import ForwardChain
from genesis.generation.verbalizer import Verbalizer


class ChatInterface:
    """Conversational mode: text in → encode → reason → verbalize → learn."""

    def __init__(self, tokenizer: Tokenizer, encoder: Encoder,
                 binder: Binder, organism: Organism,
                 working_memory: WorkingMemory, reasoner: ForwardChain,
                 learner: HebbianLearner, verbalizer: Verbalizer):
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.binder = binder
        self.organism = organism
        self.working_memory = working_memory
        self.reasoner = reasoner
        self.learner = learner
        self.verbalizer = verbalizer
        self.turn_count = 0

    def turn(self, user_input: str) -> str:
        # 1. Encode input
        tokens = self.tokenizer.tokenize(user_input)
        self.encoder.register_vocab(self.tokenizer.vocab)
        sdrs = [self.encoder.encode_token(t) for t in tokens]
        if not sdrs:
            return "<empty input>"
        query_sdr = self.binder.bind(sdrs)

        # 2. Push to working memory
        self.working_memory.push(query_sdr)

        # 3. Reason
        result = self.reasoner.reason(query_sdr, self.organism,
                                      self.working_memory)

        # 4. Learn from this interaction
        active_cells = self.organism.route(query_sdr)
        for cell in active_cells:
            self.learner.update(cell, query_sdr,
                                result.answer if result.answer else query_sdr)
            cell.update_fitness(result.confidence if result.answer else 0.1)

        # 5. Push answer to working memory
        if result.answer:
            self.working_memory.push(result.answer)

        # 6. Generate response
        if result.answer and result.confidence > 0.30:
            response = self.verbalizer.verbalize(result.answer, self.encoder,
                                                 max_tokens=12)
        else:
            response = "<I don't know yet — still learning>"

        self.turn_count += 1
        return response
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/interfaces/test_chat.py -v
```
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add genesis/interfaces/chat.py tests/interfaces/test_chat.py
git commit -m "feat: ChatInterface — encode→reason→learn→verbalize conversational loop"
```

---

### Task 17: Agent and Embed Interfaces

**Files:**
- Create: `genesis/interfaces/agent.py`
- Create: `genesis/interfaces/embed.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/interfaces/test_agent.py  (create this file)
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
```

```python
# tests/interfaces/test_embed.py  (create this file)
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/interfaces/test_agent.py tests/interfaces/test_embed.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement AgentInterface and EmbedInterface**

```python
# genesis/interfaces/agent.py
from typing import Optional
from genesis.core.organism import Organism
from genesis.core.sdr import SDR
from genesis.core.working_memory import WorkingMemory
from genesis.learning.hebbian import HebbianLearner
from genesis.perception.binder import Binder
from genesis.perception.encoder import Encoder
from genesis.perception.tokenizer import Tokenizer
from genesis.reasoning.backward_chain import BackwardChain
from genesis.reasoning.forward_chain import ForwardChain
from genesis.generation.verbalizer import Verbalizer


class AgentInterface:
    """Autonomous agent mode: goal-directed reasoning with self-monitoring."""

    def __init__(self, tokenizer: Tokenizer, encoder: Encoder, binder: Binder,
                 organism: Organism, working_memory: WorkingMemory,
                 forward: ForwardChain, backward: BackwardChain,
                 learner: HebbianLearner, verbalizer: Verbalizer):
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.binder = binder
        self.organism = organism
        self.working_memory = working_memory
        self.forward = forward
        self.backward = backward
        self.learner = learner
        self.verbalizer = verbalizer
        self.goal_sdr: Optional[SDR] = None

    def set_goal(self, goal_text: str):
        tokens = self.tokenizer.tokenize(goal_text)
        self.encoder.register_vocab(self.tokenizer.vocab)
        sdrs = [self.encoder.encode_token(t) for t in tokens]
        self.goal_sdr = self.binder.bind(sdrs)

    def step(self) -> str:
        if self.goal_sdr is None:
            return "<no goal set>"
        # Try to verify/plan toward goal
        known = self.working_memory.union()
        verification = self.backward.verify(self.goal_sdr, self.organism, known)
        if verification.verified:
            return f"<goal achieved: {self.verbalizer.verbalize(self.goal_sdr, self.encoder, 6)}>"
        # Forward chain to find next action toward goal
        result = self.forward.reason(self.goal_sdr, self.organism,
                                     self.working_memory)
        if result.answer:
            self.working_memory.push(result.answer)
            return self.verbalizer.verbalize(result.answer, self.encoder, 8)
        return "<reasoning — no path found yet>"
```

```python
# genesis/interfaces/embed.py
from typing import List
from genesis.core.cell import Cell
from genesis.core.organism import Organism
from genesis.learning.hebbian import HebbianLearner
from genesis.perception.binder import Binder
from genesis.perception.encoder import Encoder
from genesis.perception.tokenizer import Tokenizer


class EmbedInterface:
    """Embedded mode: continuously learn from a stream of text events."""

    def __init__(self, tokenizer: Tokenizer, encoder: Encoder,
                 binder: Binder, organism: Organism, learner: HebbianLearner):
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.binder = binder
        self.organism = organism
        self.learner = learner

    def process(self, events: List[str]) -> List[str]:
        outputs = []
        for event in events:
            tokens = self.tokenizer.tokenize(event)
            self.encoder.register_vocab(self.tokenizer.vocab)
            sdrs = [self.encoder.encode_token(t) for t in tokens]
            if not sdrs:
                continue
            event_sdr = self.binder.bind(sdrs)
            active = self.organism.route(event_sdr)
            if active:
                for cell in active:
                    self.learner.update(cell, event_sdr, event_sdr)
                    outputs.append(f"processed by cell {cell.id[:8]}")
            else:
                new_cell = Cell()
                new_cell.receptive_field = event_sdr
                new_cell.add_rule(event_sdr, event_sdr, confidence=0.3)
                self.organism.add_cell(new_cell)
                outputs.append(f"new cell created")
        return outputs
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/interfaces/test_agent.py tests/interfaces/test_embed.py -v
```
Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add genesis/interfaces/agent.py genesis/interfaces/embed.py
git add tests/interfaces/test_agent.py tests/interfaces/test_embed.py
git commit -m "feat: AgentInterface + EmbedInterface — goal-directed and stream processing modes"
```

---

### Task 18: main.py — CLI Entry Point

**Files:**
- Create: `genesis/main.py`

- [ ] **Step 1: Implement main.py**

No tests — this is a thin CLI glue layer that wires together all components.

```python
# genesis/main.py
"""
GENESIS CLI — Generative Evolving Network of Emergent Sparse Intelligence Systems

Commands:
  genesis bootstrap <corpus_path> [--output colony.gen] [--qa qa_pairs.txt]
  genesis chat [--colony colony.gen]
  genesis agent <goal> [--colony colony.gen]
"""
import argparse
import os
import sys

from genesis.bootstrap.imprint import Imprinter
from genesis.bootstrap.seed_loader import SeedLoader
from genesis.core.organism import Organism
from genesis.core.working_memory import WorkingMemory
from genesis.generation.verbalizer import Verbalizer
from genesis.interfaces.agent import AgentInterface
from genesis.interfaces.chat import ChatInterface
from genesis.learning.hebbian import HebbianLearner
from genesis.learning.lifecycle import LifecycleManager
from genesis.perception.binder import Binder
from genesis.perception.encoder import Encoder
from genesis.perception.tokenizer import Tokenizer
from genesis.reasoning.backward_chain import BackwardChain
from genesis.reasoning.forward_chain import ForwardChain
from genesis.storage.colony_store import ColonyStore


def _build_components(colony_path: str = None):
    tok = Tokenizer()
    enc = Encoder()
    binder = Binder()
    org = Organism()
    wm = WorkingMemory()
    store = ColonyStore()

    if colony_path and os.path.exists(colony_path):
        print(f"Loading colony from {colony_path}...")
        org = store.load(colony_path)
        print(f"Loaded {org.cell_count()} cells.")

    return tok, enc, binder, org, wm, store


def cmd_bootstrap(args):
    tok, enc, binder, org, wm, store = _build_components()
    loader = SeedLoader()
    imp = Imprinter()

    print("Phase 0: Vocabulary imprinting...")
    sentences = list(loader.load(args.corpus))
    imp.phase0(sentences, tok, enc)
    print(f"  Vocab size: {enc.vocab_size()} tokens")

    print("Phase 1: Pattern seeding...")
    imp.phase1(sentences, tok, enc, binder, org)
    print(f"  Cells: {org.cell_count()}")

    print("Phase 2: Concept emergence...")
    imp.phase2(sentences, tok, enc, binder, org)
    print(f"  Cells: {org.cell_count()}")

    if args.qa and os.path.exists(args.qa):
        print("Phase 3: Reasoning seed...")
        qa_pairs = []
        with open(args.qa) as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) == 2:
                    qa_pairs.append((parts[0].strip(), parts[1].strip()))
        imp.phase3(qa_pairs, tok, enc, binder, org)
        print(f"  Cells: {org.cell_count()}")

    output = args.output or "colony.gen"
    store.save(org, output)
    print(f"\nBootstrap complete. Colony saved to {output}")
    print(f"Total cells: {org.cell_count()}")


def cmd_chat(args):
    tok, enc, binder, org, wm, store = _build_components(args.colony)
    reasoner = ForwardChain()
    learner = HebbianLearner()
    verbalizer = Verbalizer()
    lifecycle = LifecycleManager()

    chat = ChatInterface(tok, enc, binder, org, wm, reasoner, learner, verbalizer)
    colony_path = args.colony or "colony.gen"

    print("GENESIS — Conversational Mode")
    print("Type 'quit' to exit, 'save' to save colony.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "save":
            store.save(org, colony_path)
            print(f"Colony saved ({org.cell_count()} cells).")
            continue

        response = chat.turn(user_input)
        print(f"GENESIS: {response}")

        # Periodic maintenance every 50 turns
        if chat.turn_count % 50 == 0:
            lifecycle.run_maintenance(org)
            print(f"[maintenance: {org.cell_count()} cells]")

    store.save(org, colony_path)
    print(f"\nSession ended. Colony saved to {colony_path}")


def cmd_agent(args):
    tok, enc, binder, org, wm, store = _build_components(args.colony)
    agent = AgentInterface(
        tokenizer=tok, encoder=enc, binder=binder,
        organism=org, working_memory=wm,
        forward=ForwardChain(), backward=BackwardChain(),
        learner=HebbianLearner(), verbalizer=Verbalizer(),
    )
    agent.set_goal(args.goal)
    print(f"GENESIS Agent — Goal: {args.goal}\n")

    for step in range(args.steps):
        action = agent.step()
        print(f"Step {step + 1}: {action}")
        if "goal achieved" in action:
            break


def main():
    parser = argparse.ArgumentParser(prog="genesis",
                                     description="GENESIS AI System")
    sub = parser.add_subparsers(dest="command")

    p_boot = sub.add_parser("bootstrap", help="Bootstrap from corpus")
    p_boot.add_argument("corpus", help="Path to seed corpus text file")
    p_boot.add_argument("--output", default="colony.gen")
    p_boot.add_argument("--qa", default=None,
                        help="Path to QA pairs file (q|a per line)")

    p_chat = sub.add_parser("chat", help="Conversational mode")
    p_chat.add_argument("--colony", default=None,
                        help="Path to colony file to load")

    p_agent = sub.add_parser("agent", help="Autonomous agent mode")
    p_agent.add_argument("goal", help="Goal text for the agent")
    p_agent.add_argument("--colony", default=None)
    p_agent.add_argument("--steps", type=int, default=10)

    args = parser.parse_args()
    if args.command == "bootstrap":
        cmd_bootstrap(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "agent":
        cmd_agent(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI is importable and shows help**

```bash
python -m genesis.main --help
```
Expected:
```
usage: genesis [-h] {bootstrap,chat,agent} ...
```

- [ ] **Step 3: Commit**

```bash
git add genesis/main.py
git commit -m "feat: main.py — CLI entry point for bootstrap, chat, and agent modes"
```

---

### Task 19: Integration Test — End-to-End

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
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
```

- [ ] **Step 2: Run integration test**

```bash
pytest tests/test_integration.py -v
```
Expected: All 5 tests PASS.

- [ ] **Step 3: Run full test suite**

```bash
pytest tests/ -v --tb=short
```
Expected: All tests PASS. Note total count and any failures.

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: integration — bootstrap→chat→memory→save/load end-to-end verification"
```

---

## Phase 1 Success Verification

After all tasks complete, verify all 7 success criteria:

```bash
# 1. All tests pass
pytest tests/ -v

# 2. Memory footprint check (run after bootstrapping on a real corpus)
python -c "
import tracemalloc, os
tracemalloc.start()
from genesis.bootstrap.seed_loader import SeedLoader
from genesis.bootstrap.imprint import Imprinter
from genesis.core.organism import Organism
from genesis.perception.tokenizer import Tokenizer
from genesis.perception.encoder import Encoder
from genesis.perception.binder import Binder
tok=Tokenizer(); enc=Encoder(); binder=Binder(); org=Organism()
# (point to real corpus)
current, peak = tracemalloc.get_traced_memory()
print(f'Peak memory: {peak/1024/1024:.1f} MB')
"

# 3. Response time check
python -c "
import time
from genesis.core.sdr import SDR
from genesis.core.organism import Organism
from genesis.core.working_memory import WorkingMemory
from genesis.reasoning.forward_chain import ForwardChain
org=Organism(); wm=WorkingMemory(); reasoner=ForwardChain()
start=time.perf_counter()
for _ in range(100):
    reasoner.reason(SDR.random(), org, wm)
elapsed=(time.perf_counter()-start)/100*1000
print(f'Avg response time: {elapsed:.2f}ms')
"
```
