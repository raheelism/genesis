# GENESIS Design Specification
**Date:** 2026-04-07
**Status:** Approved
**Project:** Genesis — Generative Evolving Network of Emergent Sparse Intelligence Systems

---

## Overview

GENESIS is a novel AI architecture built from first principles. It learns causality — not correlation — and stores knowledge in living *cells*: small, self-contained reasoning units that grow, divide, merge, and die. It requires no GPU, no massive weight matrices, and no retraining. It bootstraps from a small seed corpus (~200–500MB), then grows continuously from every interaction.

### Core Design Constraints
- Runs on a standard 4-core CPU, 8GB RAM laptop — no GPU
- Target memory footprint: under 300MB after bootstrap
- Continuous online learning — never stops growing
- Full reasoning trace — every answer has an explicit causal chain
- Starts with general language; architecture is domain-agnostic
- No templates, no frozen weights, no autoregressive next-token prediction

### Interaction Modes (all three unified from one core)
- **Conversational** — text in, reasoned text out
- **Autonomous Agent** — goal-directed, self-monitoring
- **Embedded** — stream processing, plug into other systems

---

## The Three Pillars

### 1. Sparse Distributed Representations (SDR)

Every concept — a word, phrase, or idea — is encoded as a **1024-bit binary vector with exactly 20 bits active** (~2% sparsity).

**Core operations:**
```
sim(A, B)      = popcount(A AND B) / popcount(A OR B)   # Jaccard similarity
compose(A, B)  = top_20(A OR B)                          # concept composition
shift(SDR, n)  = rotate active indices by offset n        # encodes word order
```

**Properties:**
- Similarity computed via single CPU bitwise instruction (SIMD popcount)
- False match probability for unrelated concepts ≈ 10⁻²⁰
- 1 million concepts stored = 128MB flat
- "dog bites man" ≠ "man bites dog" — word order encoded structurally via shift

**Sentence encoding:**
```
encode(w₁ w₂ w₃) = compose(compose(shift(w₁, 0), shift(w₂, 1)), shift(w₃, 2))
```

### 2. The Genesis Cell

The fundamental unit of knowledge. A cell specializes in one region of concept-space and fires only when input overlaps its receptive field.

```
Cell = {
  id:               UUID
  receptive_field:  SDR            # concept-space this cell owns
  rules:            List[Rule]     # causal knowledge
  fitness:          float          # rolling activation confidence
  age:              int            # total activation count
}

Rule = {
  precondition:   SDR              # pattern P
  postcondition:  SDR              # consequence Q
  confidence:     float ∈ [0, 1]  # learned certainty
}
```

Meaning of a rule: *"When I observe pattern P in context, it leads to Q with confidence c."*

**Activation threshold:** `sim(input, receptive_field) > θ_fire = 0.35`
**Rule firing threshold:** `sim(facts_union, P) > θ_rule = 0.40 AND confidence > 0.25`

### 3. The Organism — Layered Cell Colony

```
┌─────────────────────────────────────────────────────┐
│  Layer 4 — Meta Cells         (planning, self-model) │
│  Layer 3 — Reasoning Cells    (causal inference)     │
│  Layer 2 — Concept Cells      (semantics, entities)  │
│  Layer 1 — Pattern Cells      (syntax, sequences)    │
│  Layer 0 — Perception         (text → SDR encoding)  │
└─────────────────────────────────────────────────────┘
```

Only **prediction errors** propagate upward between layers (Predictive Coding). If a lower layer perfectly predicts what the upper layer expects, nothing propagates. Only surprises travel up — massively reducing computation.

**Routing:** Cells indexed by receptive field hash via Locality Sensitive Hashing (LSH). Finding relevant cells = O(1). At 10,000 cells, routing costs ~1 microsecond.

---

## Learning Algorithm — No Backpropagation

All learning is **local** — no global gradient, no weight matrix, no optimizer state.

### Operation 1: Hebbian Strengthening
When rule `(P → Q, c)` fires and observed outcome matches Q:
```
c ← c + η · (1 - c)        η = 0.05
```
Confidence asymptotically approaches 1. Never overshoots.

### Operation 2: Predictive Weakening
When rule fires but observed outcome does not match Q:
```
c ← c · (1 - η)
```
Confidence decays toward 0. Consistently wrong rules die off.

### Operation 3: Rule Creation (Growth)
When input arrives and no existing rule matches (surprise):
```
if max_i sim(input, Pᵢ) < θ_create = 0.30:
    new_rule ← Rule(precondition=input, postcondition=observed, confidence=0.3)
    append to best-fitting cell (highest receptive_field overlap)
```

### Cell Division (Specialization)
```
if len(cell.rules) > max_rules OR shannon_entropy(pairwise_sim_matrix(cell.rules)) > θ_split:
    cluster rules into 2 groups by precondition similarity
    spawn Cell_left  ← cluster_0, receptive_field = centroid_0
    spawn Cell_right ← cluster_1, receptive_field = centroid_1
    retire parent
```

### Cell Merging (Generalization)
```
if sim(R_i, R_j) > θ_merge = 0.70:
    merged_rules = union(Φ_i, Φ_j), deduplicated by rule similarity
    spawn Cell_merged, receptive_field = compose(R_i, R_j)
    retire both parents
```

### Cell Death (Pruning)
```
if cell.fitness < θ_death = 0.10 AND cell.age > min_age = 100:
    redistribute rules to nearest neighbor cells
    retire cell
```

### Why Catastrophic Forgetting Does Not Occur
- New knowledge creates new rules in new/existing cells
- Old rules are weakened only by contradicting evidence, not unrelated learning
- Cells are isolated: updating Cell_A (biology) never touches Cell_B (history)
- Memory is additive by default; subtractive only when wrong

---

## Reasoning Engine

### Forward Chaining (default — answering questions)
```
REASON(query Q, max_depth=8):
  SDR_Q      ← encode(Q)
  facts      ← {SDR_Q}
  chain      ← []
  active     ← route(SDR_Q)

  LOOP up to max_depth:
    new_facts ← ∅
    for cell in active:
      for rule (P → Q_r, c) in cell.rules:
        if sim(union(facts), P) > θ_rule AND c > θ_confidence:
          new_facts.add(Q_r, score = c × sim(union(facts), P))
          chain.append((P, Q_r, c, cell.id))

    if any fact in new_facts has score > θ_answer = 0.40:
      RETURN highest_scoring fact as answer, chain

    facts  ← facts ∪ new_facts
    active ← route(union(new_facts))
    if new_facts = ∅: BREAK

  RETURN "insufficient knowledge", partial_chain
```

Each hop: ~50μs. 8 hops = ~400μs. **Sub-millisecond multi-hop reasoning on CPU.**

### Backward Chaining (planning / verification)
Given goal G:
1. Find rules where postcondition overlaps G
2. Their preconditions become sub-goals
3. Recurse until sub-goals are known facts or max depth exceeded
4. Proven if all sub-goals satisfied; confidence = product of rule confidences along path

### Confidence Propagation
```
confidence(A → B → C) = conf(A→B) × conf(B→C) × 0.95^depth
```
If chain confidence < θ_answer = 0.40: respond with uncertainty rather than hallucinating.

---

## Memory Architecture

### Working Memory (active context)
```
WorkingMemory = CircularBuffer(capacity=12)
```
- Holds SDRs of concepts active in current conversation
- New input SDRs pushed in; oldest evicted when full
- Reasoning draws on `working_memory ∪ query_SDRs`
- No "context window" limit — conversation lives as active concept patterns, not raw text

### Long-Term Memory (the cell colony itself)
- **Semantic memory** — rules inside all cells
- **Procedural memory** — frequently fired reasoning chains cached as macro-rules
- **Episodic memory** — ring buffer of last 10,000 (query, chain, answer) triples

### Memory Consolidation
When episodic buffer fills, low-activity periods trigger consolidation:
```
for episode in episodes where confidence > 0.60 AND access_count > 3:
    extract key (premise → conclusion) pairs
    create permanent Reasoning Cell rules
    remove from episodic buffer
```
Mirrors biological sleep consolidation. Keeps episodic memory lean while preserving important experiences permanently.

---

## Output Generation — Concept-First Verbalization

GENESIS does **not** predict the next token autoregressively.

```
GENERATE(conclusion_SDR):
  candidates ← tokens where sim(token_SDR, conclusion_SDR) > θ_verbalize
  score each by: overlap × token_frequency × position_weight[slot]
  # position_weight learned from seed corpus: how likely a token type is at each position
  greedily select tokens covering maximum conclusion_SDR bits
  apply Pattern Cell (Layer 1) grammar rules for syntactic coherence
  return token sequence
```

**Generation is driven by meaning.** The output covers the concept, not what statistically follows the previous word. If the concept is certain, the sentence is accurate. If uncertain, the sentence reflects uncertainty.

---

## Bootstrap Process

### Phase 0 — Vocabulary Imprinting (~5 min)
- Assign random SDR to each unique token in seed corpus
- Memory: ~12MB for 100K tokens

### Phase 1 — Pattern Seeding (~30 min)
- Sliding window over sentences; compose phrase SDRs
- Create Pattern Cells + rules: phrase → next phrase
- Result: ~50K–200K Pattern Cells, ~50–100MB

### Phase 2 — Concept Emergence (~2 hrs)
- Extract co-occurring entities/predicates/objects from paragraphs
- Create Concept Cells with causal relational rules
- Result: ~5K–20K Concept Cells, ~20–50MB

### Phase 3 — Reasoning Seed (~1 hr)
- Process structured QA pairs / logic statements
- Create Reasoning Cells with high-confidence seed rules
- Result: ~1K–5K Reasoning Cells, ~5–10MB

**Total bootstrap: ~3–4 hrs on 4-core CPU. Total memory: ~100–170MB.**

---

## Growth Trajectory

| Milestone | Memory | Rules | Capability |
|-----------|--------|-------|------------|
| Birth | ~100MB | ~200K | Basic language understanding |
| Week 1 | ~150MB | ~500K | Conversational competence |
| Month 1 | ~200MB | ~1M | Solid reasoning |
| Month 6 | ~300MB | ~2M | Domain expertise developing |
| Year 1 | ~400MB | ~3M | Approaching narrow expert-level |

Growth is logarithmic — redundant patterns produce diminishing rule additions. Cell division/merge/death keeps colony lean.

---

## Comparison to Existing Systems

| Property | 7B LLM | GENESIS |
|---|---|---|
| Parameters/Rules | 7 billion weights | ~500K rules at maturity |
| RAM required | 14GB minimum | ~100–300MB |
| Learns after deploy | No | Yes, continuously |
| Reasoning trace | None | Full causal chain |
| GPU required | Yes | No |
| Grows with data | No — requires retrain | Yes — cell division |
| Hallucinates | Yes — pattern fill | No — no rule = "I don't know" |
| Computation per query | O(n²) attention | O(k), k ≪ n |
| Explainable | No | Yes — every answer has a proof |

---

## File Structure

```
genesis/
├── core/
│   ├── sdr.py                # SDR class: creation, similarity, compose, shift
│   ├── cell.py               # Genesis Cell: rules, fitness, lifecycle hooks
│   ├── organism.py           # Cell colony: LSH routing, layer management
│   └── working_memory.py     # 12-slot circular buffer
├── learning/
│   ├── hebbian.py            # Strengthen, weaken, rule creation
│   ├── lifecycle.py          # Division, merging, death
│   └── consolidation.py      # Episodic → semantic consolidation
├── reasoning/
│   ├── forward_chain.py      # Forward chaining inference
│   ├── backward_chain.py     # Backward chaining / planning
│   └── confidence.py         # Confidence propagation + uncertainty
├── perception/
│   ├── tokenizer.py          # Text → token sequence
│   ├── encoder.py            # Tokens → SDRs with temporal shift
│   └── binder.py             # Phrase/sentence → composed SDR
├── generation/
│   └── verbalizer.py         # Conclusion SDR → output text
├── bootstrap/
│   ├── imprint.py            # Phase 0–3 bootstrap pipeline
│   └── seed_loader.py        # Load and chunk seed corpus
├── storage/
│   └── colony_store.py       # Save/load cell colony (compact binary)
├── interfaces/
│   ├── chat.py               # Conversational mode
│   ├── agent.py              # Autonomous agent mode
│   └── embed.py              # Embedded/stream mode
└── main.py                   # Entry point + CLI
```

---

## Out of Scope (Phase 1)

- GUI (CLI only)
- Distributed / multi-machine support
- Plugin system
- Fine-tuning API
- Multimodal (text only)
- Custom hardware / SIMD optimizations

---

## Phase 1 Success Criteria

1. Bootstraps from a ~200MB text corpus in under 4 hours on a laptop CPU
2. Fits in under 300MB RAM after bootstrap
3. Answers factual questions with traceable reasoning chains
4. Learns a new fact from conversation and recalls it correctly 10 turns later
5. Maintains coherent context across a 20-turn conversation
6. Scores above random on a standard reasoning benchmark (bAbI tasks)
7. Responds in under 500ms per turn on a 4-core CPU
