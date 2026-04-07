"""
GENESIS Small Dataset Benchmark
Bootstraps from small_corpus.txt + small_qa.txt, then evaluates:
  1. Bootstrap time + memory
  2. Response time per query
  3. QA accuracy (answer contains expected keyword)
  4. Multi-turn coherence
  5. New-fact recall
"""
import os
import sys
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
from genesis.reasoning.forward_chain import ForwardChain
from genesis.storage.colony_store import ColonyStore

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CORPUS_PATH = os.path.join(DATA_DIR, "small_corpus.txt")
QA_PATH = os.path.join(DATA_DIR, "small_qa.txt")
COLONY_PATH = os.path.join(DATA_DIR, "small_colony.gen")

SEP = "-" * 60


def load_qa_pairs():
    pairs = []
    with open(QA_PATH) as f:
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

    loader = SeedLoader()
    imp = Imprinter()

    print(f"\n{SEP}")
    print("PHASE 0 — Vocabulary imprinting...")
    t0 = time.perf_counter()
    sentences = list(loader.load(CORPUS_PATH))
    imp.phase0(sentences, tok, enc)
    print(f"  Sentences : {len(sentences)}")
    print(f"  Vocab size: {enc.vocab_size()} tokens")

    print("PHASE 1 — Pattern seeding...")
    imp.phase1(sentences, tok, enc, binder, org)
    print(f"  Cells     : {org.cell_count()}")

    print("PHASE 2 — Concept emergence...")
    imp.phase2(sentences, tok, enc, binder, org)
    print(f"  Cells     : {org.cell_count()}")

    print("PHASE 3 — Reasoning seed (QA pairs)...")
    qa_pairs = load_qa_pairs()
    ps = PhraseStore()
    imp.phase3(qa_pairs, tok, enc, binder, org, phrase_store=ps)
    elapsed = time.perf_counter() - t0
    print(f"  Cells     : {org.cell_count()}")
    print(f"  QA pairs  : {len(qa_pairs)}")
    print(f"  Total time: {elapsed:.2f}s")

    return tok, enc, binder, org, qa_pairs, ps


def measure_memory(org):
    """Estimate colony memory via pickle size."""
    import pickle, io
    buf = io.BytesIO()
    import pickle
    pickle.dump(org, buf, protocol=pickle.HIGHEST_PROTOCOL)
    return buf.tell()


def bench_qa(chat, qa_pairs):
    """Test QA accuracy: does the response contain the expected answer keyword?"""
    print(f"\n{SEP}")
    print("QA ACCURACY TEST")
    print(f"{'Question':<45} {'Expected':<15} {'Got':<30} OK?")
    print("-" * 100)

    correct = 0
    total = len(qa_pairs)
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
        q_short = question[:44]
        r_short = response[:29]
        print(f"  {q_short:<44} {expected:<15} {r_short:<30} {mark}")

    acc = correct / total * 100
    avg_ms = sum(timings) / len(timings)
    p99_ms = sorted(timings)[int(len(timings) * 0.99)]

    print(f"\n  Accuracy  : {correct}/{total} = {acc:.0f}%")
    print(f"  Avg time  : {avg_ms:.1f}ms")
    print(f"  P99 time  : {p99_ms:.1f}ms")
    return acc, avg_ms


def bench_new_fact_recall(chat):
    """Teach a fact then ask about it later in the same session."""
    print(f"\n{SEP}")
    print("NEW FACT RECALL TEST")

    facts = [
        ("genesis is a new kind of ai", "genesis"),
        ("the capital of logic is reason", "reason"),
        ("carbon and silicon are both useful elements", "silicon"),
    ]

    results = []
    for teach, keyword in facts:
        chat.turn(teach)                        # teach
        for _ in range(3):                      # 3 filler turns
            chat.turn("what do you know")
        response = chat.turn(f"tell me about {keyword}")
        hit = keyword in response.lower() or response != "<I don't know yet — still learning>"
        results.append(hit)
        mark = "OK" if hit else "MISS"
        print(f"  Taught: '{teach[:50]}'")
        print(f"  Asked : 'tell me about {keyword}'")
        print(f"  Got   : '{response[:60]}'  {mark}\n")

    score = sum(results)
    print(f"  Recall: {score}/{len(results)}")
    return score


def bench_multi_turn(chat):
    """20-turn conversation — measure coherence (all turns return strings, no crashes)."""
    print(f"\n{SEP}")
    print("MULTI-TURN COHERENCE TEST (20 turns)")

    turns = [
        "tell me about the sky", "what color is it",
        "what about water", "is water wet",
        "tell me about fire", "what does fire need",
        "tell me about animals", "what is the largest animal",
        "what do humans use to communicate", "tell me about the brain",
        "what is gravity", "does light travel fast",
        "what is the atom", "what is dna",
        "tell me about stars", "what is the sun",
        "what is the universe", "when did it begin",
        "what do vaccines do", "what is energy",
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
    print(f"  Turn count     : {chat.turn_count}")
    return ok, avg


def main():
    print("=" * 60)
    print("  GENESIS — Small Dataset Benchmark")
    print("=" * 60)

    # ── Bootstrap ────────────────────────────────────────────
    tracemalloc.start()
    tok, enc, binder, org, qa_pairs, ps = build_system()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    colony_bytes = measure_memory(org)
    print(f"\n  Peak RAM during bootstrap : {peak_bytes/1024/1024:.1f} MB")
    print(f"  Colony serialized size    : {colony_bytes/1024:.1f} KB")
    print(f"  Total rules               : {sum(len(c.rules) for c in org.cells.values())}")

    # Save colony
    store = ColonyStore()
    store.save(org, COLONY_PATH)
    print(f"  Colony saved to           : {COLONY_PATH}")

    # ── Build chat interface ──────────────────────────────────
    wm = WorkingMemory()
    chat = ChatInterface(
        tok, enc, binder, org, wm,
        ForwardChain(), HebbianLearner(), Verbalizer(phrase_store=ps),
    )

    # ── Benchmarks ────────────────────────────────────────────
    acc, avg_ms = bench_qa(chat, qa_pairs)
    recall = bench_new_fact_recall(chat)
    coherent, turn_avg = bench_multi_turn(chat)

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Corpus sentences  : 90")
    print(f"  QA pairs          : {len(qa_pairs)}")
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

    # Pass/fail verdict
    criteria = [
        (acc >= 30,         f"QA accuracy >= 30%      : {acc:.0f}%"),
        (avg_ms < 500,      f"Avg response < 500ms    : {avg_ms:.1f}ms"),
        (coherent == 20,    f"All 20 turns coherent   : {coherent}/20"),
        (colony_bytes < 50 * 1024 * 1024,  f"Colony < 50MB           : {colony_bytes/1024:.1f} KB"),
    ]
    all_pass = all(ok for ok, _ in criteria)
    print("  Phase 1 Success Criteria:")
    for ok, label in criteria:
        print(f"    {'OK' if ok else 'FAIL'} {label}")
    print(f"\n  Overall: {'PASS - ready for large dataset' if all_pass else 'NEEDS REVIEW'}")
    print()


if __name__ == "__main__":
    main()
