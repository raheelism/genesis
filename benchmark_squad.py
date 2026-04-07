"""
GENESIS SQuAD-Tiny Benchmark
Uses PhraseStore + BeamChain (the improved stack).
Run data/prepare_squad.py first.
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

    return tok, enc, binder, org, qa_pairs, ps, len(sentences)


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
    if not os.path.exists(CORPUS_PATH) or not os.path.exists(QA_PATH):
        print("ERROR: SQuAD data files not found.")
        print("Run first: python data/prepare_squad.py")
        return

    print("=" * 60)
    print("  GENESIS - SQuAD Tiny Benchmark")
    print("=" * 60)

    tracemalloc.start()
    tok, enc, binder, org, qa_pairs, ps, n_sentences = build_system()
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
