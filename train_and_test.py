"""
Train GENESIS on a large dataset and test with chat.

Example:
  python train_and_test.py --corpus wikipedia --size medium --output my_colony.gen
  python train_and_test.py --corpus squad --output squad_colony.gen

Corpus options: squad, wikipedia
Size options: small (1000 sentences), medium (10K), large (100K)
"""
import argparse
import os
import re
from pathlib import Path
from typing import Iterator, List, Tuple

from genesis.bootstrap.imprint import Imprinter
from genesis.bootstrap.seed_loader import SeedLoader
from genesis.core.organism import Organism
from genesis.core.working_memory import WorkingMemory
from genesis.generation.verbalizer import Verbalizer
from genesis.interfaces.chat import ChatInterface
from genesis.learning.hebbian import HebbianLearner
from genesis.learning.lifecycle import LifecycleManager
from genesis.perception.binder import Binder
from genesis.perception.encoder import Encoder
from genesis.perception.tokenizer import Tokenizer
from genesis.reasoning.forward_chain import ForwardChain
from genesis.storage.colony_store import ColonyStore


def fetch_wikipedia(num_articles: int = 1000, num_sentences: int = 50000) -> Iterator[str]:
    """Stream Wikipedia articles and yield cleaned sentences."""
    print(f"Loading Wikipedia (up to {num_articles} articles, ~{num_sentences} sentences)...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: Install datasets: pip install datasets")
        return

    ds = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
    
    sentences = []
    seen = set()
    article_count = 0

    for row in ds:
        if article_count >= num_articles:
            break
        article_count += 1
        
        text = row["text"]
        for sent in re.split(r'(?<=[.!?])\s+', text):
            sent = sent.strip()
            if len(sent) >= 25 and sent not in seen:
                seen.add(sent)
                sentences.append(sent)
                if len(sentences) >= num_sentences:
                    break
        
        if len(sentences) >= num_sentences:
            break

    for sent in sentences:
        yield sent
    
    print(f"  Loaded {len(sentences)} sentences from {article_count} articles")


def fetch_squad(num_examples: int = 500, max_sentences: int = 10000) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Download SQuAD v1.1 validation set and extract corpus + QA pairs."""
    print(f"Loading SQuAD v1.1 (first {num_examples} examples)...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: Install datasets: pip install datasets")
        return [], []

    ds = load_dataset("squad", split="validation")
    examples = list(ds.select(range(min(num_examples, len(ds)))))

    # Extract unique sentences for corpus
    corpus = []
    seen_sentences = set()
    for ex in examples:
        for sent in re.split(r'(?<=[.!?])\s+', ex["context"]):
            sent = sent.strip()
            if len(sent) >= 25 and sent not in seen_sentences:
                seen_sentences.add(sent)
                corpus.append(sent)
                if len(corpus) >= max_sentences:
                    break
        if len(corpus) >= max_sentences:
            break

    # Extract QA pairs (keep short answers only)
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
        # Keep answers with 1-5 words
        if 1 <= len(a.split()) <= 5:
            qa_pairs.append((q, a))
            seen_questions.add(q)
        if len(qa_pairs) >= 100:
            break

    print(f"  Corpus: {len(corpus)} sentences")
    print(f"  QA pairs: {len(qa_pairs)} pairs")
    return corpus, qa_pairs


def bootstrap_colony(corpus_path: str, qa_path: str = None, output_path: str = "colony.gen"):
    """Run the 4-phase bootstrap from corpus and QA pairs."""
    print("\n" + "="*60)
    print("BOOTSTRAP PHASE")
    print("="*60)

    tok = Tokenizer()
    enc = Encoder()
    binder = Binder()
    org = Organism()
    store = ColonyStore()
    imp = Imprinter()

    # Load sentences
    loader = SeedLoader()
    sentences = list(loader.load(corpus_path))
    print(f"Loaded {len(sentences)} sentences from {corpus_path}")

    # Phase 0: Vocabulary imprinting
    print("\nPhase 0: Vocabulary imprinting...")
    imp.phase0(sentences, tok, enc)
    print(f"  ✓ Vocab size: {enc.vocab_size()} tokens")

    # Phase 1: Pattern seeding
    print("Phase 1: Pattern seeding...")
    imp.phase1(sentences, tok, enc, binder, org)
    print(f"  ✓ Cells: {org.cell_count()}")

    # Phase 2: Concept emergence
    print("Phase 2: Concept emergence...")
    imp.phase2(sentences, tok, enc, binder, org)
    print(f"  ✓ Cells: {org.cell_count()}")

    # Phase 3: Reasoning seed (if QA pairs available)
    if qa_path and os.path.exists(qa_path):
        print("Phase 3: Reasoning seed...")
        qa_pairs = []
        with open(qa_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) == 2:
                    qa_pairs.append((parts[0].strip(), parts[1].strip()))
        imp.phase3(qa_pairs, tok, enc, binder, org)
        print(f"  ✓ QA pairs: {len(qa_pairs)}, Total cells: {org.cell_count()}")
    else:
        print("Phase 3: Skipped (no QA pairs)")

    # Save colony
    store.save(org, output_path)
    print(f"\n✓ Bootstrap complete!")
    print(f"  Colony saved to: {output_path}")
    print(f"  Total cells: {org.cell_count()}")
    return org, tok, enc, binder


def test_colony(colony_path: str, test_prompts: List[str] = None):
    """Load a trained colony and test with chat."""
    print("\n" + "="*60)
    print("CHAT TEST")
    print("="*60)

    if test_prompts is None:
        test_prompts = [
            "Hello",
            "What do you know?",
            "Tell me something interesting.",
            "Can you help me?",
        ]

    from genesis.main import _build_components

    tok, enc, binder, org, wm, store = _build_components(colony_path)
    reasoner = ForwardChain()
    learner = HebbianLearner()
    verbalizer = Verbalizer()

    chat = ChatInterface(tok, enc, binder, org, wm, reasoner, learner, verbalizer)

    print(f"\nLoaded colony: {org.cell_count()} cells\n")

    for prompt in test_prompts:
        print(f"You: {prompt}")
        response = chat.turn(prompt)
        print(f"GENESIS: {response}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train GENESIS on a large dataset and test"
    )
    parser.add_argument(
        "--corpus",
        choices=["squad", "wikipedia"],
        default="squad",
        help="Corpus source (default: squad)"
    )
    parser.add_argument(
        "--size",
        choices=["small", "medium", "large"],
        default="medium",
        help="Dataset size (default: medium)"
    )
    parser.add_argument(
        "--output",
        default="trained_colony.gen",
        help="Output colony file path"
    )
    parser.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Skip bootstrap, only test existing colony"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Load and test an existing colony file"
    )

    args = parser.parse_args()

    # Determine dataset sizes
    size_config = {
        "small": {"articles": 100, "sentences": 1000},
        "medium": {"articles": 500, "sentences": 10000},
        "large": {"articles": 2000, "sentences": 50000},
    }
    config = size_config[args.size]

    # Test mode: load and chat only
    if args.test_only:
        if not os.path.exists(args.output):
            print(f"ERROR: Colony file not found: {args.output}")
            return
        test_colony(args.output)
        return

    # Prepare corpus and QA data
    corpus_file = f"corpus_{args.corpus}_{args.size}.txt"
    qa_file = f"qa_{args.corpus}_{args.size}.txt" if args.corpus == "squad" else None

    if not args.skip_bootstrap or not os.path.exists(corpus_file):
        print(f"Preparing {args.corpus} dataset ({args.size})...\n")
        
        if args.corpus == "squad":
            corpus, qa_pairs = fetch_squad(
                num_examples=config["articles"],
                max_sentences=config["sentences"]
            )
            # Write corpus and QA to files
            Path(corpus_file).write_text("\n".join(corpus), encoding="utf-8")
            if qa_pairs:
                Path(qa_file).write_text(
                    "\n".join(f"{q}|{a}" for q, a in qa_pairs),
                    encoding="utf-8"
                )
            print(f"  Saved corpus to: {corpus_file}")
            if qa_file:
                print(f"  Saved QA pairs to: {qa_file}")
        else:  # wikipedia
            corpus = list(fetch_wikipedia(
                num_articles=config["articles"],
                num_sentences=config["sentences"]
            ))
            Path(corpus_file).write_text("\n".join(corpus), encoding="utf-8")
            print(f"  Saved corpus to: {corpus_file}")
            qa_file = None

    # Bootstrap
    if not args.skip_bootstrap:
        bootstrap_colony(corpus_file, qa_file, args.output)
    else:
        print(f"Skipping bootstrap, using existing: {args.output}")

    # Test
    test_colony(args.output)


if __name__ == "__main__":
    main()
