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
