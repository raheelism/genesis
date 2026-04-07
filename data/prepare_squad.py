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

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
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

    # QA pairs: keep only short answers (1-5 words), deduplicate questions
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
