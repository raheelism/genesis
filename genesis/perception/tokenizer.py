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
