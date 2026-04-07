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
