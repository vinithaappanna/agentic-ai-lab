"""A tiny keyword retriever, usable as an agent tool.

No embeddings, no dependencies: documents are scored by how many query
terms they contain (stop words ignored), ties broken by shorter document
then id. Enough to show the retrieval-as-a-tool pattern offline; swap in a
vector store behind the same `search` interface for real use.
"""
from __future__ import annotations

import re

STOP_WORDS = frozenset(
    "a an and are as at be by for from how in is it of on or that the to what when which with".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP_WORDS]


class KeywordRetriever:
    def __init__(self, documents: dict[str, str]):
        self._docs = dict(documents)
        self._tokens = {doc_id: set(tokenize(text)) for doc_id, text in self._docs.items()}

    def search(self, query: str, k: int = 3) -> list[dict]:
        """Top-k documents with at least one matching term, best first."""
        terms = set(tokenize(query))
        scored = []
        for doc_id, toks in self._tokens.items():
            score = len(terms & toks)
            if score:
                scored.append((-score, len(self._docs[doc_id]), doc_id))
        scored.sort()
        return [{"id": doc_id, "score": -neg, "text": self._docs[doc_id]}
                for neg, _len, doc_id in scored[:max(k, 0)]]
