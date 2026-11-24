"""A tiny keyword retriever, usable as an agent tool.

No embeddings, no dependencies: documents are scored by how many query
terms they contain (stop words ignored), ties broken by shorter document
then id. Enough to show the retrieval-as-a-tool pattern offline; swap in a
vector store behind the same `search` interface for real use.
"""
from __future__ import annotations

import math
import re

STOP_WORDS = frozenset(
    "a an and are as at be by for from how in is it of on or that the to what when which with".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP_WORDS]


class KeywordRetriever:
    def __init__(self, documents: dict[str, str], weighting: str = "count"):
        """weighting="count" scores by matched terms; "idf" weights each matched
        term by how rare it is across the documents, so a rare term like
        "deductible" outranks several common ones."""
        if weighting not in ("count", "idf"):
            raise ValueError("weighting must be 'count' or 'idf'")
        self.weighting = weighting
        self._docs = dict(documents)
        self._tokens = {doc_id: set(tokenize(text)) for doc_id, text in self._docs.items()}
        n = len(self._docs)
        df: dict[str, int] = {}
        for toks in self._tokens.values():
            for t in toks:
                df[t] = df.get(t, 0) + 1
        # smoothed idf, always > 0, so a matching term never scores zero
        self._idf = {t: math.log(1 + n / c) for t, c in df.items()}

    def search(self, query: str, k: int = 3, min_score: float = 0.0) -> list[dict]:
        """Top-k documents with at least one matching term, best first. With
        min_score > 0, weak matches are dropped, so an agent can tell "nothing
        relevant" from "something weak" and decline instead of answering from a
        poor hit (scores follow the weighting: matched-term count or idf sum)."""
        terms = set(tokenize(query))
        scored = []
        for doc_id, toks in self._tokens.items():
            matched = terms & toks
            score = (sum(self._idf[t] for t in matched) if self.weighting == "idf"
                     else len(matched))
            if matched and score >= min_score:
                scored.append((-score, len(self._docs[doc_id]), doc_id, sorted(matched)))
        scored.sort()
        return [{"id": doc_id, "score": round(-neg, 4) if self.weighting == "idf" else -neg, "text": self._docs[doc_id],
                 "matched": matched}   # which query terms hit: explains a result in an audit trail
                for neg, _len, doc_id, matched in scored[:max(k, 0)]]
