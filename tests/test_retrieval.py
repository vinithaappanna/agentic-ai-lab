import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from benefits_rag_agent import build_registry, scripted_llm

from agentlab.agent import Agent
from agentlab.retrieval import KeywordRetriever, tokenize

DOCS = {"a": "MRI needs prior authorization", "b": "annual deductible is 1500",
        "c": "prior authorization for MRI and CT imaging scans"}


def test_tokenize_drops_stop_words_and_case():
    assert tokenize("What IS the Deductible?") == ["deductible"]


def test_ranks_by_overlap_then_shorter_doc():
    hits = KeywordRetriever(DOCS).search("MRI prior authorization")
    assert [h["id"] for h in hits] == ["a", "c"]  # same score, shorter first
    assert hits[0]["score"] == 3


def test_no_match_and_k_limit():
    r = KeywordRetriever(DOCS)
    assert r.search("zebra") == []
    assert len(r.search("prior authorization MRI", k=1)) == 1
    assert r.search("MRI", k=0) == []


def test_rag_example_answers_from_retrieved_entry():
    result = Agent(scripted_llm(), build_registry()).run("Does an MRI need prior auth?")
    assert "prior-auth" in result.steps[0].observation
    assert "prior authorization" in result.answer
    assert result.stopped_reason == "final"


def test_registry_tool_reports_no_matches():
    assert build_registry().call("search_faq", {"query": "zebra"}) == "no matching FAQ entries"
