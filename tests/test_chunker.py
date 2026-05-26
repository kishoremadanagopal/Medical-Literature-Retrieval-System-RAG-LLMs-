"""
Tests for src/processing/chunker.py

Run: pytest tests/test_chunker.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.processing.chunker import (
    chunk_article,
    estimate_tokens,
    split_sentences,
    split_structured_abstract,
)


# ── Helpers ─────────────────────────────────────────────────────────

def make_article(title: str = "Test Title", abstract: str = "Test abstract.", **kwargs):
    """Create a minimal article dict for testing."""
    return {
        "pmid": kwargs.get("pmid", "12345"),
        "title": title,
        "abstract": abstract,
        "authors": kwargs.get("authors", ["Smith John"]),
        "journal": kwargs.get("journal", "Nature"),
        "pub_date": kwargs.get("pub_date", "2023-01-01"),
        "mesh_terms": kwargs.get("mesh_terms", ["Genomics"]),
        "keywords": kwargs.get("keywords", []),
        "doi": kwargs.get("doi", "10.1234/test"),
        "pub_types": kwargs.get("pub_types", ["Journal Article"]),
        "language": "eng",
    }


# ── Sentence splitting ──────────────────────────────────────────────

def test_split_sentences_basic():
    result = split_sentences("First sentence. Second sentence. Third one.")
    assert len(result) == 3


def test_split_sentences_abbreviations():
    """Abbreviations like 'et al.' shouldn't split."""
    text = "Smith et al. showed this works. The results were clear."
    result = split_sentences(text)
    # "et al." doesn't end with capital after period, so it stays joined
    assert len(result) == 2


def test_split_sentences_empty():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


# ── Structured abstract detection ───────────────────────────────────

def test_structured_abstract_detected():
    abstract = (
        "BACKGROUND: Drug discovery is costly. "
        "METHODS: We reviewed 500 papers. "
        "RESULTS: AI reduced costs by 30%. "
        "CONCLUSIONS: AI is transformative."
    )
    sections = split_structured_abstract(abstract)
    assert len(sections) == 4
    assert sections[0].startswith("BACKGROUND:")
    assert sections[3].startswith("CONCLUSIONS:")


def test_unstructured_abstract_returns_whole():
    abstract = "This is a normal abstract without labeled sections."
    sections = split_structured_abstract(abstract)
    assert len(sections) == 1
    assert sections[0] == abstract


# ── Token estimation ────────────────────────────────────────────────

def test_estimate_tokens():
    text = "This is a simple sentence with eight words total."
    tokens = estimate_tokens(text)
    # 10 words × 1.3 ≈ 13 tokens
    assert 10 <= tokens <= 16


# ── chunk_article ───────────────────────────────────────────────────

def test_short_article_single_chunk():
    """A typical PubMed abstract (~200 words) should be one chunk."""
    article = make_article(
        abstract="A short abstract about protein folding mechanisms in cells."
    )
    chunks = chunk_article(article, chunk_size=512)
    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "12345_chunk_0"
    assert "Test Title" in chunks[0]["text"]
    assert "protein folding" in chunks[0]["text"]


def test_metadata_preserved():
    """Every chunk must carry full metadata for vector store filtering."""
    article = make_article(mesh_terms=["Genomics", "CRISPR"])
    chunks = chunk_article(article, chunk_size=512)
    meta = chunks[0]["metadata"]

    assert meta["pmid"] == "12345"
    assert meta["journal"] == "Nature"
    assert "Genomics" in meta["mesh_terms"]
    assert "CRISPR" in meta["mesh_terms"]
    assert meta["chunk_index"] == 0


def test_long_article_splits():
    """An article longer than chunk_size should produce multiple chunks."""
    # Create a ~200-word abstract (will exceed chunk_size=50)
    long_abstract = ". ".join(
        [f"Sentence number {i} discusses biomedical topic {i}" for i in range(20)]
    ) + "."
    article = make_article(abstract=long_abstract)
    chunks = chunk_article(article, chunk_size=50, chunk_overlap=10)
    assert len(chunks) > 1

    # Each chunk should have unique chunk_id
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_empty_article_no_chunks():
    article = make_article(title="", abstract="")
    chunks = chunk_article(article)
    assert len(chunks) == 0


def test_title_only_article():
    """Article with title but no abstract should still produce a chunk."""
    article = make_article(abstract="")
    chunks = chunk_article(article, chunk_size=512)
    assert len(chunks) == 1
    assert "Test Title" in chunks[0]["text"]


def test_structured_abstract_chunking():
    """Structured abstract sections should be respected as split points."""
    abstract = (
        "BACKGROUND: " + " ".join(["Background sentence."] * 15) + " "
        "METHODS: " + " ".join(["Methods sentence."] * 15) + " "
        "RESULTS: " + " ".join(["Results sentence."] * 15) + " "
        "CONCLUSIONS: " + " ".join(["Conclusions sentence."] * 15)
    )
    article = make_article(abstract=abstract)
    chunks = chunk_article(article, chunk_size=100, chunk_overlap=10)

    # Should split at section boundaries, producing multiple chunks
    assert len(chunks) >= 2

    # First chunk should contain title
    assert "Test Title" in chunks[0]["text"]


def test_chunk_overlap_exists():
    """Chunks should share some text when overlap > 0."""
    long_abstract = ". ".join(
        [f"Unique sentence alpha-{i} about topic beta-{i}" for i in range(30)]
    ) + "."
    article = make_article(abstract=long_abstract)
    chunks = chunk_article(article, chunk_size=80, chunk_overlap=20)

    if len(chunks) >= 2:
        # Some text from end of chunk[0] should appear in chunk[1]
        words_0 = set(chunks[0]["text"].split()[-10:])
        words_1 = set(chunks[1]["text"].split()[:10])
        overlap = words_0 & words_1
        # With overlap=20 tokens, we expect some shared words
        assert len(overlap) > 0
