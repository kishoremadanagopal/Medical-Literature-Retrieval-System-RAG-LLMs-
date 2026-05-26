"""
Domain-aware chunker for PubMed articles.

Key insight: PubMed abstracts are typically 200-400 words (~250-500 tokens).
With chunk_size=512 tokens, most abstracts fit in a SINGLE chunk. This is
intentional — an abstract is a self-contained unit of meaning, and splitting
it mid-thought hurts retrieval quality.

For longer structured abstracts (Background/Methods/Results/Conclusions),
we split at section boundaries first, then fall back to sentence boundaries.
This preserves scientific reasoning flow.

Each chunk carries full metadata (pmid, journal, date, MeSH terms) so the
vector store can filter on these fields without a separate lookup.
"""

import json
import re
from pathlib import Path
from typing import Optional

from config.config import settings
from config.logging_config import get_logger

logger = get_logger(__name__)


# ── Sentence splitting ──────────────────────────────────────────────
# We use a regex-based splitter instead of NLTK punkt to avoid the
# 35MB model download. This handles 99% of biomedical text correctly.
SENTENCE_SPLIT_RE = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z])'  # Split after .!? followed by space + capital
)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex (fast, no NLTK dependency)."""
    if not text.strip():
        return []
    sentences = SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


# ── Structured abstract detection ───────────────────────────────────
SECTION_LABELS = re.compile(
    r'^(BACKGROUND|OBJECTIVE|METHODS|RESULTS|CONCLUSIONS|'
    r'INTRODUCTION|PURPOSE|MATERIALS AND METHODS|DISCUSSION|'
    r'AIM|DESIGN|SETTING|PATIENTS|PARTICIPANTS|'
    r'MAIN OUTCOME MEASURES|FINDINGS|INTERPRETATION):',
    re.IGNORECASE
)


def split_structured_abstract(abstract: str) -> list[str]:
    """
    Split a structured abstract into its labeled sections.
    Returns a list of section strings, each starting with its label.
    If the abstract isn't structured, returns [abstract].
    """
    sections = []
    current = []

    for sentence in split_sentences(abstract):
        if SECTION_LABELS.match(sentence):
            if current:
                sections.append(" ".join(current))
            current = [sentence]
        else:
            current.append(sentence)

    if current:
        sections.append(" ".join(current))

    return sections if len(sections) > 1 else [abstract]


# ── Token estimation ────────────────────────────────────────────────
# BioBERT uses WordPiece tokenization. A rough estimate is 1.3 tokens
# per word for biomedical text (slightly higher than general English
# due to long technical terms being split into subwords).
TOKENS_PER_WORD = 1.3


def estimate_tokens(text: str) -> int:
    """Estimate token count without loading the tokenizer."""
    return int(len(text.split()) * TOKENS_PER_WORD)


# ── Core chunking logic ────────────────────────────────────────────

def chunk_article(article: dict, chunk_size: int = None, chunk_overlap: int = None) -> list[dict]:
    """
    Split a single article into chunks, each carrying full metadata.

    Strategy:
        1. Combine "Title. Abstract" as primary text
        2. If it fits in one chunk → keep as single chunk (most common case)
        3. If structured abstract → split at section boundaries first
        4. If still too long → split at sentence boundaries with overlap

    Args:
        article: Dict from JSONL (pmid, title, abstract, mesh_terms, ...)
        chunk_size: Max tokens per chunk (default from settings)
        chunk_overlap: Token overlap between chunks (default from settings)

    Returns:
        List of chunk dicts, each with 'text', 'metadata', and 'chunk_id'.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    pmid = article.get("pmid", "unknown")
    title = article.get("title", "").strip()
    abstract = article.get("abstract", "").strip()

    if not title and not abstract:
        return []

    # Metadata carried with every chunk (used for filtering in Chroma)
    metadata = {
        "pmid": pmid,
        "title": title,
        "journal": article.get("journal", ""),
        "journal_abbrev": article.get("journal_abbrev", ""),
        "pub_date": article.get("pub_date", ""),
        "mesh_terms": article.get("mesh_terms", []),
        "keywords": article.get("keywords", []),
        "doi": article.get("doi", ""),
        "authors": article.get("authors", []),
        "pub_types": article.get("pub_types", []),
        "language": article.get("language", "eng"),
    }

    # Combine title + abstract
    full_text = f"{title}. {abstract}" if title and abstract else (title or abstract)

    # Case 1: Fits in one chunk (most PubMed abstracts)
    if estimate_tokens(full_text) <= chunk_size:
        return [_make_chunk(full_text, metadata, pmid, 0)]

    # Case 2: Try structured abstract sections
    sections = split_structured_abstract(abstract)
    if len(sections) > 1:
        chunks = _chunk_sections(sections, title, metadata, pmid, chunk_size, chunk_overlap)
        if chunks:
            return chunks

    # Case 3: Fall back to sentence-level splitting
    return _chunk_by_sentences(full_text, metadata, pmid, chunk_size, chunk_overlap)


def _make_chunk(text: str, metadata: dict, pmid: str, index: int) -> dict:
    """Create a single chunk dict."""
    return {
        "chunk_id": f"{pmid}_chunk_{index}",
        "text": text,
        "metadata": {
            **metadata,
            "chunk_index": index,
            "token_estimate": estimate_tokens(text),
        },
    }


def _chunk_sections(
    sections: list[str],
    title: str,
    metadata: dict,
    pmid: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    """
    Group structured abstract sections into chunks that fit within chunk_size.
    Each chunk gets the title prepended for context.
    """
    chunks = []
    current_text = title + "." if title else ""
    chunk_idx = 0

    for section in sections:
        combined = (current_text + " " + section).strip()

        if estimate_tokens(combined) <= chunk_size:
            current_text = combined
        else:
            # Current buffer is full — emit it
            if current_text.strip():
                chunks.append(_make_chunk(current_text.strip(), metadata, pmid, chunk_idx))
                chunk_idx += 1

            # Start new chunk with title context + this section
            current_text = f"{title}. {section}" if title else section

            # If single section exceeds chunk_size, split by sentences
            if estimate_tokens(current_text) > chunk_size:
                sub_chunks = _chunk_by_sentences(
                    current_text, metadata, pmid, chunk_size, chunk_overlap,
                    start_index=chunk_idx,
                )
                chunks.extend(sub_chunks)
                chunk_idx += len(sub_chunks)
                current_text = ""

    # Don't forget the last buffer
    if current_text.strip():
        chunks.append(_make_chunk(current_text.strip(), metadata, pmid, chunk_idx))

    return chunks


def _chunk_by_sentences(
    text: str,
    metadata: dict,
    pmid: str,
    chunk_size: int,
    chunk_overlap: int,
    start_index: int = 0,
) -> list[dict]:
    """
    Split text at sentence boundaries, respecting chunk_size and overlap.
    """
    sentences = split_sentences(text)
    if not sentences:
        return [_make_chunk(text, metadata, pmid, start_index)] if text.strip() else []

    chunks = []
    current_sentences = []
    current_tokens = 0
    chunk_idx = start_index

    for sentence in sentences:
        sent_tokens = estimate_tokens(sentence)

        if current_tokens + sent_tokens <= chunk_size:
            current_sentences.append(sentence)
            current_tokens += sent_tokens
        else:
            # Emit current chunk
            if current_sentences:
                chunk_text = " ".join(current_sentences)
                chunks.append(_make_chunk(chunk_text, metadata, pmid, chunk_idx))
                chunk_idx += 1

                # Calculate overlap: keep last N sentences that fit in overlap budget
                overlap_sentences = []
                overlap_tokens = 0
                for s in reversed(current_sentences):
                    s_tokens = estimate_tokens(s)
                    if overlap_tokens + s_tokens <= chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_tokens += s_tokens
                    else:
                        break

                current_sentences = overlap_sentences + [sentence]
                current_tokens = overlap_tokens + sent_tokens
            else:
                # Single sentence exceeds chunk_size — keep it anyway
                chunks.append(_make_chunk(sentence, metadata, pmid, chunk_idx))
                chunk_idx += 1
                current_sentences = []
                current_tokens = 0

    # Emit remaining
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunks.append(_make_chunk(chunk_text, metadata, pmid, chunk_idx))

    return chunks


# ── Batch processing ────────────────────────────────────────────────

def chunk_articles_from_jsonl(
    input_path: Path,
    output_path: Optional[Path] = None,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> Path:
    """
    Process an entire JSONL file of articles into chunks.

    Args:
        input_path: Path to pubmed_articles.jsonl
        output_path: Path for output chunks JSONL (default: data/processed/chunks.jsonl)
        chunk_size: Override settings.chunk_size
        chunk_overlap: Override settings.chunk_overlap

    Returns:
        Path to output file.
    """
    output_path = output_path or settings.processed_data_dir / "chunks.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_articles = 0
    total_chunks = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            article = json.loads(line)
            chunks = chunk_article(article, chunk_size, chunk_overlap)

            for chunk in chunks:
                fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                total_chunks += 1

            total_articles += 1

            if total_articles % 5000 == 0:
                logger.info(f"Chunked {total_articles} articles → {total_chunks} chunks")

    logger.info(
        f"Chunking complete: {total_articles} articles → {total_chunks} chunks "
        f"(avg {total_chunks / max(total_articles, 1):.1f} chunks/article)"
    )
    return output_path
