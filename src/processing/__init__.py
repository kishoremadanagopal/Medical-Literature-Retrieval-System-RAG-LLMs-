"""Text processing — chunking + BioBERT embedding."""

from src.processing.chunker import chunk_article, chunk_articles_from_jsonl
from src.processing.embedder import BioBERTEmbedder, embed_chunks_from_jsonl

__all__ = [
    "chunk_article",
    "chunk_articles_from_jsonl",
    "BioBERTEmbedder",
    "embed_chunks_from_jsonl",
]
