"""
Chroma vector store — persistent storage with rich metadata filtering.

Why Chroma?
    - Persistent: survives restarts (backed by SQLite + Parquet)
    - Metadata filtering: "find papers about CRISPR published after 2022
      in Nature" — this is a WHERE clause, not similarity search
    - LangChain integration: drop-in compatible with LangChain retrievers
    - Good enough speed for most queries (< 100ms on 50K vectors)

When to use Chroma over FAISS:
    - Queries with metadata constraints (year, journal, MeSH terms)
    - When you need persistence without manual save/load
    - When you need the full metadata alongside results
"""

import json
from pathlib import Path
from typing import Optional

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from config.config import settings
from config.logging_config import get_logger

logger = get_logger(__name__)


class ChromaStore:
    """
    Manages a Chroma collection for PubMed article chunks.

    Usage:
        store = ChromaStore()
        store.build_from_embeddings(embeddings_path, metadata_path)

        results = store.search("CRISPR gene editing", top_k=5)
        results = store.search(
            "CRISPR gene editing",
            top_k=5,
            where={"journal": "Nature"},
        )
    """

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        collection_name: Optional[str] = None,
    ):
        self.persist_dir = str(persist_dir or settings.chroma_persist_dir)
        self.collection_name = collection_name or settings.chroma_collection_name

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = None

    @property
    def collection(self) -> chromadb.Collection:
        """Get or create the collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},  # Cosine similarity
            )
        return self._collection

    @property
    def count(self) -> int:
        """Number of vectors in the collection."""
        return self.collection.count()

    # ── Building the index ──────────────────────────────────────────

    def build_from_embeddings(
        self,
        embeddings_path: Path,
        metadata_path: Path,
        batch_size: int = 500,
    ) -> int:
        """
        Load pre-computed embeddings + metadata into Chroma.

        Args:
            embeddings_path: Path to embeddings.npy (N x 768)
            metadata_path: Path to chunk_metadata.jsonl (N lines)
            batch_size: Chroma upsert batch size (500 is safe limit)

        Returns:
            Number of vectors indexed.
        """
        logger.info("Loading embeddings and metadata for Chroma indexing")

        embeddings = np.load(embeddings_path)
        metadata_records = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                metadata_records.append(json.loads(line.strip()))

        assert len(embeddings) == len(metadata_records), (
            f"Mismatch: {len(embeddings)} embeddings vs {len(metadata_records)} metadata"
        )

        total = len(embeddings)
        logger.info(f"Indexing {total} vectors into Chroma collection '{self.collection_name}'")

        # Reset collection for clean rebuild (ignore if it doesn't exist yet)
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = None

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)

            ids = []
            documents = []
            metadatas = []
            batch_embeddings = []

            for i in range(start, end):
                record = metadata_records[i]
                meta = record["metadata"]

                # Chroma metadata must be flat (str, int, float, bool)
                flat_meta = _flatten_metadata(meta)

                ids.append(record["chunk_id"])
                documents.append(record["text"])
                metadatas.append(flat_meta)
                batch_embeddings.append(embeddings[i].tolist())

            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=batch_embeddings,
            )

            if (start // batch_size + 1) % 20 == 0:
                logger.info(f"  Indexed {end}/{total} vectors")

        logger.info(f"Chroma indexing complete: {self.count} vectors")
        return self.count

    # ── Searching ───────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = None,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
    ) -> list[dict]:
        """
        Search for similar vectors with optional metadata filtering.

        Args:
            query_embedding: Query vector (768-dim numpy array)
            top_k: Number of results (default from settings)
            where: Chroma metadata filter, e.g. {"journal": "Nature"}
            where_document: Chroma document content filter

        Returns:
            List of result dicts with keys: chunk_id, text, metadata, score
        """
        top_k = top_k or settings.top_k

        query_params = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_params["where"] = where
        if where_document:
            query_params["where_document"] = where_document

        results = self.collection.query(**query_params)

        # Unpack Chroma's nested list format
        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i],  # distance → similarity
            })

        return output

    def search_by_text(
        self,
        query_text: str,
        embedder,
        top_k: int = None,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """Convenience: embed query text then search."""
        query_vec = embedder.embed_single(query_text)
        return self.search(query_vec, top_k=top_k, where=where)

    # ── Utilities ───────────────────────────────────────────────────

    def delete_collection(self) -> None:
        """Delete the collection entirely."""
        self.client.delete_collection(self.collection_name)
        self._collection = None
        logger.info(f"Deleted Chroma collection: {self.collection_name}")

    def get_by_id(self, chunk_id: str) -> Optional[dict]:
        """Retrieve a single chunk by its ID."""
        result = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        if result["ids"]:
            return {
                "chunk_id": result["ids"][0],
                "text": result["documents"][0],
                "metadata": result["metadatas"][0],
            }
        return None


def _flatten_metadata(meta: dict) -> dict:
    """
    Flatten metadata for Chroma storage.
    Chroma only supports str, int, float, bool values.
    Lists get joined as comma-separated strings.
    """
    flat = {}
    for key, value in meta.items():
        if isinstance(value, list):
            flat[key] = ", ".join(str(v) for v in value) if value else ""
        elif isinstance(value, (str, int, float, bool)):
            flat[key] = value
        elif value is None:
            flat[key] = ""
        else:
            flat[key] = str(value)
    return flat
