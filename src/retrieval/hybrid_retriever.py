"""
Hybrid retriever — routes queries to the right vector store.

Strategy:
    - Query has metadata filters (year, journal, MeSH) → Chroma
    - Query is pure similarity search → FAISS (faster)
    - Both results can be merged and deduplicated for ensemble retrieval

This is the component that LangChain's RAG chain will call in Step 5.
It implements a clean interface: query in → ranked documents out.
"""

from typing import Optional

import numpy as np

from config.config import settings
from config.logging_config import get_logger
from src.retrieval.chroma_store import ChromaStore
from src.retrieval.faiss_store import FAISSStore

logger = get_logger(__name__)


class HybridRetriever:
    """
    Routes retrieval queries to Chroma or FAISS based on query characteristics.

    Usage:
        retriever = HybridRetriever(chroma_store, faiss_store)

        # Pure similarity → FAISS (fast)
        results = retriever.retrieve(query_embedding)

        # With metadata filter → Chroma
        results = retriever.retrieve(query_embedding, where={"journal": "Nature"})

        # Ensemble: merge results from both stores
        results = retriever.retrieve_ensemble(query_embedding)
    """

    def __init__(
        self,
        chroma_store: ChromaStore,
        faiss_store: FAISSStore,
    ):
        self.chroma = chroma_store
        self.faiss = faiss_store

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int = None,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
        force_store: Optional[str] = None,
    ) -> list[dict]:
        """
        Retrieve relevant documents, routing to the best store.

        Routing logic:
            1. If force_store is set → use that store
            2. If metadata filters are present → Chroma (only one that supports filtering)
            3. Otherwise → FAISS (faster for pure similarity)

        Args:
            query_embedding: 768-dim query vector
            top_k: Number of results
            where: Metadata filter dict (forces Chroma)
            where_document: Document content filter (forces Chroma)
            force_store: "chroma" or "faiss" to override routing

        Returns:
            List of result dicts: chunk_id, text, metadata, score, source
        """
        top_k = top_k or settings.top_k

        # Determine which store to use
        if force_store == "chroma":
            store_name = "chroma"
        elif force_store == "faiss":
            store_name = "faiss"
        elif where or where_document:
            store_name = "chroma"
        else:
            store_name = "faiss"

        logger.debug(f"Routing query to {store_name} (top_k={top_k})")

        if store_name == "chroma":
            results = self.chroma.search(
                query_embedding, top_k=top_k, where=where, where_document=where_document
            )
        else:
            results = self.faiss.search(query_embedding, top_k=top_k)

        # Tag results with their source
        for r in results:
            r["source"] = store_name

        return results

    def retrieve_ensemble(
        self,
        query_embedding: np.ndarray,
        top_k: int = None,
        where: Optional[dict] = None,
        weights: tuple[float, float] = (0.5, 0.5),
    ) -> list[dict]:
        """
        Merge results from both stores using Reciprocal Rank Fusion (RRF).

        RRF is a robust rank-merging algorithm that doesn't depend on score
        normalization between stores. It ranks items by:

            RRF_score = sum(1 / (k + rank_in_store_i))

        where k=60 (standard constant that dampens high-rank dominance).

        Args:
            query_embedding: 768-dim query vector
            top_k: Final number of merged results
            where: Optional metadata filter (applied to Chroma only)
            weights: (chroma_weight, faiss_weight) for weighted RRF

        Returns:
            Merged, deduplicated, re-ranked results.
        """
        top_k = top_k or settings.top_k
        fetch_k = top_k * 2  # Over-fetch from each store before merging

        # Get results from both stores
        chroma_results = self.chroma.search(
            query_embedding, top_k=fetch_k, where=where
        )
        faiss_results = self.faiss.search(query_embedding, top_k=fetch_k)

        # Apply Reciprocal Rank Fusion
        k = 60  # Standard RRF constant
        rrf_scores: dict[str, float] = {}
        all_results: dict[str, dict] = {}

        for rank, result in enumerate(chroma_results):
            chunk_id = result["chunk_id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + weights[0] / (k + rank + 1)
            all_results[chunk_id] = result

        for rank, result in enumerate(faiss_results):
            chunk_id = result["chunk_id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + weights[1] / (k + rank + 1)
            if chunk_id not in all_results:
                all_results[chunk_id] = result

        # Sort by RRF score and return top_k
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        output = []
        for chunk_id, rrf_score in ranked:
            result = all_results[chunk_id]
            result["rrf_score"] = round(rrf_score, 6)
            result["source"] = "ensemble"
            output.append(result)

        logger.debug(
            f"Ensemble retrieval: {len(chroma_results)} chroma + "
            f"{len(faiss_results)} faiss → {len(output)} merged"
        )
        return output

    def retrieve_by_text(
        self,
        query_text: str,
        embedder,
        top_k: int = None,
        where: Optional[dict] = None,
        ensemble: bool = False,
    ) -> list[dict]:
        """
        Convenience: embed query text, then retrieve.
        This is the main entry point for the RAG chain in Step 5.
        """
        query_vec = embedder.embed_single(query_text)
        if ensemble:
            return self.retrieve_ensemble(query_vec, top_k=top_k, where=where)
        return self.retrieve(query_vec, top_k=top_k, where=where)
