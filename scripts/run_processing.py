#!/usr/bin/env python3
"""
Run the chunking + embedding pipeline (Step 3).

Usage:
    python scripts/run_processing.py                        # Full pipeline
    python scripts/run_processing.py --chunk-only           # Just chunking
    python scripts/run_processing.py --embed-only           # Just embedding (chunks must exist)
    python scripts/run_processing.py --input data/raw/pubmed_articles.jsonl

Requires: Step 2 output (data/raw/pubmed_articles.jsonl) to exist.
Produces:
    - data/processed/chunks.jsonl       (chunked text + metadata)
    - data/embeddings/embeddings.npy    (numpy vectors, shape Nx768)
    - data/embeddings/chunk_metadata.jsonl (metadata aligned with vectors)
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import settings
from config.logging_config import get_logger

logger = get_logger("run_processing")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk and embed PubMed articles")
    parser.add_argument(
        "--input", type=Path,
        default=settings.raw_data_dir / "pubmed_articles.jsonl",
        help="Input JSONL from Step 2",
    )
    parser.add_argument("--chunk-only", action="store_true", help="Run chunking only")
    parser.add_argument("--embed-only", action="store_true", help="Run embedding only")
    parser.add_argument(
        "--chunk-size", type=int, default=None,
        help=f"Tokens per chunk (default: {settings.chunk_size})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help=f"Embedding batch size (default: {settings.embedding_batch_size})",
    )
    return parser.parse_args()


def run_chunking(input_path: Path, chunk_size: int = None) -> Path:
    """Run the chunking step."""
    from src.processing.chunker import chunk_articles_from_jsonl

    print("\n📄 Step 3a: Chunking articles")
    print(f"   Input:      {input_path}")
    print(f"   Chunk size: {chunk_size or settings.chunk_size} tokens")
    print(f"   Overlap:    {settings.chunk_overlap} tokens")
    print()

    start = time.time()
    chunks_path = chunk_articles_from_jsonl(input_path, chunk_size=chunk_size)
    elapsed = time.time() - start

    # Count results
    with open(chunks_path) as f:
        count = sum(1 for _ in f)

    print(f"\n   ✓ Created {count:,} chunks in {elapsed:.1f}s")
    print(f"   Output: {chunks_path}")
    return chunks_path


def run_embedding(chunks_path: Path, batch_size: int = None) -> None:
    """Run the embedding step."""
    from src.processing.embedder import embed_chunks_from_jsonl

    print(f"\n🧠 Step 3b: Embedding with BioBERT ({settings.embedding_model})")
    print(f"   Input:      {chunks_path}")
    print(f"   Device:     {settings.embedding_device}")
    print(f"   Batch size: {batch_size or settings.embedding_batch_size}")
    print()

    start = time.time()
    emb_path, meta_path = embed_chunks_from_jsonl(chunks_path, batch_size=batch_size)
    elapsed = time.time() - start

    import numpy as np
    emb = np.load(emb_path)

    print(f"\n   ✓ Embedded {emb.shape[0]:,} chunks → {emb.shape[1]}d vectors in {elapsed:.1f}s")
    print(f"   Rate:       {emb.shape[0] / max(elapsed, 1):.0f} chunks/sec")
    print(f"   Vectors:    {emb_path} ({emb.nbytes / 1e6:.1f} MB)")
    print(f"   Metadata:   {meta_path}")


def main() -> int:
    args = parse_args()
    settings.ensure_directories()

    chunks_path = settings.processed_data_dir / "chunks.jsonl"

    # Validate input
    if not args.embed_only and not args.input.exists():
        print(f"\n  ✗ Input file not found: {args.input}")
        print("    Run Step 2 first: python scripts/run_ingestion.py")
        return 1

    if args.embed_only and not chunks_path.exists():
        print(f"\n  ✗ Chunks file not found: {chunks_path}")
        print("    Run chunking first: python scripts/run_processing.py --chunk-only")
        return 1

    total_start = time.time()

    try:
        if not args.embed_only:
            chunks_path = run_chunking(args.input, args.chunk_size)

        if not args.chunk_only:
            run_embedding(chunks_path, args.batch_size)

    except KeyboardInterrupt:
        print("\n\n  Interrupted.")
        return 1
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        print(f"\n  Error: {e}")
        return 1

    elapsed = time.time() - total_start
    minutes, seconds = divmod(elapsed, 60)
    print(f"\n{'='*50}")
    print(f"  Step 3 complete in {int(minutes)}m {int(seconds)}s")
    print(f"{'='*50}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
