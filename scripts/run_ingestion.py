#!/usr/bin/env python3
"""
Run PubMed article ingestion.

Usage:
    python scripts/run_ingestion.py                    # Full 50K ingestion
    python scripts/run_ingestion.py --resume           # Resume interrupted run
    python scripts/run_ingestion.py --count 1000       # Quick test with 1K
    python scripts/run_ingestion.py --query "covid-19"  # Specific topic

The output is a JSONL file at data/raw/pubmed_articles.jsonl.
Each line is a JSON object with: pmid, title, abstract, authors,
journal, pub_date, mesh_terms, keywords, doi, pub_types, language.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import settings
from config.logging_config import get_logger
from src.ingestion.pubmed_fetcher import PubMedFetcher

logger = get_logger("run_ingestion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest PubMed articles into JSONL format"
    )
    parser.add_argument(
        "--count", type=int, default=None,
        help=f"Number of articles to fetch (default: {settings.pubmed_target_count})"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help=f"Batch size for efetch calls (default: {settings.pubmed_batch_size})"
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="PubMed search query (default: broad biomedical)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint"
    )
    return parser.parse_args()


def print_summary(output_file: Path, elapsed: float) -> None:
    """Print ingestion summary statistics."""
    if not output_file.exists():
        print("\n  No output file found.")
        return

    # Count articles and sample a few
    total = 0
    sample = []
    with open(output_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            total += 1
            if i < 3:
                sample.append(json.loads(line))

    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)

    print("\n" + "=" * 60)
    print("  INGESTION COMPLETE")
    print("=" * 60)
    print(f"  Articles saved:  {total:,}")
    print(f"  Output file:     {output_file}")
    print(f"  File size:       {output_file.stat().st_size / (1024*1024):.1f} MB")
    print(f"  Time elapsed:    {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"  Rate:            {total / max(elapsed, 1):.1f} articles/sec")
    print("=" * 60)

    if sample:
        print("\n  Sample articles:")
        for i, article in enumerate(sample, 1):
            title = article.get("title", "")[:70]
            pmid = article.get("pmid", "?")
            mesh = article.get("mesh_terms", [])[:3]
            print(f"\n  [{i}] PMID {pmid}")
            print(f"      {title}...")
            if mesh:
                print(f"      MeSH: {', '.join(mesh)}")
    print()


def main() -> int:
    args = parse_args()

    # Ensure directories exist
    settings.ensure_directories()

    print("\n🧬 PubMed Ingestion Pipeline")
    print(f"   Target:     {args.count or settings.pubmed_target_count:,} articles")
    print(f"   Batch size: {args.batch_size or settings.pubmed_batch_size}")
    print(f"   Resume:     {args.resume}")
    print(f"   API key:    {'configured' if settings.ncbi_api_key else 'not set (slower)'}")
    print()

    fetcher = PubMedFetcher(
        target_count=args.count,
        batch_size=args.batch_size,
        query=args.query,
    )

    start_time = time.time()

    try:
        output_file = fetcher.run(resume=args.resume)
    except KeyboardInterrupt:
        print("\n\n  Interrupted. Run with --resume to continue.\n")
        return 1
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        print(f"\n  Error: {e}")
        print("  Check logs at: {settings.log_file}")
        return 1

    elapsed = time.time() - start_time
    print_summary(output_file, elapsed)

    return 0


if __name__ == "__main__":
    sys.exit(main())
