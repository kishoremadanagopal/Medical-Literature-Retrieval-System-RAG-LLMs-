"""PubMed data ingestion — fetcher + XML parser."""

from src.ingestion.parser import parse_pubmed_xml
from src.ingestion.pubmed_fetcher import PubMedFetcher

__all__ = ["PubMedFetcher", "parse_pubmed_xml"]
