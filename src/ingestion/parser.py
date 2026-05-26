"""
PubMed XML parser — extracts structured article data from Entrez efetch responses.

Each PubMed record becomes a flat dict with these fields:
    pmid, title, abstract, authors, journal, pub_date, mesh_terms,
    keywords, doi, pub_type, language

Design decision: We parse with xml.etree (stdlib) instead of Biopython's
Entrez.read() because Entrez.read() silently drops some fields and is harder
to debug. Raw XML parsing gives us full control over what we extract.
"""

import xml.etree.ElementTree as ET
from typing import Optional


def parse_pubmed_xml(xml_string: str) -> list[dict]:
    """
    Parse a PubMed efetch XML response into a list of article dicts.

    Args:
        xml_string: Raw XML string from Entrez.efetch (rettype='xml')

    Returns:
        List of article dicts. Articles missing both title and abstract
        are skipped (they're useless for RAG).
    """
    articles = []

    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError:
        return articles

    for article_elem in root.findall(".//PubmedArticle"):
        parsed = _parse_single_article(article_elem)
        if parsed and (parsed["title"] or parsed["abstract"]):
            articles.append(parsed)

    return articles


def _parse_single_article(elem: ET.Element) -> Optional[dict]:
    """Extract all fields from a single PubmedArticle element."""
    try:
        medline = elem.find("MedlineCitation")
        if medline is None:
            return None

        article = medline.find("Article")
        if article is None:
            return None

        # ---- PMID ----
        pmid_elem = medline.find("PMID")
        pmid = pmid_elem.text if pmid_elem is not None else None
        if not pmid:
            return None

        # ---- Title ----
        title = _get_text(article, "ArticleTitle")

        # ---- Abstract ----
        abstract = _extract_abstract(article)

        # ---- Authors ----
        authors = _extract_authors(article)

        # ---- Journal ----
        journal = _get_text(article, "Journal/Title")
        journal_abbrev = _get_text(article, "Journal/ISOAbbreviation")

        # ---- Publication date ----
        pub_date = _extract_pub_date(article)

        # ---- MeSH terms (gold for metadata filtering) ----
        mesh_terms = _extract_mesh_terms(medline)

        # ---- Keywords ----
        keywords = _extract_keywords(medline)

        # ---- DOI ----
        doi = _extract_doi(elem)

        # ---- Publication type ----
        pub_types = [
            pt.text
            for pt in article.findall("PublicationTypeList/PublicationType")
            if pt.text
        ]

        # ---- Language ----
        language = _get_text(article, "Language")

        return {
            "pmid": pmid,
            "title": title or "",
            "abstract": abstract or "",
            "authors": authors,
            "journal": journal or "",
            "journal_abbrev": journal_abbrev or "",
            "pub_date": pub_date,
            "mesh_terms": mesh_terms,
            "keywords": keywords,
            "doi": doi or "",
            "pub_types": pub_types,
            "language": language or "eng",
        }

    except Exception:
        # Never crash on a single article — skip it and continue
        return None


def _get_text(parent: ET.Element, path: str) -> Optional[str]:
    """Safely extract text from an element path."""
    elem = parent.find(path)
    if elem is None:
        return None
    # Handle mixed content (text with inline tags like <i>, <b>)
    return "".join(elem.itertext()).strip() or None


def _extract_abstract(article: ET.Element) -> Optional[str]:
    """
    Extract abstract text, handling structured abstracts (with sections
    like Background, Methods, Results, Conclusions).
    """
    abstract_elem = article.find("Abstract")
    if abstract_elem is None:
        return None

    parts = []
    for text_elem in abstract_elem.findall("AbstractText"):
        label = text_elem.get("Label", "")
        text = "".join(text_elem.itertext()).strip()
        if text:
            if label:
                parts.append(f"{label}: {text}")
            else:
                parts.append(text)

    return " ".join(parts) if parts else None


def _extract_authors(article: ET.Element) -> list[str]:
    """Extract author names as 'LastName FirstName' strings."""
    authors = []
    for author in article.findall("AuthorList/Author"):
        last = _get_text(author, "LastName") or ""
        first = _get_text(author, "ForeName") or ""
        name = f"{last} {first}".strip()
        if name:
            authors.append(name)
    return authors


def _extract_pub_date(article: ET.Element) -> str:
    """
    Extract publication date as 'YYYY-MM-DD' (best effort).
    Falls back to 'YYYY-MM' or 'YYYY' if day/month are missing.
    """
    # Try Journal > JournalIssue > PubDate first (most reliable)
    pub_date = article.find("Journal/JournalIssue/PubDate")
    if pub_date is None:
        return ""

    year = _get_text(pub_date, "Year") or ""
    month = _get_text(pub_date, "Month") or ""
    day = _get_text(pub_date, "Day") or ""

    # Handle MedlineDate (e.g., "2023 Jan-Feb")
    if not year:
        medline_date = _get_text(pub_date, "MedlineDate")
        return medline_date or ""

    # Month might be text ("Jan") — convert to number
    month_num = _month_to_num(month)

    if year and month_num and day:
        return f"{year}-{month_num}-{day.zfill(2)}"
    elif year and month_num:
        return f"{year}-{month_num}"
    else:
        return year


def _month_to_num(month: str) -> str:
    """Convert month name or number to zero-padded number string."""
    if not month:
        return ""
    if month.isdigit():
        return month.zfill(2)
    months = {
        "jan": "01", "feb": "02", "mar": "03", "apr": "04",
        "may": "05", "jun": "06", "jul": "07", "aug": "08",
        "sep": "09", "oct": "10", "nov": "11", "dec": "12",
    }
    return months.get(month[:3].lower(), "")


def _extract_mesh_terms(medline: ET.Element) -> list[str]:
    """
    Extract MeSH descriptor names. These are controlled-vocabulary terms
    assigned by NLM indexers — incredibly valuable for metadata filtering.
    """
    terms = []
    for mesh in medline.findall("MeshHeadingList/MeshHeading"):
        descriptor = mesh.find("DescriptorName")
        if descriptor is not None and descriptor.text:
            terms.append(descriptor.text)
    return terms


def _extract_keywords(medline: ET.Element) -> list[str]:
    """Extract author-supplied keywords."""
    keywords = []
    for kw_list in medline.findall("KeywordList/Keyword"):
        if kw_list.text:
            keywords.append(kw_list.text.strip())
    return keywords


def _extract_doi(article_elem: ET.Element) -> Optional[str]:
    """Extract DOI from article ID list."""
    for id_elem in article_elem.findall(
        "PubmedData/ArticleIdList/ArticleId"
    ):
        if id_elem.get("IdType") == "doi" and id_elem.text:
            return id_elem.text.strip()
    return None
