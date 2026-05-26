"""
Tests for src/ingestion/parser.py

Run: pytest tests/test_parser.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.parser import parse_pubmed_xml


# Realistic PubMed XML snippet (2 articles, one structured abstract)
SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE PubmedArticleSet PUBLIC "-//NLM//DTD PubMedArticle, 1st January 2024//EN"
 "https://dtd.nlm.nih.gov/ncbi/pubmed/out/pubmed_240101.dtd">
<PubmedArticleSet>

  <PubmedArticle>
    <MedlineCitation Status="MEDLINE" Owner="NLM">
      <PMID Version="1">12345678</PMID>
      <Article PubModel="Print">
        <Journal>
          <ISOAbbreviation>Nat Med</ISOAbbreviation>
          <JournalIssue>
            <PubDate>
              <Year>2023</Year>
              <Month>Jun</Month>
              <Day>15</Day>
            </PubDate>
          </JournalIssue>
          <Title>Nature Medicine</Title>
        </Journal>
        <ArticleTitle>Deep learning for drug discovery: a review.</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">Drug discovery is expensive.</AbstractText>
          <AbstractText Label="METHODS">We reviewed 500 papers.</AbstractText>
          <AbstractText Label="RESULTS">Deep learning reduced costs by 30%.</AbstractText>
          <AbstractText Label="CONCLUSIONS">AI transforms pharma.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author>
            <LastName>Smith</LastName>
            <ForeName>John A</ForeName>
          </Author>
          <Author>
            <LastName>Chen</LastName>
            <ForeName>Wei</ForeName>
          </Author>
        </AuthorList>
        <Language>eng</Language>
        <PublicationTypeList>
          <PublicationType>Journal Article</PublicationType>
          <PublicationType>Review</PublicationType>
        </PublicationTypeList>
      </Article>
      <MeshHeadingList>
        <MeshHeading>
          <DescriptorName>Drug Discovery</DescriptorName>
        </MeshHeading>
        <MeshHeading>
          <DescriptorName>Deep Learning</DescriptorName>
        </MeshHeading>
      </MeshHeadingList>
      <KeywordList>
        <Keyword>artificial intelligence</Keyword>
        <Keyword>molecular screening</Keyword>
      </KeywordList>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">12345678</ArticleId>
        <ArticleId IdType="doi">10.1038/s41591-023-00001-x</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>

  <PubmedArticle>
    <MedlineCitation Status="MEDLINE" Owner="NLM">
      <PMID Version="1">87654321</PMID>
      <Article PubModel="Electronic">
        <Journal>
          <ISOAbbreviation>Lancet</ISOAbbreviation>
          <JournalIssue>
            <PubDate>
              <Year>2024</Year>
              <Month>01</Month>
            </PubDate>
          </JournalIssue>
          <Title>The Lancet</Title>
        </Journal>
        <ArticleTitle>Global burden of antimicrobial resistance.</ArticleTitle>
        <Abstract>
          <AbstractText>Antimicrobial resistance (AMR) causes 1.27 million deaths annually. This study analyzes trends across 204 countries.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author>
            <LastName>Patel</LastName>
            <ForeName>Priya</ForeName>
          </Author>
        </AuthorList>
        <Language>eng</Language>
        <PublicationTypeList>
          <PublicationType>Journal Article</PublicationType>
        </PublicationTypeList>
      </Article>
      <MeshHeadingList>
        <MeshHeading>
          <DescriptorName>Anti-Bacterial Agents</DescriptorName>
        </MeshHeading>
      </MeshHeadingList>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">87654321</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>

</PubmedArticleSet>
"""


def test_parse_returns_two_articles():
    articles = parse_pubmed_xml(SAMPLE_XML)
    assert len(articles) == 2


def test_first_article_fields():
    articles = parse_pubmed_xml(SAMPLE_XML)
    a = articles[0]

    assert a["pmid"] == "12345678"
    assert a["title"] == "Deep learning for drug discovery: a review."
    assert a["journal"] == "Nature Medicine"
    assert a["journal_abbrev"] == "Nat Med"
    assert a["language"] == "eng"
    assert a["doi"] == "10.1038/s41591-023-00001-x"


def test_structured_abstract():
    """Structured abstracts should include section labels."""
    articles = parse_pubmed_xml(SAMPLE_XML)
    abstract = articles[0]["abstract"]

    assert "BACKGROUND:" in abstract
    assert "METHODS:" in abstract
    assert "RESULTS:" in abstract
    assert "CONCLUSIONS:" in abstract
    assert "Drug discovery is expensive." in abstract


def test_authors():
    articles = parse_pubmed_xml(SAMPLE_XML)
    authors = articles[0]["authors"]

    assert len(authors) == 2
    assert "Smith John A" in authors
    assert "Chen Wei" in authors


def test_pub_date_full():
    articles = parse_pubmed_xml(SAMPLE_XML)
    assert articles[0]["pub_date"] == "2023-06-15"


def test_pub_date_partial():
    """When day is missing, should return YYYY-MM."""
    articles = parse_pubmed_xml(SAMPLE_XML)
    assert articles[1]["pub_date"] == "2024-01"


def test_mesh_terms():
    articles = parse_pubmed_xml(SAMPLE_XML)
    mesh = articles[0]["mesh_terms"]

    assert "Drug Discovery" in mesh
    assert "Deep Learning" in mesh


def test_keywords():
    articles = parse_pubmed_xml(SAMPLE_XML)
    kw = articles[0]["keywords"]

    assert "artificial intelligence" in kw
    assert "molecular screening" in kw


def test_pub_types():
    articles = parse_pubmed_xml(SAMPLE_XML)
    assert "Journal Article" in articles[0]["pub_types"]
    assert "Review" in articles[0]["pub_types"]


def test_second_article_no_doi():
    """Second article has no DOI — should return empty string."""
    articles = parse_pubmed_xml(SAMPLE_XML)
    assert articles[1]["doi"] == ""


def test_empty_xml():
    """Malformed or empty XML should return empty list, not crash."""
    assert parse_pubmed_xml("") == []
    assert parse_pubmed_xml("<PubmedArticleSet></PubmedArticleSet>") == []
    assert parse_pubmed_xml("not xml at all") == []


def test_article_without_abstract_is_skipped():
    """Articles without abstracts are useless for RAG — skip them."""
    xml = """<PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>99999</PMID>
          <Article>
            <Journal><JournalIssue><PubDate><Year>2023</Year></PubDate></JournalIssue><Title>J</Title></Journal>
            <ArticleTitle></ArticleTitle>
            <AuthorList/>
            <Language>eng</Language>
            <PublicationTypeList/>
          </Article>
        </MedlineCitation>
      </PubmedArticle>
    </PubmedArticleSet>"""
    assert parse_pubmed_xml(xml) == []
