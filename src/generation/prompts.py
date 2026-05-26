"""
Prompt templates for the Medical RAG pipeline.

Design philosophy:
    - System prompt establishes Claude as a medical literature assistant
    - Citation format uses [PMID:12345] so answers are verifiable
    - Context window is structured: each source is clearly delimited
    - Instructions explicitly tell Claude to say "I don't know" when the
      retrieved context doesn't contain the answer (reduces hallucination)

These prompts are tuned for Claude's instruction-following style.
Switching to GPT would require adjusting the formatting.
"""

# ── System Prompt ───────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a medical literature research assistant. Your role is to answer biomedical questions using ONLY the provided research paper excerpts.

RULES:
1. Answer ONLY based on the provided context. If the context doesn't contain enough information, say "Based on the available literature, I cannot fully answer this question" and explain what's missing.
2. Cite every claim using [PMID:number] format. Each claim must have at least one citation.
3. When multiple sources support a claim, cite all of them: [PMID:111, PMID:222].
4. Distinguish between findings (what studies showed) and conclusions (what authors interpreted). Use language like "Smith et al. found that..." or "According to a study in Nature Medicine..."
5. If sources disagree, present both sides with their respective citations.
6. Do NOT use information from your training data. Only use the provided excerpts.
7. For drug dosages, treatment protocols, or clinical recommendations, always note that the information comes from research literature and should be verified with current clinical guidelines.

RESPONSE FORMAT:
- Start with a direct answer to the question (1-2 sentences)
- Follow with supporting details from the literature, with citations
- End with a "Sources" section listing the PMIDs you cited with their titles"""


# ── Context Formatting ──────────────────────────────────────────────

def format_context(retrieved_docs: list[dict]) -> str:
    """
    Format retrieved documents into a structured context block for Claude.

    Each document gets a clear header with metadata so Claude can cite properly.
    We include journal name and date to help Claude assess source quality.
    """
    if not retrieved_docs:
        return "No relevant documents were found."

    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        metadata = doc.get("metadata", {})
        pmid = metadata.get("pmid", "unknown")
        title = metadata.get("title", "Untitled")
        journal = metadata.get("journal", "Unknown Journal")
        pub_date = metadata.get("pub_date", "Unknown Date")
        score = doc.get("score", 0)

        header = (
            f"[Source {i}] PMID: {pmid} | {journal} | {pub_date} | "
            f"Relevance: {score:.3f}"
        )
        context_parts.append(
            f"{header}\n"
            f"Title: {title}\n"
            f"Content: {doc.get('text', '')}\n"
        )

    return "\n---\n".join(context_parts)


# ── User Query Template ─────────────────────────────────────────────

QUERY_TEMPLATE = """Based on the following medical research excerpts, answer the question below.

RESEARCH EXCERPTS:
{context}

QUESTION: {question}

Remember: Cite every claim with [PMID:number]. Only use information from the excerpts above."""


# ── Follow-up Template (for conversational use) ────────────────────

FOLLOWUP_TEMPLATE = """Based on the same research excerpts provided earlier, plus these additional sources:

ADDITIONAL EXCERPTS:
{context}

FOLLOW-UP QUESTION: {question}

Remember: Cite every claim with [PMID:number]. Only use information from the excerpts."""


# ── Summary Template (for generating article summaries) ─────────────

SUMMARY_TEMPLATE = """Summarize the key findings from the following medical research excerpts. Organize by theme, not by paper.

RESEARCH EXCERPTS:
{context}

Provide a structured summary with:
1. Main findings (with citations)
2. Areas of consensus across studies
3. Contradictions or gaps in the literature
4. Clinical implications (if any)

Cite every claim with [PMID:number]."""
