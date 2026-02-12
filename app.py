import os
from typing import List, Tuple

import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.retrievers import WikipediaRetriever
from langchain_core.documents import Document


# -----------------------------
# Helpers
# -----------------------------
def validate_industry(industry_input: str) -> Tuple[bool, str]:
    """Q1: check that an industry is provided."""
    if industry_input is None:
        return False, "Please provide an industry name."
    cleaned = industry_input.strip()
    if not cleaned:
        return False, "Please provide an industry name (e.g., automotive, healthcare, retail)."
    return True, cleaned


def build_wikipedia_url(title: str, lang: str = "en") -> str:
    safe_title = (title or "").strip().replace(" ", "_")
    return f"https://{lang}.wikipedia.org/wiki/{safe_title}"


def docs_to_urls(docs: List[Document], lang: str = "en") -> List[str]:
    urls: List[str] = []
    for d in docs:
        title = (d.metadata or {}).get("title", "") if hasattr(d, "metadata") else ""
        if title:
            urls.append(build_wikipedia_url(title, lang=lang))
    seen = set()
    unique = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


def word_count(text: str) -> int:
    return len([w for w in (text or "").split() if w.strip()])


def truncate_to_words(text: str, max_words: int) -> str:
    words = [w for w in (text or "").split() if w.strip()]
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).strip()


def make_llm(api_key: str, model_id: str, temperature: float = 0.2) -> ChatGroq:
    return ChatGroq(api_key=api_key, model=model_id, temperature=temperature)


# -----------------------------
# Wikipedia retrieval + LLM reranking
# -----------------------------
@st.cache_data(show_spinner=False)
def retrieve_wikipedia_docs(query: str, lang: str = "en", max_docs: int = 5) -> List[Document]:
    """
    Q2: Retrieve Wikipedia candidates then deduplicate by title.
    LLM reranking is done separately (needs api_key).
    """
    retriever = WikipediaRetriever(lang=lang, top_k_results=8, load_max_docs=8)

    candidate_queries = [
        f"{query} industry",
        f"{query} market",
        f"{query} sector",
        query,
    ]

    by_title = {}
    for q in candidate_queries:
        try:
            docs = retriever.invoke(q)
        except Exception:
            docs = []
        for d in docs:
            title = (d.metadata or {}).get("title", "")
            if title and title not in by_title:
                by_title[title] = d

    return list(by_title.values())


def rerank_with_llm(llm: ChatGroq, industry: str, docs: List[Document], max_docs: int = 5) -> List[Document]:
    """
    Use the LLM to select the most relevant docs for the given industry.
    """
    if len(docs) <= max_docs:
        return docs

    titles = [d.metadata.get("title", f"Doc {i}") for i, d in enumerate(docs)]
    titles_str = "\n".join([f"{i+1}. {t}" for i, t in enumerate(titles)])

    prompt = f"""You are a market research analyst. A user wants to research the "{industry}" industry.

Below is a list of Wikipedia article titles retrieved for this query.
Select the {max_docs} titles that are MOST relevant for understanding the "{industry}" industry from a business perspective (market size, companies, trends, etc.).

Titles:
{titles_str}

Reply with ONLY the numbers of the {max_docs} most relevant titles, comma-separated (e.g. 1,3,5,7,9). Nothing else."""

    response = (llm.invoke(prompt).content or "").strip()

    # Parse the numbers returned by the LLM
    selected_indices = []
    for part in response.replace(" ", "").split(","):
        try:
            idx = int(part) - 1
            if 0 <= idx < len(docs):
                selected_indices.append(idx)
        except ValueError:
            continue

    # Fall back to first max_docs if parsing fails
    if not selected_indices:
        return docs[:max_docs]

    return [docs[i] for i in selected_indices[:max_docs]]


def generate_industry_report(llm: ChatGroq, industry: str, docs: List[Document], max_words: int = 500) -> str:
    """Q3: report < 500 words, grounded in the 5 retrieved Wikipedia pages."""
    sources_blocks = []
    for i, d in enumerate(docs[:5], start=1):
        title = (d.metadata or {}).get("title", f"Source {i}")
        snippet = (d.page_content or "")[:2500]
        sources_blocks.append(f"SOURCE [{i}] — {title}\n{snippet}")

    sources_text = "\n\n---\n\n".join(sources_blocks)

    prompt = f"""You are a business analyst writing a market research brief on the "{industry}" industry.

Use ONLY the information in the SOURCES below. Do not add outside facts.
Write a clear industry report covering:
1) Industry overview
2) Key segments / value chain
3) Major players (only if mentioned in sources)
4) Trends
5) Challenges & opportunities
6) Near-term outlook

Requirements:
- Professional business tone
- Strictly under {max_words} words
- If something is not in sources, say "Not specified in the sources".

SOURCES:
{sources_text}
""".strip()

    report = (llm.invoke(prompt).content or "").strip()

    if word_count(report) > max_words:
        tighten = f"""Shorten the report to strictly under {max_words} words.
Keep it accurate and based ONLY on the same sources. Remove repetition first.

REPORT TO SHORTEN:
{report}""".strip()
        report = (llm.invoke(tighten).content or "").strip()

    if word_count(report) > max_words:
        report = truncate_to_words(report, max_words)

    return report


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Market Research Assistant", page_icon="📊", layout="wide")
st.title("📊 Market Research Assistant")
st.caption("Generate an industry report grounded in Wikipedia pages (WikipediaRetriever + Groq LLM).")

with st.sidebar:
    st.header("⚙️ Configuration")

    llm_model = st.selectbox(
        "Select LLM Model",
        ["llama-3.1-8b-instant"],
        index=0,
    )

    api_key_input = st.text_input(
        "Enter your API Key",
        type="password",
        help="Paste your Groq API key here (from https://console.groq.com).",
    )

    st.markdown("---")
    st.markdown("### How to use")
    st.markdown("""
1) Enter an industry  
2) Click **Find Wikipedia Pages** (Q2)  
3) Click **Generate Report** (Q3)
""")

# Session state
for key in ["industry", "docs", "urls", "report"]:
    if key not in st.session_state:
        st.session_state[key] = "" if key in ["industry", "report"] else []

# Step 1 (Q1)
st.subheader("Step 1 — Provide an industry (Q1)")
with st.form("industry_form", clear_on_submit=False):
    industry_input = st.text_input(
        "What industry would you like to research?",
        value=st.session_state.industry,
        placeholder="e.g., automotive, healthcare, beauty",
    )
    find_pages = st.form_submit_button("Find Wikipedia Pages", use_container_width=True)

if find_pages:
    ok, cleaned = validate_industry(industry_input)
    if not ok:
        st.error(cleaned)
        st.stop()

    api_key = api_key_input.strip()
    if not api_key:
        st.error("Please enter your Groq API key in the sidebar first.")
        st.stop()

    st.session_state.industry = cleaned
    st.session_state.report = ""

    with st.spinner("Retrieving Wikipedia candidates..."):
        all_docs = retrieve_wikipedia_docs(cleaned, lang="en", max_docs=5)

    with st.spinner("Selecting the 5 most relevant pages..."):
        llm = make_llm(api_key=api_key, model_id=llm_model, temperature=0.2)
        selected_docs = rerank_with_llm(llm, cleaned, all_docs, max_docs=5)
        st.session_state.docs = selected_docs
        st.session_state.urls = docs_to_urls(selected_docs, lang="en")

# Step 2 (Q2)
if st.session_state.urls:
    st.markdown("---")
    st.subheader("Step 2 — Five most relevant Wikipedia pages (Q2)")
    for i, url in enumerate(st.session_state.urls[:5], start=1):
        st.markdown(f"{i}. {url}")

# Step 3 (Q3)
if st.session_state.docs:
    st.markdown("---")
    st.subheader("Step 3 — Generate industry report (Q3)")

    api_key = api_key_input.strip()

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Generate Report", type="primary", use_container_width=True):
            if not api_key:
                st.error("Please enter your Groq API key in the sidebar.")
                st.stop()
            with st.spinner("Generating report..."):
                llm = make_llm(api_key=api_key, model_id=llm_model, temperature=0.2)
                st.session_state.report = generate_industry_report(
                    llm, st.session_state.industry, st.session_state.docs, max_words=500
                )
    with col2:
        if st.button("New Research", use_container_width=True):
            for key in ["industry", "docs", "urls", "report"]:
                st.session_state[key] = "" if key in ["industry", "report"] else []
            st.rerun()

# Display report
if st.session_state.report:
    st.markdown("---")
    st.subheader("Industry Report")
    st.write(st.session_state.report)
    wc = word_count(st.session_state.report)
    st.caption(f"Word count: {wc} (limit: 500)")
    st.download_button(
        label="Download Report (.txt)",
        data=st.session_state.report,
        file_name=f"{st.session_state.industry.replace(' ', '_')}_report.txt",
        mime="text/plain",
        use_container_width=True,
    )

st.markdown("---")
st.caption("Built with Streamlit + LangChain WikipediaRetriever + Groq (Llama 3.1 8B Instant).")
