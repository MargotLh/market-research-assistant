import os
from typing import List, Tuple

import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.retrievers import WikipediaRetriever
from langchain_core.documents import Document

# -----------------------------
# Helpers (testable)
# -----------------------------
def validate_industry(industry_input: str) -> Tuple[bool, str]:
    """
    Q1 requirement: check that an industry is indeed provided.
    No LLM needed for this check.
    """
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
    # Deduplicate while preserving order
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
    return ChatGroq(
        api_key=api_key,
        model=model_id,
        temperature=temperature,
    )


@st.cache_data(show_spinner=False)
def retrieve_wikipedia_docs(query: str, lang: str = "en", max_docs: int = 5) -> List[Document]:
    """
    Q2 requirement: return URLs of the five most relevant Wikipedia pages.
    We use WikipediaRetriever (LangChain) to get the most relevant docs,
    then we derive URLs from titles.
    """
    retriever = WikipediaRetriever(lang=lang, load_max_docs=max_docs)
    # Some people get better results with "X industry" than "X"
    return retriever.invoke(f"{query} industry")


def generate_industry_report(
    llm: ChatGroq,
    industry: str,
    docs: List[Document],
    max_words: int = 500,
) -> str:
    """
    Q3 requirement: < 500 words, based on the five most relevant Wikipedia pages.
    """
    # Keep context compact and consistent
    sources_blocks = []
    for i, d in enumerate(docs[:5], start=1):
        title = (d.metadata or {}).get("title", f"Source {i}")
        snippet = (d.page_content or "")[:2500]  # limit per source
        sources_blocks.append(f"SOURCE [{i}] — {title}\n{snippet}")

    sources_text = "\n\n---\n\n".join(sources_blocks)

    prompt = f"""
You are a business analyst writing a market research brief on the "{industry}" industry.

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
        tighten = f"""
Shorten the report to strictly under {max_words} words.
Keep it accurate and based ONLY on the same sources.
Remove repetition first.
Here is the report to shorten:

{report}
""".strip()
        report = (llm.invoke(tighten).content or "").strip()

    # Final safety: truncate if still too long
    if word_count(report) > max_words:
        report = truncate_to_words(report, max_words)

    return report


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Market Research Assistant", page_icon="📊", layout="wide")
st.title("📊 Market Research Assistant")
st.caption("Generate an industry report grounded in Wikipedia pages (LangChain WikipediaRetriever + Groq LLM).")

# Sidebar (assignment requirement)
with st.sidebar:
    st.header("⚙️ Configuration")

    # Keep only ONE model in the dropdown for the final version (assignment requirement)
    llm_model = st.selectbox(
        "Select LLM Model",
        ["llama-3.1-8b-instant"],
        index=0,
        help="Final submission should include only one LLM option.",
    )

    api_key_input = st.text_input(
        "Enter your API Key",
        type="password",
        help="Paste your Groq API key here (from https://console.groq.com).",
    )

    st.markdown("---")
    st.markdown("### How to use")
    st.markdown(
        """
1) Enter an industry  
2) Click **Find Wikipedia Pages** (Q2)  
3) Paste your Groq key (if not already) and click **Generate Report** (Q3)
"""
    )

# Session state init
if "industry" not in st.session_state:
    st.session_state.industry = ""
if "docs" not in st.session_state:
    st.session_state.docs = []
if "urls" not in st.session_state:
    st.session_state.urls = []
if "report" not in st.session_state:
    st.session_state.report = ""

# Step 1 (Q1)
st.subheader("Step 1 — Provide an industry (Q1)")
with st.form("industry_form", clear_on_submit=False):
    industry_input = st.text_input(
        "What industry would you like to research?",
        value=st.session_state.industry,
        placeholder="e.g., automotive, healthcare, renewable energy",
    )
    find_pages = st.form_submit_button("Find Wikipedia Pages", use_container_width=True)

if find_pages:
    ok, result = validate_industry(industry_input)
    if not ok:
        st.error(result)
        st.stop()

    st.session_state.industry = result
    st.session_state.report = ""  # reset report when industry changes

    with st.spinner("Retrieving relevant Wikipedia pages..."):
        docs = retrieve_wikipedia_docs(st.session_state.industry, lang="en", max_docs=5)
        st.session_state.docs = docs
        st.session_state.urls = docs_to_urls(docs, lang="en")

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

    api_key = api_key_input.strip() or os.getenv("GROQ_API_KEY", "").strip()

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Generate Report", type="primary", use_container_width=True):
            if not api_key:
                st.error("Please enter your Groq API key in the sidebar to generate the report.")
                st.stop()

            with st.spinner("Generating report with Groq..."):
                llm = make_llm(api_key=api_key, model_id=llm_model, temperature=0.2)
                report = generate_industry_report(llm, st.session_state.industry, st.session_state.docs, max_words=500)
                st.session_state.report = report

    with col2:
        if st.button("New Research", use_container_width=True):
            st.session_state.industry = ""
            st.session_state.docs = []
            st.session_state.urls = []
            st.session_state.report = ""
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
