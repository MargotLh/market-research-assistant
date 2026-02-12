# --- Relevance scoring to avoid weird Wikipedia results (e.g., unrelated events) ---
def score_wiki_doc(title: str, content: str, query: str) -> int:
    t = (title or "").lower()
    c = (content or "").lower()
    q = (query or "").lower()

    score = 0

    # Strong positives: query match + core healthcare phrasing
    if q and q in t:
        score += 10
    if "health care" in t or "healthcare" in t:
        score += 8

    # Industry-ish positives
    for kw in ["industry", "sector", "market", "economics", "value chain", "supply chain"]:
        if kw in t:
            score += 5
        if kw in c:
            score += 2

    # Domain positives (health-related)
    for kw in ["medical", "medicine", "hospital", "public health", "health insurance", "pharmaceutical"]:
        if kw in t:
            score += 4
        if kw in c:
            score += 1

    # Negatives (events / media / clearly irrelevant pages)
    for bad in [
        "killing", "murder", "death", "shooting", "attack", "trial", "case",
        "episode", "film", "song", "album", "game"
    ]:
        if bad in t:
            score -= 12

    return score


@st.cache_data(show_spinner=False)
def retrieve_wikipedia_docs(query: str, lang: str = "en", max_docs: int = 5) -> List[Document]:
    """
    Q2: Return 5 relevant Wikipedia pages.
    Fix: pull many candidates, deduplicate, then rerank by an "industry relevance" score.
    """
    retriever = WikipediaRetriever(
        lang=lang,
        top_k_results=25,   # get plenty of candidates (default is 3)
        load_max_docs=25,
    )

    candidate_queries = [
        f"{query} industry",
        f"{query} market",
        f"{query} sector",
        f"{query} health care",
        query,
    ]

    # Collect unique candidates by title
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

    # Rerank by relevance score
    scored = []
    for title, doc in by_title.items():
        scored.append((score_wiki_doc(title, doc.page_content, query), doc))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [d for _, d in scored[:max_docs]]
