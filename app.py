import streamlit as st
from google import genai
import wikipedia

# Page configuration
st.set_page_config(
    page_title="Market Research Assistant",
    page_icon="📊",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    llm_model = st.selectbox(
        "Select LLM Model",
        ["gemini-2.0-flash"],
        index=0
    )

    api_key = st.text_input(
        "Enter your API Key",
        type="password",
        help="Get your free API key from https://aistudio.google.com/apikey"
    )

    st.markdown("---")
    st.markdown("### 📌 How to use")
    st.markdown("""
    1. Enter your Gemini API key
    2. Type an industry name
    3. Get your market research report
    """)

# Title
st.title("📊 Market Research Assistant")
st.markdown("Get comprehensive industry reports based on Wikipedia data")

# Session state
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'industry' not in st.session_state:
    st.session_state.industry = ""
if 'urls' not in st.session_state:
    st.session_state.urls = []
if 'report' not in st.session_state:
    st.session_state.report = ""

def get_llm_response(api_key, prompt):
    """Call Gemini API using new google-genai library"""
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return response.text.strip()

def validate_industry(industry_input, api_key):
    """Step 1: Validate that a proper industry is provided"""
    if not api_key:
        return False, "Please enter your API key in the sidebar first."
    if not industry_input or industry_input.strip() == "":
        return False, "Please provide an industry name."
    try:
        prompt = f"""Is the following text a valid industry name or sector?

Text: "{industry_input}"

Respond with only "YES" if it's a valid industry/sector (e.g., "automotive", "healthcare", "technology", "retail").
Respond with only "NO" if it's not (e.g., random words, questions, commands).

Response:"""
        result = get_llm_response(api_key, prompt).upper()
        if "YES" in result:
            return True, industry_input.strip()
        else:
            return False, "This doesn't appear to be a valid industry. Please provide an industry name (e.g., 'automotive', 'healthcare', 'technology')."
    except Exception as e:
        return False, f"Error validating industry: {str(e)}"

def get_wikipedia_urls(industry, api_key):
    """Step 2: Get the 5 most relevant Wikipedia URLs"""
    try:
        search_results = wikipedia.search(industry, results=10)
        candidates = []
        for title in search_results:
            url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            candidates.append(url)

        prompt = f"""You are helping with market research for the "{industry}" industry.

Select the 5 MOST relevant Wikipedia URLs for understanding this industry from a business perspective.

Candidates:
{chr(10).join([f"{i+1}. {url}" for i, url in enumerate(candidates)])}

Respond with ONLY the 5 URLs, one per line, no numbering or explanations:"""

        response = get_llm_response(api_key, prompt)
        selected = [u.strip() for u in response.split('\n') if u.strip().startswith('http')]
        return selected[:5] if len(selected) >= 5 else selected

    except Exception as e:
        st.error(f"Error retrieving Wikipedia URLs: {str(e)}")
        return []

def generate_report(industry, urls, api_key):
    """Step 3: Generate industry report from Wikipedia content"""
    try:
        all_content = []
        for url in urls:
            title = url.split('/wiki/')[-1].replace('_', ' ')
            try:
                page = wikipedia.page(title, auto_suggest=False)
                all_content.append(page.content[:3000])
            except:
                try:
                    results = wikipedia.search(title, results=1)
                    if results:
                        page = wikipedia.page(results[0], auto_suggest=False)
                        all_content.append(page.content[:3000])
                except:
                    continue

        combined = "\n\n---\n\n".join(all_content)

        prompt = f"""You are a business analyst writing a market research report on the {industry} industry.

Based on the Wikipedia content below, write a comprehensive industry report covering:
1. Industry overview
2. Key players and companies
3. Market trends
4. Challenges and opportunities
5. Future outlook

Requirements:
- Less than 500 words
- Professional business tone
- Based ONLY on the provided content

Wikipedia Content:
{combined[:12000]}

Write the industry report:"""

        report = get_llm_response(api_key, prompt)

        if len(report.split()) > 500:
            condense = f"Condense this report to under 500 words, keeping all key information:\n\n{report}"
            report = get_llm_response(api_key, condense)

        return report

    except Exception as e:
        st.error(f"Error generating report: {str(e)}")
        return ""

# Main workflow
st.markdown("---")
st.subheader("🔍 Step 1: Enter Industry")
industry_input = st.text_input(
    "What industry would you like to research?",
    value=st.session_state.industry,
    placeholder="e.g., automotive, healthcare, renewable energy"
)

if st.button("🚀 Start Research", type="primary", use_container_width=True):
    if not api_key:
        st.error("⚠️ Please enter your API key in the sidebar first!")
    else:
        with st.spinner("Validating industry..."):
            is_valid, result = validate_industry(industry_input, api_key)

        if is_valid:
            st.success(f"✅ Valid industry: {result}")
            st.session_state.industry = result
            st.session_state.step = 2

            with st.spinner("🔎 Finding relevant Wikipedia pages..."):
                urls = get_wikipedia_urls(st.session_state.industry, api_key)

            if urls:
                st.session_state.urls = urls
                st.session_state.step = 3

                with st.spinner("📝 Generating industry report..."):
                    report = generate_report(st.session_state.industry, st.session_state.urls, api_key)

                if report:
                    st.session_state.report = report
                    st.session_state.step = 4
                else:
                    st.error("Failed to generate report. Please try again.")
            else:
                st.error("Failed to retrieve Wikipedia URLs. Please try again.")
        else:
            st.error(f"❌ {result}")

# Display results
if st.session_state.step >= 3 and st.session_state.urls:
    st.markdown("---")
    st.subheader("📚 Step 2: Relevant Wikipedia Pages")
    for i, url in enumerate(st.session_state.urls, 1):
        st.markdown(f"{i}. [{url}]({url})")

if st.session_state.step >= 4 and st.session_state.report:
    st.markdown("---")
    st.subheader("📊 Step 3: Industry Report")
    st.markdown(st.session_state.report)
    word_count = len(st.session_state.report.split())
    st.caption(f"📝 Word count: {word_count} words")
    st.download_button(
        label="📥 Download Report",
        data=st.session_state.report,
        file_name=f"{st.session_state.industry}_report.txt",
        mime="text/plain"
    )

if st.session_state.step > 1:
    if st.button("🔄 New Research", use_container_width=True):
        st.session_state.step = 1
        st.session_state.industry = ""
        st.session_state.urls = []
        st.session_state.report = ""
        st.rerun()

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#666;font-size:0.9em;'>Built with Streamlit & Google Gemini Flash | Market Research Assistant</div>",
    unsafe_allow_html=True
)
