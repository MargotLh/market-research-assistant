import streamlit as st
import google.generativeai as genai
from langchain_community.retrievers import WikipediaRetriever
import time

# Page configuration
st.set_page_config(
    page_title="Market Research Assistant",
    page_icon="📊",
    layout="wide"
)

# Sidebar for LLM configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # LLM Selection
    llm_model = st.selectbox(
        "Select LLM Model",
        ["gemini-2.0-flash-exp"],  # Can add more models during development
        index=0
    )
    
    # API Key input
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

# Main app title
st.title("📊 Market Research Assistant")
st.markdown("Get comprehensive industry reports based on Wikipedia data")

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'industry' not in st.session_state:
    st.session_state.industry = ""
if 'urls' not in st.session_state:
    st.session_state.urls = []
if 'report' not in st.session_state:
    st.session_state.report = ""

def validate_industry(industry_input, api_key):
    """Step 1: Validate that a proper industry is provided"""
    if not api_key:
        return False, "Please enter your API key in the sidebar first."
    
    if not industry_input or industry_input.strip() == "":
        return False, "Please provide an industry name."
    
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Use LLM to validate if input is indeed an industry
        prompt = f"""Is the following text a valid industry name or sector? 
        
Text: "{industry_input}"

Respond with only "YES" if it's a valid industry/sector (e.g., "automotive", "healthcare", "technology", "retail").
Respond with only "NO" if it's not (e.g., random words, questions, commands).

Response:"""
        
        response = model.generate_content(prompt)
        validation = response.text.strip().upper()
        
        if "YES" in validation:
            return True, industry_input.strip()
        else:
            return False, "This doesn't appear to be a valid industry. Please provide an industry name (e.g., 'automotive', 'healthcare', 'technology')."
    
    except Exception as e:
        return False, f"Error validating industry: {str(e)}"

def get_wikipedia_urls(industry, api_key):
    """Step 2: Get the 5 most relevant Wikipedia URLs"""
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Use WikipediaRetriever to get relevant pages
        retriever = WikipediaRetriever(top_k_results=10, lang="en")
        docs = retriever.get_relevant_documents(industry)
        
        # Extract URLs and summaries
        candidates = []
        for doc in docs:
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                url = doc.metadata['source']
                summary = doc.page_content[:500]  # First 500 chars
                candidates.append({
                    'url': url,
                    'summary': summary
                })
        
        # Use LLM to select the 5 most relevant
        prompt = f"""You are helping with market research for the "{industry}" industry.

Below are Wikipedia pages that might be relevant. Select the 5 MOST relevant URLs for understanding this industry from a business perspective.

Consider:
- Industry overview and definition
- Key companies and players
- Market trends and history
- Related technologies or products
- Economic impact

Candidates:
{chr(10).join([f"{i+1}. {c['url']}" for i, c in enumerate(candidates)])}

Respond with ONLY the 5 URLs you selected, one per line, no numbering or explanations:"""

        response = model.generate_content(prompt)
        selected_urls = [url.strip() for url in response.text.strip().split('\n') if url.strip().startswith('http')]
        
        # Take top 5
        return selected_urls[:5] if len(selected_urls) >= 5 else selected_urls
    
    except Exception as e:
        st.error(f"Error retrieving Wikipedia URLs: {str(e)}")
        return []

def generate_report(industry, urls, api_key):
    """Step 3: Generate industry report based on Wikipedia pages"""
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Retrieve content from URLs
        retriever = WikipediaRetriever(top_k_results=5, lang="en")
        
        # Get content for each URL by extracting page titles
        all_content = []
        for url in urls:
            # Extract Wikipedia page title from URL
            page_title = url.split('/wiki/')[-1].replace('_', ' ')
            try:
                docs = retriever.get_relevant_documents(page_title)
                if docs:
                    all_content.append(docs[0].page_content[:3000])  # First 3000 chars
            except:
                continue
        
        # Combine content
        combined_content = "\n\n---\n\n".join(all_content)
        
        # Generate report
        prompt = f"""You are a business analyst writing a market research report on the {industry} industry.

Based on the Wikipedia content below, write a comprehensive industry report that covers:
1. Industry overview and definition
2. Key players and major companies
3. Market trends and recent developments
4. Challenges and opportunities
5. Future outlook

Requirements:
- Less than 500 words
- Professional business tone
- Well-structured with clear sections
- Based ONLY on the provided Wikipedia content
- Cite specific facts and figures when available

Wikipedia Content:
{combined_content[:15000]}

Write the industry report:"""

        response = model.generate_content(prompt)
        report = response.text.strip()
        
        # Ensure it's under 500 words
        word_count = len(report.split())
        if word_count > 500:
            # Ask LLM to condense
            condense_prompt = f"""The following report is {word_count} words. Please condense it to under 500 words while keeping all key information:

{report}

Condensed report (under 500 words):"""
            response = model.generate_content(condense_prompt)
            report = response.text.strip()
        
        return report
    
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")
        return ""

# Main workflow
st.markdown("---")

# Step 1: Industry Input
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
                
                # Automatically proceed to Step 2
                with st.spinner("🔎 Finding relevant Wikipedia pages..."):
                    urls = get_wikipedia_urls(st.session_state.industry, api_key)
                    if urls:
                        st.session_state.urls = urls
                        st.session_state.step = 3
                        
                        # Automatically proceed to Step 3
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

# Display results if available
if st.session_state.step >= 3 and st.session_state.urls:
    st.markdown("---")
    st.subheader("📚 Step 2: Relevant Wikipedia Pages")
    for i, url in enumerate(st.session_state.urls, 1):
        st.markdown(f"{i}. [{url}]({url})")

if st.session_state.step >= 4 and st.session_state.report:
    st.markdown("---")
    st.subheader("📊 Step 3: Industry Report")
    st.markdown(st.session_state.report)
    
    # Word count
    word_count = len(st.session_state.report.split())
    st.caption(f"📝 Word count: {word_count} words")
    
    # Download button
    st.download_button(
        label="📥 Download Report",
        data=st.session_state.report,
        file_name=f"{st.session_state.industry}_report.txt",
        mime="text/plain"
    )

# Reset button
if st.session_state.step > 1:
    if st.button("🔄 New Research", use_container_width=True):
        st.session_state.step = 1
        st.session_state.industry = ""
        st.session_state.urls = []
        st.session_state.report = ""
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
    Built with Streamlit & Google Gemini Flash | Market Research Assistant
    </div>
    """,
    unsafe_allow_html=True
)
