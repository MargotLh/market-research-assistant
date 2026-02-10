"""
Test script for Market Research Assistant
This script helps verify the core functionality without running the full Streamlit app
"""

import google.generativeai as genai
from langchain_community.retrievers import WikipediaRetriever

def test_wikipedia_retrieval():
    """Test Wikipedia retrieval"""
    print("Testing Wikipedia retrieval...")
    try:
        retriever = WikipediaRetriever(top_k_results=5, lang="en")
        docs = retriever.get_relevant_documents("automotive industry")
        print(f"✅ Retrieved {len(docs)} documents")
        if docs:
            print(f"   First doc URL: {docs[0].metadata.get('source', 'N/A')}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_gemini_connection(api_key):
    """Test Gemini API connection"""
    print("\nTesting Gemini API connection...")
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content("Say 'Hello, the connection works!'")
        print(f"✅ Gemini response: {response.text[:50]}...")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_industry_validation(api_key):
    """Test industry validation logic"""
    print("\nTesting industry validation...")
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Test valid industry
        prompt = """Is the following text a valid industry name or sector? 

Text: "healthcare"

Respond with only "YES" if it's a valid industry/sector.
Respond with only "NO" if it's not.

Response:"""
        
        response = model.generate_content(prompt)
        result = response.text.strip().upper()
        
        if "YES" in result:
            print("✅ Valid industry correctly identified")
            return True
        else:
            print("❌ Valid industry not recognized")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("Market Research Assistant - Test Suite")
    print("="*60)
    
    # Test 1: Wikipedia (no API key needed)
    test_wikipedia_retrieval()
    
    # Test 2 & 3: Gemini tests (require API key)
    print("\n" + "="*60)
    api_key = input("\nEnter your Gemini API key to test LLM features (or press Enter to skip): ").strip()
    
    if api_key:
        test_gemini_connection(api_key)
        test_industry_validation(api_key)
    else:
        print("\n⚠️  Skipping LLM tests (no API key provided)")
    
    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)
