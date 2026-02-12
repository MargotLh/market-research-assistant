# Market Research Assistant

A Streamlit app that produces an industry report grounded in Wikipedia pages.

- **Retriever (Q2):** LangChain `WikipediaRetriever` (returns the most relevant Wikipedia pages)
- **LLM (Q3):** Groq via `langchain-groq` (default: `llama-3.1-8b-instant`)
- **UI:** Streamlit sidebar includes:
  - a dropdown for selecting the LLM (final version: one option)
  - a text field for entering the API key

## Features
- Step 1: Check that an industry is provided (no LLM needed)
- Step 2: Return URLs of the 5 most relevant Wikipedia pages
- Step 3: Generate an industry report **under 500 words** using only the retrieved Wikipedia content

## Setup

### 1) Get a Groq API key
Create a key at: https://console.groq.com

### 2) Install dependencies
```bash
pip install -r requirements.txt

