# Market Research Assistant

A Streamlit application that generates industry reports based on Wikipedia data using Google Gemini Flash LLM.

## Features

- 🔍 Industry validation using LLM
- 📚 Automatic retrieval of the 5 most relevant Wikipedia pages
- 📊 AI-generated industry reports (under 500 words)
- 🎯 Simple, user-friendly interface

## Setup Instructions

### 1. Get Your Free Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy your API key

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run app.py
```

### 4. Using the App

1. Paste your Gemini API key in the sidebar
2. Enter an industry name (e.g., "automotive", "healthcare")
3. Click "Start Research"
4. View the generated report

## Deployment on Streamlit Cloud

1. Push this code to a GitHub repository
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Sign in and click "New app"
4. Connect your GitHub repo
5. Set the main file as `app.py`
6. Deploy!

## Technical Details

- **LLM**: Google Gemini 2.0 Flash (free tier)
- **Data Source**: Wikipedia via LangChain WikipediaRetriever
- **Framework**: Streamlit

## Project Structure

```
market_research_assistant/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Word Count Compliance

The report is automatically generated to be under 500 words. If it exceeds this limit, the LLM condenses it automatically.
