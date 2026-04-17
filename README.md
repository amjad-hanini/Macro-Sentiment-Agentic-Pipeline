# 📈 Autonomous Macro-Sentiment Agentic Pipeline

An end-to-end, automated data engineering pipeline and AI reasoning system that detects, predicts, and explains divergences between macroeconomic reality and public news narratives.

## 🏗️ System Architecture

**🔴 Live Interactive Dashboard:** [Click here to view the Streamlit App](https://macro-sentiment-agentic-pipeline.streamlit.app/)

This project is built on four core pillars of modern data engineering and AI:

1. **Volume (Data Ingestion & Processing):** - Utilizes Python-based MapReduce-style chunking to ingest and clean large-scale historical Reddit WorldNews datasets alongside Yahoo Finance market data (DJIA & VIX) without overloading memory limits.
2. **Prediction (Machine Learning):** - Deploys a local **FinBERT NLP** model to score the sentiment of daily news.
   - Trains a **Random Forest Classifier** to predict the probability of future market anomalies based on historical fear indexes and sentiment divergence.
3. **Agentic RAG (Retrieval-Augmented Generation):** - A multi-agent LLM system (powered by Google Gemini) autonomously detects market anomalies.
   - Integrates **DuckDuckGo Search API** to dynamically scrape the live web for historical context surrounding the anomaly date, feeding a synthesized, hallucination-free report to the user.
4. **Velocity (CI/CD Automation):** - Fully automated via **GitHub Actions**. A nightly CRON job executes the pipeline, updates the SQLite/PostgreSQL database, compiles a LaTeX PDF summary report, and securely commits the fresh data.

## 💻 Tech Stack
* **Language:** Python 3.10
* **Data & ML:** Pandas, Scikit-Learn, PyTorch, HuggingFace (FinBERT)
* **LLM / Agents:** Google Gemini 2.5 Flash, DuckDuckGo Search API
* **Database:** SQLite / PostgreSQL (via SQLAlchemy)
* **Frontend:** Streamlit, Plotly
* **DevOps:** GitHub Actions, LaTeX automation

## 🚀 How to Run Locally

1. Clone this repository:
   ```bash
   git clone [https://github.com/amjad-hanini/Macro-Sentiment-Agentic-Pipeline.git](https://github.com/amjad-hanini/Macro-Sentiment-Agentic-Pipeline.git)
