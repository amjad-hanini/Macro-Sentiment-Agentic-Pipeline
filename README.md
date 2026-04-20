# 📈 Autonomous Macro-Sentiment Agentic Pipeline

An end-to-end, automated data engineering pipeline and AI reasoning system that detects, predicts, and explains divergences between macroeconomic reality and public news narratives.

### 🌟 Enterprise-Grade Features
* **🌐 Live Data Ingestion:** Automated nightly web scraping (DuckDuckGo Search) captures real-time global financial headlines to update the AI's short-term memory.
* **🤖 Multi-Agent AI Debate Framework:** Utilizes Gemini 2.5 Flash as a "Lead Portfolio Manager" that synthesizes inputs from web-scraping agents and quantitative data models to generate a final market verdict.
* **🧠 Big Data MapReduce:** Processes thousands of historical crisis headlines to extract live "Trending Macro Themes" to identify global panics.
* **🎯 Content-Based Recommender System:** Calculates algorithmic cosine similarity to instantly match current market conditions to the mathematically closest historical market crash.
* **⚙️ Zero-Maintenance Cron Jobs:** GitHub Actions handles Python dependency management, database overwrites, and LaTeX PDF compilation every night at midnight UTC.
* **🎛️ Live ML Inference Sandbox:** An interactive "What-If" UI caches a Random Forest model, allowing users to test custom VIX and sentiment metrics for real-time anomaly predictions.
* **⏱️ Dynamic Time-Machine Filtering:** Interactive UI sliders allow users to dynamically slice historical market data and visualize specific economic events instantly.

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

## 💻 Local Quickstart & Development

> **🗝️ Prerequisite: Google Gemini API Key**
> To run the interactive AI agents locally, you will need a free API key. 
> [👉 Get your free key from Google AI Studio here](https://aistudio.google.com/app/apikey).

Want to run this AI pipeline on your own machine? Follow these steps:

**1. Clone the repository:**
```bash
git clone [https://github.com/amjad-hanini/Macro-Sentiment-Agentic-Pipeline.git](https://github.com/amjad-hanini/Macro-Sentiment-Agentic-Pipeline.git)
cd Macro-Sentiment-Agentic-Pipeline
```

**2. Install the required dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the Backend Data Pipeline (Optional):**
*This will fetch today's live news and rebuild the local SQLite database.*
```bash
python pipeline.py
```

**4. Launch the Streamlit Dashboard:**
```bash
streamlit run app.py
```
