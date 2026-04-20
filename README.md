# 📈 Autonomous Macro-Sentiment Agentic Pipeline

An end-to-end, automated data engineering pipeline and AI reasoning system that detects, predicts, and explains divergences between macroeconomic reality and public news narratives.

### 🌟 Enterprise-Grade Architecture
* **🤖 Text-to-SQL Database Agent:** Integrates Gemini 2.5 Flash as an active query agent, allowing users to interrogate the historical SQLite database using natural language.
* **🧠 Agentic Memory Loop:** The system permanently archives AI-generated crisis reports back into the database, allowing current agents to retrieve and learn from the conclusions of past agents during similar market regimes.
* **📊 GMM Market Regime Clustering:** Utilizes Unsupervised Machine Learning (Gaussian Mixture Models) to dynamically cluster historical financial data into distinct market regimes (e.g., Standard Trading, High Fear/Crisis, Correction Phase) handling elliptical data overlap perfectly.
* **🌐 Live Data Ingestion & Big Data MapReduce:** Automated nightly DuckDuckGo web scraping feeds real-time global headlines into a custom MapReduce algorithm to extract live "Trending Macro Themes" without human intervention.
* **🎯 Content-Based Recommender System:** Calculates algorithmic cosine similarity to instantly match current market conditions to the mathematically closest historical market crash.
* **⚙️ Zero-Maintenance CI/CD:** GitHub Actions handles Python dependency management, database overwrites, and nightly execution at midnight UTC.
* **⚖️ Red Team MoE Debate:** Implements a Mixture-of-Experts (MoE) prompting architecture where adversarial AI agents (Bull vs. Bear) debate historical anomalies before a Lead Judge synthesizes a final verdict.
* **💸 Autonomous Execution Webhook:** Closes the automation loop by hooking the Judge agent's verdict directly into the Alpaca Trading API, automatically executing live or paper SPY market orders based on AI risk assessments.
* **🕸️ LLM Knowledge Graph Extraction:** Dynamically reads unstructured financial news web scrapes and autonomously generates interactive Entity-Relationship Knowledge Graphs (via Mermaid.js) to map macroeconomic domino effects.

## 🏗️ System Architecture

**🔴 Live Interactive Dashboard:** [Click here to view the Streamlit App](https://macro-sentiment-agentic-pipeline.streamlit.app/)

This project is built on four core pillars of modern data engineering and AI:

1. **Volume (Big Data Ingestion & MapReduce):**
   - Ingests large-scale historical Reddit WorldNews datasets alongside Yahoo Finance market data (DJIA & VIX).
   - Utilizes a custom Python MapReduce algorithm to process thousands of financial headlines, filtering stop words to dynamically aggregate "Top Trending Crisis Themes" across market anomalies.
2. **Prediction (Supervised, Unsupervised & Recommendation ML):**
   - Deploys a FinBERT NLP model to score the sentiment of daily news.
   - Trains a Random Forest Classifier to predict the probability of future market anomalies.
   - Utilizes a Gaussian Mixture Model (GMM) to cluster data into dynamic Market Regimes (Standard vs. Crisis).
   - Employs a Content-Based Recommender using cosine similarity to mathematically match current market conditions to historical crashes.
3. **Multi-Agent RAG & Memory Integration:**
   - A multi-agent LLM system acts as a "Lead Portfolio Manager," synthesizing Recommender data, MapReduce themes, and live DuckDuckGo web scrapes to generate hallucination-free reports.
   - **Agentic Memory Loop:** The database permanently archives AI outputs, allowing the system to retrieve past AI thought processes when similar market regimes occur.
   - **Text-to-SQL Agent:** Users can interrogate the SQLite database dynamically using natural language via Gemini 2.5 Flash.
4. **Velocity (CI/CD Automation):**
   - Fully automated via GitHub Actions. A nightly CRON job executes the pipeline, updates the SQLite database, compiles a LaTeX PDF summary report, and securely commits the fresh data.

## 💻 Tech Stack
* **Language:** Python 3.10
* **Machine Learning:** Scikit-Learn (Random Forest Classification, Gaussian Mixture Models, Cosine Similarity)
* **NLP & LLMs:** HuggingFace (FinBERT), Google Gemini 2.5 Flash (Agentic RAG & Text-to-SQL)
* **Big Data Processing:** Custom Python MapReduce Architecture 
* **Data Ingestion:** DuckDuckGo Search API, `yfinance` (Yahoo Finance API)
* **Database:** SQLite / PostgreSQL (via SQLAlchemy)
* **Frontend UI:** Streamlit, Plotly (Interactive Data Visualization)
* **Algorithmic Trading:** Alpaca Trade API
* **DevOps & CI/CD:** GitHub Actions (Automated Cron Jobs), Automated LaTeX PDF Compilation

## 💻 Local Quickstart & Development

> **🗝️ Prerequisite: API Keys**
> To run the interactive AI agents locally, you will need a free Google Gemini API key.
> To enable live algorithmic trading, you will need Alpaca Paper Trading keys.

Want to run this AI pipeline on your own machine? Follow these steps:

**1. Clone the repository:**
`git clone https://github.com/amjad-hanini/Macro-Sentiment-Agentic-Pipeline.git`
`cd Macro-Sentiment-Agentic-Pipeline`

**2. Install the required dependencies:**
`pip install -r requirements.txt`

**3. Run the Backend Data Pipeline (Optional):**
*This will fetch today's live news and rebuild the local SQLite database.*
`python pipeline.py`

**4. Launch the Streamlit Dashboard:**
`streamlit run app.py`

---

## 🚀 Roadmap & Future Scaling
While currently optimized for a standalone cloud environment, the architecture is designed to scale:
* **Distributed Computing:** Migrate the custom MapReduce text-processing logic to Apache PySpark for multi-node cluster processing.
* **Alternative Data:** Integrate live SEC Edgar filings and Twitter/X firehose APIs for deeper sentiment granularity.

---

## 📝 License
MIT License - see LICENSE for details.

## 📑 Citation
If you use this work in academic research, please cite:

> @misc{hanini2026macrosentiment,
>   title={Autonomous Macro-Sentiment Agentic Pipeline},
>   author={Hanini, Amjad},
>   year={2026},
>   note={Available at: https://github.com/amjad-hanini/Macro-Sentiment-Agentic-Pipeline}
> }

> **Reminder:** This project is for educational and research purposes only. It does not constitute investment advice and is not produced in any broker-dealer or investment advisory capacity. The creators are not liable for any financial losses incurred from using this system.

## 👨‍💻 Author
**Amjad Hanini**
* GitHub: [@amjad-hanini](https://github.com/amjad-hanini)
