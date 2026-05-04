<h1 align="center">📈 Autonomous Macro-Sentiment Agentic Pipeline</h1>

<p align="center">
  <a href="https://macro-sentiment-agentic-pipeline.streamlit.app/"><strong>Live Dashboard</strong></a> ·
  <a href="#-quickstart--development"><strong>Quickstart</strong></a> ·
  <a href="#-tech-stack"><strong>Tech Stack</strong></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue.svg?logo=python&logoColor=white" alt="Python 3.10">
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/AI-Google_Gemini-8E75B2.svg?logo=google&logoColor=white" alt="Gemini AI">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License">
</p>

> **Disclaimer:** This project is produced solely for educational and academic research purposes. Nothing in this repository constitutes investment advice, a solicitation, or a recommendation to buy, sell, or hold any security or financial instrument. Algorithmic trading carries significant financial risk. The creators are not liable for any financial losses incurred from using this simulated system.

## 📖 Overview
An autonomous, multi-agent AI pipeline that synthesizes macro volatility and NLP sentiment to predict market anomalies, orchestrate MoE analyst debates, and execute algorithmic trades.

### 🌟 Enterprise-Grade Architecture
* **🤖 Text-to-SQL Database Agent:** Integrates Gemini 2.5 Flash as an active query agent, allowing users to interrogate the historical SQLite database using natural language.
* **🧠 Agentic Memory Loop:** The system permanently archives AI-generated crisis reports back into the database, allowing current agents to retrieve and learn from the conclusions of past agents during similar market regimes.
* **📊 GMM Market Regime Clustering:** Utilizes Unsupervised Machine Learning (Gaussian Mixture Models) to dynamically cluster historical financial data into distinct market regimes (e.g., Standard Trading, High Fear/Crisis, Correction Phase) handling elliptical data overlap perfectly.
* **🌐 Big Data MapReduce & Stop-Word Filtering:** A custom Python MapReduce algorithm processes thousands of financial headlines to extract live "Trending Macro Themes," utilizing advanced stop-word filtering to ensure high-value macroeconomic keywords rise to the top.
* **🎯 Content-Based Recommender System:** Calculates algorithmic cosine similarity to instantly match current market conditions to the mathematically closest historical market crash.
* **⚙️ Zero-Maintenance CI/CD:** GitHub Actions handles Python dependency management, database overwrites, and nightly execution at midnight UTC.
* **⚖️ 100% LLM-Driven MoE Debate:** Bypasses brittle web scrapers by deploying an autonomous "Researcher Agent" (Gemini 2.5 Flash) to retrieve historical market context. Adversarial AI agents (Optimistic vs. Pessimistic) then debate this context before a Lead Judge synthesizes a final verdict.
* **💸 Autonomous Execution Webhook:** Closes the automation loop by hooking the Judge agent's verdict directly into the **Alpaca Trading API**, automatically executing live or paper SPY market orders based on AI risk assessments.
* **🕸️ LLM Knowledge Graph Extraction:** Dynamically reads AI-retrieved historical financial news and autonomously generates interactive Entity-Relationship Knowledge Graphs (via Mermaid.js) to map macroeconomic domino effects.

---

## 🗂️ Repository Structure

```text
Macro-Sentiment-Agentic-Pipeline/
├── .github/workflows/
│   └── pipeline.yaml        # Zero-maintenance CI/CD cron job for nightly execution
├── app.py                   # Streamlit interactive dashboard & UI logic
├── pipeline.py              # Backend ETL, ML clustering, and LaTeX report generation
├── requirements.txt         # Version-locked Python dependencies
├── .gitignore               # Security and environment file exclusions
├── LICENSE                  # MIT License
└── README.md                # Project documentation and architecture
```

---

## 🏗️ System Architecture

This project is built on four core pillars of modern data engineering and AI:

1. **Volume (Big Data Ingestion & MapReduce):**
   - Ingests large-scale historical Reddit WorldNews datasets alongside Yahoo Finance market data (DJIA & VIX).
   - Utilizes a custom Python **MapReduce** algorithm to process thousands of financial headlines, filtering stop words to dynamically aggregate "Top Trending Crisis Themes" across market anomalies.

2. **Prediction (Supervised, Unsupervised & Recommendation ML):**
   - Deploys a **FinBERT NLP** model to score the sentiment of daily news.
   - Trains a **Random Forest Classifier** to predict the probability of future market anomalies.
   - Utilizes a **Gaussian Mixture Model (GMM)** to cluster data into dynamic Market Regimes (Standard vs. Crisis).
   - Employs a **Content-Based Recommender** using cosine similarity to mathematically match current market conditions to historical crashes.

3. **Multi-Agent RAG & Memory Integration:**
   - A multi-agent LLM system acts as a "Lead Portfolio Manager," synthesizing Recommender data, MapReduce themes, and dynamic context generated by an LLM Researcher Agent to produce hallucination-free reports.
   - **Agentic Memory Loop:** The database permanently archives AI outputs, allowing the system to retrieve past AI thought processes when similar market regimes occur.
   - **Text-to-SQL Agent:** Users can interrogate the SQLite database dynamically using natural language via Gemini 2.5 Flash.

4. **Velocity (CI/CD Automation):**
   - Fully automated via **GitHub Actions**. A nightly CRON job executes the pipeline, updates the SQLite database, compiles a LaTeX PDF summary report, and securely commits the fresh data.

---

## 💻 Tech Stack

| Category | Technologies |
|----------|--------------|
| **Core Language** | Python 3.10 |
| **Machine Learning** | Scikit-Learn (Random Forest, Gaussian Mixture Models, Cosine Similarity) |
| **NLP & LLMs** | HuggingFace (FinBERT), Google Gemini 2.5 Flash (Agentic RAG & Text-to-SQL) |
| **Big Data Processing** | Custom Python MapReduce Architecture |
| **Data Ingestion** | `yfinance` (Yahoo Finance API), Kaggle Datasets (Reddit WorldNews) |
| **Database** | SQLite / PostgreSQL (via SQLAlchemy) |
| **Frontend UI** | Streamlit, Plotly (Interactive Data Visualization), Mermaid.js (Graphs) |
| **Algorithmic Trading** | Alpaca Trade API Webhooks |
| **DevOps & CI/CD** | GitHub Actions (Cron Jobs), Automated pdflatex PDF Compilation |

---

## 🚀 Quickstart & Development

> **🗝️ Prerequisite: API Keys**
> To run the interactive AI agents locally or via the web deployment, you will need a free Google Gemini API key.
> To enable live algorithmic trading, you will need Alpaca Paper Trading keys.
> Users can securely enter these directly into the application's sidebar UI.

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

---

## ☁️ Cloud Deployment & Security
This project utilizes **Streamlit Secrets** and **GitHub Actions Secrets** to keep API keys completely out of version control, ensuring enterprise-grade security for the open-source repository.

If deploying your own fork on Streamlit Community Cloud, navigate to **Settings > Secrets** and paste your API keys in TOML format:

```toml
GEMINI_API_KEY="your_google_key"
ALPACA_API_KEY="your_alpaca_key"
ALPACA_API_SECRET="your_alpaca_secret"
```

---

## 📈 Roadmap & Future Scaling
While currently optimized for a standalone cloud environment, the architecture is designed to scale:
* **Distributed Computing:** Migrate the custom MapReduce text-processing logic to **Apache PySpark** for multi-node cluster processing.
* **Alternative Data:** Integrate live SEC Edgar filings and Twitter/X firehose APIs for deeper sentiment granularity.

---

## 📝 License
MIT License - see [LICENSE](LICENSE) for details.

## 📑 Citation
If you use this work in academic research, please cite:
```text
@misc{hanini2026macrosentiment,
   title={Autonomous Macro-Sentiment Agentic Pipeline},
   author={Hanini, Amjad and DaSilva, Brandon and Khopkar, Anushree},
   year={2026},
   note={Available at: [https://github.com/amjad-hanini/Macro-Sentiment-Agentic-Pipeline](https://github.com/amjad-hanini/Macro-Sentiment-Agentic-Pipeline)}
}
```

## 👨‍💻 Authors

**Amjad Hanini**
* GitHub: [@amjad-hanini](https://github.com/amjad-hanini)

**Brandon DaSilva**
* GitHub: [@BDaSilva03](https://github.com/BDaSilva03)    

**Anushree Khopkar**
* GitHub: [@Anushree-Khopkar](https://github.com/Anushree-Khopkar)
