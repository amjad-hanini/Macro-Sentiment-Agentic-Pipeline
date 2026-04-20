import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
import google.generativeai as genai
from duckduckgo_search import DDGS
import json
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity
import alpaca_trade_api as tradeapi

# ==========================================
# 1. APPLICATION SETUP & UI STYLING
# ==========================================
st.set_page_config(page_title="Macro-Sentiment System", layout="wide")
st.title("📈 Autonomous Agentic Analyzer")

st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] {
        white-space: normal !important;
        word-wrap: break-word !important;
        font-size: 30px !important;
        line-height: 1.2 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Multi-API Authentication Sidebar
with st.sidebar.form("api_form"):
    st.markdown("### 🗝️ Intelligence Integration")
    api_key = st.text_input("Google Gemini API Key (Required):", type="password")
    
    st.markdown("### 💸 Execution Integration (Optional)")
    st.markdown("Enter [Alpaca Paper Trading](https://alpaca.markets/) keys to enable live autonomous execution.")
    alpaca_key = st.text_input("Alpaca API Key:", type="password")
    alpaca_secret = st.text_input("Alpaca Secret Key:", type="password")
    submit_keys = st.form_submit_button("Save Architecture Keys")

if submit_keys:
    if api_key:
        st.sidebar.success("✅ Intelligence Core Online.")
    if alpaca_key and alpaca_secret:
        st.sidebar.success("✅ Trading Webhook Armed.")

# --- DISCLAIMER FOOTER ---
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚠️ Legal Disclaimer")
st.sidebar.caption(
    "This system is for **educational and research purposes only**. "
    "The AI predictions, risk scores, and autonomous executions are simulated models and do not constitute financial advice. "
    "Algorithmic trading carries significant financial risk."
)

# ==========================================
# 2. DATA INGESTION & MACHINE LEARNING
# ==========================================
@st.cache_data(ttl=600)
def fetch_from_db():
    engine = create_engine('sqlite:///macro_data.db')
    with engine.connect() as conn:
        try:
            conn.execute(text("ALTER TABLE macro_data ADD COLUMN Agent_Report TEXT"))
            conn.commit()
        except:
            pass 
            
    df = pd.read_sql('SELECT * FROM macro_data', engine)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date').reset_index(drop=True) 
    anomalies = df[df['Is_Anomaly'] == 1]
    return df, anomalies

@st.cache_resource
def train_live_models(df):
    features = ['Volatility_VIX', 'Sentiment_Score']
    X = df[features]
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, df['Is_Anomaly'])
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(X)
    return rf, gmm

@st.cache_data
def run_mapreduce(anomalies_df):
    all_words = []
    stop_words = {"the", "and", "to", "of", "a", "in", "for", "is", "on", "that", "by", "this", "with", "i", "you", "it", "not", "or", "be", "are", "from", "at", "as"}
    for text_val in anomalies_df['Combined_News'].dropna():
        mapped_words = str(text_val).lower().split()
        all_words.extend([w for w in mapped_words if w not in stop_words and len(w) > 3])
    word_counts = {}
    for word in all_words:
        word_counts[word] = word_counts.get(word, 0) + 1
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    return pd.DataFrame(sorted_words, columns=['Macro Theme', 'Frequency'])

try:
    df_joined, df_anomalies = fetch_from_db()
    rf_model, gmm_model = train_live_models(df_joined)
    trending_themes_df = run_mapreduce(df_anomalies)
except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()

# ==========================================
# 3. INTERACTIVE TIME-MACHINE FILTER
# ==========================================
st.sidebar.markdown("---")
st.sidebar.markdown("### ⏱️ Time-Machine Filter")
min_date = df_joined['Date'].min().date()
max_date = df_joined['Date'].max().date()
date_range = st.sidebar.slider("Select Historical Range:", min_date, max_date, (min_date, max_date))

mask = (df_joined['Date'].dt.date >= date_range[0]) & (df_joined['Date'].dt.date <= date_range[1])
filtered_df = df_joined.loc[mask]
filtered_anomalies = df_anomalies.loc[(df_anomalies['Date'].dt.date >= date_range[0]) & (df_anomalies['Date'].dt.date <= date_range[1])]

# ==========================================
# 4. FRONTEND UI: ENTERPRISE TABS
# ==========================================
tab_overview, tab_agents, tab_graph, tab_sql = st.tabs([
    "📊 Market Overview", 
    "⚖️ Red Team Debate & Execution", 
    "🕸️ Knowledge Graph", 
    "🕵️‍♂️ Database Query"
])

# --- TAB 1: DATA VISUALIZATION ---
with tab_overview:
    st.subheader("Market vs. Narrative Divergence")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_df['Date'].dt.strftime('%Y-%m-%d'), y=filtered_df['Price_Change_Pct'], name='DJIA % Change', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=filtered_df['Date'].dt.strftime('%Y-%m-%d'), y=filtered_df['Sentiment_Score'] * 5, name='FinBERT Sentiment', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=filtered_anomalies['Date'].dt.strftime('%Y-%m-%d'), y=filtered_anomalies['Price_Change_Pct'], mode='markers', name='Anomalies', marker=dict(color='red', size=12)))
    fig.update_layout(xaxis_title="Date", yaxis_title="Percentage / Score", xaxis=dict(type='category', nticks=15))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    col_sandbox, col_mapreduce = st.columns(2)
    with col_sandbox:
        st.subheader("🎛️ AI ML Sandbox")
        sim_vix = st.slider("Fake Volatility (VIX)", min_value=10.0, max_value=85.0, value=20.0, step=0.5)
        sim_sent = st.slider("Fake News Sentiment", min_value=-1.0, max_value=1.0, value=0.0, step=0.05)
        sim_prob = rf_model.predict_proba([[sim_vix, sim_sent]])[0][1] * 100
        sim_regime = gmm_model.predict([[sim_vix, sim_sent]])[0]
        st.metric("Anomaly Probability", f"{sim_prob:.2f}%")
        
        if sim_regime == 0:
            st.success("🟢 **Detected Market Regime:** Standard Trading")
        elif sim_regime == 1:
            st.error("🔴 **Detected Market Regime:** High Fear / Crisis")
        else:
            st.warning("🟠 **Detected Market Regime:** Correction Phase")
            
    with col_mapreduce:
        st.subheader("🗺️ MapReduce: Top Crisis Themes")
        st.dataframe(
            trending_themes_df,
            column_config={
                "Macro Theme": "Crisis Theme",
                "Frequency": st.column_config.ProgressColumn("Frequency", format="%d", min_value=0, max_value=int(trending_themes_df['Frequency'].max())),
            },
            hide_index=True, use_container_width=True
        )

# --- TAB 2 & 3 SHARED CONTEXT SETUP ---
if not filtered_anomalies.empty:
    selected_date_str = st.sidebar.selectbox("🎯 Target Anomaly Date:", filtered_anomalies['Date'].dt.strftime('%Y-%m-%d').tolist())
    sample = df_anomalies[df_anomalies['Date'].dt.strftime('%Y-%m-%d') == selected_date_str].iloc[0]
else:
    selected_date_str = None

# --- TAB 2: RED TEAM DEBATE & ALGORITHMIC TRADING ---
with tab_agents:
    st.subheader("⚖️ Mixture-of-Experts: Red Team Debate")
    st.write("Orchestrating adversarial AI agents to debate market conditions before executing algorithmic trades.")
    
    if selected_date_str:
        if st.button("Initiate Multi-Agent Debate", type="primary"):
            if not api_key:
                st.warning("Please save your Gemini API Key in the sidebar.")
            else:
                with st.spinner("Agent 1 (Researcher) scraping intelligence..."):
                    try:
                        ddg_results = DDGS().text(f"major global financial news on {selected_date_str}", max_results=3)
                        web_context = " ".join([res['body'] for res in ddg_results])
                    except:
                        web_context = "No web context retrieved."

                with st.spinner("Agents 2, 3 & 4 debating the data..."):
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    top_themes = ", ".join(trending_themes_df['Macro Theme'].head(3).tolist())
                    
                    # THE RED TEAM DEBATE PROMPT
                    debate_prompt = f"""
                    You are orchestrating a hedge fund debate regarding the market anomaly on {selected_date_str}.
                    Data: Drop: {sample['Price_Change_Pct']:.2f}%, VIX: {sample['Volatility_VIX']:.2f}, Themes: {top_themes}. Context: "{web_context}"
                    
                    Respond ONLY with a valid JSON:
                    {{
                        "bull_agent": "A strong 2-sentence argument on why this panic is an overreaction and a buying opportunity.",
                        "bear_agent": "A strong 2-sentence argument on why this anomaly signals a systemic crash.",
                        "judge_synthesis": "The Lead Manager's final verdict combining both arguments and the data.",
                        "action": "BUY" or "SELL" or "HOLD",
                        "confidence": "Integer 1-100 representing certainty."
                    }}
                    """
                    try:
                        response = model.generate_content(debate_prompt)
                        raw_json = response.text.replace('```json', '').replace('```', '').strip()
                        report = json.loads(raw_json)
                        
                        col_bull, col_bear = st.columns(2)
                        with col_bull:
                            st.success(f"🐂 **Bull Agent Thesis:**\n\n{report['bull_agent']}")
                        with col_bear:
                            st.error(f"🐻 **Bear Agent Thesis:**\n\n{report['bear_agent']}")
                            
                        st.info(f"⚖️ **Lead Judge Synthesis:**\n\n{report['judge_synthesis']}")
                        
                        col_act, col_conf = st.columns(2)
                        col_act.metric("Determined Action", report['action'])
                        col_conf.metric("System Confidence", f"{report['confidence']}%")

                        # ALGORITHMIC EXECUTION WEBHOOK
                        st.markdown("---")
                        st.subheader("🤖 Autonomous Execution Engine")
                        if report['action'] in ["BUY", "SELL"]:
                            if alpaca_key and alpaca_secret:
                                try:
                                    api = tradeapi.REST(alpaca_key, alpaca_secret, base_url='https://paper-api.alpaca.markets')
                                    order = api.submit_order(
                                        symbol='SPY',
                                        qty=1,
                                        side=report['action'].lower(),
                                        type='market',
                                        time_in_force='gtc'
                                    )
                                    st.success(f"✅ **Trade Executed!** Successfully submitted a {report['action']} order for 1 share of SPY via Alpaca.")
                                except Exception as trade_e:
                                    st.error(f"❌ Execution Failed. Verify your Alpaca keys. Error: {trade_e}")
                            else:
                                st.warning(f"⚠️ The AI recommended a {report['action']} order, but the Trading Webhook is disabled (Missing Alpaca Keys).")
                        else:
                            st.write("⏸️ The AI recommended a HOLD. No algorithmic trades executed.")
                            
                    except Exception as e:
                        st.error(f"🛑 AI Inference Error: {e}")

# --- TAB 3: KNOWLEDGE GRAPH ---
with tab_graph:
    st.subheader("🕸️ Live Entity Knowledge Graph")
    st.write("Using LLMs to extract relational structures from unstructured financial news.")
    
    if selected_date_str:
        if st.button("Generate Relational Graph"):
            if not api_key:
                st.warning("Please save your Gemini API Key.")
            else:
                with st.spinner("Extracting Knowledge Graph..."):
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    
                    try:
                        ddg_results = DDGS().text(f"major global financial news on {selected_date_str}", max_results=3)
                        web_context = " ".join([res['body'] for res in ddg_results])
                        
                        graph_prompt = f"""
                        Read this financial news context: "{web_context}"
                        Extract the 5 most important entities (like Federal Reserve, Tech Stocks, Inflation) and their relationships.
                        Output ONLY raw Mermaid.js graph syntax. Start with 'graph TD;'. 
                        Example format:
                        graph TD;
                        A[Federal Reserve] -->|Raised| B[Interest Rates];
                        """
                        graph_response = model.generate_content(graph_prompt)
                        mermaid_code = graph_response.text.replace('```mermaid', '').replace('```', '').strip()
                        
                        st.components.v1.html(
                            f"""
                            <div class="mermaid">
                            {mermaid_code}
                            </div>
                            <script type="module">
                            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                            mermaid.initialize({{ startOnLoad: true }});
                            </script>
                            """,
                            height=400
                        )
                    except Exception as e:
                        st.error("Failed to map relationships.")

# --- TAB 4: TEXT-TO-SQL ---
with tab_sql:
    st.subheader("🕵️‍♂️ Database Agent (Text-to-SQL)")
    user_q = st.text_input("Ask a question (e.g., 'What was the highest VIX level in 2020?'):")
    if user_q:
        if not api_key:
            st.warning("API key required.")
        else:
            with st.spinner("Translating..."):
                genai.configure(api_key=api_key)
                sql_model = genai.GenerativeModel('gemini-2.5-flash')
                schema_prompt = f"Write a valid SQLite query for table `macro_data`. Columns: `Date`, `Price_Change_Pct`, `Volatility_VIX`, `Sentiment_Score`, `Is_Anomaly`. Question: {user_q}. Return ONLY raw SQL string."
                try:
                    sql_resp = sql_model.generate_content(schema_prompt)
                    raw_sql = sql_resp.text.replace('```sql', '').replace('```', '').strip()
                    st.code(raw_sql, language="sql")
                    engine = create_engine('sqlite:///macro_data.db')
                    with engine.connect() as conn:
                        result_df = pd.read_sql(text(raw_sql), conn)
                    if not result_df.empty:
                        st.dataframe(result_df, hide_index=True)
                    else:
                        st.info("Query returned no results.")
                except Exception as e:
                    st.error("Could not generate query.")
