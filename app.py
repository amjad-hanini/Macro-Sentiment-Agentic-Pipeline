import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
import google.generativeai as genai
import json
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
import alpaca_trade_api as tradeapi

# ==========================================
# 1. APPLICATION SETUP & UI STYLING
# ==========================================
st.set_page_config(page_title="Macro-Sentiment System", layout="wide")
st.title("📈 Autonomous Agentic Analyzer")

# HIGH-VISIBILITY LEGAL DISCLAIMER 
st.warning("**⚠️ Legal Disclaimer:** This system is for **educational and research purposes only**. The AI predictions, risk scores, and autonomous executions are simulated models and do not constitute financial advice. Algorithmic trading carries significant financial risk.", icon="⚠️")

# Force metric text to wrap appropriately on smaller displays
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
st.sidebar.markdown("### 🗝️ Intelligence Integration")
with st.sidebar.form("gemini_form"):
    api_key = st.text_input("Google Gemini API Key (Required):", type="password")
    submit_gemini = st.form_submit_button("Connect Intelligence")

if submit_gemini and api_key:
    st.sidebar.success("✅ Intelligence Core Online.")

st.sidebar.markdown("### 💸 Execution Integration (Optional)")
with st.sidebar.form("alpaca_form"):
    st.markdown("Enter [Alpaca Paper Trading](https://alpaca.markets/) keys to enable live autonomous execution.")
    alpaca_key = st.text_input("Alpaca API Key:", type="password")
    alpaca_secret = st.text_input("Alpaca Secret Key:", type="password")
    submit_alpaca = st.form_submit_button("Connect Trading Webhook")

if submit_alpaca and alpaca_key and alpaca_secret:
    st.sidebar.success("✅ Trading Webhook Armed.")

# ==========================================
# 2. DATA INGESTION & MACHINE LEARNING
# ==========================================
@st.cache_data(ttl=60)
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
    stop_words = {"the", "and", "to", "of", "a", "in", "for", "is", "on", "that", "by", "this", "with", "i", "you", "it", "not", "or", "be", "are", "from", "at", "as", "how", "will", "affect", "impacting", "analysts", "closely", "monitor"}
    
    for text_val in anomalies_df['Combined_News'].dropna():
        clean_text = str(text_val).lower().replace('.', '').replace(',', '').replace('"', '')
        mapped_words = clean_text.split()
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
# 3. GLOBAL CONTEXT SETUP
# ==========================================
if not df_anomalies.empty:
    anomaly_dates = df_anomalies['Date'].dt.strftime('%Y-%m-%d').tolist()
else:
    anomaly_dates = []

# ==========================================
# 4. FRONTEND UI: ENTERPRISE TABS
# ==========================================
tab_overview, tab_agents, tab_graph, tab_sql = st.tabs([
    "📊 Market Overview", 
    "⚖️ AI Analyst Debate & Execution", 
    "🕸️ Knowledge Graph", 
    "🕵️‍♂️ Database Query"
])

# --- TAB 1: DATA VISUALIZATION ---
with tab_overview:
    st.subheader("Market vs. Narrative Divergence")
    
    st.markdown("### ⏱️ Time-Machine Filter")
    min_date = df_joined['Date'].min().date()
    max_date = df_joined['Date'].max().date()
    date_range = st.slider("Select Historical Range for the Chart:", min_date, max_date, (min_date, max_date))

    mask = (df_joined['Date'].dt.date >= date_range[0]) & (df_joined['Date'].dt.date <= date_range[1])
    filtered_df = df_joined.loc[mask]
    filtered_anomalies = df_anomalies.loc[(df_anomalies['Date'].dt.date >= date_range[0]) & (df_anomalies['Date'].dt.date <= date_range[1])]

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

# --- TAB 2: RED TEAM DEBATE & ALGORITHMIC TRADING ---
with tab_agents:
    st.subheader("⚖️ AI Analysts: Market Debate")
    st.write("Orchestrating AI agents to debate market conditions before executing algorithmic trades.")
    
    if anomaly_dates:
        # THE FIX: Timeline scrubber for Tab 2
        selected_date_str_tab2 = st.select_slider("🎯 Select Target Anomaly Date to Analyze:", options=anomaly_dates, key="tab2_date")
        sample = df_anomalies[df_anomalies['Date'].dt.strftime('%Y-%m-%d') == selected_date_str_tab2].iloc[0]
        
        if "live_report" not in st.session_state:
            st.session_state.live_report = None

        if st.button("Initiate Multi-Agent Debate", type="primary"):
            if not api_key:
                st.warning("Please save your Gemini API Key in the sidebar to run live inferences.")
            else:
                with st.spinner("Agent 1 (Researcher) retrieving historical intelligence..."):
                    genai.configure(api_key=api_key)
                    research_model = genai.GenerativeModel('gemini-2.5-flash')
                    try:
                        research_prompt = f"Act as a financial researcher. What were the major global macroeconomic headlines and stock market drivers exactly on or around {selected_date_str_tab2}? Provide a highly concise 3-sentence summary of the news environment."
                        research_resp = research_model.generate_content(research_prompt)
                        web_context = research_resp.text
                    except Exception as e:
                        web_context = ""
                        st.error(f"🛑 Researcher Agent Failed: {e}")

                if web_context:
                    with st.spinner("Agents 2, 3 & 4 debating the data..."):
                        top_themes = ", ".join(trending_themes_df['Macro Theme'].head(3).tolist())
                        
                        debate_prompt = f"""
                        You are orchestrating a hedge fund debate regarding the market anomaly on {selected_date_str_tab2}.
                        Data: Drop: {sample['Price_Change_Pct']:.2f}%, VIX: {sample['Volatility_VIX']:.2f}, Themes: {top_themes}. Context: "{web_context}"
                        
                        Respond ONLY with a valid JSON using plain English:
                        {{
                            "optimistic_view": "A strong 2-sentence argument on why this panic is an overreaction and a buying opportunity.",
                            "pessimistic_view": "A strong 2-sentence argument on why this anomaly signals a systemic crash.",
                            "executive_summary": "The Lead Manager's final executive summary combining both arguments and the data.",
                            "action": "BUY" or "SELL" or "HOLD",
                            "confidence": "Integer 1-100 representing certainty."
                        }}
                        """
                        try:
                            response = research_model.generate_content(debate_prompt)
                            raw_json = response.text.replace('```json', '').replace('```', '').strip()
                            st.session_state.live_report = json.loads(raw_json)
                        except Exception as e:
                            st.error(f"🛑 AI Inference Error: {e}")

        st.markdown("---")
        
        if st.session_state.live_report is None:
            st.info("💡 **System Architecture: Mixture-of-Experts (MoE)**\n\nEnter your Gemini API key in the sidebar and click 'Initiate' to trigger the autonomous workflow. Here is how the agents operate:")
            
            col_explain1, col_explain2 = st.columns(2)
            with col_explain1:
                st.success("🟢 **The Bull (Optimist):** Analyzes the VIX and sentiment data to build a case for why the market overreacted, seeking high-upside buying opportunities.")
            with col_explain2:
                st.error("🔴 **The Bear (Pessimist):** Cross-references the drop with historical news to argue why the anomaly signals a deeper systemic crash.")
                
            st.warning("⚖️ **The Lead Manager (Judge):** Synthesizes the debate, issues a final executive summary, and outputs a strict BUY, SELL, or HOLD command with a calculated confidence score.")
            st.markdown("🤖 **Autonomous Execution:** If a BUY or SELL is triggered, the system formats a webhook to the Alpaca Paper Trading API to execute a live market order for `SPY`.")

        else:
            report = st.session_state.live_report
            st.success("✅ **Live Inference Complete:** Displaying real-time autonomous analysis.")
            
            col_bull, col_bear = st.columns(2)
            with col_bull:
                st.success(f"🟢 **Optimistic AI Analyst:**\n\n{report['optimistic_view']}")
            with col_bear:
                st.error(f"🔴 **Pessimistic AI Analyst:**\n\n{report['pessimistic_view']}")
                
            st.info(f"⚖️ **Final Executive Summary:**\n\n{report['executive_summary']}")
            
            col_act, col_conf = st.columns(2)
            col_act.metric("Determined Action", report['action'])
            col_conf.metric("System Confidence", f"{report['confidence']}%")

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

# --- TAB 3: KNOWLEDGE GRAPH WITH SIDE-BY-SIDE EXPLANATION ---
with tab_graph:
    st.subheader("🕸️ Live Entity Knowledge Graph")
    st.write("Using LLMs to extract relational structures from historical financial news.")
    
    if anomaly_dates:
        # THE FIX: Timeline scrubber for Tab 3
        selected_date_str_tab3 = st.select_slider("🎯 Select Target Anomaly Date to Graph:", options=anomaly_dates, key="tab3_date")
        
        if st.button("Generate Relational Graph"):
            if not api_key:
                st.warning("Please save your Gemini API Key in the sidebar.")
            else:
                with st.spinner("Extracting Knowledge Graph and Generating Explanation..."):
                    genai.configure(api_key=api_key)
                    research_model = genai.GenerativeModel('gemini-2.5-flash')
                    try:
                        research_prompt = f"What were the major global macroeconomic headlines and stock market drivers exactly on or around {selected_date_str_tab3}? Provide a concise summary of the key financial entities involved."
                        research_resp = research_model.generate_content(research_prompt)
                        web_context = research_resp.text
                    except Exception as e:
                        web_context = ""
                        st.error(f"🛑 Context Retrieval Failed: {e}")

                    if web_context:
                        try:
                            graph_prompt = f"""
                            Read this financial news context: "{web_context}"
                            Extract the 5 most important entities (like Federal Reserve, Tech Stocks, Inflation) and their relationships.
                            
                            Respond ONLY with a valid JSON using this exact format:
                            {{
                                "mermaid_code": "Raw Mermaid.js graph syntax. Start with 'graph TD;'.",
                                "explanation": "A 2-3 sentence plain-English explanation of the cause-and-effect loop happening in this graph, written for a non-technical manager."
                            }}
                            """
                            graph_response = research_model.generate_content(graph_prompt)
                            raw_json = graph_response.text.replace('```json', '').replace('```', '').strip()
                            graph_data = json.loads(raw_json)
                            
                            mermaid_syntax = graph_data["mermaid_code"].replace('```mermaid', '').replace('```', '').strip()
                            
                            col_graph, col_info = st.columns([2, 1])
                            
                            with col_graph:
                                st.components.v1.html(
                                    f"""
                                    <div class="mermaid">
                                    {mermaid_syntax}
                                    </div>
                                    <script type="module">
                                    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                                    mermaid.initialize({{ startOnLoad: true, theme: 'dark' }});
                                    </script>
                                    """,
                                    height=450
                                )
                                
                            with col_info:
                                st.info("💡 **How to Read This Graph**\n\nThe arrows represent cause-and-effect relationships extracted autonomously from raw news headlines on the date of the market anomaly.")
                                st.success(f"🤖 **AI Translation:**\n\n{graph_data['explanation']}")
                                
                        except Exception as e:
                            st.error(f"Failed to map relationships. Check API limits or JSON formatting. Error: {e}")

# --- TAB 4: TEXT-TO-SQL ---
with tab_sql:
    st.subheader("🕵️‍♂️ Database Agent (Text-to-SQL)")
    user_q = st.text_input("Ask a question (e.g., 'What was the highest VIX level in 2020?'):")
    if user_q:
        if not api_key:
            st.warning("API key required in the sidebar.")
        else:
            with st.spinner("Translating..."):
                genai.configure(api_key=api_key)
                sql_model = genai.GenerativeModel('gemini-2.5-flash')
                schema_prompt = f"Write a valid SQLite query for table `macro_data`. Columns: `Date`, `Price_Change_Pct`, `Volatility_VIX`, `Sentiment_Score`, `Is_Anomaly`. Question: {user_q}. Return ONLY raw SQL string."
                try:
                    sql_resp = sql_model.generate_content(schema_prompt)
                    raw_sql = sql_resp.text.replace('```sql', '').replace('```', '').strip()
                    
                    st.markdown("**Generated SQL:**")
                    st.code(raw_sql, language="sql")
                    
                    engine = create_engine('sqlite:///macro_data.db')
                    with engine.connect() as conn:
                        result_df = pd.read_sql(text(raw_sql), conn)
                        
                    if not result_df.empty:
                        st.dataframe(result_df, hide_index=True)
                    else:
                        st.info("Query returned no results.")
                except Exception as e:
                    st.error(f"🛑 Crash Diagnostics: {e}")
