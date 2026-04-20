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

# ==========================================
# 1. APPLICATION SETUP & UI STYLING
# ==========================================
# Configure the main Streamlit page layout
st.set_page_config(page_title="Macro-Sentiment System", layout="wide")
st.title("📈 Autonomous Agentic Analyzer")

# Custom CSS to ensure large text metrics wrap nicely on smaller screens
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

# Sidebar form to securely accept the user's Gemini API key without hardcoding it
with st.sidebar.form("api_form"):
    st.markdown("### 🗝️ API Key Required")
    st.markdown("To run the AI Analysis Agents, you need a free Google Gemini API key.")
    st.markdown("[👉 **Click here to get a free key**](https://aistudio.google.com/app/apikey)")
    api_key = st.text_input("Paste your API Key here:", type="password")
    submit_key = st.form_submit_button("Save Key")

if submit_key:
    if api_key:
        st.sidebar.success("✅ Key Saved! Agents are ready to deploy.")
    else:
        st.sidebar.error("❌ Please paste a valid key first.")

# ==========================================
# 2. DATA INGESTION & MACHINE LEARNING BACKEND
# ==========================================

@st.cache_data(ttl=600)
def fetch_from_db():
    """
    Connects to the local SQLite database.
    Dynamically adds a column for AI Memory if it doesn't exist yet,
    then loads the historical market and sentiment data.
    """
    engine = create_engine('sqlite:///macro_data.db')
    with engine.connect() as conn:
        try:
            # AI Memory Loop: Ensure the DB has a column to save past AI reports
            conn.execute(text("ALTER TABLE macro_data ADD COLUMN Agent_Report TEXT"))
            conn.commit()
        except:
            pass # Column already exists, proceed normally
            
    df = pd.read_sql('SELECT * FROM macro_data', engine)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date').reset_index(drop=True) 
    # Filter out only the days marked as statistical anomalies
    anomalies = df[df['Is_Anomaly'] == 1]
    return df, anomalies

@st.cache_resource
def train_live_models(df):
    """
    Trains two Machine Learning models on the fly using historical data.
    1. Random Forest (Supervised) to predict anomaly probabilities.
    2. Gaussian Mixture Model (Unsupervised) to cluster data into Market Regimes.
    """
    features = ['Volatility_VIX', 'Sentiment_Score']
    X = df[features]
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, df['Is_Anomaly'])
    
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(X)
    
    return rf, gmm

@st.cache_data
def run_mapreduce(anomalies_df):
    """
    A custom Python MapReduce implementation for Big Data text processing.
    MAP: Splits thousands of news headlines into lowercase words.
    REDUCE: Filters out stop words and calculates the frequency of crisis themes.
    """
    all_words = []
    stop_words = {"the", "and", "to", "of", "a", "in", "for", "is", "on", "that", "by", "this", "with", "i", "you", "it", "not", "or", "be", "are", "from", "at", "as"}
    
    # MAP PHASE
    for text_val in anomalies_df['Combined_News'].dropna():
        mapped_words = str(text_val).lower().split()
        all_words.extend([w for w in mapped_words if w not in stop_words and len(w) > 3])
        
    # REDUCE PHASE
    word_counts = {}
    for word in all_words:
        word_counts[word] = word_counts.get(word, 0) + 1
        
    # Sort by highest frequency and return the top 10 themes
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    return pd.DataFrame(sorted_words, columns=['Macro Theme', 'Frequency'])

# Execute backend functions and halt the app if the database is missing
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
# Sidebar slider allowing users to slice the historical data dynamically
st.sidebar.markdown("---")
st.sidebar.markdown("### ⏱️ Time-Machine Filter")
min_date = df_joined['Date'].min().date()
max_date = df_joined['Date'].max().date()
date_range = st.sidebar.slider("Select Historical Range:", min_date, max_date, (min_date, max_date))

# Apply the user's date filter to the dataframes
mask = (df_joined['Date'].dt.date >= date_range[0]) & (df_joined['Date'].dt.date <= date_range[1])
filtered_df = df_joined.loc[mask]
filtered_anomalies = df_anomalies.loc[(df_anomalies['Date'].dt.date >= date_range[0]) & (df_anomalies['Date'].dt.date <= date_range[1])]

# ==========================================
# 4. FRONTEND UI: MULTI-TAB ARCHITECTURE
# ==========================================
tab_overview, tab_agents, tab_sql = st.tabs(["📊 Market Overview", "🧠 Agentic AI Suite", "🕵️‍♂️ Database Query"])

# --- TAB 1: DATA VISUALIZATION & ML SANDBOX ---
with tab_overview:
    st.subheader("Market vs. Narrative Divergence")
    
    # Plotly interactive line chart mapping market price vs NLP sentiment
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_df['Date'].dt.strftime('%Y-%m-%d'), y=filtered_df['Price_Change_Pct'], name='DJIA % Change', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=filtered_df['Date'].dt.strftime('%Y-%m-%d'), y=filtered_df['Sentiment_Score'] * 5, name='FinBERT Sentiment', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=filtered_anomalies['Date'].dt.strftime('%Y-%m-%d'), y=filtered_anomalies['Price_Change_Pct'], mode='markers', name='Anomalies', marker=dict(color='red', size=12)))
    
    fig.update_layout(xaxis_title="Date", yaxis_title="Percentage / Score", xaxis=dict(type='category', nticks=15))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    col_sandbox, col_mapreduce = st.columns(2)
    
    # Interactive UI allowing users to test fake data against the ML models
    with col_sandbox:
        st.subheader("🎛️ AI ML Sandbox")
        sim_vix = st.slider("Fake Volatility (VIX)", min_value=10.0, max_value=85.0, value=20.0, step=0.5)
        sim_sent = st.slider("Fake News Sentiment", min_value=-1.0, max_value=1.0, value=0.0, step=0.05)
        
        sim_prob = rf_model.predict_proba([[sim_vix, sim_sent]])[0][1] * 100
        sim_regime = gmm_model.predict([[sim_vix, sim_sent]])[0]
        
        st.metric("Anomaly Probability", f"{sim_prob:.2f}%")
        
        # Color psychology logic based on the Unsupervised GMM cluster output
        if sim_regime == 0:
            st.success("🟢 **Detected Market Regime:** Standard Trading")
        elif sim_regime == 1:
            st.error("🔴 **Detected Market Regime:** High Fear / Crisis")
        else:
            st.warning("🟠 **Detected Market Regime:** Correction Phase")
            
    # Visualizing the output of the MapReduce algorithm
    with col_mapreduce:
        st.subheader("🗺️ MapReduce: Top Crisis Themes")
        st.dataframe(
            trending_themes_df,
            column_config={
                "Macro Theme": "Crisis Theme",
                "Frequency": st.column_config.ProgressColumn(
                    "Frequency",
                    help="Word frequency during anomalies",
                    format="%d",
                    min_value=0,
                    max_value=int(trending_themes_df['Frequency'].max()),
                ),
            },
            hide_index=True,
            use_container_width=True
        )

# --- TAB 2: AGENTIC AI & RECOMMENDER SYSTEM ---
with tab_agents:
    st.subheader("🤖 Agentic RAG & Multi-Agent Engine")
    if not filtered_anomalies.empty:
        # Dropdown to select a specific historical market crash
        selected_date_str = st.selectbox("Select Anomaly Date to Deploy Agents:", filtered_anomalies['Date'].dt.strftime('%Y-%m-%d').tolist())
        sample = df_anomalies[df_anomalies['Date'].dt.strftime('%Y-%m-%d') == selected_date_str].iloc[0]
        
        # Content-Based Recommender System using Cosine Similarity
        features_for_sim = ['Volatility_VIX', 'Sentiment_Score', 'Price_Change_Pct']
        target_vector = [sample[features_for_sim].values]
        all_vectors = df_anomalies[features_for_sim].values
        similarities = cosine_similarity(target_vector, all_vectors)[0]
        
        # Find the most mathematically similar historical event (excluding the exact same day)
        df_anomalies_sim = df_anomalies.copy()
        df_anomalies_sim['Similarity'] = similarities
        df_anomalies_sim = df_anomalies_sim[df_anomalies_sim['Date'] != sample['Date']]
        best_match = df_anomalies_sim.sort_values(by='Similarity', ascending=False).iloc[0]
        
        st.info(f"💡 **Big Data Recommender:** Based on algorithmic cosine similarity, market conditions on **{selected_date_str}** are a **{best_match['Similarity']*100:.1f}% match** to the historical anomaly on **{best_match['Date'].strftime('%Y-%m-%d')}**.")
    
        # AI Memory Loop: Check the database to see if a past AI generated a report for this match
        if pd.notna(best_match.get('Agent_Report')) and best_match['Agent_Report'] != None:
            with st.expander(f"🧠 View AI Memory Archive for {best_match['Date'].strftime('%Y-%m-%d')}"):
                st.json(best_match['Agent_Report'])
    
        # Multi-Agent Deployment
        if st.button("Deploy Analysis Agents", type="primary"):
            if not api_key:
                st.warning("Please save your Gemini API Key in the sidebar.")
            else:
                # Agent 1: Live Web Search using DuckDuckGo
                with st.spinner("Agent 1 (Researcher) is scraping the web..."):
                    search_query = f"major global financial news on {selected_date_str}"
                    try:
                        ddg_results = DDGS().text(search_query, max_results=3)
                        web_context = " ".join([res['body'] for res in ddg_results])
                    except:
                        web_context = "No additional web context could be retrieved."
    
                # Agents 2 & 3: LLM Synthesis of Big Data and Web Scrapes
                with st.spinner("Agents 2 & 3 are analyzing Big Data metrics..."):
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    top_themes = ", ".join(trending_themes_df['Macro Theme'].head(3).tolist())
                    
                    # Strict JSON enforcement prompt
                    system_prompt = f"""
                    You are a Lead Portfolio Manager. Synthesize findings regarding the market anomaly on {selected_date_str}.
                    Data:
                    - Drop: {sample['Price_Change_Pct']:.2f}%
                    - VIX: {sample['Volatility_VIX']:.2f}
                    - MapReduce Themes: {top_themes}
                    - Web Context: "{web_context}"
                    
                    Respond ONLY with a valid JSON using this EXACT strict format:
                    {{
                        "macro_theme": "Max 3 words.",
                        "risk_score": "Strictly an integer 1-10.",
                        "quant_analysis": "One sentence acting as the Quant Agent.",
                        "synthesized_report": "A short paragraph acting as Lead Manager."
                    }}
                    """
                    try:
                        response = model.generate_content(system_prompt)
                        raw_json = response.text.replace('```json', '').replace('```', '').strip()
                        report = json.loads(raw_json)
                        
                        # AI Memory Loop: Save the LLM's conclusions back into the SQLite database for future reference
                        try:
                            conn = sqlite3.connect('macro_data.db')
                            cursor = conn.cursor()
                            cursor.execute("UPDATE macro_data SET Agent_Report = ? WHERE Date = ?", (raw_json, sample['Date'].strftime('%Y-%m-%d %H:%M:%S')))
                            conn.commit()
                            conn.close()
                        except Exception as db_e:
                            st.sidebar.warning("Could not cache to local memory.")
    
                        # Render the final synthesized AI report to the user
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Primary Macro Theme", report['macro_theme'])
                        col2.metric("Hidden Risk Score", f"{report['risk_score']}/10")
                        col3.metric("VIX Volatility Level", f"{sample['Volatility_VIX']:.2f}")
                        st.write(f"**Quantitative Analyst:** {report['quant_analysis']}")
                        st.write(f"**Final Synthesized Report:** {report['synthesized_report']}")
                    except Exception as e:
                        st.error("🛑 Free Tier Speed Limit Reached or JSON Error Occurred.")
    else:
        st.write("No anomalies detected in this time range.")

# --- TAB 3: TEXT-TO-SQL CHATBOT ---
with tab_sql:
    st.subheader("🕵️‍♂️ Database Agent (Text-to-SQL)")
    st.write("Ask the AI to query your historical Big Data framework in plain English.")
    
    # Natural language input box
    user_q = st.text_input("Ask a question (e.g., 'What was the highest VIX level in 2020?'):")
    if user_q:
        if not api_key:
            st.warning("API key required for the Database Agent.")
        else:
            with st.spinner("Translating natural language to SQL..."):
                genai.configure(api_key=api_key)
                sql_model = genai.GenerativeModel('gemini-2.5-flash')
                
                # Prompt instructing the LLM to act as a SQL engine based on our database schema
                schema_prompt = f"""
                You are a SQL Data Analyst. Given the user question, write a valid SQLite query.
                Table name: `macro_data`
                Columns: `Date` (TEXT YYYY-MM-DD HH:MM:SS), `Price_Change_Pct` (FLOAT), `Volatility_VIX` (FLOAT), `Sentiment_Score` (FLOAT), `Is_Anomaly` (INTEGER 0 or 1).
                
                Return ONLY the pure SQL string. Do not include markdown formatting or explanation.
                
                User Question: {user_q}
                """
                try:
                    sql_resp = sql_model.generate_content(schema_prompt)
                    raw_sql = sql_resp.text.replace('```sql', '').replace('```', '').strip()
                    
                    # Display the generated SQL code to the user for transparency
                    st.code(raw_sql, language="sql")
                    
                    # Safely execute the AI-generated SQL query against the local database
                    engine = create_engine('sqlite:///macro_data.db')
                    with engine.connect() as conn:
                        result_df = pd.read_sql(text(raw_sql), conn)
                    
                    # Render the query results as a table
                    if not result_df.empty:
                        st.dataframe(result_df, hide_index=True)
                    else:
                        st.info("Query returned no results.")
                        
                except Exception as e:
                    st.error("Could not generate or execute the SQL query.")
