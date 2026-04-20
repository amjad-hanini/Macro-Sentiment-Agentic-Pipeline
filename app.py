import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import create_engine
import google.generativeai as genai
from duckduckgo_search import DDGS
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
import re

st.set_page_config(page_title="Macro-Sentiment System", layout="wide")
st.title("📈 BIA678: Autonomous Agentic Analyzer")

# --- UI FIX: Force Metric Text to Wrap ---
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

@st.cache_data
def fetch_from_db():
    engine = create_engine('sqlite:///macro_data.db')
    df = pd.read_sql('SELECT * FROM macro_data', engine)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date').reset_index(drop=True) 
    anomalies = df[df['Is_Anomaly'] == 1]
    return df, anomalies

@st.cache_resource
def train_live_model(df):
    features = ['Volatility_VIX', 'Sentiment_Score']
    X = df[features]
    y = df['Is_Anomaly']
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    return rf

# --- BIG DATA FEATURE 1: Python MapReduce ---
@st.cache_data
def run_mapreduce(anomalies_df):
    # MAP PHASE: Split all text into lowercased words, remove punctuation
    all_words = []
    stop_words = {"the", "and", "to", "of", "a", "in", "for", "is", "on", "that", "by", "this", "with", "i", "you", "it", "not", "or", "be", "are", "from", "at", "as"}
    
    for text in anomalies_df['Combined_News'].dropna():
        # Clean basic punctuation to keep words pure
        clean_text = re.sub(r'[^\w\s]', '', str(text).lower())
        mapped_words = clean_text.split()
        all_words.extend([w for w in mapped_words if w not in stop_words and len(w) > 3])
        
    # REDUCE PHASE: Count frequencies
    word_counts = {}
    for word in all_words:
        word_counts[word] = word_counts.get(word, 0) + 1
        
    # Sort and return top 10
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    return pd.DataFrame(sorted_words, columns=['Macro Theme', 'Frequency'])

try:
    df_joined, df_anomalies = fetch_from_db()
    rf_model = train_live_model(df_joined)
    trending_themes_df = run_mapreduce(df_anomalies)
except Exception as e:
    st.error("Database connection failed.")
    st.stop()

# --- Time-Machine Filter ---
st.sidebar.markdown("---")
st.sidebar.markdown("### ⏱️ Time-Machine Filter")
min_date = df_joined['Date'].min().date()
max_date = df_joined['Date'].max().date()
date_range = st.sidebar.slider("Select Historical Range:", min_date, max_date, (min_date, max_date))

mask = (df_joined['Date'].dt.date >= date_range[0]) & (df_joined['Date'].dt.date <= date_range[1])
filtered_df = df_joined.loc[mask]
filtered_anomalies = df_anomalies.loc[(df_anomalies['Date'].dt.date >= date_range[0]) & (df_anomalies['Date'].dt.date <= date_range[1])]

st.subheader("Market vs. Narrative Divergence")
fig = go.Figure()
fig.add_trace(go.Scatter(x=filtered_df['Date'].dt.strftime('%Y-%m-%d'), y=filtered_df['Price_Change_Pct'], name='DJIA % Change', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=filtered_df['Date'].dt.strftime('%Y-%m-%d'), y=filtered_df['Sentiment_Score'] * 5, name='FinBERT Sentiment', line=dict(color='green')))
fig.add_trace(go.Scatter(x=filtered_anomalies['Date'].dt.strftime('%Y-%m-%d'), y=filtered_anomalies['Price_Change_Pct'], mode='markers', name='Anomalies', marker=dict(color='red', size=12)))

fig.update_layout(xaxis_title="x", yaxis_title="y", xaxis=dict(type='category', nticks=15))
st.plotly_chart(fig, use_container_width=True)

# --- The 'What-If' AI Sandbox & MapReduce Layout ---
st.markdown("---")
col_sandbox, col_mapreduce = st.columns(2)

with col_sandbox:
    st.subheader("🎛️ AI Sandbox")
    sim_vix = st.slider("Fake Volatility (VIX)", min_value=10.0, max_value=85.0, value=20.0, step=0.5)
    sim_sent = st.slider("Fake News Sentiment", min_value=-1.0, max_value=1.0, value=0.0, step=0.05)
    sim_prob = rf_model.predict_proba([[sim_vix, sim_sent]])[0][1] * 100
    st.metric("Live Anomaly Probability", f"{sim_prob:.2f}%")

with col_mapreduce:
    st.subheader("🗺️ MapReduce: Top Crisis Themes")
    st.dataframe(trending_themes_df, use_container_width=True, hide_index=True)

st.markdown("---")

# --- ORIGINAL AGENTIC RAG & BIG DATA RECOMMENDER ---
st.subheader("🤖 Agentic RAG & Content-Based Recommender")
if "saved_reports" not in st.session_state:
    st.session_state.saved_reports = {}

if not filtered_anomalies.empty:
    selected_date_str = st.selectbox("Select Anomaly Date to Deploy Agents:", filtered_anomalies['Date'].dt.strftime('%Y-%m-%d').tolist())
    sample = df_anomalies[df_anomalies['Date'].dt.strftime('%Y-%m-%d') == selected_date_str].iloc[0]
    
    # --- BIG DATA FEATURE 2: Content-Based Recommender ---
    features_for_sim = ['Volatility_VIX', 'Sentiment_Score', 'Price_Change_Pct']
    target_vector = [sample[features_for_sim].values]
    all_vectors = df_anomalies[features_for_sim].values
    similarities = cosine_similarity(target_vector, all_vectors)[0]
    
    # Find the most similar date that is NOT the exact same date
    df_anomalies_sim = df_anomalies.copy()
    df_anomalies_sim['Similarity'] = similarities
    df_anomalies_sim = df_anomalies_sim[df_anomalies_sim['Date'] != sample['Date']]
    best_match = df_anomalies_sim.sort_values(by='Similarity', ascending=False).iloc[0]
    
    st.info(f"💡 **Recommender System Output:** Based on algorithmic cosine similarity, market conditions on **{selected_date_str}** are a **{best_match['Similarity']*100:.1f}% match** to the historical anomaly on **{best_match['Date'].strftime('%Y-%m-%d')}**.")

    if st.button("Deploy Analysis Agents", type="primary"):
        if not api_key:
            st.warning("Please save your Gemini API Key in the sidebar.")
        else:
            if selected_date_str in st.session_state.saved_reports:
                report = st.session_state.saved_reports[selected_date_str]
                st.success("Loaded from local memory! (No API quota used 🧠)")
                col1, col2, col3 = st.columns(3)
                col1.metric("Primary Macro Theme", report['macro_theme'])
                col2.metric("Hidden Risk Score", f"{report['risk_score']}/10")
                col3.metric("VIX Volatility Level", f"{sample['Volatility_VIX']:.2f}")
                st.write(f"**Quantitative Analyst:** {report['quant_analysis']}")
                st.write(f"**Final Synthesized Report:** {report['synthesized_report']}")
            else:
                with st.spinner("Agent 1 (Researcher) is scraping the web..."):
                    search_query = f"major global financial news on {selected_date_str}"
                    try:
                        ddg_results = DDGS().text(search_query, max_results=3)
                        web_context = " ".join([res['body'] for res in ddg_results])
                    except:
                        web_context = "No additional web context could be retrieved."

                with st.spinner("Agents 2 & 3 are analyzing database records..."):
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    
                    system_prompt = f"""
                    You are a trio of financial AI agents. Analyze this anomaly from {selected_date_str}:
                    - Market Drop: {sample['Price_Change_Pct']:.2f}%
                    - Market Fear (VIX): {sample['Volatility_VIX']:.2f}
                    - Database Headlines: "{sample['Combined_News']}"
                    - Live Web Scrape Context: "{web_context}"
                    
                    Respond ONLY with a valid JSON using this EXACT strict format:
                    {{
                        "macro_theme": "Maximum 3 words.",
                        "risk_score": "Strictly an integer 1-10.",
                        "quant_analysis": "One sentence explaining if the VIX justified the drop",
                        "synthesized_report": "A short paragraph explaining the gap between the news and the market"
                    }}
                    """
                    try:
                        response = model.generate_content(system_prompt)
                        raw_json = response.text.replace('```json', '').replace('```', '').strip()
                        report = json.loads(raw_json)
                        st.session_state.saved_reports[selected_date_str] = report
                        
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
