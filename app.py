import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import create_engine
import google.generativeai as genai
from duckduckgo_search import DDGS
import json
from sklearn.ensemble import RandomForestClassifier

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

try:
    df_joined, df_anomalies = fetch_from_db()
    rf_model = train_live_model(df_joined)
except Exception as e:
    st.error("Database connection failed.")
    st.stop()

# --- FEATURE 1: Time-Machine Filter ---
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

# --- FEATURE 2: "What-If" AI Sandbox ---
st.markdown("---")
st.subheader("🎛️ The 'What-If' AI Sandbox")
col_vix, col_sent, col_pred = st.columns(3)

with col_vix:
    sim_vix = st.slider("Fake Volatility (VIX)", min_value=10.0, max_value=85.0, value=20.0, step=0.5)
with col_sent:
    sim_sent = st.slider("Fake News Sentiment", min_value=-1.0, max_value=1.0, value=0.0, step=0.05)
with col_pred:
    sim_prob = rf_model.predict_proba([[sim_vix, sim_sent]])[0][1] * 100
    st.metric("Live Anomaly Probability", f"{sim_prob:.2f}%")
    if sim_prob > 50:
        st.error("🚨 High Risk of Anomaly!")
    else:
        st.success("✅ Normal Market Conditions")

# --- FEATURE 3: Model Diagnostics ---
with st.expander("📊 View Model Diagnostics & Database Stats"):
    d_col1, d_col2, d_col3 = st.columns(3)
    d_col1.metric("Total Trading Days", len(df_joined))
    d_col2.metric("Total Anomalies Detected", len(df_anomalies))
    importance = rf_model.feature_importances_
    d_col3.metric("VIX vs Sentiment Weight", f"{importance[0]*100:.0f}% / {importance[1]*100:.0f}%")

st.markdown("---")

# --- ORIGINAL AGENTIC RAG ---
st.subheader("🤖 Agentic RAG: Deep Anomaly Breakdown")
if "saved_reports" not in st.session_state:
    st.session_state.saved_reports = {}

if not filtered_anomalies.empty:
    selected_date_str = st.selectbox("Select Anomaly Date to Deploy Agents:", filtered_anomalies['Date'].dt.strftime('%Y-%m-%d').tolist())
    sample = df_anomalies[df_anomalies['Date'].dt.strftime('%Y-%m-%d') == selected_date_str].iloc[0]
    
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
                st.info(f"**Quantitative Analyst:** {report['quant_analysis']}")
                st.success(f"**Final Synthesized Report:** {report['synthesized_report']}")
            else:
                with st.spinner("Agent 1 (Researcher) is scraping the web for context via DuckDuckGo..."):
                    search_query = f"major global financial news on {selected_date_str}"
                    try:
                        ddg_results = DDGS().text(search_query, max_results=3)
                        web_context = " ".join([res['body'] for res in ddg_results])
                    except:
                        web_context = "No additional web context could be retrieved."

                with st.spinner("Agents 2 & 3 are analyzing database records + web context..."):
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
                        "macro_theme": "Maximum 3 words. (e.g. 'Pharma Price Gouging')",
                        "risk_score": "Strictly a single integer from 1 to 10. No text.",
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
                        st.info(f"**Quantitative Analyst:** {report['quant_analysis']}")
                        st.success(f"**Final Synthesized Report:** {report['synthesized_report']}")
                    except Exception as e:
                        st.error("🛑 Free Tier Speed Limit Reached or JSON Error Occurred.")
else:
    st.write("No anomalies detected in this time range.")
