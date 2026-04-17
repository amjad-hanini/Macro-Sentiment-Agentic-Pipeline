import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import create_engine
import google.generativeai as genai
from duckduckgo_search import DDGS
import json

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
    api_key = st.text_input("Enter Google Gemini API Key:", type="password")
    submit_key = st.form_submit_button("Save Key")

@st.cache_data
def fetch_from_db():
    engine = create_engine('sqlite:///macro_data.db')
    df = pd.read_sql('SELECT * FROM macro_data', engine)
    df = df.sort_values(by='Date').reset_index(drop=True) 
    anomalies = df[df['Is_Anomaly'] == 1]
    return df, anomalies

try:
    df_joined, df_anomalies = fetch_from_db()
except Exception as e:
    st.error("Database connection failed.")
    st.stop()

st.subheader("Market vs. Narrative Divergence")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_joined['Date'], y=df_joined['Price_Change_Pct'], name='DJIA % Change', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=df_joined['Date'], y=df_joined['Sentiment_Score'] * 5, name='FinBERT Sentiment', line=dict(color='green')))
fig.add_trace(go.Scatter(x=df_anomalies['Date'], y=df_anomalies['Price_Change_Pct'], mode='markers', name='Anomalies', marker=dict(color='red', size=12)))

fig.update_layout(xaxis_title="x", yaxis_title="y")
st.plotly_chart(fig, use_container_width=True)

st.subheader("🤖 Agentic RAG: Deep Anomaly Breakdown")
if "saved_reports" not in st.session_state:
    st.session_state.saved_reports = {}

if not df_anomalies.empty:
    selected_date = st.selectbox("Select Anomaly Date to Deploy Agents:", df_anomalies['Date'].tolist())
    sample = df_anomalies[df_anomalies['Date'] == selected_date].iloc[0]
    
    if st.button("Deploy Analysis Agents", type="primary"):
        if not api_key:
            st.warning("Please save your Gemini API Key in the sidebar.")
        else:
            if selected_date in st.session_state.saved_reports:
                report = st.session_state.saved_reports[selected_date]
                st.success("Loaded from local memory! (No API quota used 🧠)")
                col1, col2, col3 = st.columns(3)
                col1.metric("Primary Macro Theme", report['macro_theme'])
                col2.metric("Hidden Risk Score", f"{report['risk_score']}/10")
                col3.metric("VIX Volatility Level", f"{sample['Volatility_VIX']:.2f}")
                st.info(f"**Quantitative Analyst:** {report['quant_analysis']}")
                st.success(f"**Final Synthesized Report:** {report['synthesized_report']}")
            else:
                with st.spinner("Agent 1 (Researcher) is scraping the web for context via DuckDuckGo..."):
                    search_query = f"major global financial news on {selected_date}"
                    try:
                        ddg_results = DDGS().text(search_query, max_results=3)
                        web_context = " ".join([res['body'] for res in ddg_results])
                    except:
                        web_context = "No additional web context could be retrieved."

                with st.spinner("Agents 2 & 3 are analyzing database records + web context..."):
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    
                    system_prompt = f"""
                    You are a trio of financial AI agents. Analyze this anomaly from {selected_date}:
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
                        st.session_state.saved_reports[selected_date] = report
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Primary Macro Theme", report['macro_theme'])
                        col2.metric("Hidden Risk Score", f"{report['risk_score']}/10")
                        col3.metric("VIX Volatility Level", f"{sample['Volatility_VIX']:.2f}")
                        st.info(f"**Quantitative Analyst:** {report['quant_analysis']}")
                        st.success(f"**Final Synthesized Report:** {report['synthesized_report']}")
                    except Exception as e:
                        st.error("🛑 Free Tier Speed Limit Reached or JSON Error Occurred.")
else:
    st.write("No anomalies detected.")
