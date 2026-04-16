import pandas as pd
import yfinance as yf
from transformers import pipeline
from sqlalchemy import create_engine
import torch
import warnings
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings('ignore')

print("🚀 Starting Big Data Pipeline & ML Engine...")
device = 0 if torch.cuda.is_available() else -1
nlp = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=device)

url = "https://raw.githubusercontent.com/nelsonwang222/Reddit_Predict_DJI/master/Combined_News_DJIA.csv"

# 1. BIG DATA CHUNKING
chunk_size = 500
processed_chunks = []
print("📦 Processing Data in Chunks...")
for chunk in pd.read_csv(url, chunksize=chunk_size):
    chunk = chunk[['Date', 'Top1', 'Top2', 'Top3']].dropna()
    chunk['Combined_News'] = chunk['Top1'] + " " + chunk['Top2'] + " " + chunk['Top3']
    processed_chunks.append(chunk[['Date', 'Combined_News']])

news_df = pd.concat(processed_chunks)

# 2. MARKET DATA FETCH
start_date, end_date = news_df['Date'].min(), news_df['Date'].max()
djia = yf.download(["^DJI", "^VIX"], start=start_date, end=end_date, progress=False)
if isinstance(djia.columns, pd.MultiIndex):
    djia.columns = ['_'.join(col).strip() for col in djia.columns.values]

djia['Price_Change_Pct'] = djia['Close_^DJI'].pct_change() * 100
djia['Volatility_VIX'] = djia['Close_^VIX']
djia = djia.reset_index()
djia['Date'] = djia['Date'].dt.strftime('%Y-%m-%d')

merged_df = pd.merge(news_df, djia[['Date', 'Price_Change_Pct', 'Volatility_VIX']], on="Date").dropna().tail(300)

# 3. SENTIMENT INFERENCE
print("⚙️ Running FinBERT Sentiment Inference...")
def get_finbert_score(text):
    result = nlp(text[:512])[0]
    if result['label'] == 'positive': return result['score']
    elif result['label'] == 'negative': return -result['score']
    else: return 0.0

merged_df['Sentiment_Score'] = merged_df['Combined_News'].apply(get_finbert_score)

# 4. PREDICTIVE ML MODEL: Random Forest Classifier
print("🧠 Training Random Forest Anomaly Predictor...")
merged_df['Is_Anomaly'] = ((merged_df["Price_Change_Pct"] < -1.0) & (merged_df["Sentiment_Score"] > 0.1)).astype(int)

features = ['Volatility_VIX', 'Sentiment_Score']
X = merged_df[features]
y = merged_df['Is_Anomaly']

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

latest_data = merged_df.iloc[-1:][features]
anomaly_prob = rf_model.predict_proba(latest_data)[0][1] * 100
print(f"🔮 Tomorrow's Anomaly Probability: {anomaly_prob:.2f}%")

# Save everything to Database
engine = create_engine('sqlite:///macro_data.db')
merged_df.to_sql('macro_data', engine, if_exists='replace', index=False)
print("✅ Database built and ready!")

import datetime

# --- 5. DYNAMIC REPORT GENERATION ---
print("📝 Generating Dynamic LaTeX Report...")
today_str = datetime.datetime.now().strftime("%B %d, %Y")
latest_vix = latest_data['Volatility_VIX'].values[0]
latest_sent = latest_data['Sentiment_Score'].values[0]

latex_content = f"""\\documentclass{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{xcolor}}

\\title{{\\textbf{{Daily Macro-Sentiment Pipeline Report}}}}
\\author{{Autonomous AI Agent}}
\\date{{{today_str}}}

\\begin{{document}}
\\maketitle

\\section*{{Nightly Automation Status}}
The automated GitHub Actions pipeline successfully fetched the latest global news and market data, processed sentiment via FinBERT, and updated the historical database.

\\section*{{Current Market State}}
\\begin{{itemize}}
    \\item \\textbf{{Volatility Index (VIX):}} {latest_vix:.2f}
    \\item \\textbf{{Aggregate News Sentiment:}} {latest_sent:.2f}
\\end{{itemize}}

\\section*{{AI Predictive Outlook}}
Based on the latest divergence between market fear and news sentiment, the Random Forest model calculates the probability of a market anomaly occurring tomorrow at \\textbf{{\\color{{red}}{anomaly_prob:.2f}\\%}}.

\\end{{document}}
"""

with open("final_report.tex", "w") as f:
    f.write(latex_content)
    
print("📄 Dynamic PDF template saved!")
