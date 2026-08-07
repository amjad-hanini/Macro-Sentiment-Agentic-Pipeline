import yfinance as yf
import pandas as pd
import sqlite3
import feedparser
import random
from datetime import datetime, timedelta
from transformers import pipeline
import subprocess
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "macro_data.db"
CHAMPION_MODEL_PATH = BASE_DIR / "champion_model.pkl"
GMM_MODEL_PATH = BASE_DIR / "gmm_model.pkl"
REPORT_TEX_PATH = BASE_DIR / "final_report.tex"

# ==========================================
# 1. DATA EXTRACTION (The "Volume" Pillar)
# ==========================================

def fetch_market_data():
    """Fetches real historical market data."""
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = '2014-01-01'
    
    djia = yf.download('^DJI', start=start_date, end=end_date)
    vix = yf.download('^VIX', start=start_date, end=end_date)
    
    df = pd.DataFrame(index=djia.index)
    df['Price_Change_Pct'] = djia['Close'].pct_change() * 100
    df['Volatility_VIX'] = vix['Close']
    df = df.dropna().reset_index()
    
    return df

def fetch_real_news_data(dates):
    """
    Fetches real live RSS news for today, and uses dynamic synthetic generation
    for historical dates to ensure the MapReduce algorithm has varied data.
    """
    headlines_dict = {}
    
    try:
        rss_url = "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^DJI,SPY,TLT"
        feed = feedparser.parse(rss_url)
        live_news = " ".join([entry.title for entry in feed.entries[:5]])
        today_str = datetime.today().strftime('%Y-%m-%d')
        headlines_dict[today_str] = live_news
    except Exception as e:
        print(f"RSS Feed Warning: {e}")

    subjects = ["Federal Reserve", "Inflation", "Tech stocks", "Oil prices", "Treasury yields", "Consumer spending", "Geopolitical tensions", "Supply chains", "Housing market", "Unemployment data"]
    verbs = ["surge", "plummet", "stabilize", "trigger selloff", "rally", "collapse", "spark panic", "boost confidence", "signal recession", "exceed expectations"]
    impacts = ["global markets", "investor sentiment", "future rate cuts", "corporate earnings", "emerging economies"]
    
    random.seed(42) 
    headlines = []
    
    for d in dates:
        date_str = d.strftime('%Y-%m-%d')
        if date_str in headlines_dict and headlines_dict[date_str].strip():
            headlines.append(headlines_dict[date_str])
        else:
            h1 = f"{random.choice(subjects)} {random.choice(verbs)} impacting {random.choice(impacts)}."
            h2 = f"Analysts closely monitor how {random.choice(subjects)} will affect {random.choice(impacts)}."
            headlines.append(f"{h1} {h2}")
            
    return pd.DataFrame({'Date': dates, 'Combined_News': headlines})

# ==========================================
# 2. DATA TRANSFORMATION & NLP
# ==========================================

def score_sentiment(texts):
    """FinBERT NLP Macroeconomic sentiment scoring."""
    sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    scores = []
    
    for text in texts:
        try:
            result = sentiment_analyzer(str(text)[:512])[0]
            if result['label'] == 'positive':
                scores.append(result['score'])
            elif result['label'] == 'negative':
                scores.append(-result['score'])
            else:
                scores.append(0.0)
        except:
            scores.append(0.0)
    return scores

# ==========================================
# 3. MACHINE LEARNING & EXPORT (The "Prediction" Pillar)
# ==========================================

def train_and_export_models(df):
    """
    Trains models offline, compares accuracy, and exports the best to disk (.pkl).
    This keeps the Streamlit frontend incredibly fast.
    """
    print("Training Machine Learning Models...")
    features = ['Volatility_VIX', 'Sentiment_Score']
    X = df[features]
    y = df['Is_Anomaly']
    
    # Train-Test Split for validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model 1: Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    
    # Model 2: XGBoost
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb.fit(X_train, y_train)
    xgb_acc = accuracy_score(y_test, xgb.predict(X_test))
    
    # Select Best Model
    best_model = xgb if xgb_acc > rf_acc else rf
    model_name = "XGBoost" if xgb_acc > rf_acc else "Random Forest"
    print(f"Selected Champion Model: {model_name} (Accuracy: {max(rf_acc, xgb_acc):.2f})")
    
    # Train GMM Unsupervised Clustering
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(X)
    
    # Export Models to disk for Streamlit to load
    joblib.dump(best_model, CHAMPION_MODEL_PATH)
    joblib.dump(gmm, GMM_MODEL_PATH)
    print("Models successfully exported to disk.")

# ==========================================
# 4. DATABASE PIPELINE EXECUTION
# ==========================================

def build_database():
    print("Initiating Nightly Enterprise Data Pipeline...")
    
    market_df = fetch_market_data()
    news_df = fetch_real_news_data(market_df['Date'])
    
    joined = pd.merge(market_df, news_df, on='Date')
    joined['Sentiment_Score'] = score_sentiment(joined['Combined_News'].tolist())
    joined['Is_Anomaly'] = ((joined['Price_Change_Pct'] < -2.0) | (joined['Volatility_VIX'] > 30.0)).astype(int)
    
    # Train and save models on the fresh data
    train_and_export_models(joined)
    
    # Save to SQLite Database
    conn = sqlite3.connect(DB_PATH)
    try:
        old_db = pd.read_sql('SELECT Date, Agent_Report FROM macro_data WHERE Agent_Report IS NOT NULL', conn)
        old_db['Date'] = pd.to_datetime(old_db['Date'])
        joined = pd.merge(joined, old_db, on='Date', how='left')
    except (sqlite3.OperationalError, pd.errors.DatabaseError):
        joined['Agent_Report'] = None
        
    joined.to_sql('macro_data', conn, if_exists='replace', index=False)
    conn.close()
    print("Pipeline Execution Complete. Database updated.")

def generate_latex_report(db_path=DB_PATH):
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql('SELECT * FROM macro_data ORDER BY Date DESC LIMIT 1', conn)
        if df.empty: return
        latest = df.iloc[0]
        latest_date = pd.to_datetime(latest['Date']).strftime('%B %d, %Y')
        anomaly_status = "CRISIS DETECTED" if latest['Is_Anomaly'] == 1 else "STANDARD REGIME"
        conn.close()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    latex_content = f"""
    \\documentclass{{article}}
    \\usepackage[margin=1in]{{geometry}}
    \\begin{{document}}
    \\title{{Autonomous Macro-Sentiment Report}}
    \\date{{{latest_date}}}
    \\maketitle
    \\section*{{Daily Executive Summary}}
    The enterprise pipeline executed successfully for \\textbf{{{latest_date}}}.
    \\begin{{itemize}}
        \\item \\textbf{{DJIA Price Change:}} {latest['Price_Change_Pct']:.2f}\\%
        \\item \\textbf{{Volatility (VIX):}} {latest['Volatility_VIX']:.2f}
        \\item \\textbf{{System Status:}} Market Regime categorized as \\textbf{{{anomaly_status}}}.
    \\end{{itemize}}
    \\end{{document}}
    """
    with open(REPORT_TEX_PATH, "w") as f:
        f.write(latex_content)
    try:
        subprocess.run(["pdflatex", "final_report.tex"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=BASE_DIR)
    except: pass

if __name__ == "__main__":
    build_database()
    generate_latex_report()
