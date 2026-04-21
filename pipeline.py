import yfinance as yf
import pandas as pd
import sqlite3
import random
from datetime import datetime, timedelta
from transformers import pipeline
import subprocess

# ==========================================
# 1. DATA EXTRACTION (The "Volume" Pillar)
# ==========================================

def fetch_market_data():
    """
    Dynamically fetches market data from Yahoo Finance up to the current day.
    Calculates the daily percentage change of the DJIA and extracts the VIX fear gauge.
    """
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = '2014-01-01'
    
    djia = yf.download('^DJI', start=start_date, end=end_date)
    vix = yf.download('^VIX', start=start_date, end=end_date)
    
    df = pd.DataFrame(index=djia.index)
    df['Price_Change_Pct'] = djia['Close'].pct_change() * 100
    df['Volatility_VIX'] = vix['Close']
    df = df.dropna().reset_index()
    
    return df

def fetch_news_data(dates):
    """
    Simulates global macroeconomic headlines for portfolio demonstration.
    In a true production environment, this connects to a paid firehose API.
    """
    # Pools of realistic financial terminology
    subjects = ["Federal Reserve", "Inflation", "Tech stocks", "Oil prices", "Treasury yields", "Consumer spending", "Geopolitical tensions", "Supply chains", "Housing market", "Unemployment data"]
    verbs = ["surge", "plummet", "stabilize", "trigger selloff", "rally", "collapse", "spark panic", "boost confidence", "signal recession", "exceed expectations"]
    impacts = ["global markets", "investor sentiment", "future rate cuts", "corporate earnings", "emerging economies"]
    
    headlines = []
    # Seed the randomizer so the database builds consistently
    random.seed(42) 
    
    for _ in dates:
        # Create a synthetic daily news summary mimicking real financial reporting
        h1 = f"{random.choice(subjects)} {random.choice(verbs)} impacting {random.choice(impacts)}."
        h2 = f"Analysts closely monitor how {random.choice(subjects)} will affect {random.choice(impacts)}."
        headlines.append(f"{h1} {h2}")
        
    news_df = pd.DataFrame({
        'Date': dates,
        'Combined_News': headlines
    })
    return news_df

# ==========================================
# 2. DATA TRANSFORMATION (The "Prediction" Pillar)
# ==========================================

def score_sentiment(texts):
    """
    Utilizes HuggingFace's FinBERT NLP model to analyze the macroeconomic
    sentiment of the combined daily news headlines.
    """
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
# 3. DATA LOADING & AI MEMORY PRESERVATION
# ==========================================

def build_database():
    """
    Executes the ETL pipeline: Merges market and NLP data, calculates anomalies,
    preserves historical AI outputs, and saves to SQLite.
    """
    print("Initiating Nightly Enterprise Data Pipeline...")
    
    market_df = fetch_market_data()
    news_df = fetch_news_data(market_df['Date'])
    
    joined = pd.merge(market_df, news_df, on='Date')
    joined['Sentiment_Score'] = score_sentiment(joined['Combined_News'].tolist())
    
    joined['Is_Anomaly'] = ((joined['Price_Change_Pct'] < -2.0) | (joined['Volatility_VIX'] > 30.0)).astype(int)
    
    conn = sqlite3.connect('macro_data.db')
    try:
        old_db = pd.read_sql('SELECT Date, Agent_Report FROM macro_data WHERE Agent_Report IS NOT NULL', conn)
        old_db['Date'] = pd.to_datetime(old_db['Date'])
        joined = pd.merge(joined, old_db, on='Date', how='left')
    except:
        joined['Agent_Report'] = None
        
    joined.to_sql('macro_data', conn, if_exists='replace', index=False)
    conn.close()
    print("Pipeline Execution Complete. Database successfully updated.")

# ==========================================
# 4. DYNAMIC PDF REPORT GENERATION
# ==========================================

def generate_latex_report(db_path='macro_data.db'):
    """
    Connects to the database, extracts the latest day's metrics, and dynamically 
    compiles a LaTeX PDF executive summary.
    """
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql('SELECT * FROM macro_data ORDER BY Date DESC LIMIT 1', conn)
        
        if df.empty:
            print("Database is empty. Cannot generate report.")
            return
            
        latest = df.iloc[0]
        
        raw_date = pd.to_datetime(latest['Date'])
        latest_date = raw_date.strftime('%B %d, %Y')
        
        vix = latest['Volatility_VIX']
        sentiment = latest['Sentiment_Score']
        price_change = latest['Price_Change_Pct']
        anomaly_status = "Detected 🔴" if latest['Is_Anomaly'] == 1 else "Normal 🟢"
        
        conn.close()
    except Exception as e:
        print(f"Error fetching data for report: {e}")
        return

    latex_content = f"""
    \\documentclass{{article}}
    \\usepackage[margin=1in]{{geometry}}
    \\usepackage{{booktabs}}
    
    \\title{{Autonomous Macro-Sentiment Report}}
    \\author{{Multi-Agent Intelligence System}}
    \\date{{{latest_date}}}
    
    \\begin{{document}}
    \\maketitle
    
    \\section*{{Daily Executive Summary}}
    The enterprise pipeline successfully executed its nightly data ingestion and inference update for \\textbf{{{latest_date}}}.
    
    \\section*{{Key Market Metrics}}
    \\begin{{itemize}}
        \\item \\textbf{{DJIA Price Change:}} {price_change:.2f}\\%
        \\item \\textbf{{Volatility (VIX):}} {vix:.2f}
        \\item \\textbf{{FinBERT Sentiment Score:}} {sentiment:.4f}
        \\item \\textbf{{System Status:}} Market Regime categorized as \\textbf{{{anomaly_status}}}.
    \\end{{itemize}}
    
    \\end{{document}}
    """
    
    with open("final_report.tex", "w") as f:
        f.write(latex_content)
        
    try:
        subprocess.run(["pdflatex", "report.tex"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅ Dynamic PDF Report Generated Successfully.")
    except Exception as e:
        print(f"❌ LaTeX Compilation Failed. Make sure pdflatex is installed on the server: {e}")

if __name__ == "__main__":
    build_database()
    generate_latex_report()
