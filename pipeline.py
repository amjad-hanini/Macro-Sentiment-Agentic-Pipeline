import yfinance as yf
import pandas as pd
import sqlite3
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
    # Dynamically set dates so the nightly CRON job always pulls fresh data
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = '2020-01-01'
    
    # Download Dow Jones Industrial Average (DJIA) and Volatility Index (VIX)
    djia = yf.download('^DJI', start=start_date, end=end_date)
    vix = yf.download('^VIX', start=start_date, end=end_date)
    
    # Clean and structure the market dataframe
    df = pd.DataFrame(index=djia.index)
    df['Price_Change_Pct'] = djia['Close'].pct_change() * 100
    df['Volatility_VIX'] = vix['Close']
    df = df.dropna().reset_index()
    
    return df

def fetch_news_data(dates):
    """
    Placeholder for the news ingestion logic. 
    In production, this queries the DuckDuckGo/Reddit API for global headlines.
    """
    # NOTE: Keep your actual news scraping logic here!
    # For pipeline integrity, we ensure it returns a DataFrame with 'Date' and 'Combined_News'
    news_df = pd.DataFrame({
        'Date': dates,
        'Combined_News': ["Sample macroeconomic headline regarding interest rates and global panic."] * len(dates)
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
    # Load the specialized financial sentiment analyzer
    sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    scores = []
    
    for text in texts:
        try:
            # Truncate text to fit BERT's 512 token limit
            result = sentiment_analyzer(str(text)[:512])[0]
            # Convert label to a numerical score for correlation modeling
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
    
    # 1. Fetch raw data
    market_df = fetch_market_data()
    news_df = fetch_news_data(market_df['Date'])
    
    # 2. Merge and Transform
    joined = pd.merge(market_df, news_df, on='Date')
    joined['Sentiment_Score'] = score_sentiment(joined['Combined_News'].tolist())
    
    # 3. Anomaly Detection (Rule-based flag for the downstream ML models)
    joined['Is_Anomaly'] = ((joined['Price_Change_Pct'] < -2.0) | (joined['Volatility_VIX'] > 30.0)).astype(int)
    
    # 4. Agentic Memory Preservation (CRITICAL FIX)
    conn = sqlite3.connect('macro_data.db')
    try:
        # Attempt to read the existing database to extract previously saved AI reports
        old_db = pd.read_sql('SELECT Date, Agent_Report FROM macro_data WHERE Agent_Report IS NOT NULL', conn)
        old_db['Date'] = pd.to_datetime(old_db['Date'])
        # Merge the old reports onto the fresh dataset
        joined = pd.merge(joined, old_db, on='Date', how='left')
    except:
        # If the database or column doesn't exist yet, initialize it cleanly
        joined['Agent_Report'] = None
        
    # 5. Load to Database
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
        # Fetch the most recent date's row
        df = pd.read_sql('SELECT * FROM macro_data ORDER BY Date DESC LIMIT 1', conn)
        
        if df.empty:
            print("Database is empty. Cannot generate report.")
            return
            
        latest = df.iloc[0]
        
        # Format the date nicely for the report
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

    # Inject the dynamic variables directly into the LaTeX template
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
    
    # Save and compile the PDF
    with open("report.tex", "w") as f:
        f.write(latex_content)
        
    try:
        subprocess.run(["pdflatex", "report.tex"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅ Dynamic PDF Report Generated Successfully.")
    except Exception as e:
        print(f"❌ LaTeX Compilation Failed. Make sure pdflatex is installed on the server: {e}")

if __name__ == "__main__":
    build_database()
    generate_latex_report()
