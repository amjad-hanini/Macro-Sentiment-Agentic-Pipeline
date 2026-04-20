import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from transformers import pipeline

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

if __name__ == "__main__":
    build_database()
