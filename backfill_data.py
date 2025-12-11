# backfill_data.py
import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import pytz

from data.news_handler import fetch_company_news
from data.sentiment_cache import get_cached_sentiment, update_cache
from analysis.llm_sentiment import LLMSentimentAnalyzer
from analysis.mpnet_sentiment import mpnet_analyzer
from models import clf_handler, mpnet_embedder, llm_handler
from analysis.score_calculator import calculate_fundamental_score 
from analysis.fundamentals import get_fundamentals

# --- CONFIGURATION ---
TICKER = "AAPL"
LOOKBACK_DAYS = 365    
HOLDING_PERIOD = 14    
STEP_DAYS = 7  
DATA_FILE = "data/training_data.csv"        

def get_trading_dates(hist, start_date):
    """Finds the next valid trading day and the exit day."""
    try:
        start_idx = hist.index.get_indexer([start_date], method='bfill')[0]
    except:
        return None, None

    entry_idx = start_idx + 1
    exit_idx = entry_idx + HOLDING_PERIOD
    
    if exit_idx >= len(hist):
        return None, None 
        
    return hist.iloc[entry_idx], hist.iloc[exit_idx]

def save_incremental(new_row_dict):
    """
    Saves a single row to the CSV immediately.
    Handles reading, appending, deduplicating, and writing back.
    """
    new_df = pd.DataFrame([new_row_dict])
    
    if os.path.exists(DATA_FILE):
        try:
            existing = pd.read_csv(DATA_FILE)
            # Combine old and new
            updated_df = pd.concat([existing, new_df], ignore_index=True)
        except pd.errors.EmptyDataError:
             updated_df = new_df
    else:
        updated_df = new_df
    
    # Remove duplicates to ensure clean data if we restart
    updated_df.drop_duplicates(subset=['date', 'ticker'], keep='last', inplace=True)
    
    # Save back to disk
    updated_df.to_csv(DATA_FILE, index=False)
    return len(updated_df)

def backfill_ticker(ticker):
    print(f"\n🚀 Starting Time Machine for {ticker}...")
    
    # 1. Initialize Models
    print("   [1/5] Loading AI Models...")
    clf = clf_handler.load_trained_clf()
    embedder = mpnet_embedder.get_embedder()
    llm = llm_handler.load_llm()
    llm_analyzer = LLMSentimentAnalyzer(llm)

    # 2. Fetch Market Data
    print("   [2/5] Downloading Price History...")
    stock = yf.Ticker(ticker)
    hist = stock.history(period="2y") 
    hist.index = hist.index.tz_localize(None)

    # 3. Fetch News
    print("   [3/5] Downloading News History (Chunked)...")
    all_news = fetch_company_news(ticker, ticker, max_articles=1000)
    
    # Verify Date Range
    if all_news:
        newest_date = all_news[0]['date']
        oldest_date = all_news[-1]['date']
        print(f"         Found {len(all_news)} articles.")
        print(f"         📅 Newest: {newest_date}")
        print(f"         📅 Oldest: {oldest_date}")
    else:
        print("         ❌ ERROR: No news found!")
        return

    # 4. Get Fundamentals
    print("   [4/5] Loading Fundamentals...")
    fund_data = get_fundamentals(ticker)
    fund_score = calculate_fundamental_score(fund_data)

    # 5. The Time Loop
    print("   [5/5] Running Simulation...")
    
    end_date = datetime.now()
    current_sim_date = end_date - timedelta(days=LOOKBACK_DAYS)

    while current_sim_date < end_date - timedelta(days=HOLDING_PERIOD):
        
        # --- A. Setup Dates ---
        entry_row, exit_row = get_trading_dates(hist, current_sim_date)
        
        if entry_row is None:
            current_sim_date += timedelta(days=STEP_DAYS)
            continue

        # --- B. Calculate Returns ---
        buy_price = entry_row['Open']
        sell_price = exit_row['Open']
        pct_return = (sell_price - buy_price) / buy_price

        # --- C. Filter News ---
        window_start = current_sim_date - timedelta(days=30)
        
        relevant_news = []
        for art in all_news:
            art_ts = art.get('datetime', 0)
            art_date = datetime.fromtimestamp(art_ts)
            
            if window_start <= art_date <= current_sim_date:
                relevant_news.append(art)
        
        if not relevant_news:
            current_sim_date += timedelta(days=STEP_DAYS)
            continue

        # --- D. Score News (Using Cache) ---
        # MPNet
        label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        mpnet_res = mpnet_analyzer(relevant_news, clf, embedder, label_map)
        mp_score = np.mean([n["sentiment_score"] for n in mpnet_res]) if mpnet_res else 0

        # LLaMA (Cached)
        llm_scores = []
        for article in relevant_news:
            text_key = f"{article.get('title','')}. {article.get('description','')}"
            cached = get_cached_sentiment(text_key)
            
            if cached:
                llm_scores.append(cached['score'] * cached.get('confidence', 1.0))
            else:
                # MISS - Run LLaMA and Cache
                print(f"      [LLaMA Running] {article['title'][:30]}...")
                res = llm_analyzer.analyze_single_article(article)
                update_cache(text_key, res)
                llm_scores.append(res['sentiment_score'] * res.get('confidence', 1.0))
        
        llm_final = np.mean(llm_scores) if llm_scores else 0

        # --- E. Log Data Point & INCREMENTAL SAVE ---
        date_str = current_sim_date.strftime("%Y-%m-%d")
        
        row_data = {
            "date": date_str,
            "ticker": ticker,
            "fund_score": fund_score,
            "mpnet_score": mp_score,
            "llm_score": llm_final,
            "price_at_analysis": buy_price,
            "target_return": pct_return
        }
        
        # Save immediately
        total_rows = save_incremental(row_data)
        
        print(f"   💾 Saved | {date_str} | LLaMA: {llm_final:.2f} | Return: {pct_return:.2%} | Total Rows: {total_rows}")
        
        current_sim_date += timedelta(days=STEP_DAYS)

    print(f"\n✅ Backfill complete for {ticker}")

if __name__ == "__main__":
    # You can change this list or interrupt anytime
    tickers = ["AAPL", "MSFT", "GOOG", "TSLA", "NVDA"]
    for t in tickers:
        try:
            backfill_ticker(t)
        except KeyboardInterrupt:
            print("\n\n⚠️ Stopped by user. Data saved up to this point.")
            break
        except Exception as e:
            print(f"Error processing {t}: {e}")