# data/news_handler.py
import finnhub
import time
from datetime import date, datetime, timedelta
from utils.config_loader import CONFIG

def fetch_company_news(ticker, company_name, api_key=CONFIG["finnhub"]["api_key"], max_articles=500):
    """
    Fetch company news using Finnhub API with Month-by-Month pagination
    to bypass API truncation limits.
    """
    client = finnhub.Client(api_key)
    news_articles = []
    
    # We want 1 year of data
    end_date = date.today()
    start_date = end_date - timedelta(days=365)
    
    # Create monthly chunks to ensure we get older data
    # (APIs often only return the last ~100 items per request)
    current = start_date
    while current < end_date:
        next_month = current + timedelta(days=30)
        if next_month > end_date:
            next_month = end_date
            
        _from = current.strftime("%Y-%m-%d")
        _to = next_month.strftime("%Y-%m-%d")
        
        # print(f"  [API] Fetching news for {ticker}: {_from} to {_to}")
        
        try:
            # Fetch chunk
            chunk = client.company_news(ticker, _from=_from, to=_to)
            
            # Process chunk immediately
            for art in chunk:
                related_ticker = art.get("related", "").upper()
                # Basic Filtering
                if any(t.strip() == ticker.upper() for t in related_ticker.split(',')):
                    headline = art.get("headline", "")
                    summary = art.get("summary", "")
                    
                    # Content Filter (Simple keyword match)
                    if company_name.lower() in headline.lower() or \
                       company_name.lower() in summary.lower():
                        
                        news_articles.append({
                            "title": headline,
                            "description": summary,
                            "date": date.fromtimestamp(art.get("datetime", 0)).strftime("%Y-%m-%d"),
                            "datetime": art.get("datetime", 0) # Keep raw TS
                        })
            
            # Rate Limit Protection (Finnhub Free = 60 calls/min)
            time.sleep(0.5) 
            
        except Exception as e:
            print(f"  [API Error] Failed to fetch chunk {_from}: {e}")
        
        # Move to next chunk
        current = next_month

    # Deduplicate (API might return overlapping items)
    # Use a dictionary keyed by title to remove dupes
    unique_articles = {art['title']: art for art in news_articles}.values()
    news_articles = list(unique_articles)

    # Sort by date (Newest first)
    news_articles.sort(key=lambda x: x["datetime"], reverse=True)

    # Apply Limit
    if len(news_articles) > max_articles:
        news_articles = news_articles[:max_articles]

    return news_articles