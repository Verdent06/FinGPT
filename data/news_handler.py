# data/news_handler.py
import finnhub
from datetime import date, timedelta
from utils.config_loader import CONFIG

def fetch_company_news(ticker, company_name, api_key = CONFIG["finnhub"]["api_key"] , max_articles=20):
    """
    Fetch company news using Finnhub API.
    
    Returns a list of articles with:
    - title
    - description
    - date (YYYY-MM-DD)
    """
    today = date.today()
    start_date = today - timedelta(days=365)
    
    client = finnhub.Client(api_key)
    raw_news = client.company_news(ticker, _from=start_date, to=today)

    news_articles = []
    for art in raw_news:
        related_ticker = art.get("related", "").upper()
        if any(t.strip() == ticker.upper() for t in related_ticker.split(',')):
            if company_name.lower() in art.get("headline", "").lower() or \
               company_name.lower() in art.get("summary", "").lower():
                news_articles.append({
                    "title": art.get("headline", ""),
                    "description": art.get("summary", ""),
                    "date": art.get("datetime", 0)
                })

    news_articles.sort(key=lambda x: x["date"], reverse=True)

    if len(news_articles) > max_articles:
        news_articles = news_articles[:max_articles]

    for n in news_articles:
        if isinstance(n["date"], (int, float)):
            n["date"] = date.fromtimestamp(n["date"]).strftime("%Y-%m-%d")

    return news_articles