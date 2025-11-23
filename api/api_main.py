# api/api_main.py
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import yfinance as yf

from analysis.fundamentals import get_fundamentals
from analysis.macro import get_macro_info, calc_macro_score
from analysis.score_calculator import get_hybrid_sentiment, calculate_final_score
from data.news_handler import fetch_company_news
from data.yahoo_handler import get_stock_info
from utils.helpers import extract_company_name
from models import clf_handler, mpnet_embedder, llm_handler

# Global variables to store loaded models
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at startup and keep in memory"""
    print("Loading models at startup...")
    
    # Load MPNet embedder
    models["embedder"] = mpnet_embedder.get_embedder()
    
    # Load trained classifier
    models["clf"] = clf_handler.load_trained_clf()
    
    # Load LLaMA model
    models["llm"] = llm_handler.load_llm()
    
    print("All models loaded successfully!")
    
    yield  # App runs here
    
    # Cleanup on shutdown
    print("Shutting down and cleaning up models...")
    if "llm" in models:
        models["llm"].close()

app = FastAPI(
    title="FinGPT API", 
    version="0.1",
    lifespan=lifespan
)


@app.get("/")
async def root():
    return {
        "message": "FinGPT API",
        "version": "0.1",
        "endpoints": [
            "/fundamentals/{symbol}",
            "/macro",
            "/sentiment/{symbol}",
            "/score/{symbol}",
            "/analyze/{symbol}"
        ]
    }


@app.get("/fundamentals/{symbol}")
async def fundamentals_endpoint(symbol: str):
    try:
        data = get_fundamentals(symbol)
        return {"symbol": symbol.upper(), "fundamentals": data}
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Error fetching fundamentals for {symbol}: {str(e)}"
        )


@app.get("/macro")
async def macro_endpoint():
    try:
        indicators = get_macro_info()
        macro_score = float(calc_macro_score(indicators))
        indicators_clean = {k: float(v) for k, v in indicators.items()}

        return {
            "macro": indicators_clean,
            "macro_score": macro_score
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error fetching macro data: {str(e)}"
        )


@app.get("/sentiment/{symbol}")
async def sentiment_endpoint(symbol: str):
    try:
        # Get company info
        info, _ = get_stock_info(symbol)
        company_name = extract_company_name(info.get("longName", symbol))
        
        # Fetch news
        raw_news = fetch_company_news(symbol, company_name)
        
        if not raw_news:
            return {
                "symbol": symbol.upper(),
                "sentiment": {
                    "mpnet_score": 0.0,
                    "llm_score": 0.0,
                    "llm_confidence": 0.0,
                    "combined_score": 0.0,
                    "combined_label": "Neutral",
                    "num_articles": 0,
                    "message": "No news articles found"
                }
            }
        
        # Calculate hybrid sentiment using pre-loaded models
        sentiment_result = get_hybrid_sentiment(raw_news, symbol)
        sentiment_result["num_articles"] = len(raw_news)
        
        return {
            "symbol": symbol.upper(),
            "sentiment": sentiment_result
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error fetching sentiment for {symbol}: {str(e)}"
        )


@app.get("/score/{symbol}")
async def score_endpoint(symbol: str):
    try:
        # Get fundamentals
        fundamentals = get_fundamentals(symbol)
        
        # Get macro data
        macro_indicators = get_macro_info()
        macro_score = calc_macro_score(macro_indicators)
        
        # Get sentiment
        info, _ = get_stock_info(symbol)
        company_name = extract_company_name(info.get("longName", symbol))
        raw_news = fetch_company_news(symbol, company_name)
        
        if raw_news:
            sentiment_result = get_hybrid_sentiment(raw_news, symbol)
            news_sentiment = sentiment_result["combined_score"]
        else:
            news_sentiment = 0.0
        
        # Calculate final score
        final_score, recommendation = calculate_final_score(
            fundamentals, 
            news_sentiment, 
            macro_score
        )
        
        return {
            "symbol": symbol.upper(),
            "final_score": float(final_score),
            "recommendation": recommendation,
            "components": {
                "fundamentals_score": float(
                    fundamentals["E"] * 0.35 + 
                    fundamentals["V"] * 0.20 + 
                    fundamentals["M"] * 0.15 + 
                    fundamentals["A"] * 0.10 + 
                    fundamentals["S"] * 0.15 +
                    fundamentals["C"] * 0.05
                ),
                "sentiment_score": float(news_sentiment),
                "macro_score": float(macro_score)
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error calculating score for {symbol}: {str(e)}"
        )


@app.get("/analyze/{symbol}")
async def analyze_endpoint(symbol: str):
    """
    Full analysis pipeline - combines all data sources
    """
    try:
        # 1. Get fundamentals
        fundamentals = get_fundamentals(symbol)
        
        # 2. Get macro data
        macro_indicators = get_macro_info()
        macro_score = calc_macro_score(macro_indicators)
        macro_indicators_clean = {k: float(v) for k, v in macro_indicators.items()}
        
        # 3. Get sentiment
        info, _ = get_stock_info(symbol)
        company_name = extract_company_name(info.get("longName", symbol))
        raw_news = fetch_company_news(symbol, company_name)
        
        if raw_news:
            sentiment_result = get_hybrid_sentiment(raw_news, symbol)
            news_sentiment = sentiment_result["combined_score"]
        else:
            sentiment_result = {
                "mpnet_score": 0.0,
                "llm_score": 0.0,
                "combined_score": 0.0,
                "combined_label": "Neutral",
                "num_articles": 0
            }
            news_sentiment = 0.0
        
        # 4. Calculate final score
        final_score, recommendation = calculate_final_score(
            fundamentals, 
            news_sentiment, 
            macro_score
        )
        
        # 5. Compile full response
        return {
            "symbol": symbol.upper(),
            "company_name": company_name,
            "sector": fundamentals.get("sector", "Unknown"),
            "final_score": float(final_score),
            "recommendation": recommendation,
            "fundamentals": {
                "E": float(fundamentals["E"]),
                "V": float(fundamentals["V"]),
                "M": float(fundamentals["M"]),
                "A": float(fundamentals["A"]),
                "S": float(fundamentals["S"]),
                "C": float(fundamentals["C"])
            },
            "macro": {
                "indicators": macro_indicators_clean,
                "score": float(macro_score)
            },
            "sentiment": sentiment_result,
            "raw_data": {
                "current_price": info.get("currentPrice"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE")
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error analyzing {symbol}: {str(e)}"
        )