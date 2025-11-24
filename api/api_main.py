# api/api_main.py
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from service.stock_service import StockAnalysisService 
from models import clf_handler, mpnet_embedder, llm_handler

models = {}

ANALYSIS_SERVICE: StockAnalysisService = None 

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at startup and keep in memory"""
    print("Loading models at startup...")
    
    # Load models
    models["embedder"] = mpnet_embedder.get_embedder()
    models["clf"] = clf_handler.load_trained_clf()
    models["llm"] = llm_handler.load_llm()

    global ANALYSIS_SERVICE
    ANALYSIS_SERVICE = StockAnalysisService(models=models)
    
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
        data = ANALYSIS_SERVICE.get_fundamentals_only(symbol)
        return {"symbol": symbol.upper(), "fundamentals": data}
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Error fetching fundamentals for {symbol}: {str(e)}"
        )


@app.get("/macro")
async def macro_endpoint():
    try:
        macro_data = ANALYSIS_SERVICE.get_macro_data()
        return {
            "macro": macro_data["indicators"],
            "macro_score": macro_data["score"]
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error fetching macro data: {str(e)}"
        )


@app.get("/sentiment/{symbol}")
async def sentiment_endpoint(symbol: str):
    try:
        sentiment_result = ANALYSIS_SERVICE.get_sentiment_result(symbol)
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
        score_data = ANALYSIS_SERVICE.get_score_data(symbol)
        
        return {
            "symbol": symbol.upper(),
            "final_score": score_data["final_score"],
            "recommendation": score_data["recommendation"],
            "components": score_data["components"]
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error calculating score for {symbol}: {str(e)}"
        )


@app.get("/analyze/{symbol}")
async def analyze_endpoint(symbol: str):
    """
    Full analysis pipeline - uses the centralized service.
    """
    try:
        full_analysis = ANALYSIS_SERVICE.analyze_stock(symbol)
        
        return {
            "symbol": symbol.upper(),
            "company_name": full_analysis["company_name"],
            "sector": full_analysis["sector"],
            "final_score": full_analysis["final_score"],
            "recommendation": full_analysis["recommendation"],
            "fundamentals": full_analysis["fundamentals"],
            "macro": full_analysis["macro"],
            "sentiment": full_analysis["sentiment"],
            "raw_data": full_analysis["raw_data"]
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error analyzing {symbol}: {str(e)}"
        )