from pydantic import BaseModel

class StockRequest(BaseModel):
    ticker: str

class FundamentalsResponse(BaseModel):
    ticker: str
    earnings_growth: float
    valuation: float
    momentum: float
    stability: float
    analyst_sentiment: float
