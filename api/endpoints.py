from fastapi import APIRouter, HTTPException
from api.models import StockRequest
from analysis import fundamentals

router = APIRouter()


@router.post("/fundamentals/")
def get_fundamentals_endpoint(req: StockRequest):
    try:
        result = fundamentals.get_fundamentals(req.ticker)
        return {
            "ticker": req.ticker,
            "fundamentals": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
