from fastapi import FastAPI
from api.endpoints import router

app = FastAPI(title="FinGPT API")

app.include_router(router)
