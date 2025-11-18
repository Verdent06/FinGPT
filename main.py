# main.py
from analysis import fundamentals, macro, score_calculator
from data import news_handler, yahoo_handler, phrasebank_loader
from models import  llm_handler
from utils import helpers, logging_setup
import yfinance as yf
import numpy as np
import warnings
import logging



# ------------ Warnings Suppressing ---------------
warnings.simplefilter("ignore")

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)



# ----------------- Setup Logging -----------------
logger = logging_setup.setup_logging()

# ----------------- User Input -----------------
ticker = input("\nEnter a stock ticker symbol (e.g., AAPL): ").upper()

# ----------------- Stock Info -----------------
info, hist = yahoo_handler.get_stock_info(ticker)
company_name = helpers.extract_company_name(info.get("longName", ticker))
sector = info.get("sector", "Unknown")

# ----------------- Macro Data -----------------
macro_indicators = macro.get_macro_info()

# ----------------- Sector ETF -----------------
etf_symbol = fundamentals.sector_etf_map.get(sector, "SPY")
etf_info, hist_etf = yahoo_handler.get_stock_info(etf_symbol)

# ----------------- Calculate Fundamentals -----------------
E = fundamentals.calc_earnings_growth(yf.Ticker(ticker))
V = fundamentals.calc_valuation(yf.Ticker(ticker), sector)
M = fundamentals.calc_momentum(hist, hist_etf)
A = fundamentals.calc_analyst_sentiment(yf.Ticker(ticker))
S = fundamentals.calc_sector_health(hist_etf)
C = fundamentals.calc_company_maturity(info.get("marketCap", 1e9))

fundamentals_dict = {"E": E, "V": V, "M": M, "A": A, "S": S, "C": C}

# ----------------- News Sentiment -----------------
raw_news = news_handler.fetch_company_news(ticker, company_name)
sentiment_result = score_calculator.get_hybrid_sentiment(raw_news)
hybrid_sentiment = sentiment_result["combined_score"]

# ----------------- Calculate Scores -----------------
macro_score = macro.calc_macro_score(macro_indicators)
final_score, recommendation = score_calculator.calculate_final_score(fundamentals_dict, hybrid_sentiment, macro_score)

# ----------------- LLM Recommendation -----------------
llm = llm_handler.load_llm()
llm_score, llm_recommendation, llm_justification = llm_handler.get_llm_recommendation(
    llm, ticker, info, fundamentals_dict, final_score, hybrid_sentiment, macro_score, company_name
)

# Combine scores
final_combined_score = 0.8*final_score + 0.2*llm_score

# ----------------- Display Results -----------------
helpers.display_results(
    ticker, info, raw_news, hybrid_sentiment, final_score, final_combined_score, llm_recommendation
)
llm.close()