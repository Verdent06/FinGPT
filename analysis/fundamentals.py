# analysis/fundamentals.py
import yfinance as yf
import numpy as np
from data.reference_data import sector_tickers_map, sector_etf_map, sector_pe_avg

# Earnings Growth (E)
def calc_earnings_growth(stock):
    info = stock.info
    E_raw = info.get("earningsGrowth", 0) or 0
    E_capped = np.clip(E_raw, -0.2, 0.3)
    E = np.interp(E_capped, [-0.2, 0.3], [0, 1])
    
    try:
        earnings_3y = stock.earnings.iloc[-3:]
        cagr = (earnings_3y['Earnings'].iloc[-1] / earnings_3y['Earnings'].iloc[0])**(1/3) - 1
        E_combined = 0.7 * E + 0.3 * np.interp(np.clip(cagr, -0.2, 0.3), [-0.2, 0.3], [0,1])
    except:
        E_combined = E
    
    return E_combined

# Valuation (V)
def calc_valuation(stock, sector):
    info = stock.info
    sector_pes = []
    for t in sector_tickers_map.get(sector, []):
        try:
            info_t = yf.Ticker(t).info
            pe_t = info_t.get("trailingPE")
            if pe_t and pe_t > 0:
                sector_pes.append(pe_t)
        except:
            continue
    dynamic_sector_pe = np.mean(sector_pes) if sector_pes else 25
    
    pe_forward = info.get("forwardPE")
    pe_trailing = info.get("trailingPE")
    pe = pe_forward if pe_forward and pe_forward > 0 else (pe_trailing if pe_trailing and pe_trailing > 0 else dynamic_sector_pe)
    
    V = np.clip(dynamic_sector_pe / pe, 0, 1)
    return V

# Momentum Stability (M)
def calc_momentum(hist, hist_etf):
    hist["Close"] = hist["Close"].ffill()
    hist_etf["Close"] = hist_etf["Close"].ffill()
    
    vol_1m = np.std(hist["Close"].pct_change(21).dropna()) if len(hist) > 21 else 0.02
    vol_3m = np.std(hist["Close"].pct_change(63).dropna()) if len(hist) > 63 else 0.02
    vol_1y = np.std(hist["Close"].pct_change(252).dropna()) if len(hist) > 252 else 0.02
    volatility = np.nanmean([vol_1m, vol_3m, vol_1y])
    
    vol_sector = np.std(hist_etf["Close"].pct_change().dropna())
    rel_volatility = volatility / vol_sector
    
    rolling_max = hist["Close"].cummax()
    drawdowns = (rolling_max - hist["Close"]) / rolling_max
    max_drawdown = drawdowns.max()
    
    M_vol = np.interp(rel_volatility, [0.5, 1.5], [1, 0])
    M_dd = 1 - np.clip(max_drawdown / 0.5, 0, 1)
    
    M = 0.7 * M_vol + 0.3 * M_dd
    return np.clip(M, 0, 1)

# Analyst Sentiment (A)
def calc_analyst_sentiment(stock):
    info = stock.info
    mean_rating = info.get("recommendationMean")
    if mean_rating:
        A = np.clip((mean_rating - 1) / 4, 0, 1)
    else:
        A = {"strong_buy": 1, "buy": 0.8, "hold": 0.5, "sell": 0.2, "strong_sell": 0}.get(info.get("recommendationKey", "hold"), 0.5)
    
    num_analysts = info.get("numberOfAnalysts", 5)
    confidence_weight = min(num_analysts / 20, 1)
    A = A * confidence_weight + 0.5 * (1 - confidence_weight)
    return A

# Sector Health (S)
def calc_sector_health(hist_etf):
    sector_health = (hist_etf["Close"][-1] - hist_etf["Close"][0]) / hist_etf["Close"][0]
    S = np.clip((sector_health + 0.5)/1.5, 0, 1)
    return S

# Company Maturity (C)
def calc_company_maturity(market_cap):
    return np.interp(np.log(market_cap), [np.log(1e9), np.log(2e12)], [0, 1])
