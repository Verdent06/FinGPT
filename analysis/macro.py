# analysis/macro.py
import numpy as np
from fredapi import Fred
from utils.config_loader import CONFIG

def get_macro_info(api_key=CONFIG["fred"]["api_key"]):
    fred = Fred(api_key=api_key)
    unemployment = fred.get_series('UNRATE')[-1]
    cpi = fred.get_series('CPIAUCSL')
    cpi_yoy = (cpi[-1] - cpi[-12]) / cpi[-12]
    interest_rate = fred.get_series('FEDFUNDS')[-1]
    return {"unemployment": unemployment, "cpi_yoy": cpi_yoy, "interest_rate": interest_rate}

def calc_macro_score(indicators: dict):
    interest_rate_score = np.tanh((5 - indicators["interest_rate"])/2)
    cpi_score = np.tanh((5 - indicators["cpi_yoy"]*100)/2)
    unemployment_score = np.tanh((5 - indicators["unemployment"])/2)
    
    macro_score = 0.4*interest_rate_score + 0.4*cpi_score + 0.2*unemployment_score
    return macro_score
