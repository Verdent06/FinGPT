# data/reference_data.py

# Sector P/E averages
sector_pe_avg = {
    "Technology": 30,
    "Financial Services": 15,
    "Healthcare": 25,
    "Consumer Cyclical": 22,
    "Communication Services": 28,
    "Industrials": 20,
    "Energy": 12,
    "Utilities": 18,
    "Materials": 21,
    "Real Estate": 17
}

# Sector ETFs
sector_etf_map = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Communication Services": "XLC",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Materials": "XLB",
    "Real Estate": "XLRE"
}

# Sector tickers
sector_tickers_map = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "NVDA", "AMD", "INTC", "CSCO", "ORCL", "ADBE", "CRM"],
    "Healthcare": ["JNJ", "PFE", "MRK", "ABBV", "TMO", "AMGN", "GILD", "BMY", "LLY", "REGN"],
    "Financial Services": ["JPM", "BAC", "C", "WFC", "GS", "MS", "PNC", "BK", "USB", "AXP"],
    "Consumer Cyclical": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "BKNG", "GM"],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "CHTR", "TMUS", "EA"],
    "Industrials": ["UNP", "UPS", "BA", "CAT", "DE", "LMT", "HON", "MMM", "GE", "CSX"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "PSX", "VLO", "MPC", "OXY", "KMI"],
    "Utilities": ["NEE", "DUK", "SO", "D", "EXC", "AEP", "PEG", "SRE", "EIX", "ED"],
    "Materials": ["LIN", "SHW", "NEM", "APD", "ECL", "FCX", "DD", "DDOG", "IFF", "VMC"],
    "Real Estate": ["PLD", "AMT", "CCI", "EQIX", "PSA", "SPG", "DLR", "AVB", "O", "VTR"]
}
