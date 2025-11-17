# data/yahoo_handler.py
import yfinance as yf
import sys, os, contextlib

@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, "w") as fnull:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = fnull, fnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err


def get_stock_info(ticker, period="1y"):
    """
    Returns:
    - info: dictionary of stock information
    - hist: historical price data as DataFrame
    """
    with suppress_stdout_stderr():
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period=period)
    return info, hist


