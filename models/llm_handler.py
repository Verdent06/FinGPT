# models/llm_handler.py
from llama_cpp import Llama
import warnings, sys, os, contextlib
from utils.config_loader import CONFIG

def load_llm():

    """
    Load LLaMA 2 model with suppressed stderr to avoid cluttered output.
    """

    model_path=CONFIG["llm"]["model_path"]
    USE_LLM = CONFIG["llm"].get("use_llm", True) # enable/disable llama for testing

    @contextlib.contextmanager
    def suppress_stderr():
        with open(os.devnull, "w") as devnull:
            old_stderr = sys.stderr
            sys.stderr = devnull
            try:
                yield
            finally:
                sys.stderr = old_stderr

    with suppress_stderr():
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=8,
            n_gpu_layers=35,
            verbose=False
        )
    warnings.filterwarnings("ignore")
    return llm

def get_llm_recommendation(llm, ticker, info, fundamentals_metrics, final_score, news_sentiment, macro_score, company_name):
    """
    Placeholder for LLM recommendation.
    You can replace with actual prompt & LLM call logic.

    Returns:
    - numeric score (0-100)
    - recommendation string ("Buy"/"Hold"/"Sell")
    - justification string
    """
    # TODO: Implement LLM prompt & parse output
    return 50, "Hold", "LLM placeholder justification"


