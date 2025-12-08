# models/llm_handler.py
import json
import os
import sys
import contextlib
import warnings
from llama_cpp import Llama
from utils.config_loader import CONFIG

def load_llm():
    """
    Load LLaMA model with suppressed stderr to avoid cluttered output.
    """
    model_path = CONFIG["llm"]["model_path"]
    
    # Defensive check for model path
    if not model_path or not os.path.exists(model_path):
        print(f"WARNING: Model path not found at {model_path}. LLM features will be disabled.")
        return None

    @contextlib.contextmanager
    def suppress_stderr():
        with open(os.devnull, "w") as devnull:
            old_stderr = sys.stderr
            sys.stderr = devnull
            try:
                yield
            finally:
                sys.stderr = old_stderr

    try:
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
    except Exception as e:
        print(f"Error loading LLM: {e}")
        return None

def get_llm_recommendation(llm, ticker, info, fundamentals, final_score, news_sentiment, macro_score, company_name):
    """
    Generates an investment recommendation score (-1 to 1) using the LLM 
    by synthesizing quantitative metrics and qualitative news data.
    """
    if not llm:
        # Return neutral default if LLM fails (score, analysis)
        return 0.0, "LLM not loaded."

    # 1. Construct the Context String
    # Serializing the data so the LLM can "read" the math.
    system_context = f"""
    You are a senior financial analyst. Analyze the provided data for {company_name} ({ticker}) and provide an investment recommendation.
    
    DATA SNAPSHOT:
    1. Fundamental Score (0 to 1): {fundamentals.get('E', 0):.2f} (Growth) | {fundamentals.get('V', 0):.2f} (Valuation)
    2. Macroeconomic Score (0 to 1): {macro_score:.2f} (Higher is better environment)
    3. News Sentiment Score (-1 to 1): {news_sentiment:.2f}
    
    Current Price: {info.get('currentPrice', 'N/A')}
    P/E Ratio: {info.get('trailingPE', 'N/A')}
    Sector: {info.get('sector', 'Unknown')}
    """

    # 2. The Prompt
    # We specifically ask for a float score between -1.0 and 1.0
    prompt = f"""
    {system_context}
    
    Based strictly on the data above, provide a score between -1.0 (Strong Sell) and 1.0 (Strong Buy).
    
    RESPONSE FORMAT (JSON ONLY):
    {{
        "recommendation": "Buy" | "Hold" | "Sell",
        "score": (float between -1.0 and 1.0),
        "rationale": "A concise 3-sentence explanation citing specific metrics from the data."
    }}
    
    JSON Response:
    """

    # 3. Generate
    try:
        output = llm(
            prompt, 
            max_tokens=256, 
            stop=["}", "\n\n"], 
            echo=False,
            temperature=0.2 # Low temp for consistent formatting
        )
        
        # 4. Parse Response
        text_output = output['choices'][0]['text'].strip()
        
        # Robustness: Ensure we close the JSON if the LLM cut off early
        if not text_output.endswith("}"): 
            text_output += "}"
            
        # Clean up potential preamble
        start_idx = text_output.find("{")
        if start_idx != -1:
            text_output = text_output[start_idx:]
            
        result = json.loads(text_output)
        
        # Map result to (score, analysis_text)
        score = float(result.get("score", 0.0))
        analysis_text = result.get("rationale") or result.get("analysis") or result.get("recommendation") or "No analysis generated."
        return score, analysis_text

    except Exception as e:
        print(f"LLM Generation Error: {e}")
        return 0.0, "Error generating LLM insight."