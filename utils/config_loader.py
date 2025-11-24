# utils/config_loader.py
import json
import os
from pathlib import Path
from dotenv import load_dotenv 

# Load environment variables from .env file
load_dotenv() #

CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.json" 

def load_config():
    """
    Load configuration from the JSON file and override/augment with
    environment variables for sensitive/dynamic settings.
    """
    with open(CONFIG_PATH, "r") as f:
        config_data = json.load(f)

    # 1. Inject API Keys from Environment (Priority 1: Secure)
    config_data["fred"]["api_key"] = os.getenv("FRED_API_KEY")
    config_data["finnhub"]["api_key"] = os.getenv("FINNHUB_API_KEY")
    
    # 2. Inject LLM path from Environment (Priority 1: Environment)
    # Use config file value as a low-priority fallback if ENV is not set.
    config_data["llm"]["model_path"] = os.getenv(
        "LLAMA_MODEL_PATH", 
        config_data["llm"]["model_path"] # Fallback to JSON value
    )
    
    # Convert 'use_llm' to boolean since environment variables are strings
    use_llm_env = os.getenv("USE_LLM")
    if use_llm_env is not None:
        config_data["llm"]["use_llm"] = use_llm_env.lower() in ('true', '1', 't')

    return config_data #

CONFIG = load_config() #