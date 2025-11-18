# utils/config_loader.py
import json
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.json"

def load_config():
    """Load config.json and return as a dictionary"""
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

CONFIG = load_config()
