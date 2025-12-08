# data/sentiment_cache.py
import json
import os
import hashlib

CACHE_FILE = "data/sentiment_cache.json"

def load_cache():
    """Loads the existing cache from disk."""
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        return {}

def save_cache(cache_data):
    """Saves the updated cache to disk."""
    with open(CACHE_FILE, "w") as f:
        json.dump(cache_data, f, indent=4)

def get_cached_sentiment(text):
    """
    Returns the cached score and confidence if found, else None.
    Uses an MD5 hash of the text (title + description) as the unique key.
    """
    cache = load_cache()
    # Create a unique ID for the content
    content_id = hashlib.md5(text.encode("utf-8")).hexdigest()
    
    return cache.get(content_id)

def update_cache(text, sentiment_result):
    """
    Saves a new analysis result to the cache.
    sentiment_result should be: {"score": float, "confidence": float, "label": str}
    """
    cache = load_cache()
    content_id = hashlib.md5(text.encode("utf-8")).hexdigest()
    
    cache[content_id] = {
        "score": sentiment_result.get('sentiment_score', 0),
        "confidence": sentiment_result.get('confidence', 0),
        "label": sentiment_result.get('sentiment_label', "Neutral")
    }
    save_cache(cache)