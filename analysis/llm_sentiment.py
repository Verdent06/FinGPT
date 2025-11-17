import json
from models.llm_handler import load_llm

class LLMSentimentAnalyzer:
    def __init__(self):
        self.llm = load_llm()

    def summarize_articles(self, articles):
        """
        articles: list of dicts containing {title, description}
        Returns a short combined summary for LLM use.
        """
        combined = ""
        for a in articles:
            title = a.get("title", "")
            desc = a.get("description", "")
            combined += f"- {title}: {desc}\n"

        # Keep it short for the 7B model
        return combined[:1500]

    def classify_sentiment(self, articles):
        """
        Returns:
        {
            "sentiment_label": "Positive"/"Neutral"/"Negative",
            "sentiment_score": float (-1 to 1),
            "confidence": float (0 to 1)
        }
        """
        text = self.summarize_articles(articles)

        prompt = f"""
        You are a financial sentiment analyst. 
        Analyze the overall sentiment of the company's recent news headlines.

        News:
        {text}

        Return ONLY this JSON format:
        {{
        "sentiment_label": "Positive" | "Neutral" | "Negative",
        "sentiment_score": -1.0 to 1.0,
        "confidence": 0.0 to 1.0
        }}
        """

        raw_output = self.llm.generate(prompt)

        try:
            result = json.loads(raw_output)
        except:
            # fallback in case LLM fails
            result = {
                "sentiment_label": "Neutral",
                "sentiment_score": 0.0,
                "confidence": 0.0
            }

        return result
