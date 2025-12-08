# analysis/llm_sentiment.py
import json
import re
from models.llm_handler import load_llm

class LLMSentimentAnalyzer:
    def __init__(self, llm_instance):
        self.llm = llm_instance

    def analyze_single_article(self, article):
        """
        Analyzes a single news article dictionary.
        Returns: {"sentiment_score": float, "confidence": float, "sentiment_label": str}
        """
        title = article.get("title", "")
        desc = article.get("description", "")
        # Clean text to prevent prompt injection
        text = f"{title}. {desc}".replace('"', "'").replace('\n', ' ')

        # --- REFINED PROMPT ---
        # I aligned the examples to match the JSON format exactly to reduce confusion.
        prompt = f"""### Instruction:
Act as a financial analyst. Analyze the sentiment of the news headline below.
Return a JSON object with keys: "sentiment_label", "sentiment_score", "confidence".

Examples:
Input: "Company reports record profits."
Output: {{ "sentiment_label": "Positive", "sentiment_score": 0.9, "confidence": 0.9 }}

Input: "CEO steps down amid scandal."
Output: {{ "sentiment_label": "Negative", "sentiment_score": -0.8, "confidence": 0.9 }}

Input: "{text}"
Output:
{{"""
        
        try:
            # Call LLaMA
            response = self.llm(
                prompt,
                max_tokens=128,
                stop=["}", "###"], 
                temperature=0.1, 
                echo=False 
            )
            
            raw_output = "{" + response['choices'][0]['text']
            

        except Exception as e:
            print(f"LLM Generation Error: {e}")
            raw_output = ""

        try:
            if "}" not in raw_output:
                raw_output += "}"
            
            json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
            if json_match:
                clean_json = json_match.group(0)
                result = json.loads(clean_json)
            else:
                print(f"[ERROR] No JSON found in: {raw_output}")
                raise ValueError("No JSON found")

            score = float(result.get("sentiment_score", 0.0))
            conf = float(result.get("confidence", 0.0))
            label = result.get("sentiment_label", "Neutral")
            
            return {
                "sentiment_score": score,
                "confidence": conf,
                "sentiment_label": label
            }

        except Exception as e:
            print(f"[ERROR] Parsing Failed: {e}")
            # Fallback
            return {
                "sentiment_label": "Neutral",
                "sentiment_score": 0.0,
                "confidence": 0.0
            }