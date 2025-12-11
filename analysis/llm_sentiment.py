# analysis/llm_sentiment.py
import json
import re
from models.llm_handler import load_llm

class LLMSentimentAnalyzer:
    def __init__(self, llm_instance):
        self.llm = llm_instance

    def analyze_single_article(self, article):
        title = article.get("title", "")
        desc = article.get("description", "")
        text = f"{title}. {desc}".replace('"', "'").replace('\n', ' ')

        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
Act as a financial analyst. Analyze the news headline.
1. Provide a brief rationale.
2. Classify as "Bullish", "Bearish", or "Neutral".
3. Assign a score (-1.0 to 1.0).
Return ONLY a JSON object.

### Input:
{text}

### Response:
"""
        
        try:
            response = self.llm(
                prompt,
                max_tokens=128,
                stop=["}"],      
                temperature=0.1, 
                echo=False 
            )
            
            raw_output = response['choices'][0]['text']
            
            json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
            else:
                result = json.loads("{" + raw_output + "}")

            return {
                "sentiment_score": float(result.get("sentiment_score", 0.0)),
                "confidence": float(result.get("confidence", 1.0)), 
                "sentiment_label": result.get("sentiment_label", "Neutral"),
                "rationale": result.get("rationale", "")
            }

        except Exception as e:
            print(f"[Model Error] {e}")
            return {
                "sentiment_label": "Neutral",
                "sentiment_score": 0.0,
                "confidence": 0.0,
                "rationale": "Error"
            }