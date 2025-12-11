# data/prepare_dataset.py
import json
import random

INPUT_FILE = "data/Sentences_75Agree.txt"
OUTPUT_FILE = "data/finetune_dataset.jsonl"

def prepare_data():
    print(f"Reading from {INPUT_FILE}...")
    
    with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        
    training_data = []
    
    for line in lines:
        try:
            # Parse the "Text@Label" format
            if "@" not in line: continue
            text, label = line.strip().rsplit("@", 1)
            
            # Map labels to our target JSON structure
            # We add variety to the "Rationale" to help the model learn to think
            score = 0.0
            rationale = "Neutral market event."
            
            if label == "positive":
                score = 0.8  # Standard positive
                rationale = "Positive indicator for growth or profitability."
            elif label == "negative":
                score = -0.8 # Standard negative
                rationale = "Negative indicator showing risk or decline."
            
            # The Training Prompt Format (Alpaca/Instruction Style)
            # This teaches the model: "When you see this Instruction, output this JSON"
            entry = {
                "instruction": "Analyze the financial sentiment of this headline. Return JSON.",
                "input": text.strip(),
                "output": json.dumps({
                    "sentiment_label": label.capitalize(),
                    "sentiment_score": score,
                    "confidence": 0.95,
                    "rationale": rationale
                })
            }
            
            training_data.append(entry)
            
        except Exception as e:
            print(f"Skipping line: {line[:20]}... Error: {e}")

    # Shuffle to prevent ordering bias
    random.shuffle(training_data)
    
    # Save
    with open(OUTPUT_FILE, "w") as f:
        for entry in training_data:
            f.write(json.dumps(entry) + "\n")
            
    print(f"✅ Success! Created {len(training_data)} training examples in {OUTPUT_FILE}")

if __name__ == "__main__":
    prepare_data()