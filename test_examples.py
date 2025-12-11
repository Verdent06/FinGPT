# test_examples.py
import sys
import os

# Ensure the project root is in the python path
sys.path.append(os.getcwd())

from analysis.mpnet_sentiment import mpnet_analyzer
from analysis.llm_sentiment import LLMSentimentAnalyzer
from models import clf_handler, mpnet_embedder, llm_handler

def run_test():
    print("--- Loading Models (This takes a few seconds) ---")
    # 1. Load the Brains
    clf = clf_handler.load_trained_clf()
    embedder = mpnet_embedder.get_embedder()
    llm = llm_handler.load_llm()
    llm_analyzer = LLMSentimentAnalyzer(llm)

    # 2. Define your Custom Test Cases
    # Add whatever wild or subtle examples you want to test here.
    my_examples = [
        {
            "title": "Meta announces restructuring, cutting 10% of workforce to boost margins",
            "description": "Zuckerberg states efficiency is the priority as stock reacts favorably."
        },
        {
            "title": "Nvidia posts record Q4 revenue, but CFO warns of 'significant headwinds' in 2026",
            "description": "Supply chain constraints expected to limit growth next year."
        },
        {
            "title": "Competitor AMD faces DOJ antitrust probe; Nvidia shares rally on market share hopes",
            "description": "Regulatory pressure on AMD could open doors for Nvidia's new chip line."
        },
        {
            "title": "Court rules in favor of Apple, but awards only $1 in symbolic damages",
            "description": "The jury found infringement but determined no financial loss occurred."
        },
        {
            "title": "Intel halts dividend to fund massive $10B AI infrastructure expansion",
            "description": "The company pivots strategy to catch up in the semiconductor race."
        }
    ]

    print(f"\n--- Testing {len(my_examples)} Articles ---\n")

    # 3. Run MPNet (The "Calculator")
    print(">>> MPNet Results (Pattern Matching):")
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    mpnet_results = mpnet_analyzer(my_examples, clf, embedder, label_map)
    
    for res in mpnet_results:
        print(f"Title: {res['title'][:50]}...")
        print(f"  Score: {res['sentiment_score']} ({res['tone']})")
        print(f"  Probs: {res['raw_probabilities']}")
        print("-" * 20)

    # 4. Run LLaMA (The "Smart Intern")
    print("\n>>> LLaMA Results (Reasoning):")
    for article in my_examples:
        print(f"Analyzing: {article['title'][:50]}...")
        res = llm_analyzer.analyze_single_article(article)
        
        print(f"  Label: {res['sentiment_label']}")
        print(f"  Score: {res['sentiment_score']}")
        print(f"  Conf:  {res['confidence']}")
        print("-" * 20)

if __name__ == "__main__":
    run_test()