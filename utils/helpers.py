import re
import numpy as np

def extract_company_name(full_name):
    """
    Clean company name by removing suffixes like Inc, Corp, LLC, etc.
    """
    clean_name = re.sub(
        r"\s*\b(Inc|Corporation|Corp|Ltd|LLC|PLC)\.?\b[,]?\s*", 
        "", 
        full_name, 
        flags=re.IGNORECASE
    )
    clean_name = re.sub(r"\s+", " ", clean_name)
    return clean_name.strip().rstrip(".,")
    
def display_results( info, news_sentiment, final_score, final_combined_score, recommendation, llm_score, llm_analysis):
    """
    Nicely print the analysis results for a company.
    """
    print("\n--- Company Overview ---")
    print("Name:", info.get("longName"))
    print("Sector:", info.get("sector"))
    print("Current Price:", info.get("currentPrice"))
    print("Market Cap:", info.get("marketCap"))

    print("\n--- Recent News ---")
    print(f"Average Sentiment: {news_sentiment:.2f}")

    print("\n--- Final Combined Score and Recommendation ---")
    print(f"Final Score: {final_score:.2f}")
    print(f"LLM Score: {llm_score:.2f}")
    print(f"LLM Analysis: {llm_analysis}")
    print(f"Combined Score with LLM: {final_combined_score:.2f}")
    print(f"Recommendation: {recommendation}")
