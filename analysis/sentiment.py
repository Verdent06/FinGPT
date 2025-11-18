# analysis/sentiment.py
import numpy as np

def mpnet_analyzer(news_articles, clf, embedder, label_map):
    texts = [a.get('title','') + ". " + a.get('description','') for a in news_articles]
    numbers = embedder.encode(texts, batch_size=len(texts), show_progress_bar=False)
    probs = clf.predict_proba(numbers)
    print(probs)

    results = []
    for i, p in enumerate(probs):
        score = (-0.5 * p[0]) + (0 * p[1]) + (1.5 * p[2])
        label = label_map[np.argmax(p)]
        results.append({
            'article_index': i,
            'tone': label,
            'sentiment_score': round(score,2),
            'raw_probabilities': [round(float(x),2) for x in p]
        })
    return results
