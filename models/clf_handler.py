# models/clf_handler.py
import joblib

def load_trained_clf(file_path="models/trained_financial_clf.pkl"):
    """
    Load a trained financial news classifier.
    Returns a scikit-learn compatible classifier object.
    """
    return joblib.load(file_path)
