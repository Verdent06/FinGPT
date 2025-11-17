# models/mpnet_embedder.py
from sentence_transformers import SentenceTransformer

def get_embedder(model_path="sentence-transformers/all-mpnet-base-v2"):
    """
    Load MPNet embedding model for news/article text embeddings.
    Downloads automatically from Hugging Face if not present locally.
    """
    return SentenceTransformer(model_path)
