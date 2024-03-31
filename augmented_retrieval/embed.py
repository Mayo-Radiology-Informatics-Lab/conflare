"""
Embedding utilities for augmented retrieval.
"""

from typing import List
import torch
from sentence_transformers import SentenceTransformer


class TextEmbedding:
    """
    A wrapper for sentece transformers to get embeddings for input text
    """

    def __init__(
        self,
        model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_id, device=device)
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

    def __call__(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=self.normalize_embeddings
        ).tolist()

        return embeddings

