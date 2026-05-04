"""Sentence-transformer encoding + on-disk caching for movie embeddings."""
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer


def build_or_load_embeddings(
    soups: list[str],
    cache_path: Path,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> np.ndarray:
    """Encode a list of texts with a sentence-transformer and L2-normalise.

    Caches the result to ``cache_path`` (a .npy file). On subsequent calls the
    cached array is returned immediately, so encoding only runs once.

    Returns
    -------
    np.ndarray of shape (n_texts, embedding_dim), L2-normalised.
    """
    if cache_path.exists():
        return np.load(cache_path)

    model = SentenceTransformer(model_name)
    embeddings = model.encode(soups, show_progress_bar=True, batch_size=batch_size)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-9)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    return embeddings
