"""SHAP and LIME helpers for explaining content-based recommendations."""
import numpy as np


def extract_class_shap(shap_values, expected_value, class_idx: int = 1):
    """Robustly extract SHAP values + base value for one class.

    Handles both SHAP < 0.41 (returns a list of arrays) and SHAP >= 0.41
    (returns a 3D ndarray of shape (n_samples, n_features, n_classes)).
    """
    if isinstance(shap_values, list):
        sv = shap_values[class_idx]
    elif np.ndim(shap_values) == 3:
        sv = shap_values[:, :, class_idx]
    else:
        sv = shap_values  # already 2D

    base = (
        expected_value[class_idx]
        if hasattr(expected_value, "__len__")
        else float(expected_value)
    )
    return sv, base


def make_lime_predict_fn(query_embedding: np.ndarray, st_model):
    """Build a LIME-compatible predict_fn that scores perturbed text against
    a reference query embedding via cosine similarity.

    Returned callable: list[str] -> ndarray of shape (len, 2) with columns
    [prob_not_similar, prob_similar].
    """
    q = query_embedding

    def predict_fn(texts):
        embs = st_model.encode(texts, show_progress_bar=False)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / np.maximum(norms, 1e-9)
        sims = np.clip(embs @ q, 0.0, 1.0)
        return np.column_stack([1.0 - sims, sims])

    return predict_fn
