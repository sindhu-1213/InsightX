"""Medical NLP preprocessing, TF-IDF vectorization, and label assignment."""

import re
import string

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Clinical negation tokens preserved during preprocessing
CRITICAL_NEGATIONS: set[str] = {
    "no",
    "not",
    "none",
    "neither",
    "never",
    "without",
    "negative",
}


def get_medical_stopwords() -> set[str]:
    """Retrieve NLTK stopwords with critical clinical negations retained.

    Returns:
        Set of stop words excluding clinical negation tokens.
    """
    try:
        base_stops = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        base_stops = set(stopwords.words("english"))

    return base_stops - CRITICAL_NEGATIONS


def clean_medical_text(text: str, medical_stops: set[str] | None = None) -> str:
    """Clean clinical radiology text while preserving negation words.

    Args:
        text: Raw clinical finding or impression text.
        medical_stops: Custom set of stopwords to remove.

    Returns:
        Cleaned, normalized string.
    """
    if medical_stops is None:
        medical_stops = get_medical_stopwords()

    text_str = str(text).lower()
    # Replace punctuation with whitespace
    cleaned = re.sub(f"[{re.escape(string.punctuation)}]", " ", text_str)
    tokens = [w for w in cleaned.split() if w not in medical_stops]
    return " ".join(tokens).strip()


def fit_transform_tfidf(
    texts: pd.Series | list[str],
    max_features: int = 1000,
) -> tuple[np.ndarray, TfidfVectorizer]:
    """Fit a TF-IDF vectorizer and extract numerical text representations.

    Args:
        texts: Series or list of cleaned clinical findings.
        max_features: Maximum vocabulary size.

    Returns:
        Tuple of (feature_vectors_array, fitted_vectorizer).
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    sparse_matrix = vectorizer.fit_transform(texts)
    return sparse_matrix.toarray(), vectorizer


def derive_finding_labels(texts: pd.Series | list[str]) -> pd.Series:
    """Derive binary prediction target (0 = Normal, 1 = Abnormal) from impression text.

    Derived from radiologist conclusion/impression rather than input findings
    to avoid label leakage.

    Args:
        texts: Cleaned impression texts.

    Returns:
        Series of binary labels (0 = Normal, 1 = Abnormal).
    """
    normal_signals = {"normal", "unremarkable", "clear", "no acute"}
    abnormal_signals = {
        "opacity",
        "effusion",
        "infiltrate",
        "consolidation",
        "pneumonia",
        "mass",
        "nodule",
        "cardiomegaly",
        "fracture",
        "atelectasis",
        "edema",
        "pleural",
    }

    def _classify_single(txt: str) -> int:
        norm_txt = str(txt).lower()
        has_abnormal = any(sig in norm_txt for sig in abnormal_signals)
        has_normal = any(sig in norm_txt for sig in normal_signals)
        if has_abnormal:
            return 1
        if has_normal:
            return 0
        return 1  # Conservative default in clinical research setting

    if isinstance(texts, list):
        return pd.Series([_classify_single(t) for t in texts])
    return texts.apply(_classify_single)
