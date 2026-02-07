# ==========================================
# Content-Based Feature Vectorizer (Silent)
# FAISS + TF-IDF + Genre Encoding
# ==========================================

import faiss
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

# ==========================================
# Silent Logger (THIS FILE ONLY)
# ==========================================
logger = logging.getLogger("ContentVectorizer")
logger.propagate = False
logger.handlers.clear()
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.CRITICAL)

# ==========================================
# ContentVectorizer Class
# ==========================================
class ContentVectorizer:
    """
    Build content-based item representations using:
    - Genre encoding (MultiLabelBinarizer)
    - Tag text features (TF-IDF)
    - FAISS index for fast similarity search

    Design:
    - 🔕 Completely silent (no logging, no printing)
    - 🚀 Optimized for large-scale pipelines
    """

    # -------------------------
    # Build Item Feature Vectors
    # -------------------------
    def build_item_features(
        self,
        movies: pd.DataFrame,
        tags: pd.DataFrame
    ):
        """
        Construct combined item feature vectors.

        Returns
        -------
        item_features : np.ndarray (float32, L2-normalized, C-contiguous)
        mlb           : MultiLabelBinarizer
        vectorizer    : TfidfVectorizer
        """

        movies = movies.copy()

        # -------------------------
        # Genre Features
        # -------------------------
        movies["genres_list"] = movies["genres"].str.split("|")

        mlb = MultiLabelBinarizer()
        genre_features = pd.DataFrame(
            mlb.fit_transform(movies["genres_list"]),
            columns=mlb.classes_,
            index=movies["movieId"]
        )

        # -------------------------
        # Tag Features (TF-IDF)
        # -------------------------
        tag_text = (
            tags.groupby("movieId")["tag"]
            .apply(list)
            .apply(lambda x: " ".join(x))
        )

        vectorizer = TfidfVectorizer(max_features=2000)
        tag_matrix = vectorizer.fit_transform(tag_text).toarray()

        tag_matrix = pd.DataFrame(
            tag_matrix,
            index=tag_text.index
        )

        # -------------------------
        # Combine Features
        # -------------------------
        item_features = pd.concat([genre_features, tag_matrix], axis=1).fillna(0.0).values

        # -------------------------
        # Ensure float32 & C-contiguous
        # -------------------------
        item_features = np.ascontiguousarray(item_features, dtype=np.float32)

        # -------------------------
        # Normalize for Cosine Similarity
        # -------------------------
        faiss.normalize_L2(item_features)

        return item_features, mlb, vectorizer

    # -------------------------
    # Build FAISS Index
    # -------------------------
    def build_faiss_index(self, item_features: np.ndarray):
        """
        Build FAISS index using Inner Product (cosine similarity).

        Parameters
        ----------
        item_features : np.ndarray (float32, C-contiguous, L2-normalized)

        Returns
        -------
        index : faiss.IndexFlatIP
        """
        # Ensure C-contiguous just in case
        item_features = np.ascontiguousarray(item_features, dtype=np.float32)

        dim = item_features.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(item_features)
        return index

    # -------------------------
    # Build ID Mappings
    # -------------------------
    def build_id_mappings(self, movies: pd.DataFrame):
        """
        Build movieId <-> index mappings.
        """
        movie_ids = movies["movieId"].values
        movieId_to_index = {m: i for i, m in enumerate(movie_ids)}
        index_to_movieId = {i: m for m, i in movieId_to_index.items()}
        return movieId_to_index, index_to_movieId
