# ==========================================
# RECSYS_PROJECT/src/inference.py
# Production-Level Recommender Engine
# (Class-Based ALS, Content-Based, Hybrid)
# ==========================================

import pickle
import faiss
import numpy as np
import pandas as pd
from scipy import sparse
from pathlib import Path

# Import recommendation logic
from src.als.recommend import ALSRecommender
from src.content_based.search import ContentSearcher
from src.hybrid.hybrid import HybridRecommender

# ==========================================
# Paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

# ==========================================
# Recommender Engine
# ==========================================

class RecommenderEngine:

    def __init__(self):
        self._load_data()
        self._load_models()
        self._build_als_engine()
        self._build_content_engine()
        self._build_hybrid_engine()

    # --------------------------------------
    # Load Display Data
    # --------------------------------------
    def _load_data(self):
        self.movies = pd.read_csv(DATA_DIR / "clean_movies.csv")

    # --------------------------------------
    # Load All Models
    # --------------------------------------
    def _load_models(self):
        print("🔄 Loading models...")

        # ALS
        with open(MODELS_DIR / "als_model.pkl", "rb") as f:
            self.als_model = pickle.load(f)

        # Content-based
        with open(MODELS_DIR / "tfidf.pkl", "rb") as f:
            self.tfidf = pickle.load(f)
        with open(MODELS_DIR / "mlb.pkl", "rb") as f:
            self.mlb = pickle.load(f)

        # Mappings
        with open(MODELS_DIR / "item_map.pkl", "rb") as f:
            self.item_map = pickle.load(f)
        with open(MODELS_DIR / "user_map.pkl", "rb") as f:
            self.user_map = pickle.load(f)
        with open(MODELS_DIR / "movieId_to_index.pkl", "rb") as f:
            self.movieId_to_index = pickle.load(f)

        # Inverse map for ALS
        self.inv_item_map = {v: k for k, v in self.item_map.items()}

        # FAISS index & features
        self.faiss_index = faiss.read_index(str(MODELS_DIR / "faiss.index"))
        self.item_features = np.load(MODELS_DIR / "item_features.npy")

        # User-item sparse matrix
        self.X_sparse = sparse.load_npz(MODELS_DIR / "X_sparse.npz")

        print("✅ Models loaded successfully")

    # --------------------------------------
    # Build ALS Engine
    # --------------------------------------
    def _build_als_engine(self):
        self.als_engine = ALSRecommender(
            model=self.als_model,
            X=self.X_sparse,
            user_map=self.user_map,
            item_map=self.item_map,
            inv_item_map=self.inv_item_map
        )

    # --------------------------------------
    # Build Content-Based Engine
    # --------------------------------------
    def _build_content_engine(self):
        self.content_engine = ContentSearcher(
            train_df=pd.read_csv(DATA_DIR / "clean_interactions.csv"),
            item_features=self.item_features,
            faiss_index=self.faiss_index,
            movieId_to_index=self.movieId_to_index,
            index_to_movieId={v: k for k, v in self.movieId_to_index.items()}
        )

    # --------------------------------------
    # Build Hybrid Engine
    # --------------------------------------
    def _build_hybrid_engine(self):
        self.hybrid_engine = HybridRecommender(
            als_recommender=self.als_engine,
            content_searcher=self.content_engine,
            train_df=pd.read_csv(DATA_DIR / "clean_interactions.csv")
        )

    # --------------------------------------
    # ALS Recommendation
    # --------------------------------------
    def recommend_als(self, user_id, top_k=10):
        recs = self.als_engine.recommend_als(user_id=user_id, top_k=top_k)
        if recs is None:
            return pd.DataFrame()
        rec_df = pd.DataFrame(recs, columns=["movieId", "score"])
        rec_df = rec_df.merge(self.movies, on="movieId", how="left")
        return rec_df

    # --------------------------------------
    # Content-Based Recommendation
    # --------------------------------------
    def recommend_content(self, user_id, top_k=10):
        recs = self.content_engine.recommend(user_id=user_id, top_k=top_k)
        if recs is None:
            return pd.DataFrame()
        rec_df = pd.DataFrame(recs, columns=["movieId", "score"])
        rec_df = rec_df.merge(self.movies, on="movieId", how="left")
        return rec_df

    # --------------------------------------
    # Hybrid Recommendation
    # --------------------------------------
    def recommend_hybrid(self, user_id, top_k=10, alpha=0.7):
        recs = self.hybrid_engine.recommend_weighted(
            user_id=user_id,
            top_k=top_k,
            alpha=alpha
        )
        if recs is None:
            return pd.DataFrame()
        rec_df = pd.DataFrame(recs, columns=["movieId", "score"])
        rec_df = rec_df.merge(self.movies, on="movieId", how="left")
        return rec_df
