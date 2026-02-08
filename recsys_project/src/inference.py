# ==========================================
# RECSYS_PROJECT/src/inference.py
# Production-Grade Recommender Engine
# Static Models - Lazy Loading - Robust I/O
# ==========================================

import logging
import pickle
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from scipy import sparse

from src.als.recommend import ALSRecommender
from src.content_based.search import ContentSearcher
from src.hybrid.hybrid import HybridRecommender


# ==========================================
# Logging Configuration
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ==========================================
# Paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"


# ==========================================
# Utility Loaders
# ==========================================

def load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


# ==========================================
# Recommender Engine
# ==========================================

class RecommenderEngine:

    def __init__(self):

        logger.info("Initializing Recommender Engine...")

        self._load_static_data()
        self._load_models()

        self._build_als_engine()
        self._build_content_engine()
        self._build_hybrid_engine()

        logger.info("Recommender Engine Ready")

    # --------------------------------------
    # Static Data
    # --------------------------------------
    def _load_static_data(self):

        logger.info("Loading static datasets...")

        self.movies = load_csv(DATA_DIR / "clean_movies.csv")
        self.interactions = load_csv(DATA_DIR / "clean_interactions.csv")

    # --------------------------------------
    # Models
    # --------------------------------------
    def _load_models(self):

        logger.info("Loading models...")

        self.als_model = load_pickle(MODELS_DIR / "als_model.pkl")
        self.tfidf = load_pickle(MODELS_DIR / "tfidf.pkl")
        self.mlb = load_pickle(MODELS_DIR / "mlb.pkl")

        self.item_map = load_pickle(MODELS_DIR / "item_map.pkl")
        self.user_map = load_pickle(MODELS_DIR / "user_map.pkl")
        self.movieId_to_index = load_pickle(MODELS_DIR / "movieId_to_index.pkl")

        self.inv_item_map = {v: k for k, v in self.item_map.items()}

        self.faiss_index = faiss.read_index(
            str(MODELS_DIR / "faiss.index")
        )

        self.item_features = np.load(
            MODELS_DIR / "item_features.npy",
            allow_pickle=False
        )

        self.X_sparse = sparse.load_npz(
            MODELS_DIR / "X_sparse.npz"
        )

        logger.info("Models loaded successfully")

    # --------------------------------------
    # ALS Engine
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
    # Content Engine
    # --------------------------------------
    def _build_content_engine(self):

        self.content_engine = ContentSearcher(
            train_df=self.interactions,
            item_features=self.item_features,
            faiss_index=self.faiss_index,
            movieId_to_index=self.movieId_to_index,
            index_to_movieId={
                v: k for k, v in self.movieId_to_index.items()
            }
        )

    # --------------------------------------
    # Hybrid Engine
    # --------------------------------------
    def _build_hybrid_engine(self):

        self.hybrid_engine = HybridRecommender(
            als_recommender=self.als_engine,
            content_searcher=self.content_engine,
            train_df=self.interactions
        )

    # --------------------------------------
    # Utilities
    # --------------------------------------
    def _format_output(self, recs):

        if recs is None or len(recs) == 0:
            return pd.DataFrame()

        rec_df = pd.DataFrame(
            recs,
            columns=["movieId", "score"]
        )

        return rec_df.merge(
            self.movies,
            on="movieId",
            how="left"
        )

    # --------------------------------------
    # Public APIs
    # --------------------------------------
    def recommend_als(self, user_id, top_k=10):

        recs = self.als_engine.recommend_als(
            user_id=user_id,
            top_k=top_k
        )
        return self._format_output(recs)

    def recommend_content(self, user_id, top_k=10):

        recs = self.content_engine.recommend(
            user_id=user_id,
            top_k=top_k
        )
        return self._format_output(recs)

    def recommend_hybrid(self, user_id, top_k=10, alpha=0.7):

        recs = self.hybrid_engine.recommend_weighted(
            user_id=user_id,
            top_k=top_k,
            alpha=alpha
        )
        return self._format_output(recs)
