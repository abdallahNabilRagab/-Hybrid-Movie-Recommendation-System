# ==========================================
# Hybrid Recommender System (Optimized)
# ALS + Content-Based + Stability Layer
# ==========================================

import numpy as np
import logging
from tqdm import tqdm


# ==========================================
# Silent Logger Configuration (THIS FILE ONLY)
# ==========================================

logger = logging.getLogger("HybridRecommender")

logger.propagate = False
logger.handlers.clear()

logger.addHandler(logging.NullHandler())
logger.setLevel(logging.CRITICAL)


# ==========================================
# Utility Functions
# ==========================================

def normalize_scores(score_dict: dict) -> dict:
    """
    Min-Max normalize scores to [0,1].

    Prevents domination of one model
    in hybrid fusion.
    """

    if not score_dict:
        return {}

    values = np.asarray(
        list(score_dict.values()),
        dtype=np.float32
    )

    min_v = values.min()
    max_v = values.max()

    if max_v - min_v < 1e-9:
        return {k: 1.0 for k in score_dict}

    return {
        k: float((v - min_v) / (max_v - min_v))
        for k, v in score_dict.items()
    }


# ==========================================
# HybridRecommender Class
# ==========================================

class HybridRecommender:
    """
    Production-Grade Hybrid Recommender.

    Combines:
    ----------
    ✔ ALS Collaborative Filtering
    ✔ Content-Based Similarity

    Features:
    ----------
    ✔ Cold-start protection
    ✔ Score normalization
    ✔ Stable fusion
    ✔ Research-standard candidate pool
    ✔ tqdm-safe evaluation
    ✔ Silent logging
    """


    # ======================================
    # Initialization
    # ======================================

    def __init__(
        self,
        als_recommender,
        content_searcher,
        train_df,
    ):

        self.als = als_recommender
        self.content = content_searcher
        self.train_df = train_df


    # ======================================
    # Internal: Popularity Fallback
    # ======================================

    def _popular_items(self, top_k=10):
        """
        Fallback recommender (cold-start safety).
        """

        if self.train_df is None:
            return []

        popular = (
            self.train_df.groupby("movieId")["rating"]
            .count()
            .sort_values(ascending=False)
            .head(top_k)
            .index
            .tolist()
        )

        return [(m, 1.0) for m in popular]


    # ======================================
    # Main Recommendation Function
    # ======================================

    def recommend_weighted(
        self,
        user_id: int,
        top_k: int = 10,
        alpha: float = 0.8,
        verbose: bool = False,
        show_progress: bool = False,
    ):
        """
        Generate hybrid recommendations.

        Parameters
        ----------
        user_id : int
            External user ID

        top_k : int
            Number of recommendations

        alpha : float
            Weight of ALS (0-1)

        verbose : bool
            Enable debug logs

        show_progress : bool
            Enable tqdm
        """


        # ============================
        # Validate Alpha
        # ============================

        alpha = float(np.clip(alpha, 0.0, 1.0))


        # ============================
        # Optional Debug Logging
        # ============================

        if verbose:

            logger.setLevel(logging.INFO)

            handler = logging.StreamHandler()

            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)s | %(message)s"
                )
            )

            logger.handlers = [handler]

            logger.info(
                f"Hybrid inference for user {user_id}"
            )


        # ============================
        # Candidate Pool
        # ============================

        candidate_k = max(top_k * 50, 500)


        # ============================
        # ALS Candidates
        # ============================

        try:

            recs_cf = self.als.recommend_als(
                user_id,
                candidate_k,
                normalize=True,
                verbose=False,
            )

        except Exception as e:

            logger.error(f"ALS failed: {e}")
            recs_cf = []


        recs_cf_dict = dict(recs_cf) if recs_cf else {}


        # ============================
        # Content Candidates
        # ============================

        try:

            recs_cb = self.content.recommend(
                user_id,
                candidate_k
            )

        except Exception as e:

            logger.error(f"Content failed: {e}")
            recs_cb = []


        recs_cb_dict = dict(recs_cb) if recs_cb else {}


        # ============================
        # Cold Start Protection
        # ============================

        if not recs_cf_dict and not recs_cb_dict:

            logger.warning("Full cold-start detected.")

            return self._popular_items(top_k)


        # ============================
        # Score Normalization
        # ============================

        recs_cf_norm = normalize_scores(recs_cf_dict)
        recs_cb_norm = normalize_scores(recs_cb_dict)


        if verbose:

            logger.info(
                f"ALS candidates    : {len(recs_cf_norm)}"
            )

            logger.info(
                f"Content candidates: {len(recs_cb_norm)}"
            )


        # ============================
        # Merge Candidates
        # ============================

        all_items = set(recs_cf_norm) | set(recs_cb_norm)

        iterator = all_items

        if show_progress:

            iterator = tqdm(
                all_items,
                desc="Hybrid Fusion",
                unit="item",
                leave=False
            )


        # ============================
        # Weighted Fusion
        # ============================

        hybrid_scores = {}

        for item in iterator:

            cf = recs_cf_norm.get(item, 0.0)
            cb = recs_cb_norm.get(item, 0.0)

            hybrid_scores[item] = (
                alpha * cf +
                (1.0 - alpha) * cb
            )


        # ============================
        # Remove Seen Items
        # ============================

        try:

            seen_items = set(
                self.train_df.loc[
                    self.train_df["userId"] == user_id,
                    "movieId"
                ].values
            )

        except Exception:

            seen_items = set()


        for item in list(hybrid_scores.keys()):

            if item in seen_items:
                hybrid_scores.pop(item)


        # ============================
        # Final Ranking
        # ============================

        top_items = sorted(
            hybrid_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]


        if verbose:

            logger.info(
                f"Returned {len(top_items)} items"
            )

            if top_items:
                logger.info(
                    f"Top score: {top_items[0][1]:.4f}"
                )


        return top_items
