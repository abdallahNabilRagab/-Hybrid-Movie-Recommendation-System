# ==========================================
# ALS Recommender Module (Optimized Version)
# Silent • Stable • Hybrid-Ready
# ==========================================

import logging
import numpy as np
import scipy.sparse as sp


# ==========================================
# Logger Configuration (Silent by Default)
# ==========================================

logger = logging.getLogger(__name__)

logger.propagate = False
logger.handlers.clear()

logger.addHandler(logging.NullHandler())
logger.setLevel(logging.CRITICAL)


# ==========================================
# ALSRecommender Class
# ==========================================

class ALSRecommender:
    """
    Production-Ready ALS Recommender.

    Features
    --------
    ✔ Robust ID mapping
    ✔ Cold-start fallback
    ✔ Safe boundary checks
    ✔ Stable score output
    ✔ Hybrid-compatible
    ✔ Silent by default
    """


    # ======================================
    # Initialization
    # ======================================

    def __init__(
        self,
        model,
        X,
        user_map,
        item_map,
        inv_item_map,
        train_df=None,
    ):
        """
        Initialize ALS recommender.

        Parameters
        ----------
        model : AlternatingLeastSquares
            Trained ALS model

        X : csr_matrix
            User-item interaction matrix

        user_map : dict
            userId -> internal id

        item_map : dict
            itemId -> internal id

        inv_item_map : dict
            internal id -> itemId

        train_df : pd.DataFrame, optional
            For popularity fallback
        """

        self.model = model
        self.X = X

        self.user_map = user_map
        self.item_map = item_map
        self.inv_item_map = inv_item_map

        self.train_df = train_df


        # =============================
        # Safety Checks (CRITICAL)
        # =============================

        if not sp.issparse(self.X):

            raise ValueError(
                "ALSRecommender Error: "
                "X must be a scipy sparse matrix."
            )


        if self.X.shape[0] != self.model.user_factors.shape[0]:

            raise ValueError(
                "ALSRecommender Error: "
                "X rows != user_factors size."
            )


        logger.info("ALSRecommender initialized successfully.")


    # ======================================
    # Popularity Fallback (Cold Start)
    # ======================================

    def _popular_items(self, top_k=10):
        """
        Return most popular items as fallback.
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
    # Main Recommendation Interface
    # ======================================

    def recommend_als(
        self,
        user_id: int,
        top_k: int = 10,
        normalize: bool = False,
        verbose: bool = False,
    ):
        """
        Generate Top-K ALS recommendations.

        Parameters
        ----------
        user_id : int
            External user ID

        top_k : int
            Number of recommendations

        normalize : bool
            Min-Max normalize scores

        verbose : bool
            Enable debug logging

        Returns
        -------
        list[(itemId, score)]
        """


        # =============================
        # Optional Debug Logging
        # =============================

        if verbose:

            logger.setLevel(logging.INFO)

            handler = logging.StreamHandler()

            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)s | %(message)s"
                )
            )

            logger.handlers = [handler]


        # =============================
        # Validate Input
        # =============================

        if not isinstance(user_id, int):

            logger.warning("Invalid user_id type.")
            return self._popular_items(top_k)


        # =============================
        # Check User Exists
        # =============================

        if user_id not in self.user_map:

            logger.warning(f"Cold-start user: {user_id}")
            return self._popular_items(top_k)


        u = self.user_map[user_id]


        # =============================
        # Boundary Check
        # =============================

        if u < 0 or u >= self.model.user_factors.shape[0]:

            logger.error(f"User index out of bounds: {u}")
            return self._popular_items(top_k)


        # =============================
        # Get User Vector
        # =============================

        try:

            user_items = self.X[u]

        except Exception as e:

            logger.error(f"Failed to access X[{u}]: {e}")
            return self._popular_items(top_k)


        # =============================
        # Generate Recommendations
        # =============================

        try:

            item_ids, scores = self.model.recommend(
                userid=u,
                user_items=user_items,
                N=top_k,
                filter_already_liked_items=True,
            )

        except Exception as e:

            logger.error(f"ALS recommend failed: {e}")
            return self._popular_items(top_k)


        # =============================
        # Post-process Scores
        # =============================

        scores = np.asarray(scores, dtype=np.float32)

        if normalize and len(scores) > 0:

            min_v = scores.min()
            max_v = scores.max()

            if max_v - min_v > 1e-9:

                scores = (scores - min_v) / (max_v - min_v)

            else:

                scores = np.ones_like(scores)


        # =============================
        # Map to External IDs
        # =============================

        recommendations = []

        for i, s in zip(item_ids, scores):

            movie_id = self.inv_item_map.get(i)

            if movie_id is not None:

                recommendations.append(
                    (movie_id, float(s))
                )


        return recommendations
