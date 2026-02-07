# ==========================================
# ALS Evaluator Module
# Evaluate ALS-based recommender
# With Clean Metrics and Progress Tracking
# ==========================================

import numpy as np
import logging
from tqdm import tqdm


# ==========================================
# Logger Configuration
# ==========================================

logger = logging.getLogger(__name__)


# ==========================================
# ALSEvaluator Class
# ==========================================

class ALSEvaluator:
    """
    Evaluate ALS recommender using:

    - Precision@K
    - Recall@K
    - NDCG@K

    Only evaluates users with valid predictions.
    """


    def __init__(self, recommender):
        """
        Initialize ALS evaluator.

        Args:
            recommender: Trained ALS-based recommender
        """

        self.recommender = recommender

        logger.info("🧪 ALSEvaluator initialized")


    # ==========================================
    # ALS Prediction
    # ==========================================

    def als_predict(self, user_id, k=10):
        """
        Generate Top-K recommendations.

        Returns empty list if user is invalid.
        """

        recs = self.recommender.recommend_als(user_id, k)

        if recs is None:
            return []

        return [r[0] for r in recs]


    # ==========================================
    # Precompute Predictions
    # ==========================================

    def precompute(self, test_data, k=10):
        """
        Precompute predictions for valid users only.

        Args:
            test_data: pd.Series (user -> true items)
            k: Top-K

        Returns:
            dict: user -> predicted items
        """

        logger.info("📊 Precomputing ALS predictions...")

        preds = {}

        for user in tqdm(test_data.index, desc="Predicting users"):

            recs = self.als_predict(user, k)

            # ✅ Skip users with no predictions
            if len(recs) == 0:
                continue

            preds[user] = recs


        logger.info(f"✅ Predictions ready | Users evaluated: {len(preds)}")

        return preds


    # ==========================================
    # Recall@K
    # ==========================================

    def recall(self, test_data, preds, k=10):
        """
        Compute Recall@K (only valid users).
        """

        logger.info(f"📈 Computing Recall@{k}...")

        scores = []


        for user, pred_items in preds.items():

            true_items = test_data[user]

            if len(true_items) == 0:
                continue

            hit = len(set(pred_items) & true_items)

            scores.append(hit / len(true_items))


        recall_score = np.mean(scores) if scores else 0.0

        logger.info(f"✅ Recall@{k}: {recall_score:.4f}")

        return recall_score


    # ==========================================
    # Precision@K
    # ==========================================

    def precision(self, test_data, preds, k=10):
        """
        Compute Precision@K (only valid users).
        """

        logger.info(f"📏 Computing Precision@{k}...")

        scores = []


        for user, pred_items in preds.items():

            true_items = test_data[user]

            hit = len(set(pred_items) & true_items)

            scores.append(hit / k)


        precision_score = np.mean(scores) if scores else 0.0

        logger.info(f"✅ Precision@{k}: {precision_score:.4f}")

        return precision_score


    # ==========================================
    # NDCG@K
    # ==========================================

    def ndcg(self, test_data, preds, k=10):
        """
        Compute NDCG@K (only valid users).
        """

        logger.info(f"📐 Computing NDCG@{k}...")

        scores = []


        for user, pred_items in preds.items():

            true_items = test_data[user]


            # -----------------------
            # DCG
            # -----------------------

            dcg = 0.0

            for i, item in enumerate(pred_items):

                if item in true_items:

                    dcg += 1 / np.log2(i + 2)


            # -----------------------
            # IDCG
            # -----------------------

            idcg = 0.0

            max_rank = min(len(true_items), k)

            for i in range(max_rank):

                idcg += 1 / np.log2(i + 2)


            if idcg > 0:

                scores.append(dcg / idcg)


        ndcg_score = np.mean(scores) if scores else 0.0

        logger.info(f"✅ NDCG@{k}: {ndcg_score:.4f}")

        return ndcg_score
