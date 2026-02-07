# ==========================================
# Hybrid Recommender Evaluator
# Clean & Reliable Evaluation Module
# ==========================================

from tqdm import tqdm

from src.utils.metrics import (
    recall_at_k,
    precision_at_k,
    ndcg_at_k
)


# ==========================================
# Evaluator Class
# ==========================================

class Evaluator:
    """
    Evaluator for Hybrid Recommendation Models.

    Features
    --------
    ✔ Filters invalid users
    ✔ Uses proper ground-truth (user -> set(items))
    ✔ Prevents empty predictions from biasing results
    ✔ Single progress bar
    ✔ Compatible with large-scale evaluation
    """


    # ======================================
    # Main Evaluation Function
    # ======================================

    def evaluate_model(
        self,
        recommender,
        test_df,
        k: int = 10,
        show_progress: bool = True,
    ):
        """
        Evaluate recommender using Precision@K, Recall@K, and NDCG@K.

        Parameters
        ----------
        recommender : object
            Must implement recommend_weighted()

        test_df : pd.DataFrame
            Test interactions dataframe

        k : int
            Top-K cutoff

        show_progress : bool
            Enable tqdm progress bar

        Returns
        -------
        precision : float
        recall    : float
        ndcg      : float
        evaluated_users : int
        """


        # ============================
        # Build Ground Truth
        # ============================

        # user -> set of true items
        test_user_items = (
            test_df
            .groupby("userId")["movieId"]
            .apply(set)
        )


        # ============================
        # Prediction Storage
        # ============================

        predictions = {}


        # ============================
        # Evaluation Users
        # ============================

        users = test_user_items.index


        iterator = users

        if show_progress:
            iterator = tqdm(
                users,
                desc="Evaluating Hybrid Model",
                unit="user",
            )


        # ============================
        # Generate Predictions
        # ============================

        for user_id in iterator:

            recs = recommender.recommend_weighted(
                user_id=user_id,
                top_k=k,
                verbose=False,
                show_progress=False,
            )

            # ✅ Skip invalid users
            if not recs:
                continue

            predictions[user_id] = [
                item_id for item_id, _ in recs
            ]


        # ============================
        # Filter Ground Truth
        # ============================

        # Keep only evaluated users
        ground_truth = {
            user: test_user_items[user]
            for user in predictions.keys()
        }


        # ============================
        # Compute Metrics
        # ============================

        recall = recall_at_k(
            ground_truth,
            predictions,
            k
        )

        precision = precision_at_k(
            ground_truth,
            predictions,
            k
        )

        ndcg = ndcg_at_k(
            ground_truth,
            predictions,
            k
        )


        # ============================
        # Return Results
        # ============================

        return (
            precision,
            recall,
            ndcg,
            len(predictions)
        )
