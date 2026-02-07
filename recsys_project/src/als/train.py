# ==========================================
# ALS Trainer Module (Optimized Version)
# Implicit Feedback + Smart Weighting
# ==========================================

import os
import logging
import numpy as np
import scipy.sparse as sp

from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from threadpoolctl import threadpool_limits


# ==========================================
# Logger Configuration
# ==========================================

logger = logging.getLogger(__name__)


# ==========================================
# ALSTrainer Class
# ==========================================

class ALSTrainer:
    """
    Optimized ALS Trainer for Implicit Feedback.

    Features
    --------
    ✔ Advanced confidence weighting
    ✔ Active user/item filtering
    ✔ Large-scale sparse handling
    ✔ Thread-safe training
    ✔ Research-grade hyperparameters
    """


    # ======================================
    # Initialization
    # ======================================

    def __init__(
        self,
        factors: int = 128,
        iterations: int = 40,
        reg: float = 0.01,
        num_threads: int = 8,
        alpha: float = 50.0,
    ):
        """
        Parameters
        ----------
        factors : int
            Latent dimension

        iterations : int
            ALS iterations

        reg : float
            Regularization strength

        num_threads : int
            Parallel threads

        alpha : float
            Confidence scaling factor
        """

        self.alpha = alpha


        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=reg,
            iterations=iterations,
            num_threads=num_threads,
        )


        logger.info(
            "ALSTrainer initialized | "
            f"Factors={factors} | "
            f"Iter={iterations} | "
            f"Reg={reg} | "
            f"Alpha={alpha} | "
            f"Threads={num_threads}"
        )


    # ======================================
    # Confidence Weighting
    # ======================================

    def build_confidence(self, train_df):
        """
        Build implicit confidence using log-scaled rating.

        Formula
        -------
        confidence = 1 + alpha * log1p(rating)
        """

        logger.info("Building confidence weights...")

        df = train_df.copy()

        df["confidence"] = (
            1.0 +
            self.alpha * np.log1p(df["rating"])
        )

        return df


    # ======================================
    # Filter Sparse Users & Items
    # ======================================

    def filter_sparse_entities(
        self,
        train_df,
        min_user_interactions: int = 5,
        min_item_interactions: int = 5,
    ):
        """
        Remove very inactive users and items.
        """

        logger.info("Filtering sparse users/items...")

        df = train_df.copy()


        # User activity
        user_counts = df["u"].value_counts()
        active_users = user_counts[
            user_counts >= min_user_interactions
        ].index


        # Item activity
        item_counts = df["i"].value_counts()
        active_items = item_counts[
            item_counts >= min_item_interactions
        ].index


        df = df[
            df["u"].isin(active_users) &
            df["i"].isin(active_items)
        ]


        logger.info(
            f"Filtered users: {len(active_users)} | "
            f"Filtered items: {len(active_items)}"
        )

        logger.info(f"Remaining interactions: {len(df)}")

        return df


    # ======================================
    # Sparse Matrix Construction
    # ======================================

    def build_sparse_matrix(
        self,
        train_df,
        user_map,
        item_map
    ):
        """
        Build CSR interaction matrix.
        """

        logger.info("Building sparse interaction matrix...")


        # ----------------------------
        # Safety Checks
        # ----------------------------

        if train_df.empty:
            raise ValueError("Training DataFrame is empty!")


        if not user_map or not item_map:
            raise ValueError("User/Item mapping is empty!")


        # ----------------------------
        # Build CSR Matrix
        # ----------------------------

        X = csr_matrix(
            (
                train_df["confidence"].astype("float32").values,

                (
                    train_df["u"].astype("int32").values,
                    train_df["i"].astype("int32").values,
                ),
            ),
            shape=(
                len(user_map),
                len(item_map),
            ),
        )


        # ----------------------------
        # Validation
        # ----------------------------

        if not sp.issparse(X):
            raise ValueError("Matrix is not sparse!")


        if X.nnz == 0:
            raise ValueError("Sparse matrix has no interactions!")


        sparsity = 1.0 - (X.nnz / (X.shape[0] * X.shape[1]))

        logger.info(
            f"Sparse matrix ready | "
            f"Shape={X.shape} | "
            f"NNZ={X.nnz} | "
            f"Sparsity={sparsity:.4f}"
        )


        return X


    # ======================================
    # Training Pipeline
    # ======================================

    def fit(
        self,
        train_df,
        user_map,
        item_map,
    ):
        """
        Full ALS Training Pipeline.
        """

        logger.info("Starting ALS training pipeline...")


        # =============================
        # Step 1: Build Confidence
        # =============================

        train_df = self.build_confidence(train_df)


        # =============================
        # Step 2: Filter Sparse Data
        # =============================

        train_df = self.filter_sparse_entities(train_df)


        # =============================
        # Step 3: Build Matrix
        # =============================

        X = self.build_sparse_matrix(
            train_df,
            user_map,
            item_map,
        )


        # =============================
        # Step 4: Thread Control
        # =============================

        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"

        threadpool_limits(1, "blas")


        # =============================
        # Step 5: Train Model
        # =============================

        logger.info("Training ALS model...")

        try:
            self.model.fit(X)

        except Exception as e:
            logger.error(f"ALS training failed: {e}")
            raise


        logger.info("ALS training completed successfully.")


        # =============================
        # Final Validation
        # =============================

        if self.model.user_factors.shape[0] != X.shape[0]:

            raise ValueError(
                "Mismatch: user_factors vs interaction matrix!"
            )


        logger.info("ALS model validated successfully.")


        return self.model, X
