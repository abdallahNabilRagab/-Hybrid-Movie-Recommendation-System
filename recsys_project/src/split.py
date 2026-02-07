# ==========================================
# Temporal Splitter Module
# Ensures Train-Test Consistency for ALS
# ==========================================

import pandas as pd
import logging


# ==========================================
# Logger Configuration
# ==========================================

logger = logging.getLogger(__name__)


# ==========================================
# TemporalSplitter Class
# ==========================================

class TemporalSplitter:
    """
    Perform time-aware train/test split.

    Guarantees:
    - Every user in test appears in train
    - At least one interaction in train per user
    - No cold-start users in evaluation
    """


    # ==========================================
    # Prepare ALS DataFrame
    # ==========================================

    def prepare_als_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare ALS-ready dataframe.

        Keeps only required columns and sorts by time.
        """

        als_df = df[["userId", "movieId", "rating", "timestamp"]].copy()

        als_df = als_df.sort_values("timestamp")

        logger.info(f"ALS dataframe prepared | Rows: {len(als_df)}")

        return als_df


    # ==========================================
    # Temporal Split
    # ==========================================

    def temporal_split(self, df: pd.DataFrame, test_ratio: float = 0.2):
        """
        Perform per-user temporal split.

        Rules:
        - Last interactions go to test
        - At least 1 interaction stays in train
        - Users with <2 interactions stay fully in train
        """

        logger.info("⏳ Starting temporal train/test split...")

        train_parts = []
        test_parts = []


        for user_id, g in df.groupby("userId"):

            g = g.sort_values("timestamp")

            n = len(g)


            # ----------------------------------
            # Case 1: Too few interactions
            # ----------------------------------
            if n < 2:
                # Cannot split → keep in train only
                train_parts.append(g)
                continue


            # ----------------------------------
            # Compute split index
            # ----------------------------------
            split_idx = int(n * (1 - test_ratio))


            # ----------------------------------
            # Guarantee at least 1 in train
            # ----------------------------------
            if split_idx < 1:
                split_idx = 1


            # ----------------------------------
            # Perform split
            # ----------------------------------
            train_parts.append(g.iloc[:split_idx])
            test_parts.append(g.iloc[split_idx:])


        train_df = pd.concat(train_parts).reset_index(drop=True)

        test_df = pd.concat(test_parts).reset_index(drop=True)


        logger.info(
            f"✅ Split completed | "
            f"Train: {len(train_df)} | "
            f"Test: {len(test_df)}"
        )

        return train_df, test_df


    # ==========================================
    # Backward Compatibility
    # ==========================================

    def split(self, df: pd.DataFrame, test_ratio: float = 0.2):
        """
        Alias for temporal_split.
        """

        return self.temporal_split(df, test_ratio)
