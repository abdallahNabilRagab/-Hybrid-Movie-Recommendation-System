# ==========================================
# Data Preprocessing Module
# Handles data cleaning, filtering, and analysis
# With Detailed Logging
# ==========================================

import pandas as pd
import logging
import time
from typing import Dict

from .config import Config


# ==========================================
# Logger Configuration
# ==========================================

logger = logging.getLogger(__name__)


# ==========================================
# DataPreprocessor Class
# ==========================================

class DataPreprocessor:
    """
    Responsible for:
    - Inspecting datasets
    - Cleaning and filtering interactions
    - Processing timestamps
    - Removing duplicates
    - Computing sparsity and activity
    - Cleaning metadata (movies, tags, links)
    """


    def __init__(self, config: Config = Config):
        """
        Initialize DataPreprocessor.

        Args:
            config (Config): Project configuration class
        """
        self.config = config

        logger.info("🧹 DataPreprocessor initialized")


    # ==========================================
    # Dataset Inspection
    # ==========================================

    def inspect_datasets(self, datasets: Dict[str, pd.DataFrame]):
        """
        Analyze missing values and duplicates in datasets.

        Args:
            datasets (Dict[str, DataFrame]): Loaded datasets

        Returns:
            Dict: Inspection report
        """

        logger.info("🔍 Inspecting datasets for missing values and duplicates...")

        report = {}

        for name, df in datasets.items():

            missing = df.isna().sum().sum()
            duplicates = df.duplicated().sum()

            logger.info(
                f"📊 {name} | Missing: {missing} | Duplicates: {duplicates}"
            )

            report[name] = {
                "missing": df.isna().sum(),
                "duplicates": duplicates
            }

        logger.info("✅ Dataset inspection completed")

        return report


    # ==========================================
    # User & Movie Activity Analysis
    # ==========================================

    def compute_activity(self, ratings: pd.DataFrame):
        """
        Compute interaction counts per user and per movie.
        """

        logger.info("📈 Computing user and movie activity...")

        user_activity = ratings.groupby("userId").size()
        movie_activity = ratings.groupby("movieId").size()

        logger.info(
            f"👤 Active users: {len(user_activity)} | 🎬 Active movies: {len(movie_activity)}"
        )

        return user_activity, movie_activity


    # ==========================================
    # Interaction Filtering
    # ==========================================

    def filter_interactions(
        self,
        ratings: pd.DataFrame,
        user_activity: pd.Series,
        movie_activity: pd.Series
    ):
        """
        Filter users and movies with low activity.
        """

        logger.info("✂️ Filtering low-activity interactions...")

        before = len(ratings)

        valid_users = user_activity[
            user_activity >= self.config.MIN_USER_RATINGS
        ].index

        valid_movies = movie_activity[
            movie_activity >= self.config.MIN_MOVIE_RATINGS
        ].index

        filtered = ratings[
            ratings["userId"].isin(valid_users)
            &
            ratings["movieId"].isin(valid_movies)
        ].copy()

        after = len(filtered)

        logger.info(
            f"✅ Filtering completed | Before: {before} → After: {after}"
        )

        return filtered


    # ==========================================
    # Sparsity Computation
    # ==========================================

    def compute_sparsity(self, df: pd.DataFrame):
        """
        Compute user-item matrix sparsity.
        """

        logger.info("📉 Computing sparsity level...")

        num_users = df["userId"].nunique()
        num_movies = df["movieId"].nunique()
        num_ratings = len(df)

        sparsity = 1 - (num_ratings / (num_users * num_movies))

        logger.info(f"📊 Sparsity: {sparsity:.4f}")

        return sparsity


    # ==========================================
    # Timestamp Processing
    # ==========================================

    def process_timestamps(self, df: pd.DataFrame):
        """
        Convert timestamps to datetime and remove invalid values.
        """

        logger.info("⏱️ Processing timestamps...")

        before = len(df)

        df = df.copy()

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            unit="s",
            errors="coerce"
        )

        df = df.dropna(subset=["timestamp"])

        after = len(df)

        logger.info(
            f"✅ Timestamp processing | Before: {before} → After: {after}"
        )

        return df


    # ==========================================
    # Duplicate Interaction Removal
    # ==========================================

    def remove_duplicate_interactions(self, df: pd.DataFrame):
        """
        Keep only the latest interaction per user-item pair.
        """

        logger.info("🗑️ Removing duplicate interactions...")

        before = len(df)

        df = (
            df.sort_values("timestamp")
            .drop_duplicates(
                subset=["userId", "movieId"],
                keep="last"
            )
        )

        after = len(df)

        logger.info(
            f"✅ Duplicates removed | Before: {before} → After: {after}"
        )

        return df


    # ==========================================
    # Rating Cleaning
    # ==========================================

    def clean_ratings(self, df: pd.DataFrame):
        """
        Remove ratings outside valid range.
        """

        logger.info("🧼 Cleaning rating values...")

        before = len(df)

        df = df[
            (df["rating"] >= self.config.MIN_RATING)
            &
            (df["rating"] <= self.config.MAX_RATING)
        ]

        after = len(df)

        logger.info(
            f"✅ Rating cleaning | Before: {before} → After: {after}"
        )

        return df


    # ==========================================
    # Movie Metadata Cleaning
    # ==========================================

    def clean_movies(self, movies: pd.DataFrame):
        """
        Fill missing genres and clean movie metadata.
        """

        logger.info("🎬 Cleaning movie metadata...")

        movies = movies.copy()

        missing_before = movies["genres"].isna().sum()

        movies["genres"] = movies["genres"].fillna("Unknown")

        missing_after = movies["genres"].isna().sum()

        logger.info(
            f"✅ Genres cleaned | Missing: {missing_before} → {missing_after}"
        )

        return movies


    # ==========================================
    # Links Cleaning
    # ==========================================

    def clean_links(self, links: pd.DataFrame):
        """
        Remove rows with missing external IDs.
        """

        logger.info("🔗 Cleaning links dataset...")

        before = len(links)

        links = links.dropna(
            subset=["imdbId", "tmdbId"],
            how="all"
        )

        after = len(links)

        logger.info(
            f"✅ Links cleaned | Before: {before} → After: {after}"
        )

        return links


    # ==========================================
    # Tags Cleaning
    # ==========================================

    def clean_tags(self, tags: pd.DataFrame):
        """
        Normalize tags and remove rare/generic ones.
        """

        logger.info("🏷️ Cleaning tags dataset...")

        before = len(tags)

        tags = tags.copy()

        # Remove missing tags
        tags = tags.dropna(subset=["tag"])

        # Normalize text
        tags["tag"] = (
            tags["tag"]
            .str.lower()
            .str.strip()
        )

        # Remove generic tags
        tags = tags[
            ~tags["tag"].isin(self.config.GENERIC_TAGS)
        ]

        # Remove rare tags
        tag_counts = tags["tag"].value_counts()

        rare_tags = tag_counts[
            tag_counts < self.config.MIN_TAG_FREQUENCY
        ].index

        tags = tags[
            ~tags["tag"].isin(rare_tags)
        ]

        after = len(tags)

        logger.info(
            f"✅ Tags cleaned | Before: {before} → After: {after}"
        )

        return tags
