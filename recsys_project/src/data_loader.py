# ==========================================
# Data Loader Module
# Handles dataset validation and loading
# With Detailed Logging
# ==========================================

import pandas as pd
from typing import Dict
from pathlib import Path
import logging
import time

from .config import Config


# ==========================================
# Logger Configuration
# ==========================================

logger = logging.getLogger(__name__)


# ==========================================
# DataLoader Class
# ==========================================

class DataLoader:
    """
    Responsible for:
    - Verifying dataset files
    - Loading raw CSV data
    - Applying basic type optimization
    """


    def __init__(self, config: Config = Config):
        """
        Initialize DataLoader with configuration.

        Args:
            config (Config): Project configuration class
        """
        self.config = config

        logger.info("📁 DataLoader initialized")


    # ==========================================
    # Verify Required Dataset Files
    # ==========================================

    def verify_files(self) -> Dict[str, Path]:
        """
        Check if all required dataset files exist.

        Returns:
            Dict[str, Path]: Dictionary of verified file paths

        Raises:
            FileNotFoundError: If any required file is missing
        """

        logger.info("🔍 Verifying dataset files...")

        required_files = {
            "ratings": self.config.RATINGS_FILE,
            "movies": self.config.MOVIES_FILE,
            "tags": self.config.TAGS_FILE,
            "links": self.config.LINKS_FILE,
        }

        # Check existence of each file
        for name, path in required_files.items():

            logger.info(f"📄 Checking {name} file: {path}")

            if not path.exists():

                logger.error(f"❌ Missing file: {name} at {path}")

                raise FileNotFoundError(
                    f"{name} dataset not found at: {path}"
                )

        logger.info("✅ All dataset files verified successfully")

        return required_files


    # ==========================================
    # Load Datasets
    # ==========================================

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all datasets into Pandas DataFrames.

        Returns:
            Dict[str, pd.DataFrame]: Loaded datasets
        """

        start_time = time.time()

        logger.info("📥 Starting dataset loading process...")

        # Verify files before loading
        self.verify_files()

        # ------------------------------
        # Load Ratings Dataset
        # ------------------------------

        logger.info("⭐ Loading ratings dataset...")

        ratings = pd.read_csv(
            self.config.RATINGS_FILE,
            dtype={
                "userId": "int32",
                "movieId": "int32",
                "rating": "float32",
                "timestamp": "int64"
            }
        )

        logger.info(
            f"✅ Ratings loaded | Shape: {ratings.shape}"
        )

        # ------------------------------
        # Load Movies Dataset
        # ------------------------------

        logger.info("🎬 Loading movies dataset...")

        movies = pd.read_csv(self.config.MOVIES_FILE)

        logger.info(
            f"✅ Movies loaded | Shape: {movies.shape}"
        )

        # ------------------------------
        # Load Tags Dataset
        # ------------------------------

        logger.info("🏷️ Loading tags dataset...")

        tags = pd.read_csv(self.config.TAGS_FILE)

        logger.info(
            f"✅ Tags loaded | Shape: {tags.shape}"
        )

        # ------------------------------
        # Load Links Dataset
        # ------------------------------

        logger.info("🔗 Loading links dataset...")

        links = pd.read_csv(self.config.LINKS_FILE)

        logger.info(
            f"✅ Links loaded | Shape: {links.shape}"
        )

        # ------------------------------
        # Finish Loading
        # ------------------------------

        end_time = time.time()
        elapsed = end_time - start_time

        logger.info("🎉 All datasets loaded successfully")
        logger.info(f"⏱️ Data loading time: {elapsed:.2f} seconds")

        return {
            "ratings": ratings,
            "movies": movies,
            "tags": tags,
            "links": links
        }
