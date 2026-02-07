# 📁 src/config.py
from pathlib import Path


class Config:

    # Root Project
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # Data Paths
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    SPLITS_DIR = DATA_DIR / "splits"

    # Dataset Files
    RATINGS_FILE = RAW_DATA_DIR / "ratings.csv"
    MOVIES_FILE = RAW_DATA_DIR / "movies.csv"
    TAGS_FILE = RAW_DATA_DIR / "tags.csv"
    LINKS_FILE = RAW_DATA_DIR / "links.csv"

    # Cleaning Thresholds
    MIN_USER_RATINGS = 5
    MIN_MOVIE_RATINGS = 5
    COLD_START_THRESHOLD = 10

    # Rating Range
    MIN_RATING = 0.5
    MAX_RATING = 5.0

    # Tag Filtering
    GENERIC_TAGS = [
        "movie", "film", "cinema", "actor", "actress"
    ]

    MIN_TAG_FREQUENCY = 5
