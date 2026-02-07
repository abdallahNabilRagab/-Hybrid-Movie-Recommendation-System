# ==========================================
# FeatureBuilder Module
# Builds all features for recommendation
# With Detailed Logging and Timing
# ==========================================

import pandas as pd
import logging
import time
from sklearn.preprocessing import MultiLabelBinarizer

# ==========================================
# Logger Configuration
# ==========================================

logger = logging.getLogger(__name__)

# ==========================================
# FeatureBuilder Class
# ==========================================

class FeatureBuilder:
    """
    Build all features required for recommendation models:
    - User statistics
    - Movie statistics
    - Temporal features
    - Yearly activity
    - Genre encoding
    - Tag features
    - Rating deviations
    - Index mappings
    """

    def __init__(self):
        logger.info("🛠️ FeatureBuilder initialized")

    # ------------------------------------------
    # User Features
    # ------------------------------------------
    def build_user_features(self, df: pd.DataFrame):
        logger.info("👤 Building user features...")
        count = df.groupby("userId")["rating"].count().rename("user_rating_count")
        avg = df.groupby("userId")["rating"].mean().rename("user_avg_rating")
        std = df.groupby("userId")["rating"].std().rename("user_std_rating")
        user_features = pd.concat([count, avg, std], axis=1)
        logger.info(f"✅ User features built | Shape: {user_features.shape}")
        return user_features

    # ------------------------------------------
    # Movie Features
    # ------------------------------------------
    def build_movie_features(self, df: pd.DataFrame):
        logger.info("🎬 Building movie features...")
        count = df.groupby("movieId")["rating"].count().rename("movie_rating_count")
        avg = df.groupby("movieId")["rating"].mean().rename("movie_avg_rating")
        std = df.groupby("movieId")["rating"].std().rename("movie_std_rating")
        movie_features = pd.concat([count, avg, std], axis=1)
        logger.info(f"✅ Movie features built | Shape: {movie_features.shape}")
        return movie_features

    # ------------------------------------------
    # Temporal Features
    # ------------------------------------------
    def build_temporal_features(self, df: pd.DataFrame):
        logger.info("⏱️ Building temporal features...")
        df = df.copy()
        df["year"] = df["timestamp"].dt.year
        df["month"] = df["timestamp"].dt.month
        df["day"] = df["timestamp"].dt.day
        logger.info("✅ Temporal features added")
        return df

    # ------------------------------------------
    # Yearly Activity
    # ------------------------------------------
    def build_yearly_activity(self, df: pd.DataFrame):
        logger.info("📅 Computing yearly activity...")
        user_yearly = df.groupby(["userId", "year"]).size().rename("user_yearly_ratings")
        movie_yearly = df.groupby(["movieId", "year"]).size().rename("movie_yearly_ratings")
        logger.info("✅ Yearly activity computed")
        return user_yearly, movie_yearly

    # ------------------------------------------
    # Genre Features
    # ------------------------------------------
    def encode_genres(self, movies: pd.DataFrame):
        logger.info("🎭 Encoding movie genres...")
        movies = movies.copy()
        movies["genres_list"] = movies["genres"].str.split("|")
        mlb = MultiLabelBinarizer()
        genre_features = pd.DataFrame(
            mlb.fit_transform(movies["genres_list"]),
            columns=mlb.classes_,
            index=movies["movieId"]
        )
        logger.info(f"✅ Genre encoding completed | Shape: {genre_features.shape}")
        return genre_features, mlb

    # ------------------------------------------
    # Tag Features
    # ------------------------------------------
    def build_tag_features(self, tags: pd.DataFrame):
        logger.info("🏷️ Building tag features...")
        tag_features = tags.groupby("movieId")["tag"].apply(list)
        tag_features = tag_features.apply(lambda x: " ".join(x))
        logger.info(f"✅ Tag features built | Movies: {len(tag_features)}")
        return tag_features

    # ------------------------------------------
    # Rating Deviations
    # ------------------------------------------
    def add_rating_deviation(self, df, user_avg, movie_avg):
        logger.info("📐 Computing rating deviations...")
        df = df.merge(user_avg, on="userId", how="left")
        df["rating_dev_user"] = df["rating"] - df["user_avg_rating"]
        df = df.merge(movie_avg, on="movieId", how="left")
        df["rating_dev_movie"] = df["rating"] - df["movie_avg_rating"]
        logger.info("✅ Rating deviations added")
        return df

    # ------------------------------------------
    # Index Mappings
    # ------------------------------------------
    def build_index_mapping(self, df):
        logger.info("🔢 Building index mappings...")
        user2idx = {u: i for i, u in enumerate(df["userId"].unique())}
        movie2idx = {m: i for i, m in enumerate(df["movieId"].unique())}
        df = df.copy()
        df["user_idx"] = df["userId"].map(user2idx)
        df["movie_idx"] = df["movieId"].map(movie2idx)
        logger.info(f"✅ Index mapping done | Users: {len(user2idx)} | Movies: {len(movie2idx)}")
        return df, user2idx, movie2idx

    # ------------------------------------------
    # Unified Feature Pipeline
    # ------------------------------------------
    def build_features(self, df, movies, tags):
        logger.info("🚀 Starting unified feature engineering pipeline...")
        start_time = time.time()

        # Step 1: User & Movie Features
        user_features = self.build_user_features(df)
        movie_features = self.build_movie_features(df)

        # Step 2: Temporal Features
        df = self.build_temporal_features(df)

        # Step 3: Yearly Activity
        user_yearly, movie_yearly = self.build_yearly_activity(df)

        # Step 4: Genre Features
        genre_features, mlb = self.encode_genres(movies)

        # Step 5: Tag Features
        tag_features = self.build_tag_features(tags)

        # Step 6: Rating Deviations
        df = self.add_rating_deviation(df, user_features[["user_avg_rating"]], movie_features[["movie_avg_rating"]])

        # Step 7: Index Mappings
        df, user2idx, movie2idx = self.build_index_mapping(df)

        elapsed = time.time() - start_time
        logger.info(f"🎉 Feature engineering pipeline completed in {elapsed:.2f}s")

        return {
            "df": df,
            "user_features": user_features,
            "movie_features": movie_features,
            "user_yearly": user_yearly,
            "movie_yearly": movie_yearly,
            "genre_features": genre_features,
            "tag_features": tag_features,
            "user2idx": user2idx,
            "movie2idx": movie2idx,
            "mlb": mlb
        }
