# ==========================================
# Content-Based Movie Searcher (Silent)
# FAISS-based Similarity Search
# ==========================================

import numpy as np
import logging

# ==========================================
# Silent Logger (THIS FILE ONLY)
# ==========================================
logger = logging.getLogger("ContentSearcher")
logger.propagate = False
logger.handlers.clear()
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.CRITICAL)

# ==========================================
# ContentSearcher Class
# ==========================================
class ContentSearcher:
    """
    Content-based recommender using:
    - Item feature vectors
    - FAISS similarity search

    Design:
    - 🔕 Completely silent (no logging)
    - 🚀 Optimized for large-scale evaluation
    """

    def __init__(
        self,
        train_df,
        item_features,
        faiss_index,
        movieId_to_index,
        index_to_movieId
    ):
        self.train_df = train_df
        self.item_features = item_features
        self.faiss_index = faiss_index
        self.movieId_to_index = movieId_to_index
        self.index_to_movieId = index_to_movieId

    # -------------------------
    # Recommend Items
    # -------------------------
    def recommend(self, user_id: int, top_k: int = 10):
        """
        Generate top-K content-based recommendations for a user.

        Strategy:
        - Cold-start → popular items
        - User profile → mean of positive item vectors
        - FAISS similarity search
        - Filter seen items
        """

        # -------------------------
        # Cold-start: popular movies
        # -------------------------
        if user_id not in self.train_df["userId"].values:
            top_movies = (
                self.train_df.groupby("movieId")["rating"]
                .count()
                .sort_values(ascending=False)
                .head(top_k)
                .index
                .tolist()
            )
            return [(m, 1.0) for m in top_movies]

        # -------------------------
        # User interaction history
        # -------------------------
        user_rated = self.train_df[self.train_df["userId"] == user_id]
        rated_movie_set = set(user_rated["movieId"].values)

        # -------------------------
        # Select positive interactions
        # -------------------------
        positive_movies = user_rated[
            user_rated["rating"] >= user_rated["rating"].mean()
        ]["movieId"].values

        if len(positive_movies) == 0:
            positive_movies = user_rated["movieId"].values

        # keep only valid movies
        positive_movies = [
            m for m in positive_movies if m in self.movieId_to_index
        ]

        if len(positive_movies) == 0:
            positive_movies = list(rated_movie_set)

        # -------------------------
        # Build user profile vector
        # -------------------------
        indices = [self.movieId_to_index[m] for m in positive_movies]
        user_vector = np.mean(
            self.item_features[indices],
            axis=0,
            keepdims=True
        )

        # -------------------------
        # FAISS similarity search
        # -------------------------
        distances, indices = self.faiss_index.search(
            user_vector,
            top_k * 2
        )

        # -------------------------
        # Filter seen items
        # -------------------------
        recommendations = []
        for idx, score in zip(indices[0], distances[0]):
            movie_id = self.index_to_movieId.get(idx)
            if movie_id and movie_id not in rated_movie_set:
                recommendations.append((movie_id, float(score)))
            if len(recommendations) >= top_k:
                break

        return recommendations
