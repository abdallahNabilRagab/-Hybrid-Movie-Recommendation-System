# ==========================================
# Main Recommendation Pipeline (Full Saving Version)
# Unified, Safe Logging to Avoid Duplicate Messages
# ==========================================

import logging
import time
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import shutil
import faiss

# =====================
# Import Project Modules
# =====================

# Data
from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor

# Features & Split
from src.features import FeatureBuilder
from src.split import TemporalSplitter

# Content-Based
from src.content_based.vectorize import ContentVectorizer
from src.content_based.search import ContentSearcher

# Hybrid & Evaluation
from src.hybrid.hybrid import HybridRecommender
from src.evaluation import Evaluator

# ALS Modules
from src.als.train import ALSTrainer
from src.als.recommend import ALSRecommender
from src.als.evaluate import ALSEvaluator

# =====================
# Unified Logging Configuration
# =====================
logger = logging.getLogger("recommender_pipeline")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# =====================
# Project Paths
# =====================

ROOT = Path(__file__).resolve().parents[1]

DATA_PROCESSED = ROOT / "data" / "processed"
DATA_SPLITS = ROOT / "data" / "splits"
MODELS_DIR = ROOT / "models"
EVAL_DIR = DATA_PROCESSED / "evaluation"

for p in [DATA_PROCESSED, DATA_SPLITS, MODELS_DIR, EVAL_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# =====================
# Helper Save Functions
# =====================

def save_pickle(obj, path):
    """Save object as pickle if file does not exist."""
    if path.exists():
        logger.info(f"File exists, skipping save: {path.name}")
    else:
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Saved pickle: {path.name}")

# =====================
# Main Function
# =====================

def main():
    start_time = time.time()
    logger.info("🚀 Starting Recommendation Pipeline...")

    # ---------------------
    # Initialize Components
    # ---------------------
    loader = DataLoader()
    preprocessor = DataPreprocessor()
    builder = FeatureBuilder()
    splitter = TemporalSplitter()
    vectorizer = ContentVectorizer()
    evaluator = Evaluator()

    # ---------------------
    # Load Raw Data
    # ---------------------
    logger.info("📥 Loading raw datasets...")
    data = loader.load_data()
    ratings, movies, tags, links = data["ratings"], data["movies"], data["tags"], data["links"]

    # ---------------------
    # Data Preprocessing
    # ---------------------
    logger.info("🧹 Preprocessing data...")

    user_activity, movie_activity = preprocessor.compute_activity(ratings)
    filtered = preprocessor.filter_interactions(ratings, user_activity, movie_activity)
    filtered = preprocessor.process_timestamps(filtered)
    filtered = preprocessor.clean_ratings(filtered)
    filtered = preprocessor.remove_duplicate_interactions(filtered)

    movies = preprocessor.clean_movies(movies)
    tags = preprocessor.clean_tags(tags)
    links = preprocessor.clean_links(links)

    clean_interactions_path = DATA_PROCESSED / "clean_interactions.csv"
    clean_movies_path = DATA_PROCESSED / "clean_movies.csv"
    clean_tags_path = DATA_PROCESSED / "clean_tags.csv"

    if not clean_interactions_path.exists():
        filtered.to_csv(clean_interactions_path, index=False)
    if not clean_movies_path.exists():
        movies.to_csv(clean_movies_path, index=False)
    if not clean_tags_path.exists():
        tags.to_csv(clean_tags_path, index=False)

    # ---------------------
    # Feature Engineering
    # ---------------------
    logger.info("🛠️ Building features...")
    features_path = DATA_PROCESSED / "features.pkl"
    if features_path.exists():
        logger.info("Features already exist, loading...")
        with open(features_path, "rb") as f:
            features = pickle.load(f)
    else:
        features = builder.build_features(filtered, movies, tags)
        save_pickle(features, features_path)

    # ---------------------
    # Train/Test Split
    # ---------------------
    logger.info("📆 Temporal split...")
    train_path = DATA_SPLITS / "train.csv"
    test_path = DATA_SPLITS / "test.csv"

    if train_path.exists() and test_path.exists():
        logger.info("Train/Test split already exists, loading...")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    else:
        train_df, test_df = splitter.split(filtered)
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

    # ---------------------
    # ALS Preparation & Training
    # ---------------------
    logger.info("💪 Preparing ALS pipeline...")

    # Compute confidence
    if "confidence" not in train_df.columns:
        alpha = 40
        train_df["confidence"] = 1 + alpha * train_df["rating"]

    # User/Item mappings
    user_map_path = MODELS_DIR / "user_map.pkl"
    item_map_path = MODELS_DIR / "item_map.pkl"

    if user_map_path.exists() and item_map_path.exists():
        logger.info("User and item mappings already exist, loading...")
        with open(user_map_path, "rb") as f:
            user_map = pickle.load(f)
        with open(item_map_path, "rb") as f:
            item_map = pickle.load(f)
    else:
        logger.info("Creating user and item mappings...")
        user_map = {u: i for i, u in enumerate(train_df["userId"].unique())}
        item_map = {m: i for i, m in enumerate(train_df["movieId"].unique())}
        save_pickle(user_map, user_map_path)
        save_pickle(item_map, item_map_path)

    inv_item_map = {i: m for m, i in item_map.items()}

    if "u" not in train_df.columns or "i" not in train_df.columns:
        train_df["u"] = train_df["userId"].map(user_map)
        train_df["i"] = train_df["movieId"].map(item_map)

    # ALS Model Training
    from scipy.sparse import save_npz, load_npz

    als_model_path = MODELS_DIR / "als_model.pkl"
    X_sparse_path = MODELS_DIR / "X_sparse.npz"

    als_trainer = ALSTrainer(factors=64, iterations=20)

    if als_model_path.exists() and X_sparse_path.exists():
        logger.info("ALS model and sparse matrix already exist, loading...")
        with open(als_model_path, "rb") as f:
            als_model_raw = pickle.load(f)
        X_sparse = load_npz(X_sparse_path)
    else:
        logger.info("Training ALS model...")
        als_model_raw, X_sparse = als_trainer.fit(train_df, user_map, item_map)
        save_pickle(als_model_raw, als_model_path)
        save_npz(X_sparse_path, X_sparse)

    als_recommender = ALSRecommender(
        model=als_model_raw,
        X=X_sparse,
        user_map=user_map,
        item_map=item_map,
        inv_item_map=inv_item_map
    )

    # ALS Evaluation
    als_results_path = EVAL_DIR / "als_results.csv"

    if als_results_path.exists():
        logger.info("ALS evaluation results already exist, loading...")
        als_results = pd.read_csv(als_results_path)
    else:
        logger.info("Evaluating ALS model...")
        als_evaluator = ALSEvaluator(als_recommender)
        test_user_items = test_df.groupby("userId")["movieId"].apply(set)
        preds = als_evaluator.precompute(test_user_items, k=10)
        precision_als = als_evaluator.precision(test_user_items, preds, k=10)
        recall_als = als_evaluator.recall(test_user_items, preds, k=10)
        ndcg_als = als_evaluator.ndcg(test_user_items, preds, k=10)
        als_results = pd.DataFrame([{
            "precision@10": precision_als,
            "recall@10": recall_als,
            "ndcg@10": ndcg_als
        }])
        als_results.to_csv(als_results_path, index=False)
        logger.info(f"✅ ALS evaluation results saved to {als_results_path}")

    # ---------------------
    # Content-Based System
    # ---------------------
    logger.info("📚 Building content-based system...")

    mlb_path = MODELS_DIR / "mlb.pkl"
    tfidf_path = MODELS_DIR / "tfidf.pkl"
    movieId_to_index_path = MODELS_DIR / "movieId_to_index.pkl"
    item_features_path = MODELS_DIR / "item_features.npy"
    faiss_index_path = MODELS_DIR / "faiss.index"

    models_exist = all(p.exists() for p in [mlb_path, tfidf_path, movieId_to_index_path, item_features_path, faiss_index_path])

    if models_exist:
        logger.info("Content-based models already exist, loading...")
        with open(mlb_path, "rb") as f:
            mlb = pickle.load(f)
        with open(tfidf_path, "rb") as f:
            tfidf = pickle.load(f)
        with open(movieId_to_index_path, "rb") as f:
            movieId_to_index = pickle.load(f)
        item_features = np.load(item_features_path)
        faiss_index = faiss.read_index(str(faiss_index_path))
        index_to_movieId = {i: m for m, i in movieId_to_index.items()}
    else:
        # Build features
        logger.info("🛠️ Building item features from movies and tags...")
        item_features, mlb, tfidf = vectorizer.build_item_features(movies, tags)
        save_pickle(mlb, mlb_path)
        save_pickle(tfidf, tfidf_path)
        np.save(item_features_path, item_features)

        # Build FAISS index
        temp_faiss_dir = Path(r"D:\Books & Courses")
        temp_faiss_dir.mkdir(parents=True, exist_ok=True)
        temp_faiss_path = temp_faiss_dir / "faiss_temp.index"

        faiss_index = vectorizer.build_faiss_index(item_features)
        faiss.write_index(faiss_index, str(temp_faiss_path))
        shutil.move(str(temp_faiss_path), str(faiss_index_path))
        logger.info(f"✅ FAISS index saved to {faiss_index_path}")

        # MovieId mappings
        movieId_to_index, index_to_movieId = vectorizer.build_id_mappings(movies)
        save_pickle(movieId_to_index, movieId_to_index_path)

    content_searcher = ContentSearcher(
        train_df,
        item_features,
        faiss_index,
        movieId_to_index,
        index_to_movieId
    )

    # ---------------------
    # Hybrid Model (No Pickle Saving)
    # ---------------------
    logger.info("🤝 Building hybrid recommender...")
    hybrid_model = HybridRecommender(
        als_recommender,
        content_searcher,
        train_df
    )
    logger.info("HybridRecommender ready for evaluation.")

    # ---------------------
    # Hybrid Evaluation (Sampled Users)
    # ---------------------

    EVAL_USER_SAMPLE = 100
    logger.info(f"📊 Evaluating Hybrid on a sample of {EVAL_USER_SAMPLE} users...")

    # ---------------------
    # Ensure test_df has correct columns
    # ---------------------
    if "userId" not in test_df.columns and "u" in test_df.columns:
        test_df = test_df.rename(columns={"u": "userId", "i": "movieId"})

    # Sample users from test set (only users present in test_df)
    available_users = test_df["userId"].drop_duplicates()
    sampled_users = available_users.sample(
        min(EVAL_USER_SAMPLE, len(available_users)),
        random_state=42
    )

    test_df_sampled = test_df[test_df["userId"].isin(sampled_users)].copy()

    logger.info(
        f"✅ Number of users used in Hybrid evaluation: {test_df_sampled['userId'].nunique()}"
    )

    # ---------------------
    # Run evaluation
    # ---------------------
    precision, recall, ndcg, evaluated_users = evaluator.evaluate_model(
        hybrid_model,
        test_df_sampled,
        k=10
    )

    # ---------------------
    # Save results
    # ---------------------
    hybrid_results_path = EVAL_DIR / "hybrid_results.csv"
    hybrid_results = pd.DataFrame([{
        "precision@10": precision,
        "recall@10": recall,
        "ndcg@10": ndcg,
        "evaluated_users": evaluated_users
    }])
    hybrid_results.to_csv(hybrid_results_path, index=False)

    logger.info(f"✅ Hybrid results saved to {hybrid_results_path}")


    # ---------------------
    # Finish
    # ---------------------
    total_time = time.time() - start_time
    logger.info("🎉 Pipeline finished successfully")
    logger.info(f"⏳ Runtime: {total_time:.2f} sec")


# =====================
# Run Script
# =====================
if __name__ == "__main__":
    main()
