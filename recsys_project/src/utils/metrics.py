# ==========================================
# metrics.py - Evaluation Metrics (Dict-Based)
# Compatible with Evaluator's ground_truth dict
# ==========================================

import numpy as np

# -----------------------------
# Recall@K
# -----------------------------
def recall_at_k(ground_truth, predictions, k=10):
    """
    Compute Recall@K for a set of users.

    Parameters
    ----------
    ground_truth : dict
        {user_id: set of true item_ids}

    predictions : dict
        {user_id: list of predicted item_ids}

    k : int
        Top-K cutoff

    Returns
    -------
    float
        Mean Recall@K over all users
    """
    scores = []

    for user, true_items in ground_truth.items():
        pred_items = predictions.get(user, [])
        if not true_items:
            continue
        hits = len(set(pred_items[:k]) & set(true_items))
        scores.append(hits / len(true_items))

    return np.mean(scores)


# -----------------------------
# Precision@K
# -----------------------------
def precision_at_k(ground_truth, predictions, k=10):
    """
    Compute Precision@K for a set of users.

    Parameters
    ----------
    ground_truth : dict
        {user_id: set of true item_ids}

    predictions : dict
        {user_id: list of predicted item_ids}

    k : int
        Top-K cutoff

    Returns
    -------
    float
        Mean Precision@K over all users
    """
    scores = []

    for user, true_items in ground_truth.items():
        pred_items = predictions.get(user, [])[:k]
        if not pred_items:
            continue
        hits = len(set(pred_items) & set(true_items))
        scores.append(hits / k)

    return np.mean(scores)


# -----------------------------
# NDCG@K
# -----------------------------
def ndcg_at_k(ground_truth, predictions, k=10):
    """
    Compute Normalized Discounted Cumulative Gain (NDCG)@K.

    Parameters
    ----------
    ground_truth : dict
        {user_id: set of true item_ids}

    predictions : dict
        {user_id: list of predicted item_ids}

    k : int
        Top-K cutoff

    Returns
    -------
    float
        Mean NDCG@K over all users
    """
    scores = []

    for user, true_items in ground_truth.items():
        pred_items = predictions.get(user, [])[:k]
        dcg = 0
        idcg = 0

        for i, item in enumerate(pred_items):
            if item in true_items:
                dcg += 1 / np.log2(i + 2)

        for i in range(min(len(true_items), k)):
            idcg += 1 / np.log2(i + 2)

        if idcg > 0:
            scores.append(dcg / idcg)

    return np.mean(scores)
