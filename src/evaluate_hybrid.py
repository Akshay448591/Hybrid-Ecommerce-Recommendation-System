from pathlib import Path
import json
import time

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

RECOMMENDATIONS_FILE = OUTPUT_DIR / "hybrid_validation_recommendations.parquet"
VALIDATION_FILE = DATA_DIR / "transactions_val.parquet"
TRAIN_FILE = DATA_DIR / "transactions_train.parquet"

OUTPUT_METRICS_FILE = OUTPUT_DIR / "hybrid_validation_metrics.json"

K_VALUES = [10, 20]


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def precision_at_k(recommended, relevant, k):
    recommended = recommended[:k]
    if not recommended:
        return 0.0

    hits = sum(item in relevant for item in recommended)
    return hits / k


def recall_at_k(recommended, relevant, k):
    recommended = recommended[:k]

    if len(relevant) == 0:
        return 0.0

    hits = sum(item in relevant for item in recommended)
    return hits / len(relevant)


def hit_rate_at_k(recommended, relevant, k):
    recommended = recommended[:k]

    return float(
        any(item in relevant for item in recommended)
    )


def ndcg_at_k(recommended, relevant, k):
    recommended = recommended[:k]

    if not relevant:
        return 0.0

    dcg = 0.0

    for rank, item in enumerate(recommended, start=1):
        if item in relevant:
            dcg += 1.0 / np.log2(rank + 1)

    ideal_hits = min(len(relevant), k)

    idcg = sum(
        1.0 / np.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


def average_precision_at_k(recommended, relevant, k):
    recommended = recommended[:k]

    if not relevant:
        return 0.0

    hits = 0
    score = 0.0

    for rank, item in enumerate(recommended, start=1):

        if item in relevant:
            hits += 1
            score += hits / rank

    denominator = min(len(relevant), k)

    if denominator == 0:
        return 0.0

    return score / denominator


# ============================================================
# COVERAGE
# ============================================================

def coverage_at_k(recommendations, total_items, k):
    recommended_items = set()

    for items in recommendations.values():
        recommended_items.update(items[:k])

    if total_items == 0:
        return 0.0

    return len(recommended_items) / total_items


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print("=" * 70)
    print("HYBRID VALIDATION EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load files
    # --------------------------------------------------------

    print("\n[1] Loading files")

    recommendations = pd.read_parquet(RECOMMENDATIONS_FILE)
    validation = pd.read_parquet(VALIDATION_FILE)
    training = pd.read_parquet(TRAIN_FILE)

    print(f"Recommendation rows: {len(recommendations):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Training rows: {len(training):,}")

    # --------------------------------------------------------
    # 2. Identify warm-start population
    # --------------------------------------------------------

    print("\n[2] Building warm-start population")

    train_users = set(training["user_index"].unique())
    train_items = set(training["item_index"].unique())

    validation["user_known"] = validation["user_index"].isin(
        train_users
    )

    validation["item_known"] = validation["item_index"].isin(
        train_items
    )

    warm_validation = validation[
        validation["user_known"]
        & validation["item_known"]
    ].copy()

    warm_users = sorted(
        warm_validation["user_index"].unique()
    )

    print(
        f"Warm-start interactions: "
        f"{len(warm_validation):,}"
    )

    print(
        f"Warm-start users: "
        f"{len(warm_users):,}"
    )

    assert len(warm_validation) == 751_993
    assert len(warm_users) == 173_203

    print("Warm-start population: PASS")

    # --------------------------------------------------------
    # 3. Build ground truth
    # --------------------------------------------------------

    print("\n[3] Building validation ground truth")

    ground_truth = (
        warm_validation
        .groupby("user_index")["item_index"]
        .apply(set)
        .to_dict()
    )

    assert len(ground_truth) == 173_203

    print(
        f"Ground-truth users: "
        f"{len(ground_truth):,}"
    )

    # --------------------------------------------------------
    # 4. Build recommendations
    # --------------------------------------------------------

    print("\n[4] Building recommendation lists")

    recommendations = recommendations.sort_values(
        ["train_user_id", "rank"]
    )

    recommendation_lists = (
        recommendations
        .groupby("train_user_id")["train_item_id"]
        .apply(list)
        .to_dict()
    )

    assert len(recommendation_lists) == 173_203

    print("Recommendation lists: PASS")

    # --------------------------------------------------------
    # 5. Map model user IDs to global user IDs
    # --------------------------------------------------------

    print("\n[5] Mapping recommendation users")

    user_mapping = pd.read_parquet(
        DATA_DIR / "train_user_mapping.parquet"
    )

    model_to_global_user = dict(
        zip(
            user_mapping["train_user_id"],
            user_mapping["user_index"]
        )
    )

    # --------------------------------------------------------
    # 6. Calculate ranking metrics
    # --------------------------------------------------------

    print("\n[6] Calculating ranking metrics")

    metric_values = {}

    for k in K_VALUES:

        precision_scores = []
        recall_scores = []
        hit_scores = []
        ndcg_scores = []
        map_scores = []

        for train_user_id, recommended in recommendation_lists.items():

            global_user_id = model_to_global_user[
                train_user_id
            ]

            relevant = ground_truth.get(
                global_user_id,
                set()
            )

            precision_scores.append(
                precision_at_k(
                    recommended,
                    relevant,
                    k
                )
            )

            recall_scores.append(
                recall_at_k(
                    recommended,
                    relevant,
                    k
                )
            )

            hit_scores.append(
                hit_rate_at_k(
                    recommended,
                    relevant,
                    k
                )
            )

            ndcg_scores.append(
                ndcg_at_k(
                    recommended,
                    relevant,
                    k
                )
            )

            map_scores.append(
                average_precision_at_k(
                    recommended,
                    relevant,
                    k
                )
            )

        metric_values[f"P@{k}"] = float(
            np.mean(precision_scores)
        )

        metric_values[f"R@{k}"] = float(
            np.mean(recall_scores)
        )

        metric_values[f"HR@{k}"] = float(
            np.mean(hit_scores)
        )

        metric_values[f"NDCG@{k}"] = float(
            np.mean(ndcg_scores)
        )

        metric_values[f"MAP@{k}"] = float(
            np.mean(map_scores)
        )

    # --------------------------------------------------------
    # 7. Coverage
    # --------------------------------------------------------

    print("\n[7] Calculating coverage")

    training_item_count = len(train_items)

    metric_values["Coverage@10"] = coverage_at_k(
        recommendation_lists,
        training_item_count,
        10
    )

    metric_values["Coverage@20"] = coverage_at_k(
        recommendation_lists,
        training_item_count,
        20
    )

    # --------------------------------------------------------
    # 8. Diversity and novelty
    # --------------------------------------------------------

    print("\n[8] Calculating diversity and novelty")

    # Item popularity in training data
    item_popularity = (
        training["item_index"]
        .value_counts()
        .to_dict()
    )

    total_training_interactions = len(training)

    diversity_scores = {}
    novelty_scores = {}

    for k in K_VALUES:

        diversity_user_scores = []
        novelty_user_scores = []

        for recommended in recommendation_lists.values():

            items = recommended[:k]

            # ------------------------------------------------
            # Diversity
            #
            # Pairwise distinct-item ratio.
            # ------------------------------------------------

            if len(items) <= 1:
                diversity = 0.0

            else:
                diversity = (
                    len(set(items))
                    / len(items)
                )

            diversity_user_scores.append(
                diversity
            )

            # ------------------------------------------------
            # Novelty
            #
            # Mean self-information:
            # -log2(popularity / total)
            # ------------------------------------------------

            item_novelties = []

            for item in items:

                count = item_popularity.get(
                    item,
                    0
                )

                if count > 0:

                    probability = (
                        count
                        / total_training_interactions
                    )

                    item_novelties.append(
                        -np.log2(probability)
                    )

            if item_novelties:

                novelty_user_scores.append(
                    np.mean(item_novelties)
                )

            else:

                novelty_user_scores.append(0.0)

        diversity_scores[k] = float(
            np.mean(diversity_user_scores)
        )

        novelty_scores[k] = float(
            np.mean(novelty_user_scores)
        )

        metric_values[f"Diversity@{k}"] = (
            diversity_scores[k]
        )

        metric_values[f"Novelty@{k}"] = (
            novelty_scores[k]
        )

    # --------------------------------------------------------
    # 9. Recommendation validation
    # --------------------------------------------------------

    print("\n[9] Validating metrics")

    for key, value in metric_values.items():

        assert np.isfinite(value), (
            f"{key} is not finite"
        )

    for k in K_VALUES:

        assert 0 <= metric_values[f"P@{k}"] <= 1
        assert 0 <= metric_values[f"R@{k}"] <= 1
        assert 0 <= metric_values[f"HR@{k}"] <= 1
        assert 0 <= metric_values[f"NDCG@{k}"] <= 1
        assert 0 <= metric_values[f"MAP@{k}"] <= 1
        assert 0 <= metric_values[f"Coverage@{k}"] <= 1
        assert 0 <= metric_values[f"Diversity@{k}"] <= 1
        assert metric_values[f"Novelty@{k}"] >= 0

    print("Metric ranges: PASS")

    # --------------------------------------------------------
    # 10. Evaluation time
    # --------------------------------------------------------

    evaluation_time = time.time() - start_time

    # --------------------------------------------------------
    # 11. Save
    # --------------------------------------------------------

    results = {
        "model": "Hybrid LightGCN + SASRec",
        "evaluation_population": "warm_start",
        "warm_start_users": len(warm_users),
        "warm_start_interactions": len(warm_validation),
        "training_users": len(train_users),
        "training_items": len(train_items),
        "recommendations_per_user": 20,
        "metrics": metric_values,
        "evaluation_time_seconds": evaluation_time,
    }

    with open(
        OUTPUT_METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # 12. Print results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("HYBRID VALIDATION RESULTS")
    print("=" * 70)

    print("\nRanking Metrics:")

    for k in K_VALUES:

        print(
            f"P@{k}:    "
            f"{metric_values[f'P@{k}']:.6f}"
        )

        print(
            f"R@{k}:    "
            f"{metric_values[f'R@{k}']:.6f}"
        )

        print(
            f"HR@{k}:   "
            f"{metric_values[f'HR@{k}']:.6f}"
        )

        print(
            f"NDCG@{k}: "
            f"{metric_values[f'NDCG@{k}']:.6f}"
        )

        print(
            f"MAP@{k}:  "
            f"{metric_values[f'MAP@{k}']:.6f}"
        )

    print("\nRecommendation Quality:")

    for k in K_VALUES:

        print(
            f"Coverage@{k}:  "
            f"{metric_values[f'Coverage@{k}']:.6f}"
        )

        print(
            f"Diversity@{k}: "
            f"{metric_values[f'Diversity@{k}']:.6f}"
        )

        print(
            f"Novelty@{k}:   "
            f"{metric_values[f'Novelty@{k}']:.6f}"
        )

    print(
        f"\nEvaluation time: "
        f"{evaluation_time:.2f}s"
    )

    print(
        f"\nSaved: "
        f"{OUTPUT_METRICS_FILE}"
    )

    print("\nHYBRID VALIDATION EVALUATION PASSED")


if __name__ == "__main__":
    main()