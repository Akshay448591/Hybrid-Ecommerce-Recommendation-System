"""
Evaluate Hybrid V3-A on the warm-start validation set.

Metrics:
Ranking:
    P@10, P@20
    R@10, R@20
    HR@10, HR@20
    NDCG@10, NDCG@20
    MAP@10, MAP@20

Recommendation quality:
    Coverage@10, Coverage@20
    Diversity@10, Diversity@20
    Novelty@10, Novelty@20

Efficiency:
    Evaluation time

Evaluation population:
    Warm-start validation interactions only:
        user seen in training
        item seen in training

Recommendations:
    output/hybrid_v3a_validation_recommendations.parquet

Model:
    output/hybrid_v3a_model.pt
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hybrid_model import (
    HybridLightGCNSASRecV3,
    LightGCN,
    SASRec,
)


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

TRAIN_PATH = DATA_DIR / "transactions_train.parquet"
VAL_PATH = DATA_DIR / "transactions_val.parquet"

TRAIN_MODEL_IDS_PATH = (
    DATA_DIR / "transactions_train_model_ids.parquet"
)

TRAIN_USER_MAPPING_PATH = (
    DATA_DIR / "train_user_mapping.parquet"
)

TRAIN_ITEM_MAPPING_PATH = (
    DATA_DIR / "train_item_mapping.parquet"
)

RECOMMENDATIONS_PATH = (
    OUTPUT_DIR / "hybrid_v3a_validation_recommendations.parquet"
)

MODEL_PATH = (
    OUTPUT_DIR / "hybrid_v3a_model.pt"
)

LIGHTGCN_MODEL_PATH = (
    OUTPUT_DIR / "lightgcn_model.pt"
)

SASREC_MODEL_PATH = (
    OUTPUT_DIR / "sasrec_model.pt"
)

LIGHTGCN_USER_EMBEDDINGS_PATH = (
    OUTPUT_DIR / "lightgcn_user_embeddings.npy"
)

LIGHTGCN_ITEM_EMBEDDINGS_PATH = (
    OUTPUT_DIR / "lightgcn_item_embeddings.npy"
)

METRICS_PATH = (
    OUTPUT_DIR / "hybrid_v3a_validation_metrics.json"
)


# ============================================================
# MODEL CONFIG
# ============================================================

NUM_USERS = 778_421
NUM_ITEMS = 74_541

EMBEDDING_DIM = 64
MAX_SEQUENCE_LENGTH = 50

LIGHTGCN_LAYERS = 3

SASREC_HEADS = 2
SASREC_LAYERS = 2
SASREC_DROPOUT = 0.2

TOP_KS = [10, 20]

DEVICE = torch.device("cpu")


# ============================================================
# HELPERS
# ============================================================


def print_header(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def load_parquet(path, name):
    print(f"Loading {name}: {path}")
    df = pd.read_parquet(path)

    print(
        f"{name}: "
        f"{len(df):,} rows | "
        f"{df['user_index'].nunique():,} users | "
        f"{df['item_index'].nunique():,} items"
    )

    return df


# ============================================================
# LOAD DATA
# ============================================================


def load_data():
    print_header("LOADING DATA")

    train = pd.read_parquet(TRAIN_PATH)

    val = pd.read_parquet(VAL_PATH)

    recommendations = pd.read_parquet(
        RECOMMENDATIONS_PATH
    )

    train_model_ids = pd.read_parquet(
        TRAIN_MODEL_IDS_PATH,
        columns=[
            "user_index",
            "item_index",
            "train_user_id",
            "train_item_id",
        ],
    )

    train_user_mapping = pd.read_parquet(
        TRAIN_USER_MAPPING_PATH
    )

    train_item_mapping = pd.read_parquet(
        TRAIN_ITEM_MAPPING_PATH
    )

    print(
        f"Train interactions: "
        f"{len(train):,}"
    )

    print(
        f"Validation interactions: "
        f"{len(val):,}"
    )

    print(
        f"Recommendation rows: "
        f"{len(recommendations):,}"
    )

    print(
        f"Recommendation users: "
        f"{recommendations['user_index'].nunique():,}"
    )

    return (
        train,
        val,
        recommendations,
        train_model_ids,
        train_user_mapping,
        train_item_mapping,
    )


# ============================================================
# BUILD TRAINING VOCABULARY
# ============================================================


def build_training_vocabulary(
    train_user_mapping,
    train_item_mapping,
):
    print_header("BUILDING TRAINING VOCABULARY")

    train_users = set(
        train_user_mapping["user_index"]
        .astype(np.int64)
        .tolist()
    )

    train_items = set(
        train_item_mapping["item_index"]
        .astype(np.int64)
        .tolist()
    )

    print(
        f"Training users: "
        f"{len(train_users):,}"
    )

    print(
        f"Training items: "
        f"{len(train_items):,}"
    )

    return train_users, train_items


# ============================================================
# WARM-START VALIDATION
# ============================================================


def build_warm_start_validation(
    val,
    train_users,
    train_items,
):
    print_header("BUILDING WARM-START VALIDATION SET")

    val_users = set(
        val["user_index"]
        .astype(np.int64)
        .tolist()
    )

    val_items = set(
        val["item_index"]
        .astype(np.int64)
        .tolist()
    )

    known_users = val_users.intersection(
        train_users
    )

    known_items = val_items.intersection(
        train_items
    )

    print(
        f"Validation users: "
        f"{len(val_users):,}"
    )

    print(
        f"Validation items: "
        f"{len(val_items):,}"
    )

    print(
        f"Known validation users: "
        f"{len(known_users):,}"
    )

    print(
        f"Known validation items: "
        f"{len(known_items):,}"
    )

    warm_start = val[
        val["user_index"].isin(known_users)
        & val["item_index"].isin(known_items)
    ].copy()

    print(
        f"Warm-start interactions: "
        f"{len(warm_start):,}"
    )

    print(
        f"Warm-start users: "
        f"{warm_start['user_index'].nunique():,}"
    )

    print(
        f"Warm-start items: "
        f"{warm_start['item_index'].nunique():,}"
    )

    return warm_start


# ============================================================
# GROUND TRUTH
# ============================================================


def build_ground_truth(warm_start):
    """
    Build unique future-item targets for each user.

    Repeated purchases of the same item are treated as
    one relevant item for ranking metrics.
    """

    print_header("BUILDING GROUND TRUTH")

    grouped = (
        warm_start
        .groupby("user_index")["item_index"]
        .apply(lambda x: set(x.astype(np.int64)))
    )

    ground_truth = grouped.to_dict()

    print(
        f"Ground-truth users: "
        f"{len(ground_truth):,}"
    )

    target_sizes = np.array(
        [len(v) for v in ground_truth.values()],
        dtype=np.int64,
    )

    print(
        f"Target items/user - "
        f"mean: {target_sizes.mean():.4f} | "
        f"median: {np.median(target_sizes):.0f} | "
        f"max: {target_sizes.max():,}"
    )

    return ground_truth


# ============================================================
# RECOMMENDATION VALIDATION
# ============================================================


def validate_recommendations(
    recommendations,
    ground_truth,
):
    print_header("VALIDATING RECOMMENDATIONS")

    required_columns = {
        "user_index",
        "item_index",
        "rank",
    }

    missing = required_columns.difference(
        recommendations.columns
    )

    if missing:
        raise ValueError(
            f"Missing recommendation columns: {missing}"
        )

    expected_users = set(
        ground_truth.keys()
    )

    recommendation_users = set(
        recommendations["user_index"]
        .astype(np.int64)
        .tolist()
    )

    if expected_users != recommendation_users:
        missing_users = (
            expected_users - recommendation_users
        )
        extra_users = (
            recommendation_users - expected_users
        )

        raise ValueError(
            "Recommendation users do not match "
            "evaluation users. "
            f"Missing={len(missing_users):,}, "
            f"Extra={len(extra_users):,}"
        )

    duplicate_rows = recommendations.duplicated(
        subset=["user_index", "rank"]
    ).sum()

    if duplicate_rows:
        raise ValueError(
            f"Duplicate user/rank rows: "
            f"{duplicate_rows:,}"
        )

    max_rank = (
        recommendations
        .groupby("user_index")["rank"]
        .max()
    )

    if not (max_rank == 20).all():
        raise ValueError(
            "Not every user has exactly 20 recommendations."
        )

    duplicate_items = (
        recommendations
        .groupby("user_index")["item_index"]
        .apply(lambda x: x.nunique() != len(x))
        .sum()
    )

    if duplicate_items:
        raise ValueError(
            f"Users with duplicate recommended items: "
            f"{duplicate_items:,}"
        )

    print(
        "[PASS] Recommendation structure valid"
    )

    print(
        f"Users: "
        f"{recommendation_users.__len__():,}"
    )

    print(
        f"Rows: "
        f"{len(recommendations):,}"
    )


# ============================================================
# RANKING METRICS
# ============================================================


def precision_at_k(recommended, relevant, k):
    rec = recommended[:k]

    if len(rec) == 0:
        return 0.0

    hits = sum(
        1 for item in rec
        if item in relevant
    )

    return hits / k


def recall_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0

    rec = recommended[:k]

    hits = sum(
        1 for item in rec
        if item in relevant
    )

    return hits / len(relevant)


def hit_rate_at_k(recommended, relevant, k):
    rec = recommended[:k]

    return float(
        any(item in relevant for item in rec)
    )


def ndcg_at_k(recommended, relevant, k):
    rec = recommended[:k]

    dcg = 0.0

    for rank, item in enumerate(rec, start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(
        len(relevant),
        k,
    )

    if ideal_hits == 0:
        return 0.0

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / idcg


def map_at_k(recommended, relevant, k):
    """
    Mean Average Precision for one user.

    Binary relevance.
    """

    if not relevant:
        return 0.0

    rec = recommended[:k]

    hits = 0
    precision_sum = 0.0

    for rank, item in enumerate(rec, start=1):
        if item in relevant:
            hits += 1
            precision_sum += hits / rank

    denominator = min(
        len(relevant),
        k,
    )

    if denominator == 0:
        return 0.0

    return precision_sum / denominator


def calculate_ranking_metrics(
    recommendations,
    ground_truth,
):
    print_header("CALCULATING RANKING METRICS")

    recommendation_groups = (
        recommendations
        .sort_values(
            ["user_index", "rank"]
        )
        .groupby("user_index")["item_index"]
        .apply(list)
        .to_dict()
    )

    results = {}

    for k in TOP_KS:

        precisions = []
        recalls = []
        hit_rates = []
        ndcgs = []
        maps = []

        for user_id, relevant in ground_truth.items():

            recommended = recommendation_groups[
                user_id
            ]

            precisions.append(
                precision_at_k(
                    recommended,
                    relevant,
                    k,
                )
            )

            recalls.append(
                recall_at_k(
                    recommended,
                    relevant,
                    k,
                )
            )

            hit_rates.append(
                hit_rate_at_k(
                    recommended,
                    relevant,
                    k,
                )
            )

            ndcgs.append(
                ndcg_at_k(
                    recommended,
                    relevant,
                    k,
                )
            )

            maps.append(
                map_at_k(
                    recommended,
                    relevant,
                    k,
                )
            )

        results[f"P@{k}"] = float(
            np.mean(precisions)
        )

        results[f"R@{k}"] = float(
            np.mean(recalls)
        )

        results[f"HR@{k}"] = float(
            np.mean(hit_rates)
        )

        results[f"NDCG@{k}"] = float(
            np.mean(ndcgs)
        )

        results[f"MAP@{k}"] = float(
            np.mean(maps)
        )

    for key, value in results.items():
        print(
            f"{key:10s}: {value:.6f}"
        )

    return results, recommendation_groups


# ============================================================
# LOAD V3-A ITEM REPRESENTATIONS
# ============================================================


def load_v3a_model():
    print_header("LOADING V3-A MODEL")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model = HybridLightGCNSASRecV3(
        num_users=NUM_USERS,
        num_items=NUM_ITEMS,
        embedding_dim=EMBEDDING_DIM,
        max_seq_len=MAX_SEQUENCE_LENGTH,
        num_heads=SASREC_HEADS,
        num_layers=SASREC_LAYERS,
        dropout=SASREC_DROPOUT,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    print(
        f"Checkpoint epochs: "
        f"{checkpoint.get('epochs')}"
    )

    print(
        f"Trainable parameters recorded: "
        f"{checkpoint.get('trainable_parameters'):,}"
    )

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Total parameters: "
        f"{total_parameters:,}"
    )

    return model


# ============================================================
# BUILD ITEM REPRESENTATIONS
# ============================================================


def build_v3a_item_representations(
    model,
):
    """
    Reconstruct the exact V3-A item representation used
    during recommendation generation.

    Item:
        L2-normalized LightGCN item
        +
        L2-normalized SASRec item embedding

        -> item_fusion
    """

    print_header(
        "BUILDING V3-A ITEM REPRESENTATIONS"
    )

    lightgcn_items = np.load(
        LIGHTGCN_ITEM_EMBEDDINGS_PATH,
        mmap_mode="r",
    )

    lightgcn_items = np.asarray(
        lightgcn_items,
        dtype=np.float32,
    )

    lightgcn_items = torch.from_numpy(
        lightgcn_items.copy()
    )

    print(
        f"LightGCN item embeddings: "
        f"{tuple(lightgcn_items.shape)}"
    )

    with torch.no_grad():

        graph_items = model.l2_normalize(
            lightgcn_items
        )

        sasrec_items = (
            model.sasrec.item_embedding.weight[1:]
            .detach()
        )

        sasrec_items = model.l2_normalize(
            sasrec_items
        )

        combined_items = torch.cat(
            [
                graph_items,
                sasrec_items,
            ],
            dim=1,
        )

        hybrid_items = model.item_fusion(
            combined_items
        )

        hybrid_items = torch.relu(
            hybrid_items
        )

    print(
        f"V3-A item representations: "
        f"{tuple(hybrid_items.shape)}"
    )

    if not torch.isfinite(
        hybrid_items
    ).all():
        raise ValueError(
            "Non-finite V3-A item representations."
        )

    return hybrid_items


# ============================================================
# COVERAGE
# ============================================================


def coverage_at_k(
    recommendation_groups,
    k,
    num_items,
):
    unique_items = set()

    for recommendations in recommendation_groups.values():
        unique_items.update(
            recommendations[:k]
        )

    return len(unique_items) / num_items


# ============================================================
# DIVERSITY
# ============================================================


def diversity_at_k(
    recommendation_groups,
    item_embeddings,
    k,
    batch_size=512,
):
    """
    Intra-list diversity.

    For each recommendation list:
        diversity = mean pairwise cosine distance

    cosine distance:
        1 - cosine_similarity

    Users with fewer than two items receive 0.
    """

    all_diversities = []

    item_embeddings = item_embeddings.cpu()

    for user_id, recommendations in recommendation_groups.items():

        rec_items = recommendations[:k]

        if len(rec_items) < 2:
            all_diversities.append(0.0)
            continue

        indices = torch.tensor(
            rec_items,
            dtype=torch.long,
        )

        embeddings = item_embeddings[
            indices
        ]

        embeddings = torch.nn.functional.normalize(
            embeddings,
            p=2,
            dim=1,
        )

        similarity = (
            embeddings @ embeddings.T
        )

        n = similarity.shape[0]

        upper_triangle = torch.triu(
            similarity,
            diagonal=1,
        )

        pair_count = (
            n * (n - 1)
        ) / 2

        similarity_sum = (
            upper_triangle.sum().item()
        )

        mean_similarity = (
            similarity_sum / pair_count
        )

        diversity = (
            1.0 - mean_similarity
        )

        all_diversities.append(
            diversity
        )

    return float(
        np.mean(all_diversities)
    )


# ============================================================
# NOVELTY
# ============================================================


def calculate_training_popularity(
    train,
):
    """
    Item popularity based on number of training
    interactions.

    Novelty:
        -log2(popularity probability)

    where:
        popularity probability =
            item_interactions / total_training_interactions
    """

    counts = (
        train["item_index"]
        .value_counts()
        .astype(np.float64)
    )

    total = counts.sum()

    popularity = counts / total

    novelty = -np.log2(
        popularity
    )

    return novelty.to_dict()


def novelty_at_k(
    recommendation_groups,
    item_novelty,
    k,
):
    values = []

    for recommendations in recommendation_groups.values():

        rec_items = recommendations[:k]

        for item in rec_items:

            if item in item_novelty:
                values.append(
                    item_novelty[item]
                )

    if not values:
        return 0.0

    return float(
        np.mean(values)
    )


# ============================================================
# QUALITY METRICS
# ============================================================


def calculate_quality_metrics(
    recommendation_groups,
    train,
):
    print_header(
        "CALCULATING RECOMMENDATION QUALITY"
    )

    item_novelty = (
        calculate_training_popularity(
            train
        )
    )

    results = {}

    for k in TOP_KS:

        coverage = coverage_at_k(
            recommendation_groups,
            k,
            NUM_ITEMS,
        )

        diversity = diversity_at_k(
            recommendation_groups,
            None,
            k,
        )

        novelty = novelty_at_k(
            recommendation_groups,
            item_novelty,
            k,
        )

        results[
            f"Coverage@{k}"
        ] = coverage

        results[
            f"Diversity@{k}"
        ] = diversity

        results[
            f"Novelty@{k}"
        ] = novelty

        print(
            f"Coverage@{k:2d}: "
            f"{coverage:.6f}"
        )

        print(
            f"Diversity@{k:2d}: "
            f"{diversity:.6f}"
        )

        print(
            f"Novelty@{k:2d}: "
            f"{novelty:.6f}"
        )

    return results


# ============================================================
# MAIN
# ============================================================


def main():

    print_header(
        "HYBRID V3-A VALIDATION EVALUATION"
    )

    evaluation_start = time.time()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    (
        train,
        val,
        recommendations,
        train_model_ids,
        train_user_mapping,
        train_item_mapping,
    ) = load_data()

    # --------------------------------------------------------
    # Training vocabulary
    # --------------------------------------------------------

    train_users, train_items = (
        build_training_vocabulary(
            train_user_mapping,
            train_item_mapping,
        )
    )

    # --------------------------------------------------------
    # Warm-start validation
    # --------------------------------------------------------

    warm_start = (
        build_warm_start_validation(
            val,
            train_users,
            train_items,
        )
    )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    ground_truth = build_ground_truth(
        warm_start
    )

    # --------------------------------------------------------
    # Recommendation validation
    # --------------------------------------------------------

    validate_recommendations(
        recommendations,
        ground_truth,
    )

    # --------------------------------------------------------
    # Ranking metrics
    # --------------------------------------------------------

    ranking_metrics, recommendation_groups = (
        calculate_ranking_metrics(
            recommendations,
            ground_truth,
        )
    )

    # --------------------------------------------------------
    # V3-A model
    # --------------------------------------------------------

    model = load_v3a_model()

    # --------------------------------------------------------
    # Item representations
    # --------------------------------------------------------

    item_embeddings = (
        build_v3a_item_representations(
            model
        )
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Recalculate diversity using actual V3-A
    # item representations.
    # --------------------------------------------------------

    quality_metrics = {}

    item_novelty = (
        calculate_training_popularity(
            train
        )
    )

    for k in TOP_KS:

        coverage = coverage_at_k(
            recommendation_groups,
            k,
            NUM_ITEMS,
        )

        diversity = diversity_at_k(
            recommendation_groups,
            item_embeddings,
            k,
        )

        novelty = novelty_at_k(
            recommendation_groups,
            item_novelty,
            k,
        )

        quality_metrics[
            f"Coverage@{k}"
        ] = coverage

        quality_metrics[
            f"Diversity@{k}"
        ] = diversity

        quality_metrics[
            f"Novelty@{k}"
        ] = novelty

    # --------------------------------------------------------
    # Evaluation time
    # --------------------------------------------------------

    evaluation_time = (
        time.time() - evaluation_start
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    metrics = {
        **ranking_metrics,
        **quality_metrics,
        "Evaluation Time": evaluation_time,
        "Evaluation Users": len(ground_truth),
        "Warm-start Interactions": int(
            len(warm_start)
        ),
        "Recommendation Rows": int(
            len(recommendations)
        ),
        "Recommendation K": 20,
        "Training Users": NUM_USERS,
        "Training Items": NUM_ITEMS,
        "Model": "Hybrid V3-A",
        "Architecture": (
            "Adaptive graph-sequential hybrid "
            "with user gate"
        ),
        "Trainable Parameters": 41_280,
        "Total Parameters": 59_504_832,
        "Seed": 42,
    }

    # --------------------------------------------------------
    # Print final result
    # --------------------------------------------------------

    print_header(
        "HYBRID V3-A FINAL VALIDATION METRICS"
    )

    print(
        f"Evaluation users: "
        f"{len(ground_truth):,}"
    )

    print(
        f"Warm-start interactions: "
        f"{len(warm_start):,}"
    )

    print()

    print("Ranking metrics:")

    for k in TOP_KS:

        print(
            f"P@{k:<2}     "
            f"{metrics[f'P@{k}']:.6f}"
        )

        print(
            f"R@{k:<2}     "
            f"{metrics[f'R@{k}']:.6f}"
        )

        print(
            f"HR@{k:<2}    "
            f"{metrics[f'HR@{k}']:.6f}"
        )

        print(
            f"NDCG@{k:<2}  "
            f"{metrics[f'NDCG@{k}']:.6f}"
        )

        print(
            f"MAP@{k:<2}   "
            f"{metrics[f'MAP@{k}']:.6f}"
        )

        print()

    print("Recommendation quality:")

    for k in TOP_KS:

        print(
            f"Coverage@{k:<2}   "
            f"{metrics[f'Coverage@{k}']:.6f}"
        )

        print(
            f"Diversity@{k:<2}  "
            f"{metrics[f'Diversity@{k}']:.6f}"
        )

        print(
            f"Novelty@{k:<2}    "
            f"{metrics[f'Novelty@{k}']:.6f}"
        )

        print()

    print(
        f"Evaluation time: "
        f"{evaluation_time:.2f}s"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
        )

    print()
    print(
        f"Saved metrics to:\n"
        f"{METRICS_PATH}"
    )

    print()
    print(
        "HYBRID V3-A EVALUATION PASSED"
    )


if __name__ == "__main__":
    main()