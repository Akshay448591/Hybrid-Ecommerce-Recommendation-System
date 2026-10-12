"""
Evaluate Hybrid LightGCN + SASRec Recommendation Model - V2

Evaluation protocol
-------------------
Validation period:
    2020-03-01 to 2020-03-31

Primary evaluation population:
    Warm-start validation interactions only.

Warm-start definition:
    User was present in training AND
    item was present in training.

Recommendation file:
    output/hybrid_v2_validation_recommendations.parquet

Metrics:
    Ranking:
        P@10
        P@20
        R@10
        R@20
        HR@10
        HR@20
        NDCG@10
        NDCG@20
        MAP@10
        MAP@20

    Recommendation quality:
        Coverage@10
        Coverage@20
        Diversity@10
        Diversity@20
        Novelty@10
        Novelty@20

    Efficiency:
        Evaluation time

Important:
    Diversity is NOT calculated as within-list uniqueness.

    Diversity is calculated using pairwise cosine distance
    between recommended item representations.

    Novelty uses training interaction popularity:
        novelty(item) = -log2(popularity(item))

    where:
        popularity(item) =
            training interaction count / total training interactions

All item representations used for diversity are the final
Hybrid V2 item representations.

No model retraining occurs in this script.
"""

from pathlib import Path
from collections import defaultdict
import json
import math
import time

import numpy as np
import pandas as pd
import torch

from hybrid_model import HybridLightGCNSASRec


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"


# ============================================================
# INPUT FILES
# ============================================================

TRAIN_FILE = (
    DATA_DIR /
    "transactions_train.parquet"
)

VALIDATION_FILE = (
    DATA_DIR /
    "transactions_val.parquet"
)

TRAIN_MODEL_IDS_FILE = (
    DATA_DIR /
    "transactions_train_model_ids.parquet"
)

TRAIN_USER_MAPPING_FILE = (
    DATA_DIR /
    "train_user_mapping.parquet"
)

TRAIN_ITEM_MAPPING_FILE = (
    DATA_DIR /
    "train_item_mapping.parquet"
)

LIGHTGCN_USER_EMBEDDINGS_FILE = (
    OUTPUT_DIR /
    "lightgcn_user_embeddings.npy"
)

LIGHTGCN_ITEM_EMBEDDINGS_FILE = (
    OUTPUT_DIR /
    "lightgcn_item_embeddings.npy"
)

SASREC_MODEL_FILE = (
    OUTPUT_DIR /
    "sasrec_model.pt"
)

HYBRID_MODEL_FILE = (
    OUTPUT_DIR /
    "hybrid_v2_model.pt"
)

RECOMMENDATIONS_FILE = (
    OUTPUT_DIR /
    "hybrid_v2_validation_recommendations.parquet"
)


# ============================================================
# OUTPUT FILES
# ============================================================

OUTPUT_METRICS_FILE = (
    OUTPUT_DIR /
    "hybrid_v2_validation_metrics.json"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

NUM_USERS = 778_421

NUM_ITEMS = 74_541

EMBEDDING_DIM = 64

MAX_SEQUENCE_LENGTH = 50

NUM_HEADS = 2

NUM_LAYERS = 2

DROPOUT = 0.2


# ============================================================
# EVALUATION CONFIGURATION
# ============================================================

K_VALUES = [10, 20]

RECOMMENDATIONS_PER_USER = 20

USER_BATCH_SIZE = 512

DIVERSITY_BATCH_SIZE = 2048

EPSILON = 1e-12


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_header(title):
    """
    Print a consistent section header.
    """

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def require_file(path):
    """
    Verify that a required file exists.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    print(f"[FOUND] {path}")


def assert_finite(array, name):
    """
    Assert that a NumPy array contains only finite values.
    """

    if not np.isfinite(array).all():
        raise AssertionError(
            f"{name} contains non-finite values."
        )


def safe_divide(numerator, denominator):
    """
    Safe scalar division.
    """

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

def check_required_files():

    print_header(
        "HYBRID V2 VALIDATION EVALUATION"
    )

    print("Checking required files...")

    required_files = [
        TRAIN_FILE,
        VALIDATION_FILE,
        TRAIN_MODEL_IDS_FILE,
        TRAIN_USER_MAPPING_FILE,
        TRAIN_ITEM_MAPPING_FILE,
        LIGHTGCN_USER_EMBEDDINGS_FILE,
        LIGHTGCN_ITEM_EMBEDDINGS_FILE,
        SASREC_MODEL_FILE,
        HYBRID_MODEL_FILE,
        RECOMMENDATIONS_FILE,
    ]

    for path in required_files:
        require_file(path)

    print()
    print(
        "[PASS] All required files found"
    )


# ============================================================
# BUILD GLOBAL -> TRAINING MAPPINGS
# ============================================================

def build_training_mappings():

    print_header(
        "BUILDING GLOBAL -> TRAINING MAPPINGS"
    )

    user_mapping = pd.read_parquet(
        TRAIN_USER_MAPPING_FILE
    )

    item_mapping = pd.read_parquet(
        TRAIN_ITEM_MAPPING_FILE
    )

    expected_user_columns = {
        "user_index",
        "train_user_id",
    }

    expected_item_columns = {
        "item_index",
        "train_item_id",
    }

    if not expected_user_columns.issubset(
        user_mapping.columns
    ):
        raise AssertionError(
            "Unexpected train_user_mapping columns."
        )

    if not expected_item_columns.issubset(
        item_mapping.columns
    ):
        raise AssertionError(
            "Unexpected train_item_mapping columns."
        )

    global_to_train_user = dict(
        zip(
            user_mapping["user_index"].astype(
                np.int64
            ),
            user_mapping["train_user_id"].astype(
                np.int64
            ),
        )
    )

    global_to_train_item = dict(
        zip(
            item_mapping["item_index"].astype(
                np.int64
            ),
            item_mapping["train_item_id"].astype(
                np.int64
            ),
        )
    )

    print(
        f"Mapped training users: "
        f"{len(global_to_train_user):,}"
    )

    print(
        f"Mapped training items: "
        f"{len(global_to_train_item):,}"
    )

    if len(global_to_train_user) != NUM_USERS:
        raise AssertionError(
            "Training user mapping count mismatch."
        )

    if len(global_to_train_item) != NUM_ITEMS:
        raise AssertionError(
            "Training item mapping count mismatch."
        )

    print(
        "[PASS] Global -> training mappings"
    )

    return (
        global_to_train_user,
        global_to_train_item,
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print_header(
        "LOADING DATA"
    )

    train = pd.read_parquet(
        TRAIN_FILE
    )

    validation = pd.read_parquet(
        VALIDATION_FILE
    )

    train_model_ids = pd.read_parquet(
        TRAIN_MODEL_IDS_FILE
    )

    print(
        f"Training interactions: "
        f"{len(train):,}"
    )

    print(
        f"Validation interactions: "
        f"{len(validation):,}"
    )

    print(
        f"Training model-ID interactions: "
        f"{len(train_model_ids):,}"
    )

    if len(train) != 19_836_738:
        raise AssertionError(
            "Unexpected training interaction count."
        )

    if len(validation) != 891_329:
        raise AssertionError(
            "Unexpected validation interaction count."
        )

    if len(train_model_ids) != len(train):
        raise AssertionError(
            "Training model-ID row count mismatch."
        )

    print(
        "[PASS] Dataset sizes"
    )

    return (
        train,
        validation,
        train_model_ids,
    )


# ============================================================
# BUILD WARM-START VALIDATION SET
# ============================================================

def build_warm_start_validation(
    validation,
    global_to_train_user,
    global_to_train_item,
):

    print_header(
        "BUILDING WARM-START VALIDATION SET"
    )

    validation_users = set(
        validation["user_index"].unique()
    )

    validation_items = set(
        validation["item_index"].unique()
    )

    training_users = set(
        global_to_train_user.keys()
    )

    training_items = set(
        global_to_train_item.keys()
    )

    known_users = (
        validation_users
        & training_users
    )

    known_items = (
        validation_items
        & training_items
    )

    warm_start_mask = (
        validation["user_index"].isin(
            known_users
        )
        &
        validation["item_index"].isin(
            known_items
        )
    )

    warm_start = (
        validation.loc[
            warm_start_mask
        ]
        .copy()
    )

    warm_start["train_user_id"] = (
        warm_start["user_index"]
        .map(global_to_train_user)
        .astype(np.int64)
    )

    warm_start["train_item_id"] = (
        warm_start["item_index"]
        .map(global_to_train_item)
        .astype(np.int64)
    )

    print(
        f"Validation interactions: "
        f"{len(validation):,}"
    )

    print(
        f"Warm-start validation interactions: "
        f"{len(warm_start):,}"
    )

    print(
        f"Warm-start validation users: "
        f"{warm_start['train_user_id'].nunique():,}"
    )

    if len(warm_start) != 751_993:
        raise AssertionError(
            "Warm-start validation interaction count "
            "does not match expected 751,993."
        )

    if (
        warm_start["train_user_id"].nunique()
        != 173_203
    ):
        raise AssertionError(
            "Warm-start validation user count "
            "does not match expected 173,203."
        )

    print(
        "[PASS] Warm-start evaluation population"
    )

    return warm_start


# ============================================================
# BUILD RELEVANT ITEM SETS
# ============================================================

def build_relevant_items(warm_start):

    print_header(
        "BUILDING RELEVANT VALIDATION ITEMS"
    )

    relevant_items = (
        warm_start
        .groupby("train_user_id")[
            "train_item_id"
        ]
        .apply(
            lambda values:
                set(
                    values.astype(
                        np.int64
                    )
                )
        )
        .to_dict()
    )

    users_with_targets = len(
        relevant_items
    )

    print(
        f"Users with validation targets: "
        f"{users_with_targets:,}"
    )

    if users_with_targets != 173_203:
        raise AssertionError(
            "Unexpected number of evaluation users."
        )

    total_targets = sum(
        len(items)
        for items in relevant_items.values()
    )

    print(
        f"Unique user-item validation targets: "
        f"{total_targets:,}"
    )

    if total_targets == 0:
        raise AssertionError(
            "No validation targets found."
        )

    print(
        "[PASS] Validation target sets"
    )

    return relevant_items


# ============================================================
# LOAD HYBRID MODEL
# ============================================================

def load_hybrid_model():

    print_header(
        "LOADING HYBRID V2 MODEL"
    )

    model = HybridLightGCNSASRec(
        num_users=NUM_USERS,
        num_items=NUM_ITEMS,
        embedding_dim=EMBEDDING_DIM,
        max_seq_len=MAX_SEQUENCE_LENGTH,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )

    checkpoint = torch.load(
        HYBRID_MODEL_FILE,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Hybrid V2 checkpoint does not contain "
            "'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    model.lightgcn.eval()

    model.sasrec.eval()

    # Freeze pretrained components.
    for parameter in model.lightgcn.parameters():
        parameter.requires_grad = False

    for parameter in model.sasrec.parameters():
        parameter.requires_grad = False

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    trainable_count = sum(
        parameter.numel()
        for parameter in trainable_parameters
    )

    print(
        "[PASS] Hybrid V2 checkpoint loaded"
    )

    print(
        "[PASS] LightGCN frozen"
    )

    print(
        "[PASS] SASRec frozen"
    )

    print(
        "[PASS] Encoders in evaluation mode"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_count:,}"
    )

    if trainable_count != 33_024:
        raise AssertionError(
            "Unexpected number of trainable parameters."
        )

    print(
        "[PASS] Exactly 33,024 fusion parameters"
    )

    return model


# ============================================================
# LOAD LIGHTGCN EMBEDDINGS
# ============================================================

def load_lightgcn_embeddings():

    print_header(
        "LOADING LIGHTGCN EMBEDDINGS"
    )

    user_embeddings = np.load(
        LIGHTGCN_USER_EMBEDDINGS_FILE,
        mmap_mode="r",
    )

    item_embeddings = np.load(
        LIGHTGCN_ITEM_EMBEDDINGS_FILE,
        mmap_mode="r",
    )

    print(
        f"LightGCN user embeddings: "
        f"{user_embeddings.shape}"
    )

    print(
        f"LightGCN item embeddings: "
        f"{item_embeddings.shape}"
    )

    if user_embeddings.shape != (
        NUM_USERS,
        EMBEDDING_DIM,
    ):
        raise AssertionError(
            "Unexpected LightGCN user embedding shape."
        )

    if item_embeddings.shape != (
        NUM_ITEMS,
        EMBEDDING_DIM,
    ):
        raise AssertionError(
            "Unexpected LightGCN item embedding shape."
        )

    # Copy mmap arrays so torch receives writable memory.
    user_embeddings = np.array(
        user_embeddings,
        dtype=np.float32,
        copy=True,
    )

    item_embeddings = np.array(
        item_embeddings,
        dtype=np.float32,
        copy=True,
    )

    assert_finite(
        user_embeddings,
        "LightGCN user embeddings",
    )

    assert_finite(
        item_embeddings,
        "LightGCN item embeddings",
    )

    print(
        "[PASS] LightGCN embedding shapes"
    )

    print(
        "[PASS] LightGCN embeddings finite"
    )

    print(
        "[PASS] Memory-mapped arrays copied safely"
    )

    return (
        torch.from_numpy(
            user_embeddings
        ),
        torch.from_numpy(
            item_embeddings
        ),
    )


# ============================================================
# LOAD RECOMMENDATIONS
# ============================================================

def load_recommendations():

    print_header(
        "LOADING HYBRID V2 RECOMMENDATIONS"
    )

    recommendations = pd.read_parquet(
        RECOMMENDATIONS_FILE
    )

    print(
        f"Recommendation rows: "
        f"{len(recommendations):,}"
    )

    expected_rows = (
        173_203
        *
        RECOMMENDATIONS_PER_USER
    )

    if len(recommendations) != expected_rows:
        raise AssertionError(
            "Recommendation row count mismatch."
        )

    required_columns = {
        "train_user_id",
        "train_item_id",
        "rank",
        "score",
    }

    if not required_columns.issubset(
        recommendations.columns
    ):
        raise AssertionError(
            "Recommendation file is missing "
            "required columns."
        )

    if recommendations[
        "train_user_id"
    ].nunique() != 173_203:
        raise AssertionError(
            "Recommendation user count mismatch."
        )

    counts = (
        recommendations
        .groupby("train_user_id")
        .size()
    )

    if not (
        counts == RECOMMENDATIONS_PER_USER
    ).all():
        raise AssertionError(
            "Not every user has exactly 20 recommendations."
        )

    if recommendations[
        "train_item_id"
    ].min() < 0:
        raise AssertionError(
            "Negative training item ID found."
        )

    if recommendations[
        "train_item_id"
    ].max() >= NUM_ITEMS:
        raise AssertionError(
            "Recommendation item ID exceeds "
            "training vocabulary."
        )

    if not np.isfinite(
        recommendations["score"]
        .to_numpy()
    ).all():
        raise AssertionError(
            "Recommendation scores contain "
            "non-finite values."
        )

    print(
        "[PASS] Recommendation file"
    )

    print(
        "[PASS] Recommendation row count"
    )

    print(
        "[PASS] Exactly 20 recommendations/user"
    )

    return recommendations


# ============================================================
# ALIGN RECOMMENDATIONS AND TARGETS
# ============================================================

def build_recommendation_lists(
    recommendations,
):

    recommendation_lists = (
        recommendations
        .sort_values(
            [
                "train_user_id",
                "rank",
            ]
        )
        .groupby(
            "train_user_id"
        )[
            "train_item_id"
        ]
        .apply(
            lambda values:
                values.to_numpy(
                    dtype=np.int64
                )
        )
        .to_dict()
    )

    if len(
        recommendation_lists
    ) != 173_203:
        raise AssertionError(
            "Recommendation list user count mismatch."
        )

    for user_id, items in (
        recommendation_lists.items()
    ):

        if len(items) != 20:
            raise AssertionError(
                f"User {user_id} does not have "
                f"exactly 20 recommendations."
            )

        if len(
            np.unique(items)
        ) != len(items):
            raise AssertionError(
                f"Duplicate recommendation found "
                f"for user {user_id}."
            )

    print(
        "[PASS] Recommendation lists constructed"
    )

    return recommendation_lists


# ============================================================
# RANKING METRICS
# ============================================================

def precision_at_k(
    recommended,
    relevant,
    k,
):

    top_k = recommended[:k]

    hits = sum(
        1
        for item in top_k
        if item in relevant
    )

    return hits / k


def recall_at_k(
    recommended,
    relevant,
    k,
):

    if not relevant:
        return 0.0

    top_k = recommended[:k]

    hits = sum(
        1
        for item in top_k
        if item in relevant
    )

    return hits / len(relevant)


def hit_rate_at_k(
    recommended,
    relevant,
    k,
):

    top_k = recommended[:k]

    for item in top_k:

        if item in relevant:
            return 1.0

    return 0.0


def ndcg_at_k(
    recommended,
    relevant,
    k,
):

    if not relevant:
        return 0.0

    top_k = recommended[:k]

    dcg = 0.0

    for rank, item in enumerate(
        top_k,
        start=1,
    ):

        if item in relevant:

            dcg += (
                1.0
                /
                math.log2(
                    rank + 1
                )
            )

    ideal_hits = min(
        len(relevant),
        k,
    )

    if ideal_hits == 0:
        return 0.0

    idcg = sum(
        1.0
        /
        math.log2(
            rank + 1
        )
        for rank in range(
            1,
            ideal_hits + 1,
        )
    )

    return dcg / idcg


def average_precision_at_k(
    recommended,
    relevant,
    k,
):

    if not relevant:
        return 0.0

    top_k = recommended[:k]

    hits = 0

    precision_sum = 0.0

    for rank, item in enumerate(
        top_k,
        start=1,
    ):

        if item in relevant:

            hits += 1

            precision_sum += (
                hits / rank
            )

    denominator = min(
        len(relevant),
        k,
    )

    if denominator == 0:
        return 0.0

    return (
        precision_sum
        /
        denominator
    )


# ============================================================
# RANKING EVALUATION
# ============================================================

def evaluate_ranking_metrics(
    recommendation_lists,
    relevant_items,
):

    print_header(
        "CALCULATING RANKING METRICS"
    )

    metric_values = {
        "P@10": [],
        "P@20": [],
        "R@10": [],
        "R@20": [],
        "HR@10": [],
        "HR@20": [],
        "NDCG@10": [],
        "NDCG@20": [],
        "MAP@10": [],
        "MAP@20": [],
    }

    evaluated_users = 0

    for user_id, recommended in (
        recommendation_lists.items()
    ):

        if user_id not in relevant_items:
            continue

        relevant = relevant_items[
            user_id
        ]

        evaluated_users += 1

        metric_values["P@10"].append(
            precision_at_k(
                recommended,
                relevant,
                10,
            )
        )

        metric_values["P@20"].append(
            precision_at_k(
                recommended,
                relevant,
                20,
            )
        )

        metric_values["R@10"].append(
            recall_at_k(
                recommended,
                relevant,
                10,
            )
        )

        metric_values["R@20"].append(
            recall_at_k(
                recommended,
                relevant,
                20,
            )
        )

        metric_values["HR@10"].append(
            hit_rate_at_k(
                recommended,
                relevant,
                10,
            )
        )

        metric_values["HR@20"].append(
            hit_rate_at_k(
                recommended,
                relevant,
                20,
            )
        )

        metric_values["NDCG@10"].append(
            ndcg_at_k(
                recommended,
                relevant,
                10,
            )
        )

        metric_values["NDCG@20"].append(
            ndcg_at_k(
                recommended,
                relevant,
                20,
            )
        )

        metric_values["MAP@10"].append(
            average_precision_at_k(
                recommended,
                relevant,
                10,
            )
        )

        metric_values["MAP@20"].append(
            average_precision_at_k(
                recommended,
                relevant,
                20,
            )
        )

    if evaluated_users != 173_203:
        raise AssertionError(
            "Ranking evaluation user count mismatch."
        )

    metrics = {}

    for metric_name, values in (
        metric_values.items()
    ):

        metrics[metric_name] = float(
            np.mean(
                np.asarray(
                    values,
                    dtype=np.float64,
                )
            )
        )

    print(
        f"Evaluated users: "
        f"{evaluated_users:,}"
    )

    print()

    for metric_name in [
        "P@10",
        "P@20",
        "R@10",
        "R@20",
        "HR@10",
        "HR@20",
        "NDCG@10",
        "NDCG@20",
        "MAP@10",
        "MAP@20",
    ]:

        print(
            f"{metric_name:<8}: "
            f"{metrics[metric_name]:.6f}"
        )

    print(
        "[PASS] Ranking metrics"
    )

    return metrics


# ============================================================
# COVERAGE
# ============================================================

def calculate_coverage(
    recommendation_lists,
    k,
):

    recommended_items = set()

    for items in (
        recommendation_lists.values()
    ):

        recommended_items.update(
            items[:k]
        )

    coverage = (
        len(recommended_items)
        /
        NUM_ITEMS
    )

    return float(
        coverage
    )


# ============================================================
# LOAD HYBRID ITEM REPRESENTATIONS
# ============================================================

def generate_hybrid_item_representations(
    model,
    lightgcn_item_embeddings,
):

    print_header(
        "GENERATING HYBRID ITEM REPRESENTATIONS"
    )

    model.eval()

    with torch.no_grad():

        item_ids = torch.arange(
            NUM_ITEMS,
            dtype=torch.long,
        )

        hybrid_items = (
            model.get_item_representation(
                item_ids=item_ids,
                lightgcn_item_embeddings=(
                    lightgcn_item_embeddings
                ),
            )
        )

    if hybrid_items.shape != (
        NUM_ITEMS,
        128,
    ):
        raise AssertionError(
            "Unexpected Hybrid V2 item "
            "representation shape."
        )

    if not torch.isfinite(
        hybrid_items
    ).all():
        raise AssertionError(
            "Hybrid V2 item representations "
            "contain non-finite values."
        )

    print(
        f"Hybrid item representation shape: "
        f"{tuple(hybrid_items.shape)}"
    )

    print(
        "[PASS] Hybrid item representations"
    )

    return hybrid_items


# ============================================================
# DIVERSITY
# ============================================================

def calculate_list_diversity(
    item_ids,
    item_embeddings,
):

    """
    Intra-list diversity.

    Diversity is defined as the average pairwise
    cosine distance between items in a recommendation list.

    cosine distance:
        1 - cosine_similarity

    A higher value means recommended items are
    less similar to each other in the model's
    representation space.
    """

    if len(item_ids) < 2:
        return 0.0

    embeddings = (
        item_embeddings[
            torch.as_tensor(
                item_ids,
                dtype=torch.long,
            )
        ]
    )

    embeddings = torch.nn.functional.normalize(
        embeddings,
        p=2,
        dim=1,
        eps=EPSILON,
    )

    similarity = (
        embeddings
        @
        embeddings.T
    )

    n = similarity.shape[0]

    upper_triangle = torch.triu(
        similarity,
        diagonal=1,
    )

    pair_count = (
        n * (n - 1)
    ) // 2

    similarity_sum = (
        upper_triangle.sum()
    )

    mean_similarity = (
        similarity_sum
        /
        pair_count
    )

    diversity = (
        1.0
        -
        mean_similarity.item()
    )

    return float(
        diversity
    )


def calculate_diversity(
    recommendation_lists,
    item_embeddings,
    k,
):

    values = []

    for items in (
        recommendation_lists.values()
    ):

        values.append(
            calculate_list_diversity(
                items[:k],
                item_embeddings,
            )
        )

    if not values:
        return 0.0

    return float(
        np.mean(
            np.asarray(
                values,
                dtype=np.float64,
            )
        )
    )


# ============================================================
# TRAINING ITEM POPULARITY
# ============================================================

def calculate_training_item_popularity(
    train_model_ids,
):

    print_header(
        "CALCULATING TRAINING ITEM POPULARITY"
    )

    counts = (
        train_model_ids[
            "train_item_id"
        ]
        .value_counts()
        .sort_index()
    )

    popularity = np.zeros(
        NUM_ITEMS,
        dtype=np.float64,
    )

    item_ids = (
        counts.index
        .to_numpy(
            dtype=np.int64
        )
    )

    frequencies = (
        counts.values
        .astype(np.float64)
    )

    popularity[
        item_ids
    ] = frequencies

    total_interactions = (
        len(train_model_ids)
    )

    if total_interactions <= 0:
        raise AssertionError(
            "Training interaction count is zero."
        )

    popularity /= (
        total_interactions
    )

    if not np.isfinite(
        popularity
    ).all():
        raise AssertionError(
            "Popularity contains non-finite values."
        )

    if (
        popularity.sum()
        <= 0
    ):
        raise AssertionError(
            "Popularity distribution is empty."
        )

    print(
        f"Training interactions: "
        f"{total_interactions:,}"
    )

    print(
        f"Items with non-zero popularity: "
        f"{np.count_nonzero(popularity):,}"
    )

    print(
        "[PASS] Training popularity distribution"
    )

    return popularity


# ============================================================
# NOVELTY
# ============================================================

def calculate_novelty(
    recommendation_lists,
    popularity,
    k,
):

    """
    Novelty definition:

        novelty(i) = -log2(popularity(i))

    The user's recommendation novelty is the
    mean item novelty over the top-k list.

    Higher values correspond to less popular
    items.
    """

    values = []

    for items in (
        recommendation_lists.values()
    ):

        top_k = items[:k]

        item_popularity = (
            popularity[
                top_k
            ]
        )

        # Every recommended item should be
        # present in the training vocabulary.
        if np.any(
            item_popularity <= 0
        ):
            raise AssertionError(
                "Recommended item has zero "
                "training popularity."
            )

        item_novelty = (
            -np.log2(
                item_popularity
            )
        )

        values.append(
            float(
                np.mean(
                    item_novelty
                )
            )
        )

    if not values:
        return 0.0

    return float(
        np.mean(
            np.asarray(
                values,
                dtype=np.float64,
            )
        )
    )


# ============================================================
# QUALITY METRICS
# ============================================================

def evaluate_quality_metrics(
    recommendation_lists,
    hybrid_item_embeddings,
    popularity,
):

    print_header(
        "CALCULATING RECOMMENDATION QUALITY METRICS"
    )

    metrics = {}

    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    metrics["Coverage@10"] = (
        calculate_coverage(
            recommendation_lists,
            10,
        )
    )

    metrics["Coverage@20"] = (
        calculate_coverage(
            recommendation_lists,
            20,
        )
    )

    # --------------------------------------------------------
    # Diversity
    # --------------------------------------------------------

    print(
        "Calculating Diversity@10..."
    )

    metrics["Diversity@10"] = (
        calculate_diversity(
            recommendation_lists,
            hybrid_item_embeddings,
            10,
        )
    )

    print(
        "Calculating Diversity@20..."
    )

    metrics["Diversity@20"] = (
        calculate_diversity(
            recommendation_lists,
            hybrid_item_embeddings,
            20,
        )
    )

    # --------------------------------------------------------
    # Novelty
    # --------------------------------------------------------

    metrics["Novelty@10"] = (
        calculate_novelty(
            recommendation_lists,
            popularity,
            10,
        )
    )

    metrics["Novelty@20"] = (
        calculate_novelty(
            recommendation_lists,
            popularity,
            20,
        )
    )

    print()

    print(
        f"Coverage@10 : "
        f"{metrics['Coverage@10']:.6f}"
    )

    print(
        f"Coverage@20 : "
        f"{metrics['Coverage@20']:.6f}"
    )

    print(
        f"Diversity@10: "
        f"{metrics['Diversity@10']:.6f}"
    )

    print(
        f"Diversity@20: "
        f"{metrics['Diversity@20']:.6f}"
    )

    print(
        f"Novelty@10  : "
        f"{metrics['Novelty@10']:.6f}"
    )

    print(
        f"Novelty@20  : "
        f"{metrics['Novelty@20']:.6f}"
    )

    print(
        "[PASS] Recommendation quality metrics"
    )

    return metrics


# ============================================================
# RECOMMENDATION INTEGRITY CHECKS
# ============================================================

def validate_recommendation_integrity(
    recommendations,
    train_model_ids,
):

    print_header(
        "VALIDATING RECOMMENDATION INTEGRITY"
    )

    # --------------------------------------------------------
    # User count
    # --------------------------------------------------------

    users = recommendations[
        "train_user_id"
    ].unique()

    if len(users) != 173_203:
        raise AssertionError(
            "Unexpected recommendation user count."
        )

    print(
        "[PASS] Recommendation user count"
    )

    # --------------------------------------------------------
    # Exactly 20 per user
    # --------------------------------------------------------

    counts = (
        recommendations
        .groupby(
            "train_user_id"
        )
        .size()
    )

    if not (
        counts == 20
    ).all():
        raise AssertionError(
            "Not all users have exactly 20 recommendations."
        )

    print(
        "[PASS] Exactly 20 recommendations/user"
    )

    # --------------------------------------------------------
    # Rank correctness
    # --------------------------------------------------------

    for _, group in (
        recommendations
        .groupby("train_user_id")
    ):

        ranks = (
            group["rank"]
            .to_numpy()
        )

        expected = np.arange(
            1,
            21,
            dtype=np.int64,
        )

        if not np.array_equal(
            ranks,
            expected,
        ):
            raise AssertionError(
                "Invalid rank sequence."
            )

    print(
        "[PASS] Rank sequences"
    )

    # --------------------------------------------------------
    # Duplicate user-item pairs
    # --------------------------------------------------------

    duplicate_count = (
        recommendations
        .duplicated(
            subset=[
                "train_user_id",
                "train_item_id",
            ]
        )
        .sum()
    )

    if duplicate_count != 0:
        raise AssertionError(
            "Duplicate user-item recommendation found."
        )

    print(
        "[PASS] No duplicate user-item pairs"
    )

    # --------------------------------------------------------
    # Item range
    # --------------------------------------------------------

    if (
        recommendations[
            "train_item_id"
        ].min()
        < 0
    ):

        raise AssertionError(
            "Negative item ID found."
        )

    if (
        recommendations[
            "train_item_id"
        ].max()
        >= NUM_ITEMS
    ):

        raise AssertionError(
            "Item ID outside training vocabulary."
        )

    print(
        "[PASS] Item ID range"
    )

    # --------------------------------------------------------
    # Scores
    # --------------------------------------------------------

    if not np.isfinite(
        recommendations[
            "score"
        ].to_numpy()
    ).all():

        raise AssertionError(
            "Recommendation scores contain "
            "non-finite values."
        )

    print(
        "[PASS] Scores finite"
    )

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    user_seen_items = (
        train_model_ids
        .groupby(
            "train_user_id"
        )[
            "train_item_id"
        ]
        .apply(
            lambda values:
                set(
                    values.astype(
                        np.int64
                    )
                )
        )
        .to_dict()
    )

    for user_id, group in (
        recommendations
        .groupby(
            "train_user_id"
        )
    ):

        seen = user_seen_items[
            int(user_id)
        ]

        recommended = set(
            group[
                "train_item_id"
            ]
            .astype(np.int64)
        )

        overlap = (
            recommended
            &
            seen
        )

        if overlap:
            raise AssertionError(
                f"Training-seen item recommended "
                f"for user {user_id}: "
                f"{list(overlap)[:5]}"
            )

    print(
        "[PASS] Training-seen items excluded"
    )


# ============================================================
# SAVE METRICS
# ============================================================

def save_metrics(
    metrics,
    evaluation_time,
    recommendation_count,
    evaluation_users,
):

    print_header(
        "SAVING HYBRID V2 METRICS"
    )

    output = {
        "model": (
            "Hybrid LightGCN + SASRec V2"
        ),

        "evaluation_protocol": {
            "population": (
                "warm-start validation interactions"
            ),
            "validation_interactions": 891_329,
            "warm_start_interactions": 751_993,
            "warm_start_users": 173_203,
            "recommendations_per_user": 20,
            "training_users": NUM_USERS,
            "training_items": NUM_ITEMS,
        },

        "architecture": {
            "lightgcn_embedding_dimension": 64,
            "lightgcn_layers": 3,
            "sasrec_embedding_dimension": 64,
            "sasrec_heads": 2,
            "sasrec_layers": 2,
            "max_sequence_length": 50,
            "component_normalization": "L2",
            "user_projection": (
                "Linear(128,128) + ReLU + Dropout(0.2)"
            ),
            "item_projection": (
                "Linear(128,128) + ReLU"
            ),
            "pretrained_encoders_frozen": True,
            "fusion_trainable_parameters": 33_024,
            "loss": "BPR",
        },

        "metrics": metrics,

        "efficiency": {
            "evaluation_time_seconds": (
                float(evaluation_time)
            ),
            "recommendation_rows_evaluated": (
                int(recommendation_count)
            ),
            "evaluation_users": (
                int(evaluation_users)
            ),
        },

        "metric_definitions": {
            "precision": (
                "hits_at_k / k"
            ),
            "recall": (
                "hits_at_k / number_of_validation_targets"
            ),
            "hit_rate": (
                "1 if at least one validation target "
                "appears in top-k, else 0"
            ),
            "ndcg": (
                "binary relevance with logarithmic "
                "rank discount"
            ),
            "map": (
                "mean average precision using "
                "binary relevance"
            ),
            "coverage": (
                "unique recommended items / "
                "training item catalogue size"
            ),
            "diversity": (
                "mean pairwise cosine distance between "
                "recommended item representations"
            ),
            "novelty": (
                "-log2(training interaction popularity)"
            ),
        },

        "recommendation_file": str(
            RECOMMENDATIONS_FILE
        ),
    }

    with open(
        OUTPUT_METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
        )

    print(
        "Saved metrics:"
    )

    print(
        OUTPUT_METRICS_FILE
    )

    return output


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

def print_final_summary(
    metrics,
    evaluation_time,
):

    print_header(
        "HYBRID V2 FINAL VALIDATION METRICS"
    )

    print(
        "Ranking"
    )

    print(
        f"P@10    : {metrics['P@10']:.6f}"
    )

    print(
        f"P@20    : {metrics['P@20']:.6f}"
    )

    print(
        f"R@10    : {metrics['R@10']:.6f}"
    )

    print(
        f"R@20    : {metrics['R@20']:.6f}"
    )

    print(
        f"HR@10   : {metrics['HR@10']:.6f}"
    )

    print(
        f"HR@20   : {metrics['HR@20']:.6f}"
    )

    print(
        f"NDCG@10 : {metrics['NDCG@10']:.6f}"
    )

    print(
        f"NDCG@20 : {metrics['NDCG@20']:.6f}"
    )

    print(
        f"MAP@10  : {metrics['MAP@10']:.6f}"
    )

    print(
        f"MAP@20  : {metrics['MAP@20']:.6f}"
    )

    print()

    print(
        "Recommendation Quality"
    )

    print(
        f"Coverage@10  : "
        f"{metrics['Coverage@10']:.6f}"
    )

    print(
        f"Coverage@20  : "
        f"{metrics['Coverage@20']:.6f}"
    )

    print(
        f"Diversity@10 : "
        f"{metrics['Diversity@10']:.6f}"
    )

    print(
        f"Diversity@20 : "
        f"{metrics['Diversity@20']:.6f}"
    )

    print(
        f"Novelty@10   : "
        f"{metrics['Novelty@10']:.6f}"
    )

    print(
        f"Novelty@20   : "
        f"{metrics['Novelty@20']:.6f}"
    )

    print()

    print(
        f"Evaluation time: "
        f"{evaluation_time:.2f}s"
    )

    print(
        f"Evaluation time: "
        f"{evaluation_time / 60:.2f} min"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    evaluation_start = (
        time.perf_counter()
    )

    # --------------------------------------------------------
    # Files
    # --------------------------------------------------------

    check_required_files()

    # --------------------------------------------------------
    # Mappings
    # --------------------------------------------------------

    (
        global_to_train_user,
        global_to_train_item,
    ) = build_training_mappings()

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    (
        train,
        validation,
        train_model_ids,
    ) = load_data()

    # --------------------------------------------------------
    # Warm-start validation
    # --------------------------------------------------------

    warm_start = (
        build_warm_start_validation(
            validation,
            global_to_train_user,
            global_to_train_item,
        )
    )

    # --------------------------------------------------------
    # Relevant validation targets
    # --------------------------------------------------------

    relevant_items = (
        build_relevant_items(
            warm_start
        )
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = load_hybrid_model()

    # --------------------------------------------------------
    # LightGCN embeddings
    # --------------------------------------------------------

    (
        lightgcn_user_embeddings,
        lightgcn_item_embeddings,
    ) = load_lightgcn_embeddings()

    # --------------------------------------------------------
    # Recommendations
    # --------------------------------------------------------

    recommendations = (
        load_recommendations()
    )

    # --------------------------------------------------------
    # Integrity
    # --------------------------------------------------------

    validate_recommendation_integrity(
        recommendations,
        train_model_ids,
    )

    # --------------------------------------------------------
    # Recommendation lists
    # --------------------------------------------------------

    recommendation_lists = (
        build_recommendation_lists(
            recommendations
        )
    )

    # --------------------------------------------------------
    # Ranking metrics
    # --------------------------------------------------------

    ranking_metrics = (
        evaluate_ranking_metrics(
            recommendation_lists,
            relevant_items,
        )
    )

    # --------------------------------------------------------
    # Hybrid item representations
    # --------------------------------------------------------

    hybrid_item_embeddings = (
        generate_hybrid_item_representations(
            model,
            lightgcn_item_embeddings,
        )
    )

    # --------------------------------------------------------
    # Training popularity
    # --------------------------------------------------------

    popularity = (
        calculate_training_item_popularity(
            train_model_ids
        )
    )

    # --------------------------------------------------------
    # Quality metrics
    # --------------------------------------------------------

    quality_metrics = (
        evaluate_quality_metrics(
            recommendation_lists,
            hybrid_item_embeddings,
            popularity,
        )
    )

    # --------------------------------------------------------
    # Combine metrics
    # --------------------------------------------------------

    metrics = {}

    metrics.update(
        ranking_metrics
    )

    metrics.update(
        quality_metrics
    )

    # --------------------------------------------------------
    # Evaluation time
    # --------------------------------------------------------

    evaluation_time = (
        time.perf_counter()
        -
        evaluation_start
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_metrics(
        metrics=metrics,
        evaluation_time=evaluation_time,
        recommendation_count=len(
            recommendations
        ),
        evaluation_users=len(
            recommendation_lists
        ),
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print_final_summary(
        metrics,
        evaluation_time,
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print(
        "HYBRID V2 EVALUATION PASSED"
    )
    print("=" * 78)

    print()
    print(
        f"Warm-start users: "
        f"{len(recommendation_lists):,}"
    )

    print(
        f"Recommendation rows: "
        f"{len(recommendations):,}"
    )

    print(
        f"Metrics saved to:"
    )

    print(
        OUTPUT_METRICS_FILE
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()