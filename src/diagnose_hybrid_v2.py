from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

TRAIN_FILE = DATA_DIR / "transactions_train.parquet"
VAL_FILE = DATA_DIR / "transactions_val.parquet"

TRAIN_MODEL_IDS_FILE = (
    DATA_DIR / "transactions_train_model_ids.parquet"
)

TRAIN_USER_MAPPING_FILE = (
    DATA_DIR / "train_user_mapping.parquet"
)

TRAIN_ITEM_MAPPING_FILE = (
    DATA_DIR / "train_item_mapping.parquet"
)

LIGHTGCN_USER_EMBEDDINGS_FILE = (
    OUTPUT_DIR / "lightgcn_user_embeddings.npy"
)

LIGHTGCN_ITEM_EMBEDDINGS_FILE = (
    OUTPUT_DIR / "lightgcn_item_embeddings.npy"
)

SASREC_MODEL_FILE = (
    OUTPUT_DIR / "sasrec_model.pt"
)

HYBRID_MODEL_FILE = (
    OUTPUT_DIR / "hybrid_v2_model.pt"
)

HYBRID_V2_RECOMMENDATIONS_FILE = (
    OUTPUT_DIR / "hybrid_v2_validation_recommendations.parquet"
)

OUTPUT_FILE = (
    OUTPUT_DIR / "hybrid_v2_component_diagnostic.json"
)


# ------------------------------------------------------------
# Dataset / model dimensions
# ------------------------------------------------------------

GLOBAL_NUM_USERS = 797_782
GLOBAL_NUM_ITEMS = 79_026

NUM_USERS = 778_421
NUM_ITEMS = 74_541

EMBEDDING_DIM = 64
HYBRID_DIM = 128

MAX_SEQUENCE_LENGTH = 50

TOP_K = 20

# Number of warm-start users to inspect.
#
# The diagnostic is deliberately sampled because calculating
# full score matrices for all 173k validation users is expensive.
#
# 1,000 gives a stable diagnostic while keeping runtime reasonable.
DIAGNOSTIC_USERS = 1_000

SEED = 42

BATCH_SIZE = 256


# ============================================================
# V1 REFERENCE VALUES
# ============================================================
#
# These values come from the previously completed Hybrid V1
# diagnostic. They are used ONLY for comparison.
#
# They are not recomputed here.
# ============================================================

V1_REFERENCE = {
    "unique_items": 149,
    "coverage_percent": 0.1999,

    "top10_concentration_percent": 43.10,
    "top20_concentration_percent": 69.13,
    "top50_concentration_percent": 95.17,
    "top100_concentration_percent": 99.69,

    "avg_top20_jaccard": 0.45479170,

    "lightgcn_score_std": 0.00603444,
    "sasrec_score_std": 1.45463276,
    "hybrid_score_std": 1.75188911,

    "lightgcn_hybrid_score_correlation": 0.10712785,
    "sasrec_hybrid_score_correlation": 0.45486573,

    "lightgcn_component_score_std": 0.09114906,
    "sasrec_component_score_std": 1.70762408,

    "lightgcn_component_hybrid_correlation": 0.50551278,
    "sasrec_component_hybrid_correlation": 0.99893889,

    "lightgcn_user_variance": 0.0006483811,
    "sasrec_user_variance": 0.0535164252,
    "hybrid_user_variance": 0.0518055521,

    "lightgcn_item_variance": 0.0008285219,
    "sasrec_item_variance": 0.0529511906,
    "hybrid_item_variance": 0.0268898569,
}


# ============================================================
# IMPORT MODEL
# ============================================================

from hybrid_model import HybridLightGCNSASRec


# ============================================================
# GENERAL UTILITIES
# ============================================================

def set_seed(seed=42):

    np.random.seed(seed)
    torch.manual_seed(seed)


def check_file(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    print(f"[FOUND] {path}")


def summarize_array(name, array):

    array = np.asarray(array)

    print(
        f"{name}: "
        f"shape={array.shape}, "
        f"dtype={array.dtype}, "
        f"mean={array.mean():.8f}, "
        f"std={array.std():.8f}, "
        f"min={array.min():.8f}, "
        f"max={array.max():.8f}"
    )


# ============================================================
# SASREC CHECKPOINT KEY CONVERSION
# ============================================================

def remap_sasrec_checkpoint_keys(state_dict):

    converted = {}

    for key, value in state_dict.items():

        new_key = key

        if key.startswith(
            "transformer_encoder.layers."
        ):
            new_key = key.replace(
                "transformer_encoder.layers.",
                "transformer_layers.",
                1,
            )

        elif key.startswith("layer_norm."):

            new_key = key.replace(
                "layer_norm.",
                "final_layer_norm.",
                1,
            )

        converted[new_key] = value

    return converted


# ============================================================
# CREATE MODEL
# ============================================================

def create_model():

    print("\n" + "=" * 70)
    print("LOADING HYBRID V2 MODEL")
    print("=" * 70)

    model = HybridLightGCNSASRec(
        num_users=NUM_USERS,
        num_items=NUM_ITEMS,
        embedding_dim=EMBEDDING_DIM,
        max_seq_len=MAX_SEQUENCE_LENGTH,
        num_heads=2,
        num_layers=2,
        dropout=0.2,
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
        checkpoint["model_state_dict"],
        strict=True,
    )

    print(
        "[PASS] Hybrid V2 checkpoint loaded"
    )

    model.freeze_pretrained_encoders()

    model.eval()
    model.lightgcn.eval()
    model.sasrec.eval()

    print(
        "[PASS] LightGCN frozen"
    )

    print(
        "[PASS] SASRec frozen"
    )

    print(
        "[PASS] Frozen encoders in eval mode"
    )

    trainable_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    trainable_count = sum(
        parameter.numel()
        for _, parameter in trainable_parameters
    )

    print(
        f"Trainable parameters: "
        f"{trainable_count}"
    )

    if trainable_count != 33_024:

        raise AssertionError(
            "Unexpected number of trainable "
            f"parameters: {trainable_count}"
        )

    print(
        "[PASS] Trainable parameter count = 33,024"
    )

    return model


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

def load_embeddings():

    print("\n" + "=" * 70)
    print("LOADING PRECOMPUTED EMBEDDINGS")
    print("=" * 70)

    lightgcn_user = np.load(
        LIGHTGCN_USER_EMBEDDINGS_FILE,
        mmap_mode="r",
    )

    lightgcn_item = np.load(
        LIGHTGCN_ITEM_EMBEDDINGS_FILE,
        mmap_mode="r",
    )

    print(
        f"LightGCN user embeddings: "
        f"{lightgcn_user.shape}"
    )

    print(
        f"LightGCN item embeddings: "
        f"{lightgcn_item.shape}"
    )

    if lightgcn_user.shape != (
        NUM_USERS,
        EMBEDDING_DIM,
    ):

        raise AssertionError(
            "Unexpected LightGCN user embedding shape."
        )

    if lightgcn_item.shape != (
        NUM_ITEMS,
        EMBEDDING_DIM,
    ):

        raise AssertionError(
            "Unexpected LightGCN item embedding shape."
        )

    # Copy mmap arrays so they are ordinary writable arrays.
    lightgcn_user = np.array(
        lightgcn_user,
        dtype=np.float32,
        copy=True,
    )

    lightgcn_item = np.array(
        lightgcn_item,
        dtype=np.float32,
        copy=True,
    )

    if not np.isfinite(lightgcn_user).all():

        raise AssertionError(
            "Non-finite LightGCN user embeddings."
        )

    if not np.isfinite(lightgcn_item).all():

        raise AssertionError(
            "Non-finite LightGCN item embeddings."
        )

    print(
        "[PASS] LightGCN embeddings loaded"
    )

    return (
        torch.from_numpy(lightgcn_user),
        torch.from_numpy(lightgcn_item),
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    train = pd.read_parquet(
        TRAIN_FILE
    )

    val = pd.read_parquet(
        VAL_FILE
    )

    train_model_ids = pd.read_parquet(
        TRAIN_MODEL_IDS_FILE
    )

    train_user_mapping = pd.read_parquet(
        TRAIN_USER_MAPPING_FILE
    )

    train_item_mapping = pd.read_parquet(
        TRAIN_ITEM_MAPPING_FILE
    )

    print(
        f"Training interactions: "
        f"{len(train):,}"
    )

    print(
        f"Validation interactions: "
        f"{len(val):,}"
    )

    print(
        f"Training model-ID interactions: "
        f"{len(train_model_ids):,}"
    )

    return (
        train,
        val,
        train_model_ids,
        train_user_mapping,
        train_item_mapping,
    )


# ============================================================
# GLOBAL → TRAINING MAPPINGS
# ============================================================

def build_global_to_training_maps(
    train_user_mapping,
    train_item_mapping,
):

    print("\n" + "=" * 70)
    print("BUILDING GLOBAL → TRAINING MAPPINGS")
    print("=" * 70)

    global_user_to_train = np.full(
        GLOBAL_NUM_USERS,
        -1,
        dtype=np.int32,
    )

    user_global = (
        train_user_mapping["user_index"]
        .to_numpy(dtype=np.int64)
    )

    user_train = (
        train_user_mapping["train_user_id"]
        .to_numpy(dtype=np.int32)
    )

    global_user_to_train[
        user_global
    ] = user_train

    global_item_to_train = np.full(
        GLOBAL_NUM_ITEMS,
        -1,
        dtype=np.int32,
    )

    item_global = (
        train_item_mapping["item_index"]
        .to_numpy(dtype=np.int64)
    )

    item_train = (
        train_item_mapping["train_item_id"]
        .to_numpy(dtype=np.int32)
    )

    global_item_to_train[
        item_global
    ] = item_train

    mapped_users = np.sum(
        global_user_to_train >= 0
    )

    mapped_items = np.sum(
        global_item_to_train >= 0
    )

    print(
        f"Mapped global users: "
        f"{mapped_users:,}"
    )

    print(
        f"Mapped global items: "
        f"{mapped_items:,}"
    )

    if mapped_users != NUM_USERS:

        raise AssertionError(
            "Training user mapping count mismatch."
        )

    if mapped_items != NUM_ITEMS:

        raise AssertionError(
            "Training item mapping count mismatch."
        )

    print(
        "[PASS] Global → training mappings"
    )

    return (
        global_user_to_train,
        global_item_to_train,
    )


# ============================================================
# TRAINING HISTORIES
# ============================================================

def build_training_histories(
    train_model_ids,
):

    print("\n" + "=" * 70)
    print("BUILDING TRAINING HISTORIES")
    print("=" * 70)

    sorted_train = train_model_ids.sort_values(
        [
            "train_user_id",
            "t_dat",
            "train_item_id",
        ],
        kind="mergesort",
    )

    histories = (
        sorted_train
        .groupby("train_user_id")[
            "train_item_id"
        ]
        .apply(list)
        .to_dict()
    )

    print(
        f"Training histories: "
        f"{len(histories):,}"
    )

    if len(histories) != NUM_USERS:

        raise AssertionError(
            "Training history user count mismatch."
        )

    lengths = np.fromiter(
        (
            len(history)
            for history in histories.values()
        ),
        dtype=np.int32,
    )

    print(
        f"Minimum history length: "
        f"{lengths.min()}"
    )

    print(
        f"Maximum history length: "
        f"{lengths.max()}"
    )

    print(
        f"Mean history length: "
        f"{lengths.mean():.2f}"
    )

    return histories


# ============================================================
# SASREC SEQUENCE PREPARATION
# ============================================================

def build_sequence(
    history,
    max_len=50,
):

    history = list(history)

    if len(history) > max_len:
        history = history[-max_len:]

    sequence = np.zeros(
        max_len,
        dtype=np.int64,
    )

    if len(history) > 0:

        sequence[
            -len(history):
        ] = (
            np.asarray(
                history,
                dtype=np.int64,
            )
            + 1
        )

    return sequence


# ============================================================
# WARM-START USERS
# ============================================================

def select_warm_start_users(
    val,
    global_user_to_train,
    global_item_to_train,
):

    print("\n" + "=" * 70)
    print("SELECTING WARM-START VALIDATION USERS")
    print("=" * 70)

    validation_users = (
        val["user_index"]
        .to_numpy(dtype=np.int64)
    )

    validation_items = (
        val["item_index"]
        .to_numpy(dtype=np.int64)
    )

    mapped_users = np.full(
        len(val),
        -1,
        dtype=np.int32,
    )

    mapped_items = np.full(
        len(val),
        -1,
        dtype=np.int32,
    )

    valid_users = (
        (validation_users >= 0)
        & (
            validation_users
            < GLOBAL_NUM_USERS
        )
    )

    valid_items = (
        (validation_items >= 0)
        & (
            validation_items
            < GLOBAL_NUM_ITEMS
        )
    )

    mapped_users[valid_users] = (
        global_user_to_train[
            validation_users[valid_users]
        ]
    )

    mapped_items[valid_items] = (
        global_item_to_train[
            validation_items[valid_items]
        ]
    )

    warm_start = (
        (mapped_users >= 0)
        & (mapped_items >= 0)
    )

    warm_start_users = np.unique(
        mapped_users[warm_start]
    )

    warm_start_interactions = int(
        warm_start.sum()
    )

    print(
        f"Warm-start validation interactions: "
        f"{warm_start_interactions:,}"
    )

    print(
        f"Warm-start validation users: "
        f"{len(warm_start_users):,}"
    )

    if warm_start_interactions != 751_993:

        raise AssertionError(
            "Unexpected warm-start interaction count."
        )

    if len(warm_start_users) != 173_203:

        raise AssertionError(
            "Unexpected warm-start user count."
        )

    print(
        "[PASS] Warm-start population"
    )

    return warm_start_users


# ============================================================
# GENERATE USER REPRESENTATIONS
# ============================================================

def generate_hybrid_user_representations(
    model,
    warm_start_users,
    training_histories,
    lightgcn_user_embeddings,
):

    print("\n" + "=" * 70)
    print("GENERATING HYBRID USER REPRESENTATIONS")
    print("=" * 70)

    representations = []

    start_time = time.time()

    with torch.no_grad():

        for start in range(
            0,
            len(warm_start_users),
            BATCH_SIZE,
        ):

            end = min(
                start + BATCH_SIZE,
                len(warm_start_users),
            )

            batch_users = (
                warm_start_users[start:end]
            )

            batch_sequences = np.stack(
                [
                    build_sequence(
                        training_histories[
                            int(user_id)
                        ],
                        MAX_SEQUENCE_LENGTH,
                    )
                    for user_id in batch_users
                ]
            )

            user_tensor = torch.from_numpy(
                batch_users.astype(np.int64)
            )

            sequence_tensor = torch.from_numpy(
                batch_sequences
            )

            # IMPORTANT:
            #
            # Pass the COMPLETE LightGCN user
            # embedding matrix.
            #
            # The model indexes it internally.
            #
            batch_representation = (
                model.get_user_representation(
                    user_ids=user_tensor,
                    sequences=sequence_tensor,
                    lightgcn_user_embeddings=(
                        lightgcn_user_embeddings
                    ),
                )
            )

            if batch_representation.shape != (
                end - start,
                HYBRID_DIM,
            ):

                raise AssertionError(
                    "Unexpected hybrid user "
                    "representation shape."
                )

            if not torch.isfinite(
                batch_representation
            ).all():

                raise AssertionError(
                    "Non-finite hybrid user "
                    "representation."
                )

            representations.append(
                batch_representation.cpu()
            )

    hybrid_users = torch.cat(
        representations,
        dim=0,
    )

    elapsed = time.time() - start_time

    print(
        f"Hybrid user representation shape: "
        f"{tuple(hybrid_users.shape)}"
    )

    print(
        f"Generation time: "
        f"{elapsed:.2f}s"
    )

    return hybrid_users


# ============================================================
# GENERATE ITEM REPRESENTATIONS
# ============================================================

def generate_hybrid_item_representations(
    model,
    lightgcn_item_embeddings,
):

    print("\n" + "=" * 70)
    print("GENERATING HYBRID ITEM REPRESENTATIONS")
    print("=" * 70)

    item_ids = torch.arange(
        NUM_ITEMS,
        dtype=torch.long,
    )

    with torch.no_grad():

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
        HYBRID_DIM,
    ):

        raise AssertionError(
            "Unexpected hybrid item "
            "representation shape."
        )

    if not torch.isfinite(
        hybrid_items
    ).all():

        raise AssertionError(
            "Non-finite hybrid item "
            "representation."
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
# REPRESENTATION STATISTICS
# ============================================================

def calculate_representation_statistics(
    name,
    representations,
):

    values = representations.detach().cpu().numpy()

    norms = np.linalg.norm(
        values,
        axis=1,
    )

    variance = float(
        values.var()
    )

    feature_variance = float(
        values.var(axis=0).mean()
    )

    statistics = {
        "shape": list(values.shape),
        "mean_norm": float(norms.mean()),
        "median_norm": float(np.median(norms)),
        "std_norm": float(norms.std()),
        "min_norm": float(norms.min()),
        "max_norm": float(norms.max()),
        "variance": variance,
        "mean_feature_variance": feature_variance,
        "mean_value": float(values.mean()),
        "std_value": float(values.std()),
        "min_value": float(values.min()),
        "max_value": float(values.max()),
    }

    print(
        f"\n{name}"
    )

    print(
        f"  mean norm: "
        f"{statistics['mean_norm']:.8f}"
    )

    print(
        f"  median norm: "
        f"{statistics['median_norm']:.8f}"
    )

    print(
        f"  std norm: "
        f"{statistics['std_norm']:.8f}"
    )

    print(
        f"  variance: "
        f"{statistics['variance']:.10f}"
    )

    return statistics


# ============================================================
# COMPONENT REPRESENTATIONS
# ============================================================

def extract_component_representations(
    model,
    warm_start_users,
    training_histories,
    lightgcn_user_embeddings,
    lightgcn_item_embeddings,
):

    print("\n" + "=" * 70)
    print("EXTRACTING COMPONENT REPRESENTATIONS")
    print("=" * 70)

    # --------------------------------------------------------
    # LightGCN users
    # --------------------------------------------------------

    user_ids = torch.from_numpy(
        warm_start_users.astype(np.int64)
    )

    lightgcn_users = (
        lightgcn_user_embeddings[
            user_ids
        ]
    )

    # --------------------------------------------------------
    # SASRec users
    # --------------------------------------------------------

    sequence_rows = np.stack(
        [
            build_sequence(
                training_histories[
                    int(user_id)
                ],
                MAX_SEQUENCE_LENGTH,
            )
            for user_id in warm_start_users
        ]
    )

    sequences = torch.from_numpy(
        sequence_rows
    )

    with torch.no_grad():

        sasrec_users = (
            model.sasrec(
                sequences
            )
        )

        # The hybrid model uses the representation
        # of the final sequence position.
        #
        # This follows the model's own implementation.
        sasrec_user_representations = (
            sasrec_users[:, -1, :]
        )

    # --------------------------------------------------------
    # LightGCN items
    # --------------------------------------------------------

    lightgcn_items = (
        lightgcn_item_embeddings
    )

    # --------------------------------------------------------
    # SASRec items
    # --------------------------------------------------------

    sasrec_items = (
        model.sasrec.item_embedding.weight[
            1:
        ]
        .detach()
        .cpu()
    )

    if sasrec_items.shape != (
        NUM_ITEMS,
        EMBEDDING_DIM,
    ):

        raise AssertionError(
            "Unexpected SASRec item embedding shape."
        )

    print(
        f"LightGCN user component: "
        f"{tuple(lightgcn_users.shape)}"
    )

    print(
        f"SASRec user component: "
        f"{tuple(sasrec_user_representations.shape)}"
    )

    print(
        f"LightGCN item component: "
        f"{tuple(lightgcn_items.shape)}"
    )

    print(
        f"SASRec item component: "
        f"{tuple(sasrec_items.shape)}"
    )

    return (
        lightgcn_users,
        sasrec_user_representations,
        lightgcn_items,
        sasrec_items,
    )


# ============================================================
# SCORE DISTRIBUTIONS
# ============================================================

def calculate_score_distributions(
    lightgcn_users,
    sasrec_users,
    hybrid_users,
    lightgcn_items,
    sasrec_items,
    hybrid_items,
):

    print("\n" + "=" * 70)
    print("CALCULATING SCORE DISTRIBUTIONS")
    print("=" * 70)

    results = {}

    # --------------------------------------------------------
    # LightGCN-only
    # --------------------------------------------------------

    lightgcn_scores = (
        lightgcn_users
        @ lightgcn_items.T
    )

    # --------------------------------------------------------
    # SASRec-only
    # --------------------------------------------------------

    sasrec_scores = (
        sasrec_users
        @ sasrec_items.T
    )

    # --------------------------------------------------------
    # Hybrid
    # --------------------------------------------------------

    hybrid_scores = (
        hybrid_users
        @ hybrid_items.T
    )

    for name, scores in [
        ("lightgcn", lightgcn_scores),
        ("sasrec", sasrec_scores),
        ("hybrid", hybrid_scores),
    ]:

        scores_np = (
            scores.detach()
            .cpu()
            .numpy()
        )

        if not np.isfinite(scores_np).all():

            raise AssertionError(
                f"Non-finite {name} scores."
            )

        results[name] = {
            "mean": float(
                scores_np.mean()
            ),
            "std": float(
                scores_np.std()
            ),
            "min": float(
                scores_np.min()
            ),
            "max": float(
                scores_np.max()
            ),
            "median": float(
                np.median(scores_np)
            ),
        }

        print(
            f"{name}: "
            f"mean={results[name]['mean']:.8f}, "
            f"std={results[name]['std']:.8f}, "
            f"min={results[name]['min']:.8f}, "
            f"max={results[name]['max']:.8f}"
        )

    return (
        results,
        lightgcn_scores,
        sasrec_scores,
        hybrid_scores,
    )


# ============================================================
# SCORE CORRELATIONS
# ============================================================

def calculate_score_correlations(
    lightgcn_scores,
    sasrec_scores,
    hybrid_scores,
):

    print("\n" + "=" * 70)
    print("CALCULATING SCORE CORRELATIONS")
    print("=" * 70)

    lightgcn = (
        lightgcn_scores
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    sasrec = (
        sasrec_scores
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    hybrid = (
        hybrid_scores
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    correlation_matrix = np.corrcoef(
        np.vstack(
            [
                lightgcn,
                sasrec,
                hybrid,
            ]
        )
    )

    result = {
        "lightgcn_vs_sasrec": float(
            correlation_matrix[0, 1]
        ),
        "lightgcn_vs_hybrid": float(
            correlation_matrix[0, 2]
        ),
        "sasrec_vs_hybrid": float(
            correlation_matrix[1, 2]
        ),
    }

    print(
        f"LightGCN vs SASRec: "
        f"{result['lightgcn_vs_sasrec']:.8f}"
    )

    print(
        f"LightGCN vs Hybrid: "
        f"{result['lightgcn_vs_hybrid']:.8f}"
    )

    print(
        f"SASRec vs Hybrid: "
        f"{result['sasrec_vs_hybrid']:.8f}"
    )

    return result


# ============================================================
# COMPONENT CONTRIBUTION
# ============================================================

def calculate_component_contribution(
    lightgcn_users,
    sasrec_users,
    hybrid_users,
    lightgcn_items,
    sasrec_items,
    hybrid_items,
):

    print("\n" + "=" * 70)
    print("ANALYZING HYBRID COMPONENT CONTRIBUTION")
    print("=" * 70)

    # --------------------------------------------------------
    # The V2 model first normalizes each component, then
    # concatenates the normalized representations before
    # passing them through the separate fusion projections.
    #
    # To measure the contribution of each half of the final
    # hybrid representation, use the actual trained hybrid
    # user/item representations and split them into:
    #
    #   first 64 dimensions  -> LightGCN side
    #   second 64 dimensions -> SASRec side
    #
    # This directly measures contribution at the final
    # scoring representation.
    # --------------------------------------------------------

    hybrid_user_np = (
        hybrid_users
        .detach()
        .cpu()
        .numpy()
    )

    hybrid_item_np = (
        hybrid_items
        .detach()
        .cpu()
        .numpy()
    )

    first_user = hybrid_user_np[:, :64]
    second_user = hybrid_user_np[:, 64:]

    first_item = hybrid_item_np[:, :64]
    second_item = hybrid_item_np[:, 64:]

    first_scores = (
        first_user
        @ first_item.T
    )

    second_scores = (
        second_user
        @ second_item.T
    )

    full_scores = (
        hybrid_user_np
        @ hybrid_item_np.T
    )

    # --------------------------------------------------------
    # Cross terms
    #
    # Full hybrid score:
    #
    #   U1·I1 + U1·I2 + U2·I1 + U2·I2
    #
    # Therefore also calculate the cross terms.
    # --------------------------------------------------------

    cross_user_first_item_second = (
        first_user
        @ second_item.T
    )

    cross_user_second_item_first = (
        second_user
        @ first_item.T
    )

    def stats(values):

        return {
            "mean": float(
                values.mean()
            ),
            "std": float(
                values.std()
            ),
            "min": float(
                values.min()
            ),
            "max": float(
                values.max()
            ),
        }

    result = {
        "lightgcn_component": stats(
            first_scores
        ),
        "sasrec_component": stats(
            second_scores
        ),
        "cross_lightgcn_user_sasrec_item": stats(
            cross_user_first_item_second
        ),
        "cross_sasrec_user_lightgcn_item": stats(
            cross_user_second_item_first
        ),
        "full_hybrid": stats(
            full_scores
        ),
    }

    # --------------------------------------------------------
    # Correlations with final score
    # --------------------------------------------------------

    flattened_full = (
        full_scores.reshape(-1)
    )

    flattened_first = (
        first_scores.reshape(-1)
    )

    flattened_second = (
        second_scores.reshape(-1)
    )

    flattened_cross_1 = (
        cross_user_first_item_second
        .reshape(-1)
    )

    flattened_cross_2 = (
        cross_user_second_item_first
        .reshape(-1)
    )

    result["correlations_with_full_hybrid"] = {
        "lightgcn_component": float(
            np.corrcoef(
                flattened_first,
                flattened_full,
            )[0, 1]
        ),
        "sasrec_component": float(
            np.corrcoef(
                flattened_second,
                flattened_full,
            )[0, 1]
        ),
        "cross_lightgcn_user_sasrec_item": float(
            np.corrcoef(
                flattened_cross_1,
                flattened_full,
            )[0, 1]
        ),
        "cross_sasrec_user_lightgcn_item": float(
            np.corrcoef(
                flattened_cross_2,
                flattened_full,
            )[0, 1]
        ),
    }

    print(
        "LightGCN-side score std: "
        f"{result['lightgcn_component']['std']:.8f}"
    )

    print(
        "SASRec-side score std: "
        f"{result['sasrec_component']['std']:.8f}"
    )

    print(
        "LightGCN-side correlation with hybrid: "
        f"{result['correlations_with_full_hybrid']['lightgcn_component']:.8f}"
    )

    print(
        "SASRec-side correlation with hybrid: "
        f"{result['correlations_with_full_hybrid']['sasrec_component']:.8f}"
    )

    return result


# ============================================================
# RECOMMENDATION CONCENTRATION
# ============================================================

def calculate_concentration(
    recommendations,
    num_items=NUM_ITEMS,
):

    print("\n" + "=" * 70)
    print("CALCULATING HYBRID V2 RECOMMENDATION CONCENTRATION")
    print("=" * 70)

    required_columns = {
        "train_user_id",
        "train_item_id",
        "rank",
        "score",
    }

    missing = (
        required_columns
        - set(recommendations.columns)
    )

    if missing:

        raise AssertionError(
            f"Recommendation file missing columns: "
            f"{missing}"
        )

    unique_items = (
        recommendations["train_item_id"]
        .nunique()
    )

    coverage_percent = (
        unique_items
        / num_items
        * 100
    )

    print(
        f"Unique recommended items: "
        f"{unique_items:,}"
    )

    print(
        f"Catalogue coverage: "
        f"{coverage_percent:.6f}%"
    )

    concentration = {}

    for k in [
        10,
        20,
        50,
        100,
    ]:

        top_k = recommendations[
            recommendations["rank"] <= k
        ]

        unique_top_k = (
            top_k["train_item_id"]
            .nunique()
        )

        percent = (
            unique_top_k
            / num_items
            * 100
        )

        concentration[
            f"top{k}_unique_items"
        ] = int(
            unique_top_k
        )

        concentration[
            f"top{k}_catalogue_coverage_percent"
        ] = float(
            percent
        )

        print(
            f"Top-{k}: "
            f"{unique_top_k:,} unique items "
            f"= {percent:.6f}% catalogue"
        )

    # --------------------------------------------------------
    # Item recommendation frequency
    # --------------------------------------------------------

    item_frequency = (
        recommendations[
            "train_item_id"
        ]
        .value_counts()
    )

    concentration[
        "most_recommended_item_count"
    ] = int(
        item_frequency.iloc[0]
    )

    concentration[
        "median_recommendation_count_per_item"
    ] = float(
        item_frequency.median()
    )

    concentration[
        "mean_recommendation_count_per_item"
    ] = float(
        item_frequency.mean()
    )

    print(
        f"Most recommended item count: "
        f"{item_frequency.iloc[0]:,}"
    )

    # --------------------------------------------------------
    # Top-20 item frequency share
    # --------------------------------------------------------

    top20_frequency = (
        item_frequency
        .head(20)
        .sum()
    )

    total_recommendations = len(
        recommendations
    )

    top20_share = (
        top20_frequency
        / total_recommendations
        * 100
    )

    concentration[
        "top20_items_recommendation_share_percent"
    ] = float(
        top20_share
    )

    print(
        f"Top-20 items recommendation share: "
        f"{top20_share:.6f}%"
    )

    concentration[
        "unique_items"
    ] = int(
        unique_items
    )

    concentration[
        "catalogue_coverage_percent"
    ] = float(
        coverage_percent
    )

    return concentration


# ============================================================
# TOP-K JACCARD
# ============================================================

def calculate_top20_jaccard(
    recommendations,
):

    print("\n" + "=" * 70)
    print("CALCULATING TOP-20 JACCARD SIMILARITY")
    print("=" * 70)

    user_groups = (
        recommendations
        .sort_values(
            [
                "train_user_id",
                "rank",
            ]
        )
        .groupby("train_user_id")
    )

    user_sets = []

    for _, group in user_groups:

        items = set(
            group[
                group["rank"] <= 20
            ]["train_item_id"]
            .tolist()
        )

        if len(items) == 20:
            user_sets.append(items)

    if len(user_sets) < 2:

        raise AssertionError(
            "Not enough users for Jaccard analysis."
        )

    # To avoid O(n^2) on all users, compare each
    # user against the next 20 users.
    #
    # This gives a deterministic sample of pairwise
    # list similarity.
    MAX_COMPARISONS_PER_USER = 20

    jaccards = []

    for i in range(
        len(user_sets)
    ):

        current = user_sets[i]

        upper = min(
            len(user_sets),
            i + 1 + MAX_COMPARISONS_PER_USER,
        )

        for j in range(
            i + 1,
            upper,
        ):

            other = user_sets[j]

            intersection = len(
                current & other
            )

            union = len(
                current | other
            )

            if union > 0:

                jaccards.append(
                    intersection / union
                )

    avg_jaccard = float(
        np.mean(jaccards)
    )

    result = {
        "users_compared": int(
            len(user_sets)
        ),
        "pairwise_comparisons": int(
            len(jaccards)
        ),
        "average_top20_jaccard": (
            avg_jaccard
        ),
    }

    print(
        f"Users compared: "
        f"{result['users_compared']:,}"
    )

    print(
        f"Pairwise comparisons: "
        f"{result['pairwise_comparisons']:,}"
    )

    print(
        f"Average Top-20 Jaccard: "
        f"{avg_jaccard:.8f}"
    )

    return result


# ============================================================
# COMPARE WITH V1
# ============================================================

def compare_with_v1(
    concentration,
    jaccard,
    score_statistics,
    score_correlations,
):

    print("\n" + "=" * 70)
    print("HYBRID V1 vs V2 COMPARISON")
    print("=" * 70)

    v2_unique = concentration[
        "unique_items"
    ]

    v2_coverage = concentration[
        "catalogue_coverage_percent"
    ]

    v2_top10 = concentration[
        "top10_catalogue_coverage_percent"
    ]

    v2_top20 = concentration[
        "top20_catalogue_coverage_percent"
    ]

    v2_top50 = concentration[
        "top50_catalogue_coverage_percent"
    ]

    v2_top100 = concentration[
        "top100_catalogue_coverage_percent"
    ]

    v2_jaccard = jaccard[
        "average_top20_jaccard"
    ]

    comparison = {
        "unique_items": {
            "v1": V1_REFERENCE[
                "unique_items"
            ],
            "v2": int(v2_unique),
            "change": int(
                v2_unique
                - V1_REFERENCE[
                    "unique_items"
                ]
            ),
        },

        "coverage_percent": {
            "v1": V1_REFERENCE[
                "coverage_percent"
            ],
            "v2": float(v2_coverage),
            "change": float(
                v2_coverage
                - V1_REFERENCE[
                    "coverage_percent"
                ]
            ),
        },

        "top10_coverage_percent": {
            "v1": V1_REFERENCE[
                "top10_concentration_percent"
            ],
            "v2": float(v2_top10),
            "change": float(
                v2_top10
                - V1_REFERENCE[
                    "top10_concentration_percent"
                ]
            ),
        },

        "top20_coverage_percent": {
            "v1": V1_REFERENCE[
                "top20_concentration_percent"
            ],
            "v2": float(v2_top20),
            "change": float(
                v2_top20
                - V1_REFERENCE[
                    "top20_concentration_percent"
                ]
            ),
        },

        "top50_coverage_percent": {
            "v1": V1_REFERENCE[
                "top50_concentration_percent"
            ],
            "v2": float(v2_top50),
            "change": float(
                v2_top50
                - V1_REFERENCE[
                    "top50_concentration_percent"
                ]
            ),
        },

        "top100_coverage_percent": {
            "v1": V1_REFERENCE[
                "top100_concentration_percent"
            ],
            "v2": float(v2_top100),
            "change": float(
                v2_top100
                - V1_REFERENCE[
                    "top100_concentration_percent"
                ]
            ),
        },

        "average_top20_jaccard": {
            "v1": V1_REFERENCE[
                "avg_top20_jaccard"
            ],
            "v2": float(v2_jaccard),
            "change": float(
                v2_jaccard
                - V1_REFERENCE[
                    "avg_top20_jaccard"
                ]
            ),
        },
    }

    print(
        f"Unique items: "
        f"V1={comparison['unique_items']['v1']:,} "
        f"→ V2={comparison['unique_items']['v2']:,}"
    )

    print(
        f"Coverage: "
        f"V1={comparison['coverage_percent']['v1']:.6f}% "
        f"→ V2={comparison['coverage_percent']['v2']:.6f}%"
    )

    print(
        f"Top-20 coverage: "
        f"V1={comparison['top20_coverage_percent']['v1']:.6f}% "
        f"→ V2={comparison['top20_coverage_percent']['v2']:.6f}%"
    )

    print(
        f"Top-100 coverage: "
        f"V1={comparison['top100_coverage_percent']['v1']:.6f}% "
        f"→ V2={comparison['top100_coverage_percent']['v2']:.6f}%"
    )

    print(
        f"Average Top-20 Jaccard: "
        f"V1={comparison['average_top20_jaccard']['v1']:.8f} "
        f"→ V2={comparison['average_top20_jaccard']['v2']:.8f}"
    )

    # --------------------------------------------------------
    # Component score comparison
    # --------------------------------------------------------

    component_comparison = {
        "score_std": {
            "lightgcn": {
                "v1": V1_REFERENCE[
                    "lightgcn_score_std"
                ],
                "v2": score_statistics[
                    "lightgcn"
                ]["std"],
            },
            "sasrec": {
                "v1": V1_REFERENCE[
                    "sasrec_score_std"
                ],
                "v2": score_statistics[
                    "sasrec"
                ]["std"],
            },
            "hybrid": {
                "v1": V1_REFERENCE[
                    "hybrid_score_std"
                ],
                "v2": score_statistics[
                    "hybrid"
                ]["std"],
            },
        },

        "hybrid_correlation": {
            "lightgcn": {
                "v1": V1_REFERENCE[
                    "lightgcn_hybrid_score_correlation"
                ],
                "v2": score_correlations[
                    "lightgcn_vs_hybrid"
                ],
            },
            "sasrec": {
                "v1": V1_REFERENCE[
                    "sasrec_hybrid_score_correlation"
                ],
                "v2": score_correlations[
                    "sasrec_vs_hybrid"
                ],
            },
        },
    }

    return {
        "recommendation_concentration": comparison,
        "component_score_comparison": (
            component_comparison
        ),
    }


# ============================================================
# FINAL INTERPRETATION FLAGS
# ============================================================

def create_diagnostic_flags(
    concentration,
    jaccard,
    score_statistics,
    score_correlations,
):

    print("\n" + "=" * 70)
    print("DIAGNOSTIC FLAGS")
    print("=" * 70)

    unique_items = concentration[
        "unique_items"
    ]

    coverage = concentration[
        "catalogue_coverage_percent"
    ]

    top100 = concentration[
        "top100_catalogue_coverage_percent"
    ]

    avg_jaccard = jaccard[
        "average_top20_jaccard"
    ]

    lightgcn_std = score_statistics[
        "lightgcn"
    ]["std"]

    sasrec_std = score_statistics[
        "sasrec"
    ]["std"]

    hybrid_std = score_statistics[
        "hybrid"
    ]["std"]

    lightgcn_corr = score_correlations[
        "lightgcn_vs_hybrid"
    ]

    sasrec_corr = score_correlations[
        "sasrec_vs_hybrid"
    ]

    flags = {}

    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    flags[
        "severe_catalogue_concentration"
    ] = bool(
        coverage < 1.0
    )

    flags[
        "extreme_top100_concentration"
    ] = bool(
        top100 > 95.0
    )

    flags[
        "high_list_similarity"
    ] = bool(
        avg_jaccard > 0.30
    )

    # --------------------------------------------------------
    # Score dominance
    # --------------------------------------------------------

    flags[
        "sasrec_score_variance_dominates_lightgcn"
    ] = bool(
        sasrec_std
        > lightgcn_std * 5
    )

    flags[
        "hybrid_more_correlated_with_sasrec"
    ] = bool(
        sasrec_corr
        > lightgcn_corr
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    for name, value in flags.items():

        print(
            f"{name}: "
            f"{'YES' if value else 'NO'}"
        )

    print(
        f"\nV2 unique recommended items: "
        f"{unique_items:,}"
    )

    print(
        f"V2 catalogue coverage: "
        f"{coverage:.6f}%"
    )

    print(
        f"V2 Top-100 catalogue coverage: "
        f"{top100:.6f}%"
    )

    print(
        f"V2 average Top-20 Jaccard: "
        f"{avg_jaccard:.8f}"
    )

    print(
        f"LightGCN score std: "
        f"{lightgcn_std:.8f}"
    )

    print(
        f"SASRec score std: "
        f"{sasrec_std:.8f}"
    )

    print(
        f"Hybrid score std: "
        f"{hybrid_std:.8f}"
    )

    print(
        f"LightGCN ↔ Hybrid correlation: "
        f"{lightgcn_corr:.8f}"
    )

    print(
        f"SASRec ↔ Hybrid correlation: "
        f"{sasrec_corr:.8f}"
    )

    return flags


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("HYBRID V2 COMPONENT & CONCENTRATION DIAGNOSTIC")
    print("=" * 70)

    set_seed(SEED)

    # --------------------------------------------------------
    # Required files
    # --------------------------------------------------------

    required_files = [
        TRAIN_FILE,
        VAL_FILE,
        TRAIN_MODEL_IDS_FILE,
        TRAIN_USER_MAPPING_FILE,
        TRAIN_ITEM_MAPPING_FILE,
        LIGHTGCN_USER_EMBEDDINGS_FILE,
        LIGHTGCN_ITEM_EMBEDDINGS_FILE,
        SASREC_MODEL_FILE,
        HYBRID_MODEL_FILE,
        HYBRID_V2_RECOMMENDATIONS_FILE,
    ]

    for path in required_files:
        check_file(path)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model()

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    (
        lightgcn_user_embeddings,
        lightgcn_item_embeddings,
    ) = load_embeddings()

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    (
        train,
        val,
        train_model_ids,
        train_user_mapping,
        train_item_mapping,
    ) = load_data()

    # --------------------------------------------------------
    # Mappings
    # --------------------------------------------------------

    (
        global_user_to_train,
        global_item_to_train,
    ) = build_global_to_training_maps(
        train_user_mapping,
        train_item_mapping,
    )

    # --------------------------------------------------------
    # Histories
    # --------------------------------------------------------

    training_histories = (
        build_training_histories(
            train_model_ids
        )
    )

    # --------------------------------------------------------
    # Warm-start population
    # --------------------------------------------------------

    warm_start_users = (
        select_warm_start_users(
            val,
            global_user_to_train,
            global_item_to_train,
        )
    )

    # --------------------------------------------------------
    # Diagnostic sampling
    # --------------------------------------------------------

    diagnostic_users = (
        warm_start_users[
            :DIAGNOSTIC_USERS
        ]
    )

    print(
        "\nDiagnostic users: "
        f"{len(diagnostic_users):,}"
    )

    # --------------------------------------------------------
    # Hybrid representations
    # --------------------------------------------------------

    hybrid_users = (
        generate_hybrid_user_representations(
            model=model,
            warm_start_users=diagnostic_users,
            training_histories=training_histories,
            lightgcn_user_embeddings=(
                lightgcn_user_embeddings
            ),
        )
    )

    hybrid_items = (
        generate_hybrid_item_representations(
            model=model,
            lightgcn_item_embeddings=(
                lightgcn_item_embeddings
            ),
        )
    )

    # --------------------------------------------------------
    # Representation statistics
    # --------------------------------------------------------

    representation_statistics = {}

    representation_statistics[
        "lightgcn_user"
    ] = calculate_representation_statistics(
        "LightGCN user representations",
        lightgcn_user_embeddings[
            torch.from_numpy(
                diagnostic_users.astype(
                    np.int64
                )
            )
        ],
    )

    # Build SASRec users separately.
    (
        lightgcn_users,
        sasrec_users,
        lightgcn_items,
        sasrec_items,
    ) = extract_component_representations(
        model=model,
        warm_start_users=diagnostic_users,
        training_histories=training_histories,
        lightgcn_user_embeddings=(
            lightgcn_user_embeddings
        ),
        lightgcn_item_embeddings=(
            lightgcn_item_embeddings
        ),
    )

    representation_statistics[
        "sasrec_user"
    ] = calculate_representation_statistics(
        "SASRec user representations",
        sasrec_users,
    )

    representation_statistics[
        "hybrid_user"
    ] = calculate_representation_statistics(
        "Hybrid user representations",
        hybrid_users,
    )

    representation_statistics[
        "lightgcn_item"
    ] = calculate_representation_statistics(
        "LightGCN item representations",
        lightgcn_items,
    )

    representation_statistics[
        "sasrec_item"
    ] = calculate_representation_statistics(
        "SASRec item representations",
        sasrec_items,
    )

    representation_statistics[
        "hybrid_item"
    ] = calculate_representation_statistics(
        "Hybrid item representations",
        hybrid_items,
    )

    # --------------------------------------------------------
    # Scores
    # --------------------------------------------------------

    (
        score_statistics,
        lightgcn_scores,
        sasrec_scores,
        hybrid_scores,
    ) = calculate_score_distributions(
        lightgcn_users=lightgcn_users,
        sasrec_users=sasrec_users,
        hybrid_users=hybrid_users,
        lightgcn_items=lightgcn_items,
        sasrec_items=sasrec_items,
        hybrid_items=hybrid_items,
    )

    # --------------------------------------------------------
    # Correlations
    # --------------------------------------------------------

    score_correlations = (
        calculate_score_correlations(
            lightgcn_scores,
            sasrec_scores,
            hybrid_scores,
        )
    )

    # --------------------------------------------------------
    # Component contribution
    # --------------------------------------------------------

    component_contribution = (
        calculate_component_contribution(
            lightgcn_users=lightgcn_users,
            sasrec_users=sasrec_users,
            hybrid_users=hybrid_users,
            lightgcn_items=lightgcn_items,
            sasrec_items=sasrec_items,
            hybrid_items=hybrid_items,
        )
    )

    # --------------------------------------------------------
    # Load recommendations
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOADING HYBRID V2 RECOMMENDATIONS")
    print("=" * 70)

    recommendations = pd.read_parquet(
        HYBRID_V2_RECOMMENDATIONS_FILE
    )

    print(
        f"Recommendation rows: "
        f"{len(recommendations):,}"
    )

    expected_rows = (
        173_203 * TOP_K
    )

    if len(recommendations) != expected_rows:

        raise AssertionError(
            f"Expected {expected_rows:,} "
            f"recommendation rows."
        )

    print(
        "[PASS] Recommendation file"
    )

    # --------------------------------------------------------
    # Concentration
    # --------------------------------------------------------

    concentration = (
        calculate_concentration(
            recommendations
        )
    )

    # --------------------------------------------------------
    # Jaccard
    # --------------------------------------------------------

    jaccard = (
        calculate_top20_jaccard(
            recommendations
        )
    )

    # --------------------------------------------------------
    # V1 comparison
    # --------------------------------------------------------

    v1_vs_v2 = compare_with_v1(
        concentration=concentration,
        jaccard=jaccard,
        score_statistics=score_statistics,
        score_correlations=score_correlations,
    )

    # --------------------------------------------------------
    # Diagnostic flags
    # --------------------------------------------------------

    flags = create_diagnostic_flags(
        concentration=concentration,
        jaccard=jaccard,
        score_statistics=score_statistics,
        score_correlations=score_correlations,
    )

    # --------------------------------------------------------
    # Final JSON
    # --------------------------------------------------------

    result = {
        "model": "Hybrid LightGCN + SASRec V2",

        "configuration": {
            "num_users": NUM_USERS,
            "num_items": NUM_ITEMS,
            "embedding_dim": EMBEDDING_DIM,
            "hybrid_dimension": HYBRID_DIM,
            "max_sequence_length": (
                MAX_SEQUENCE_LENGTH
            ),
            "diagnostic_users": int(
                len(diagnostic_users)
            ),
            "seed": SEED,
        },

        "data": {
            "training_interactions": int(
                len(train)
            ),
            "validation_interactions": int(
                len(val)
            ),
            "warm_start_validation_users": int(
                len(warm_start_users)
            ),
            "diagnostic_users": int(
                len(diagnostic_users)
            ),
        },

        "representation_statistics": (
            representation_statistics
        ),

        "score_statistics": (
            score_statistics
        ),

        "score_correlations": (
            score_correlations
        ),

        "component_contribution": (
            component_contribution
        ),

        "recommendation_concentration": (
            concentration
        ),

        "top20_jaccard": (
            jaccard
        ),

        "v1_vs_v2": (
            v1_vs_v2
        ),

        "diagnostic_flags": (
            flags
        ),
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("HYBRID V2 DIAGNOSTIC COMPLETE")
    print("=" * 70)

    print(
        f"Unique recommended items: "
        f"{concentration['unique_items']:,}"
    )

    print(
        f"Catalogue coverage: "
        f"{concentration['catalogue_coverage_percent']:.6f}%"
    )

    print(
        f"Top-100 catalogue coverage: "
        f"{concentration['top100_catalogue_coverage_percent']:.6f}%"
    )

    print(
        f"Average Top-20 Jaccard: "
        f"{jaccard['average_top20_jaccard']:.8f}"
    )

    print(
        f"SASRec ↔ Hybrid score correlation: "
        f"{score_correlations['sasrec_vs_hybrid']:.8f}"
    )

    print(
        f"LightGCN ↔ Hybrid score correlation: "
        f"{score_correlations['lightgcn_vs_hybrid']:.8f}"
    )

    print(
        f"\nSaved diagnostic:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        "\nHYBRID V2 DIAGNOSTIC PASSED"
    )


if __name__ == "__main__":
    main()