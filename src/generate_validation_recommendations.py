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

TRAIN_MODEL_IDS_FILE = DATA_DIR / "transactions_train_model_ids.parquet"
TRAIN_USER_MAPPING_FILE = DATA_DIR / "train_user_mapping.parquet"
TRAIN_ITEM_MAPPING_FILE = DATA_DIR / "train_item_mapping.parquet"

LIGHTGCN_MODEL_FILE = OUTPUT_DIR / "lightgcn_model.pt"
LIGHTGCN_USER_EMBEDDINGS_FILE = OUTPUT_DIR / "lightgcn_user_embeddings.npy"
LIGHTGCN_ITEM_EMBEDDINGS_FILE = OUTPUT_DIR / "lightgcn_item_embeddings.npy"

SASREC_MODEL_FILE = OUTPUT_DIR / "sasrec_model.pt"

HYBRID_MODEL_FILE = OUTPUT_DIR / "hybrid_v2_model.pt"

RECOMMENDATION_FILE = (
    OUTPUT_DIR / "hybrid_v2_validation_recommendations.parquet"
)

METADATA_FILE = (
    OUTPUT_DIR / "hybrid_v2_validation_recommendations_metadata.json"
)

TOP_K = 20
MAX_SEQUENCE_LENGTH = 50

BATCH_SIZE = 256

NUM_USERS = 778_421
NUM_ITEMS = 74_541
EMBEDDING_DIM = 64


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def check_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    print(f"[FOUND] {path}")


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def left_pad_sequence(item_ids, max_len=50):
    """
    Convert a user's training history into SASRec input format.

    SASRec convention:
        0       -> padding
        1..N    -> actual item IDs

    The most recent max_len interactions are retained.
    """

    item_ids = list(item_ids)

    if len(item_ids) > max_len:
        item_ids = item_ids[-max_len:]

    sequence = np.zeros(max_len, dtype=np.int64)

    if len(item_ids) > 0:
        sequence[-len(item_ids):] = np.asarray(item_ids, dtype=np.int64) + 1

    return sequence


def remap_sasrec_checkpoint_keys(state_dict):
    """
    The standalone SASRec checkpoint uses:

        transformer_encoder.layers.*
        layer_norm.*

    while the SASRec implementation inside the hybrid model uses:

        transformer_layers.*
        final_layer_norm.*

    Only parameter names are changed.
    Parameter values remain unchanged.
    """

    converted = {}

    transformer_count = 0
    layer_norm_count = 0

    for key, value in state_dict.items():

        new_key = key

        if key.startswith("transformer_encoder.layers."):
            new_key = key.replace(
                "transformer_encoder.layers.",
                "transformer_layers.",
                1,
            )
            transformer_count += 1

        elif key.startswith("layer_norm."):
            new_key = key.replace(
                "layer_norm.",
                "final_layer_norm.",
                1,
            )
            layer_norm_count += 1

        converted[new_key] = value

    print(
        f"SASRec checkpoint key conversion: "
        f"transformer keys={transformer_count}, "
        f"layer-norm keys={layer_norm_count}"
    )

    return converted


# ============================================================
# MODEL IMPORT
# ============================================================

from hybrid_model import HybridLightGCNSASRec


# ============================================================
# MODEL CREATION
# ============================================================

def create_model():

    print("\n" + "=" * 70)
    print("CREATING HYBRID V2 MODEL")
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

    print(
        f"Model created: "
        f"users={NUM_USERS}, "
        f"items={NUM_ITEMS}, "
        f"embedding_dim={EMBEDDING_DIM}"
    )

    # --------------------------------------------------------
    # Load complete Hybrid V2 checkpoint
    # --------------------------------------------------------

    print("\nLoading Hybrid V2 checkpoint...")

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

    print("[PASS] Hybrid V2 checkpoint loaded")

    # --------------------------------------------------------
    # Freeze pretrained encoders
    # --------------------------------------------------------

    model.freeze_pretrained_encoders()

    model.lightgcn.eval()
    model.sasrec.eval()

    print("[PASS] LightGCN frozen")
    print("[PASS] SASRec frozen")
    print("[PASS] Frozen encoders in evaluation mode")

    # --------------------------------------------------------
    # Validate trainable parameters
    # --------------------------------------------------------

    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    trainable_count = sum(
        parameter.numel()
        for _, parameter in trainable
    )

    print(f"Trainable parameters: {trainable_count}")

    if trainable_count != 33_024:
        raise AssertionError(
            f"Expected 33,024 trainable parameters, "
            f"got {trainable_count}"
        )

    expected_names = {
        "user_fusion.0.weight",
        "user_fusion.0.bias",
        "item_fusion.0.weight",
        "item_fusion.0.bias",
    }

    actual_names = {
        name
        for name, _ in trainable
    }

    if actual_names != expected_names:
        raise AssertionError(
            "Unexpected trainable parameter set.\n"
            f"Expected: {expected_names}\n"
            f"Actual: {actual_names}"
        )

    print("[PASS] Exactly 33,024 fusion parameters trainable")

    return model


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    train = pd.read_parquet(TRAIN_FILE)
    val = pd.read_parquet(VAL_FILE)

    train_model_ids = pd.read_parquet(
        TRAIN_MODEL_IDS_FILE
    )

    train_user_mapping = pd.read_parquet(
        TRAIN_USER_MAPPING_FILE
    )

    train_item_mapping = pd.read_parquet(
        TRAIN_ITEM_MAPPING_FILE
    )

    print(f"Training interactions: {len(train):,}")
    print(f"Validation interactions: {len(val):,}")
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
# CREATE GLOBAL → TRAINING MAPPINGS
# ============================================================

def create_global_to_training_maps(
    train_user_mapping,
    train_item_mapping,
):

    print("\n" + "=" * 70)
    print("BUILDING GLOBAL → TRAINING ID MAPS")
    print("=" * 70)

    # --------------------------------------------------------
    # User mapping
    #
    # global user_index -> train_user_id
    # --------------------------------------------------------

    global_user_to_train = np.full(
        NUM_USERS + 100_000,
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

    max_global_user = int(user_global.max())

    global_user_to_train = np.full(
        max_global_user + 1,
        -1,
        dtype=np.int32,
    )

    global_user_to_train[user_global] = user_train

    # --------------------------------------------------------
    # Item mapping
    #
    # global item_index -> train_item_id
    # --------------------------------------------------------

    item_global = (
        train_item_mapping["item_index"]
        .to_numpy(dtype=np.int64)
    )

    item_train = (
        train_item_mapping["train_item_id"]
        .to_numpy(dtype=np.int32)
    )

    max_global_item = int(item_global.max())

    global_item_to_train = np.full(
        max_global_item + 1,
        -1,
        dtype=np.int32,
    )

    global_item_to_train[item_global] = item_train

    print(
        f"Global users represented in training: "
        f"{np.sum(global_user_to_train >= 0):,}"
    )

    print(
        f"Global items represented in training: "
        f"{np.sum(global_item_to_train >= 0):,}"
    )

    return (
        global_user_to_train,
        global_item_to_train,
    )


# ============================================================
# BUILD TRAINING HISTORY
# ============================================================

def build_training_histories(train_model_ids):

    print("\n" + "=" * 70)
    print("BUILDING TRAINING HISTORIES")
    print("=" * 70)

    required_columns = {
        "train_user_id",
        "train_item_id",
        "t_dat",
    }

    missing = required_columns - set(
        train_model_ids.columns
    )

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Deterministic chronological ordering.
    #
    # train_item_id is used only as a deterministic tie-breaker
    # for same-day transactions.
    train_model_ids = train_model_ids.sort_values(
        [
            "train_user_id",
            "t_dat",
            "train_item_id",
        ],
        kind="mergesort",
    )

    histories = (
        train_model_ids
        .groupby("train_user_id")["train_item_id"]
        .apply(list)
        .to_dict()
    )

    print(
        f"Training histories created: "
        f"{len(histories):,}"
    )

    lengths = np.fromiter(
        (len(x) for x in histories.values()),
        dtype=np.int32,
    )

    print(f"Minimum history length: {lengths.min()}")
    print(f"Maximum history length: {lengths.max()}")
    print(f"Mean history length: {lengths.mean():.2f}")

    return histories


# ============================================================
# SELECT WARM-START VALIDATION USERS
# ============================================================

def prepare_warm_start_validation(
    val,
    global_user_to_train,
    global_item_to_train,
):

    print("\n" + "=" * 70)
    print("PREPARING WARM-START VALIDATION")
    print("=" * 70)

    validation_users = val["user_index"].to_numpy(
        dtype=np.int64
    )

    validation_items = val["item_index"].to_numpy(
        dtype=np.int64
    )

    valid_user_mask = (
        validation_users >= 0
    ) & (
        validation_users < len(global_user_to_train)
    )

    valid_item_mask = (
        validation_items >= 0
    ) & (
        validation_items < len(global_item_to_train)
    )

    user_train_ids = np.full(
        len(val),
        -1,
        dtype=np.int32,
    )

    item_train_ids = np.full(
        len(val),
        -1,
        dtype=np.int32,
    )

    user_train_ids[valid_user_mask] = (
        global_user_to_train[
            validation_users[valid_user_mask]
        ]
    )

    item_train_ids[valid_item_mask] = (
        global_item_to_train[
            validation_items[valid_item_mask]
        ]
    )

    warm_start_mask = (
        (user_train_ids >= 0)
        & (item_train_ids >= 0)
    )

    warm_start_user_ids = user_train_ids[
        warm_start_mask
    ]

    warm_start_item_ids = item_train_ids[
        warm_start_mask
    ]

    warm_start_users = np.unique(
        warm_start_user_ids
    )

    print(
        f"Validation interactions: "
        f"{len(val):,}"
    )

    print(
        f"Warm-start validation interactions: "
        f"{warm_start_mask.sum():,}"
    )

    print(
        f"Warm-start validation users: "
        f"{len(warm_start_users):,}"
    )

    expected_interactions = 751_993
    expected_users = 173_203

    if len(warm_start_user_ids) != expected_interactions:
        raise AssertionError(
            "Unexpected warm-start validation interaction count.\n"
            f"Expected {expected_interactions:,}, "
            f"got {len(warm_start_user_ids):,}"
        )

    if len(warm_start_users) != expected_users:
        raise AssertionError(
            "Unexpected warm-start validation user count.\n"
            f"Expected {expected_users:,}, "
            f"got {len(warm_start_users):,}"
        )

    print("[PASS] Warm-start interaction count")
    print("[PASS] Warm-start user count")

    return (
        warm_start_users,
        warm_start_user_ids,
        warm_start_item_ids,
    )


# ============================================================
# BUILD USER SEQUENCE MATRIX
# ============================================================

def build_validation_sequences(
    warm_start_users,
    training_histories,
):

    print("\n" + "=" * 70)
    print("BUILDING SASREC VALIDATION SEQUENCES")
    print("=" * 70)

    sequences = np.zeros(
        (
            len(warm_start_users),
            MAX_SEQUENCE_LENGTH,
        ),
        dtype=np.int64,
    )

    missing_users = 0

    for row_idx, user_id in enumerate(
        warm_start_users
    ):

        history = training_histories.get(
            int(user_id)
        )

        if history is None:
            missing_users += 1
            continue

        sequences[row_idx] = left_pad_sequence(
            history,
            MAX_SEQUENCE_LENGTH,
        )

    if missing_users != 0:
        raise AssertionError(
            f"{missing_users} warm-start users "
            "do not have training histories."
        )

    nonzero_lengths = (
        sequences > 0
    ).sum(axis=1)

    print(
        f"Sequence matrix: {sequences.shape}"
    )

    print(
        f"Minimum sequence length: "
        f"{nonzero_lengths.min()}"
    )

    print(
        f"Maximum sequence length: "
        f"{nonzero_lengths.max()}"
    )

    print(
        f"Mean sequence length: "
        f"{nonzero_lengths.mean():.2f}"
    )

    if sequences.min() < 0:
        raise AssertionError(
            "Negative SASRec sequence ID found."
        )

    if sequences.max() > NUM_ITEMS:
        raise AssertionError(
            "SASRec sequence ID exceeds "
            "training vocabulary."
        )

    print("[PASS] SASRec sequence range")

    return sequences


# ============================================================
# LOAD PRECOMPUTED LIGHTGCN EMBEDDINGS
# ============================================================

def load_lightgcn_embeddings():

    print("\n" + "=" * 70)
    print("LOADING LIGHTGCN EMBEDDINGS")
    print("=" * 70)

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

    # Convert mmap arrays into writable/normal arrays.
    #
    # This avoids the PyTorch warning:
    #
    # "The given NumPy array is not writable..."
    #
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

    if not np.isfinite(user_embeddings).all():
        raise AssertionError(
            "Non-finite LightGCN user embeddings."
        )

    if not np.isfinite(item_embeddings).all():
        raise AssertionError(
            "Non-finite LightGCN item embeddings."
        )

    print("[PASS] LightGCN embedding shapes")
    print("[PASS] LightGCN embeddings finite")
    print("[PASS] Memory-mapped arrays copied safely")

    return (
        user_embeddings,
        item_embeddings,
    )


# ============================================================
# GENERATE HYBRID USER REPRESENTATIONS
# ============================================================

def generate_user_representations(
    model,
    warm_start_users,
    sequences,
    lightgcn_user_embeddings,
):

    print("\n" + "=" * 70)
    print("GENERATING HYBRID USER REPRESENTATIONS")
    print("=" * 70)

    device = torch.device("cpu")

    model.eval()
    model.lightgcn.eval()
    model.sasrec.eval()

    all_user_representations = []

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

            batch_user_ids = torch.from_numpy(
                np.asarray(
                    warm_start_users[start:end],
                    dtype=np.int64,
                )
            ).to(device)

            batch_sequences = torch.from_numpy(
                np.asarray(
                    sequences[start:end],
                    dtype=np.int64,
                )
            ).to(device)

            # IMPORTANT:
            #
            # Pass the FULL LightGCN user embedding
            # matrix here.
            #
            # The model internally indexes it using
            # batch_user_ids.
            #
            batch_hybrid_users = (
                model.get_user_representation(
                    user_ids=batch_user_ids,
                    sequences=batch_sequences,
                    lightgcn_user_embeddings=(
                        lightgcn_user_embeddings
                    ),
                )
            )

            if batch_hybrid_users.shape != (
                end - start,
                128,
            ):
                raise AssertionError(
                    "Unexpected hybrid user representation shape."
                )

            if not torch.isfinite(
                batch_hybrid_users
            ).all():
                raise AssertionError(
                    "Non-finite hybrid user representation."
                )

            all_user_representations.append(
                batch_hybrid_users.cpu()
            )

            if (
                (start == 0)
                or (end == len(warm_start_users))
                or (
                    start % (BATCH_SIZE * 50)
                    == 0
                )
            ):
                print(
                    f"Processed "
                    f"{end:,} / "
                    f"{len(warm_start_users):,} users"
                )

    user_representations = torch.cat(
        all_user_representations,
        dim=0,
    )

    elapsed = time.time() - start_time

    print(
        f"User representation shape: "
        f"{tuple(user_representations.shape)}"
    )

    print(
        f"User representation generation: "
        f"{elapsed:.2f}s"
    )

    if user_representations.shape != (
        len(warm_start_users),
        128,
    ):
        raise AssertionError(
            "Final hybrid user representation "
            "shape mismatch."
        )

    print("[PASS] Hybrid user representations")

    return user_representations


# ============================================================
# GENERATE HYBRID ITEM REPRESENTATIONS
# ============================================================

def generate_item_representations(
    model,
    lightgcn_item_embeddings,
):

    print("\n" + "=" * 70)
    print("GENERATING HYBRID ITEM REPRESENTATIONS")
    print("=" * 70)

    device = torch.device("cpu")

    model.eval()
    model.lightgcn.eval()
    model.sasrec.eval()

    all_item_ids = torch.arange(
        NUM_ITEMS,
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():

        hybrid_items = model.get_item_representation(
            item_ids=all_item_ids,
            lightgcn_item_embeddings=(
                lightgcn_item_embeddings
            ),
        )

    print(
        f"Hybrid item representation shape: "
        f"{tuple(hybrid_items.shape)}"
    )

    if hybrid_items.shape != (
        NUM_ITEMS,
        128,
    ):
        raise AssertionError(
            "Unexpected hybrid item representation shape."
        )

    if not torch.isfinite(hybrid_items).all():
        raise AssertionError(
            "Non-finite hybrid item representations."
        )

    print("[PASS] Hybrid item representations")

    return hybrid_items


# ============================================================
# BUILD SEEN-ITEM SETS
# ============================================================

def build_seen_item_sets(
    train_model_ids,
):

    print("\n" + "=" * 70)
    print("BUILDING SEEN-ITEM SETS")
    print("=" * 70)

    seen_items = (
        train_model_ids
        .groupby("train_user_id")["train_item_id"]
        .apply(set)
        .to_dict()
    )

    print(
        f"Users with seen-item sets: "
        f"{len(seen_items):,}"
    )

    if len(seen_items) != NUM_USERS:
        raise AssertionError(
            "Training users do not match "
            "training vocabulary."
        )

    print("[PASS] Seen-item sets")

    return seen_items


# ============================================================
# GENERATE TOP-K RECOMMENDATIONS
# ============================================================

def generate_recommendations(
    model,
    warm_start_users,
    user_representations,
    hybrid_item_representations,
    seen_items,
):

    print("\n" + "=" * 70)
    print("GENERATING TOP-K RECOMMENDATIONS")
    print("=" * 70)

    start_time = time.time()

    recommendation_rows = []

    item_matrix = hybrid_item_representations

    num_users = len(warm_start_users)

    with torch.no_grad():

        for start in range(
            0,
            num_users,
            BATCH_SIZE,
        ):

            end = min(
                start + BATCH_SIZE,
                num_users,
            )

            batch_users = (
                user_representations[start:end]
            )

            scores = (
                batch_users @ item_matrix.T
            )

            if not torch.isfinite(scores).all():
                raise AssertionError(
                    "Non-finite recommendation scores."
                )

            # ------------------------------------------------
            # Remove items already seen during training.
            # ------------------------------------------------

            for local_idx in range(
                end - start
            ):

                user_id = int(
                    warm_start_users[
                        start + local_idx
                    ]
                )

                user_scores = scores[
                    local_idx
                ].clone()

                seen = seen_items.get(
                    user_id,
                    set(),
                )

                if len(seen) > 0:

                    seen_tensor = torch.tensor(
                        list(seen),
                        dtype=torch.long,
                    )

                    user_scores[
                        seen_tensor
                    ] = -torch.inf

                top_scores, top_items = torch.topk(
                    user_scores,
                    k=TOP_K,
                )

                top_items = (
                    top_items
                    .cpu()
                    .numpy()
                )

                top_scores = (
                    top_scores
                    .cpu()
                    .numpy()
                )

                for rank, (
                    item_id,
                    score,
                ) in enumerate(
                    zip(
                        top_items,
                        top_scores,
                    ),
                    start=1,
                ):

                    recommendation_rows.append(
                        (
                            user_id,
                            int(item_id),
                            int(rank),
                            float(score),
                        )
                    )

            if (
                start == 0
                or end == num_users
                or start % (BATCH_SIZE * 20) == 0
            ):
                print(
                    f"Processed "
                    f"{end:,} / "
                    f"{num_users:,} users"
                )

    elapsed = time.time() - start_time

    recommendations = pd.DataFrame(
        recommendation_rows,
        columns=[
            "train_user_id",
            "train_item_id",
            "rank",
            "score",
        ],
    )

    print(
        f"\nRecommendation rows: "
        f"{len(recommendations):,}"
    )

    print(
        f"Recommendation generation time: "
        f"{elapsed:.2f}s"
    )

    expected_rows = (
        num_users * TOP_K
    )

    if len(recommendations) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows:,} rows, "
            f"got {len(recommendations):,}"
        )

    # --------------------------------------------------------
    # Structural validation
    # --------------------------------------------------------

    counts = (
        recommendations
        .groupby("train_user_id")
        .size()
    )

    if counts.min() != TOP_K:
        raise AssertionError(
            "Some users do not have exactly "
            f"{TOP_K} recommendations."
        )

    if counts.max() != TOP_K:
        raise AssertionError(
            "Some users have more than "
            f"{TOP_K} recommendations."
        )

    rank_counts = (
        recommendations
        .groupby("train_user_id")["rank"]
        .apply(list)
    )

    for ranks in rank_counts:

        if ranks != list(
            range(1, TOP_K + 1)
        ):
            raise AssertionError(
                "Invalid ranking sequence."
            )

    duplicate_pairs = recommendations.duplicated(
        subset=[
            "train_user_id",
            "train_item_id",
        ]
    ).sum()

    if duplicate_pairs != 0:
        raise AssertionError(
            "Duplicate user-item recommendation found."
        )

    if recommendations["train_item_id"].min() < 0:
        raise AssertionError(
            "Negative training item ID."
        )

    if recommendations["train_item_id"].max() >= NUM_ITEMS:
        raise AssertionError(
            "Recommendation item ID exceeds "
            "training vocabulary."
        )

    if not np.isfinite(
        recommendations["score"]
    ).all():
        raise AssertionError(
            "Non-finite recommendation score."
        )

    # --------------------------------------------------------
    # Confirm seen items were excluded
    # --------------------------------------------------------

    violations = 0

    for row in recommendations.itertuples(
        index=False
    ):

        if row.train_item_id in seen_items[
            row.train_user_id
        ]:
            violations += 1

            if violations <= 5:
                print(
                    "[ERROR] Seen item recommended:",
                    row.train_user_id,
                    row.train_item_id,
                )

    if violations != 0:
        raise AssertionError(
            f"Found {violations} seen-item violations."
        )

    print("[PASS] Recommendation row count")
    print("[PASS] Exactly 20 recommendations/user")
    print("[PASS] Rank sequence")
    print("[PASS] No duplicate user-item pairs")
    print("[PASS] Item ID range")
    print("[PASS] Scores finite")
    print("[PASS] Training-seen items excluded")

    return recommendations, elapsed


# ============================================================
# SAVE OUTPUT
# ============================================================

def save_output(
    recommendations,
    generation_time,
    warm_start_users,
):

    print("\n" + "=" * 70)
    print("SAVING RECOMMENDATIONS")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    recommendations.to_parquet(
        RECOMMENDATION_FILE,
        index=False,
    )

    metadata = {
        "model": "Hybrid LightGCN + SASRec V2",
        "checkpoint": str(
            HYBRID_MODEL_FILE
        ),
        "num_users_training": NUM_USERS,
        "num_items_training": NUM_ITEMS,
        "embedding_dim": EMBEDDING_DIM,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "top_k": TOP_K,
        "warm_start_users": int(
            len(warm_start_users)
        ),
        "recommendation_rows": int(
            len(recommendations)
        ),
        "recommendations_per_user": TOP_K,
        "generation_time_seconds": float(
            generation_time
        ),
        "device": "cpu",
        "sequence_source": (
            "training history only"
        ),
        "evaluation_population": (
            "warm-start validation users "
            "with known validation items"
        ),
        "seen_item_policy": (
            "exclude training-seen items"
        ),
    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(
        f"Saved recommendations:\n"
        f"{RECOMMENDATION_FILE}"
    )

    print(
        f"Saved metadata:\n"
        f"{METADATA_FILE}"
    )

    return metadata


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("HYBRID V2 VALIDATION RECOMMENDATION GENERATION")
    print("=" * 70)

    set_seed(42)

    # --------------------------------------------------------
    # Check required files
    # --------------------------------------------------------

    required_files = [
        TRAIN_FILE,
        VAL_FILE,
        TRAIN_MODEL_IDS_FILE,
        TRAIN_USER_MAPPING_FILE,
        TRAIN_ITEM_MAPPING_FILE,
        LIGHTGCN_MODEL_FILE,
        LIGHTGCN_USER_EMBEDDINGS_FILE,
        LIGHTGCN_ITEM_EMBEDDINGS_FILE,
        SASREC_MODEL_FILE,
        HYBRID_MODEL_FILE,
    ]

    for path in required_files:
        check_file(path)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = create_model()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    (
        train,
        val,
        train_model_ids,
        train_user_mapping,
        train_item_mapping,
    ) = load_data()

    # --------------------------------------------------------
    # Global -> training mappings
    # --------------------------------------------------------

    (
        global_user_to_train,
        global_item_to_train,
    ) = create_global_to_training_maps(
        train_user_mapping,
        train_item_mapping,
    )

    # --------------------------------------------------------
    # Training histories
    # --------------------------------------------------------

    training_histories = build_training_histories(
        train_model_ids
    )

    # --------------------------------------------------------
    # Warm-start validation population
    # --------------------------------------------------------

    (
        warm_start_users,
        warm_start_user_ids,
        warm_start_item_ids,
    ) = prepare_warm_start_validation(
        val,
        global_user_to_train,
        global_item_to_train,
    )

    # --------------------------------------------------------
    # SASRec sequences
    # --------------------------------------------------------

    validation_sequences = (
        build_validation_sequences(
            warm_start_users,
            training_histories,
        )
    )

    # --------------------------------------------------------
    # LightGCN embeddings
    # --------------------------------------------------------

    (
        lightgcn_user_embeddings,
        lightgcn_item_embeddings,
    ) = load_lightgcn_embeddings()

    # --------------------------------------------------------
    # Convert LightGCN embeddings to tensors
    # --------------------------------------------------------

    lightgcn_user_tensor = torch.from_numpy(
        lightgcn_user_embeddings
    )

    lightgcn_item_tensor = torch.from_numpy(
        lightgcn_item_embeddings
    )

    # --------------------------------------------------------
    # Generate hybrid user representations
    # --------------------------------------------------------

    hybrid_user_representations = (
        generate_user_representations(
            model=model,
            warm_start_users=warm_start_users,
            sequences=validation_sequences,
            lightgcn_user_embeddings=(
                lightgcn_user_tensor
            ),
        )
    )

    # --------------------------------------------------------
    # Generate hybrid item representations
    # --------------------------------------------------------

    hybrid_item_representations = (
        generate_item_representations(
            model=model,
            lightgcn_item_embeddings=(
                lightgcn_item_tensor
            ),
        )
    )

    # --------------------------------------------------------
    # Build seen-item sets
    # --------------------------------------------------------

    seen_items = build_seen_item_sets(
        train_model_ids
    )

    # --------------------------------------------------------
    # Generate recommendations
    # --------------------------------------------------------

    (
        recommendations,
        generation_time,
    ) = generate_recommendations(
        model=model,
        warm_start_users=warm_start_users,
        user_representations=(
            hybrid_user_representations
        ),
        hybrid_item_representations=(
            hybrid_item_representations
        ),
        seen_items=seen_items,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    metadata = save_output(
        recommendations=recommendations,
        generation_time=generation_time,
        warm_start_users=warm_start_users,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("HYBRID V2 RECOMMENDATION GENERATION PASSED")
    print("=" * 70)

    print(
        f"Warm-start users: "
        f"{len(warm_start_users):,}"
    )

    print(
        f"Recommendations/user: "
        f"{TOP_K}"
    )

    print(
        f"Total recommendation rows: "
        f"{len(recommendations):,}"
    )

    print(
        f"Generation time: "
        f"{generation_time:.2f}s"
    )

    print(
        f"Output: "
        f"{RECOMMENDATION_FILE}"
    )


if __name__ == "__main__":
    main()