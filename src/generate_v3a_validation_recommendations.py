"""
Generate V3-A validation recommendations.

V3-A:
    Adaptive graph-sequential hybrid
    + random-negative training
    + frozen LightGCN
    + frozen SASRec
    + trained user gate
    + trained user fusion
    + trained item fusion

Validation protocol:
    - Warm-start users only
    - Warm-start items only
    - User representation uses TRAIN history only
    - SASRec sequence = latest 50 training interactions
    - Full training-item catalogue = 74,541 items
    - Training-seen items excluded
    - Top-20 recommendations per validation user

Output:
    output/hybrid_v3a_validation_recommendations.parquet
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch

from hybrid_model import (
    HybridLightGCNSASRecV3,
    load_lightgcn,
    load_sasrec,
)


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"

TRAIN_MODEL_IDS_FILE = (
    DATA_DIR / "transactions_train_model_ids.parquet"
)

VAL_FILE = (
    DATA_DIR / "transactions_val.parquet"
)

TRAIN_USER_MAPPING_FILE = (
    DATA_DIR / "train_user_mapping.parquet"
)

TRAIN_ITEM_MAPPING_FILE = (
    DATA_DIR / "train_item_mapping.parquet"
)

LIGHTGCN_MODEL_FILE = (
    OUTPUT_DIR / "lightgcn_model.pt"
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

V3A_MODEL_FILE = (
    OUTPUT_DIR / "hybrid_v3a_model.pt"
)

OUTPUT_RECOMMENDATIONS_FILE = (
    OUTPUT_DIR /
    "hybrid_v3a_validation_recommendations.parquet"
)

OUTPUT_METADATA_FILE = (
    OUTPUT_DIR /
    "hybrid_v3a_validation_recommendations_metadata.json"
)


# Model configuration
NUM_USERS = 778_421
NUM_ITEMS = 74_541

EMBEDDING_DIM = 64
MAX_SEQUENCE_LENGTH = 50

NUM_HEADS = 2
NUM_LAYERS = 2
DROPOUT = 0.2

TOP_K = 20

# CPU-friendly batches
USER_BATCH_SIZE = 512
ITEM_SCORE_BATCH_SIZE = 4096

SEED = 42

DEVICE = torch.device("cpu")


# ============================================================
# REQUIRED FILE CHECK
# ============================================================

def check_required_files():

    print()
    print("=" * 80)
    print("CHECKING REQUIRED FILES")
    print("=" * 80)

    required_files = [
        TRAIN_MODEL_IDS_FILE,
        VAL_FILE,
        TRAIN_USER_MAPPING_FILE,
        TRAIN_ITEM_MAPPING_FILE,
        LIGHTGCN_MODEL_FILE,
        LIGHTGCN_USER_EMBEDDINGS_FILE,
        LIGHTGCN_ITEM_EMBEDDINGS_FILE,
        SASREC_MODEL_FILE,
        V3A_MODEL_FILE,
    ]

    for path in required_files:

        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file:\n{path}"
            )

        print(f"[PASS] {path.name}")


# ============================================================
# LOAD V3 CHECKPOINT
# ============================================================

def load_v3a_checkpoint(model):

    print()
    print("=" * 80)
    print("LOADING V3-A CHECKPOINT")
    print("=" * 80)

    checkpoint = torch.load(
        V3A_MODEL_FILE,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = checkpoint["model_state_dict"]

    converted_state_dict = {}

    for key, value in state_dict.items():

        new_key = key

        # Compatibility with checkpoints created from
        # older SASRec module names.
        if new_key.startswith(
            "sasrec.transformer_encoder.layers."
        ):
            new_key = new_key.replace(
                "sasrec.transformer_encoder.layers.",
                "sasrec.transformer_layers.",
                1,
            )

        if new_key.startswith(
            "sasrec.layer_norm."
        ):
            new_key = new_key.replace(
                "sasrec.layer_norm.",
                "sasrec.final_layer_norm.",
                1,
            )

        converted_state_dict[new_key] = value

    missing, unexpected = model.load_state_dict(
        converted_state_dict,
        strict=False,
    )

    if missing:
        raise RuntimeError(
            "Missing checkpoint keys:\n"
            + "\n".join(missing)
        )

    if unexpected:
        raise RuntimeError(
            "Unexpected checkpoint keys:\n"
            + "\n".join(unexpected)
        )

    print("[PASS] V3-A checkpoint loaded")
    print(
        f"Checkpoint epochs: "
        f"{checkpoint.get('epochs', 'unknown')}"
    )

    return checkpoint


# ============================================================
# BUILD MODEL
# ============================================================

def build_model():

    print()
    print("=" * 80)
    print("BUILDING V3-A MODEL")
    print("=" * 80)

    model = HybridLightGCNSASRecV3(
        num_users=NUM_USERS,
        num_items=NUM_ITEMS,
        embedding_dim=EMBEDDING_DIM,
        max_seq_len=MAX_SEQUENCE_LENGTH,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Total parameters before loading: "
        f"{total_parameters:,}"
    )

    if total_parameters != 59_504_832:
        raise RuntimeError(
            f"Unexpected total parameters: "
            f"{total_parameters:,}"
        )

    # --------------------------------------------------------
    # Load LightGCN
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("LOADING LIGHTGCN")
    print("=" * 80)

    lightgcn = load_lightgcn(
        LIGHTGCN_MODEL_FILE,
        num_users=NUM_USERS,
        num_items=NUM_ITEMS,
        embedding_dim=EMBEDDING_DIM,
        num_layers=3,
    )

    model.lightgcn.load_state_dict(
        lightgcn.state_dict(),
        strict=True,
    )

    print("[PASS] LightGCN checkpoint loaded")

    # --------------------------------------------------------
    # Load SASRec
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("LOADING SASREC")
    print("=" * 80)

    sasrec = load_sasrec(
        SASREC_MODEL_FILE
    )

    model.sasrec.load_state_dict(
        sasrec.state_dict(),
        strict=True,
    )

    print("[PASS] SASRec checkpoint loaded")

    # --------------------------------------------------------
    # Load V3-A fusion/gate
    # --------------------------------------------------------

    load_v3a_checkpoint(model)

    # Freeze pretrained encoders.
    model.freeze_pretrained_encoders()

    model.lightgcn.eval()
    model.sasrec.eval()

    model.to(DEVICE)

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    if trainable_parameters != 41_280:
        raise RuntimeError(
            f"Unexpected V3-A trainable parameters: "
            f"{trainable_parameters:,}"
        )

    print("[PASS] Trainable parameter count")
    print("[PASS] LightGCN frozen")
    print("[PASS] SASRec frozen")
    print("[PASS] Encoders in evaluation mode")

    return model


# ============================================================
# LOAD LIGHTGCN EMBEDDINGS
# ============================================================

def load_lightgcn_embeddings():

    print()
    print("=" * 80)
    print("LOADING PRECOMPUTED LIGHTGCN EMBEDDINGS")
    print("=" * 80)

    user_embeddings = np.load(
        LIGHTGCN_USER_EMBEDDINGS_FILE,
        mmap_mode="r",
    ).copy()

    item_embeddings = np.load(
        LIGHTGCN_ITEM_EMBEDDINGS_FILE,
        mmap_mode="r",
    ).copy()

    print(
        f"User embeddings: "
        f"{user_embeddings.shape}"
    )

    print(
        f"Item embeddings: "
        f"{item_embeddings.shape}"
    )

    if user_embeddings.shape != (
        NUM_USERS,
        EMBEDDING_DIM,
    ):
        raise RuntimeError(
            f"Unexpected user embedding shape: "
            f"{user_embeddings.shape}"
        )

    if item_embeddings.shape != (
        NUM_ITEMS,
        EMBEDDING_DIM,
    ):
        raise RuntimeError(
            f"Unexpected item embedding shape: "
            f"{item_embeddings.shape}"
        )

    if not np.isfinite(
        user_embeddings
    ).all():

        raise RuntimeError(
            "LightGCN user embeddings contain "
            "non-finite values"
        )

    if not np.isfinite(
        item_embeddings
    ).all():

        raise RuntimeError(
            "LightGCN item embeddings contain "
            "non-finite values"
        )

    user_embeddings = torch.from_numpy(
        user_embeddings
    ).float()

    item_embeddings = torch.from_numpy(
        item_embeddings
    ).float()

    # V3-A requires normalized components.
    user_embeddings = torch.nn.functional.normalize(
        user_embeddings,
        p=2,
        dim=1,
    )

    item_embeddings = torch.nn.functional.normalize(
        item_embeddings,
        p=2,
        dim=1,
    )

    print(
        f"Normalized users: "
        f"{tuple(user_embeddings.shape)}"
    )

    print(
        f"Normalized items: "
        f"{tuple(item_embeddings.shape)}"
    )

    print("[PASS] LightGCN embeddings prepared")

    return user_embeddings, item_embeddings


# ============================================================
# BUILD TRAINING HISTORIES
# ============================================================

def build_training_histories():

    print()
    print("=" * 80)
    print("BUILDING TRAINING HISTORIES")
    print("=" * 80)

    train_df = pd.read_parquet(
        TRAIN_MODEL_IDS_FILE
    )

    required_columns = {
        "train_user_id",
        "train_item_id",
        "t_dat",
        "item_index",
    }

    missing = (
        required_columns
        - set(train_df.columns)
    )

    if missing:
        raise RuntimeError(
            f"Missing columns from training data: "
            f"{sorted(missing)}"
        )

    # Explicit deterministic chronological order.
    #
    # t_dat has date-level resolution only.
    # item_index is retained as the project's
    # deterministic same-day tie-breaker.
    train_df = train_df.sort_values(
        [
            "train_user_id",
            "t_dat",
            "item_index",
        ],
        kind="mergesort",
    )

    histories = {}

    for user_id, group in train_df.groupby(
        "train_user_id",
        sort=False,
    ):

        histories[int(user_id)] = (
            group["train_item_id"]
            .astype(np.int64)
            .tolist()
        )

    print(
        f"Training users with histories: "
        f"{len(histories):,}"
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
        f"{lengths.min():,}"
    )

    print(
        f"Maximum history length: "
        f"{lengths.max():,}"
    )

    print(
        f"Mean history length: "
        f"{lengths.mean():.2f}"
    )

    if len(histories) != NUM_USERS:
        raise RuntimeError(
            f"Expected {NUM_USERS:,} training users, "
            f"got {len(histories):,}"
        )

    return histories


# ============================================================
# GLOBAL -> TRAINING ID MAPPINGS
# ============================================================

def build_global_to_train_mappings():

    print()
    print("=" * 80)
    print("BUILDING GLOBAL -> TRAINING ID MAPPINGS")
    print("=" * 80)

    user_mapping = pd.read_parquet(
        TRAIN_USER_MAPPING_FILE
    )

    item_mapping = pd.read_parquet(
        TRAIN_ITEM_MAPPING_FILE
    )

    required_user_columns = {
        "user_index",
        "train_user_id",
    }

    required_item_columns = {
        "item_index",
        "train_item_id",
    }

    if not required_user_columns.issubset(
        user_mapping.columns
    ):
        raise RuntimeError(
            "Invalid training user mapping"
        )

    if not required_item_columns.issubset(
        item_mapping.columns
    ):
        raise RuntimeError(
            "Invalid training item mapping"
        )

    global_to_train_user = dict(
        zip(
            user_mapping[
                "user_index"
            ].astype(np.int64),
            user_mapping[
                "train_user_id"
            ].astype(np.int64),
        )
    )

    global_to_train_item = dict(
        zip(
            item_mapping[
                "item_index"
            ].astype(np.int64),
            item_mapping[
                "train_item_id"
            ].astype(np.int64),
        )
    )

    print(
        f"User mappings: "
        f"{len(global_to_train_user):,}"
    )

    print(
        f"Item mappings: "
        f"{len(global_to_train_item):,}"
    )

    if len(global_to_train_user) != NUM_USERS:
        raise RuntimeError(
            "Unexpected training-user mapping size"
        )

    if len(global_to_train_item) != NUM_ITEMS:
        raise RuntimeError(
            "Unexpected training-item mapping size"
        )

    print("[PASS] Global -> training mappings")

    return (
        global_to_train_user,
        global_to_train_item,
    )


# ============================================================
# WARM-START VALIDATION POPULATION
# ============================================================

def build_warm_start_validation_population(
    val_df,
    global_to_train_user,
    global_to_train_item,
):

    print()
    print("=" * 80)
    print("BUILDING WARM-START VALIDATION POPULATION")
    print("=" * 80)

    known_users = set(
        global_to_train_user.keys()
    )

    known_items = set(
        global_to_train_item.keys()
    )

    user_known = val_df[
        "user_index"
    ].isin(known_users)

    item_known = val_df[
        "item_index"
    ].isin(known_items)

    warm_start = val_df[
        user_known & item_known
    ].copy()

    print(
        f"Validation interactions: "
        f"{len(val_df):,}"
    )

    print(
        f"Known-user interactions: "
        f"{int(user_known.sum()):,}"
    )

    print(
        f"Known-item interactions: "
        f"{int(item_known.sum()):,}"
    )

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

    # This is the established warm-start population.
    expected_interactions = 751_993
    expected_users = 173_203

    if len(warm_start) != expected_interactions:
        raise RuntimeError(
            f"Unexpected warm-start interactions: "
            f"{len(warm_start):,}; "
            f"expected {expected_interactions:,}"
        )

    if (
        warm_start["user_index"].nunique()
        != expected_users
    ):
        raise RuntimeError(
            f"Unexpected warm-start users: "
            f"{warm_start['user_index'].nunique():,}; "
            f"expected {expected_users:,}"
        )

    print(
        "[PASS] Warm-start validation population"
    )

    return warm_start


# ============================================================
# BUILD VALIDATION USER SEQUENCES
# ============================================================

def build_validation_sequences(
    warm_start,
    global_to_train_user,
    histories,
):

    print()
    print("=" * 80)
    print("BUILDING VALIDATION USER SEQUENCES")
    print("=" * 80)

    global_user_ids = (
        warm_start[
            "user_index"
        ]
        .drop_duplicates()
        .astype(np.int64)
        .to_numpy()
    )

    sequences = np.zeros(
        (
            len(global_user_ids),
            MAX_SEQUENCE_LENGTH,
        ),
        dtype=np.int64,
    )

    train_user_ids = np.empty(
        len(global_user_ids),
        dtype=np.int64,
    )

    seen_items = []

    for row, global_user_id in enumerate(
        global_user_ids
    ):

        global_user_id = int(
            global_user_id
        )

        train_user_id = int(
            global_to_train_user[
                global_user_id
            ]
        )

        train_user_ids[row] = (
            train_user_id
        )

        history = histories[
            train_user_id
        ]

        # Latest 50 training interactions.
        recent_history = history[
            -MAX_SEQUENCE_LENGTH:
        ]

        # SASRec reserves ID 0 for padding.
        encoded = (
            np.asarray(
                recent_history,
                dtype=np.int64,
            )
            + 1
        )

        start = (
            MAX_SEQUENCE_LENGTH
            - len(encoded)
        )

        sequences[
            row,
            start:
        ] = encoded

        # Keep complete training history for
        # seen-item exclusion.
        seen_items.append(
            set(history)
        )

    lengths = np.count_nonzero(
        sequences,
        axis=1,
    )

    print(
        f"Validation users: "
        f"{len(global_user_ids):,}"
    )

    print(
        f"Sequence shape: "
        f"{sequences.shape}"
    )

    print(
        f"Sequence length min: "
        f"{lengths.min():,}"
    )

    print(
        f"Sequence length max: "
        f"{lengths.max():,}"
    )

    print(
        f"Sequence length mean: "
        f"{lengths.mean():.2f}"
    )

    if sequences.min() < 0:
        raise RuntimeError(
            "Invalid sequence value"
        )

    if sequences.max() > NUM_ITEMS:
        raise RuntimeError(
            "Invalid shifted item ID"
        )

    # Check left padding.
    for row in range(
        min(1000, len(sequences))
    ):

        sequence = sequences[row]

        nonzero = np.flatnonzero(
            sequence
        )

        if len(nonzero) == 0:
            raise RuntimeError(
                f"Empty sequence for row {row}"
            )

        first = nonzero[0]

        if np.any(
            sequence[first:] == 0
        ):
            raise RuntimeError(
                f"Internal padding detected "
                f"for row {row}"
            )

    print(
        "[PASS] Validation sequences"
    )

    return (
        global_user_ids,
        train_user_ids,
        sequences,
        seen_items,
    )


# ============================================================
# SASREC USER REPRESENTATIONS
# ============================================================

@torch.no_grad()
def build_sasrec_user_representations(
    model,
    sequences,
):

    print()
    print("=" * 80)
    print("BUILDING SASREC USER REPRESENTATIONS")
    print("=" * 80)

    sequence_tensor = torch.from_numpy(
        sequences
    ).long()

    representations = []

    total = len(sequence_tensor)

    for start in range(
        0,
        total,
        USER_BATCH_SIZE,
    ):

        end = min(
            start + USER_BATCH_SIZE,
            total,
        )

        batch = sequence_tensor[
            start:end
        ].to(DEVICE)

        output = (
            model.sasrec
            .get_sequence_representation(
                batch
            )
        )

        expected_shape = (
            end - start,
            EMBEDDING_DIM,
        )

        if tuple(output.shape) != expected_shape:
            raise RuntimeError(
                "Unexpected SASRec representation "
                f"shape: {tuple(output.shape)}; "
                f"expected {expected_shape}"
            )

        representations.append(
            output.cpu()
        )

    representations = torch.cat(
        representations,
        dim=0,
    )

    if not torch.isfinite(
        representations
    ).all():

        raise RuntimeError(
            "SASRec representations contain "
            "non-finite values"
        )

    representations = (
        torch.nn.functional.normalize(
            representations,
            p=2,
            dim=1,
        )
    )

    print(
        f"SASRec user representations: "
        f"{tuple(representations.shape)}"
    )

    print(
        "[PASS] SASRec representations"
    )

    return representations


# ============================================================
# V3-A USER REPRESENTATIONS
# ============================================================

@torch.no_grad()
def build_v3a_user_representations(
    model,
    graph_users,
    sasrec_users,
):

    print()
    print("=" * 80)
    print("BUILDING V3-A USER REPRESENTATIONS")
    print("=" * 80)

    outputs = []

    total = graph_users.shape[0]

    for start in range(
        0,
        total,
        USER_BATCH_SIZE,
    ):

        end = min(
            start + USER_BATCH_SIZE,
            total,
        )

        graph_batch = graph_users[
            start:end
        ]

        sequence_batch = sasrec_users[
            start:end
        ]

        # Components are already normalized,
        # but normalize again for explicit V3-A
        # inference consistency.
        graph_batch = (
            torch.nn.functional.normalize(
                graph_batch,
                p=2,
                dim=1,
            )
        )

        sequence_batch = (
            torch.nn.functional.normalize(
                sequence_batch,
                p=2,
                dim=1,
            )
        )

        fusion_input = torch.cat(
            [
                graph_batch,
                sequence_batch,
            ],
            dim=1,
        )

        # Adaptive graph/sequence gate.
        gate = torch.sigmoid(
            model.user_gate(
                fusion_input
            )
        )

        adaptive_graph = (
            gate * graph_batch
        )

        adaptive_sequence = (
            (1.0 - gate)
            * sequence_batch
        )

        user_input = torch.cat(
            [
                adaptive_graph,
                adaptive_sequence,
            ],
            dim=1,
        )

        hybrid_user = (
            model.user_fusion(
                user_input
            )
        )

        outputs.append(
            hybrid_user.cpu()
        )

    outputs = torch.cat(
        outputs,
        dim=0,
    )

    expected_shape = (
        total,
        EMBEDDING_DIM * 2,
    )

    if tuple(outputs.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected V3-A user shape: "
            f"{tuple(outputs.shape)}"
        )

    if not torch.isfinite(
        outputs
    ).all():

        raise RuntimeError(
            "V3-A user representations "
            "contain non-finite values"
        )

    print(
        f"V3-A user representations: "
        f"{tuple(outputs.shape)}"
    )

    print(
        "[PASS] V3-A user representations"
    )

    return outputs


# ============================================================
# FINAL V3-A ITEM REPRESENTATIONS
# ============================================================

@torch.no_grad()
def build_v3a_item_representations(
    model,
    graph_items,
):

    print()
    print("=" * 80)
    print("BUILDING V3-A ITEM REPRESENTATIONS")
    print("=" * 80)

    # Remove SASRec padding row.
    sasrec_items = (
        model.sasrec
        .item_embedding
        .weight[1:]
    )

    sasrec_items = (
        torch.nn.functional.normalize(
            sasrec_items,
            p=2,
            dim=1,
        )
    )

    graph_items = (
        torch.nn.functional.normalize(
            graph_items,
            p=2,
            dim=1,
        )
    )

    if graph_items.shape != (
        NUM_ITEMS,
        EMBEDDING_DIM,
    ):
        raise RuntimeError(
            f"Unexpected graph item shape: "
            f"{tuple(graph_items.shape)}"
        )

    if sasrec_items.shape != (
        NUM_ITEMS,
        EMBEDDING_DIM,
    ):
        raise RuntimeError(
            f"Unexpected SASRec item shape: "
            f"{tuple(sasrec_items.shape)}"
        )

    item_input = torch.cat(
        [
            graph_items,
            sasrec_items,
        ],
        dim=1,
    )

    # IMPORTANT:
    # During evaluation item_fusion is already trained,
    # so it is safe to compute the final item representation
    # once under no_grad().
    hybrid_items = (
        model.item_fusion(
            item_input
        )
    )

    expected_shape = (
        NUM_ITEMS,
        EMBEDDING_DIM * 2,
    )

    if tuple(hybrid_items.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected V3-A item shape: "
            f"{tuple(hybrid_items.shape)}"
        )

    if not torch.isfinite(
        hybrid_items
    ).all():

        raise RuntimeError(
            "V3-A item representations "
            "contain non-finite values"
        )

    print(
        f"V3-A item representations: "
        f"{tuple(hybrid_items.shape)}"
    )

    print(
        "[PASS] V3-A item representations"
    )

    return hybrid_items


# ============================================================
# RECOMMENDATION GENERATION
# ============================================================

@torch.no_grad()
def generate_recommendations(
    global_user_ids,
    hybrid_users,
    hybrid_items,
    seen_items,
):

    print()
    print("=" * 80)
    print("GENERATING V3-A RECOMMENDATIONS")
    print("=" * 80)

    recommendation_rows = []

    total_users = len(
        global_user_ids
    )

    for user_start in range(
        0,
        total_users,
        USER_BATCH_SIZE,
    ):

        user_end = min(
            user_start + USER_BATCH_SIZE,
            total_users,
        )

        user_batch = hybrid_users[
            user_start:user_end
        ]

        # ----------------------------------------------------
        # Score full catalogue in chunks.
        # ----------------------------------------------------

        score_chunks = []

        for item_start in range(
            0,
            NUM_ITEMS,
            ITEM_SCORE_BATCH_SIZE,
        ):

            item_end = min(
                item_start
                + ITEM_SCORE_BATCH_SIZE,
                NUM_ITEMS,
            )

            item_batch = hybrid_items[
                item_start:item_end
            ]

            scores = (
                user_batch
                @ item_batch.T
            )

            score_chunks.append(
                scores
            )

        scores = torch.cat(
            score_chunks,
            dim=1,
        )

        # ----------------------------------------------------
        # Exclude complete training history.
        # ----------------------------------------------------

        for local_row in range(
            user_end - user_start
        ):

            history = seen_items[
                user_start + local_row
            ]

            if history:

                history_array = np.fromiter(
                    history,
                    dtype=np.int64,
                )

                history_tensor = (
                    torch.from_numpy(
                        history_array
                    ).long()
                )

                scores[
                    local_row,
                    history_tensor,
                ] = -float("inf")

        # ----------------------------------------------------
        # Top 20.
        # ----------------------------------------------------

        top_scores, top_items = (
            torch.topk(
                scores,
                k=TOP_K,
                dim=1,
            )
        )

        if not torch.isfinite(
            top_scores
        ).all():

            raise RuntimeError(
                "Top-K scores contain non-finite values"
            )

        top_items = (
            top_items.cpu().numpy()
        )

        for local_row in range(
            top_items.shape[0]
        ):

            global_user_id = int(
                global_user_ids[
                    user_start
                    + local_row
                ]
            )

            for rank in range(
                TOP_K
            ):

                recommendation_rows.append(
                    (
                        global_user_id,
                        int(
                            top_items[
                                local_row,
                                rank,
                            ]
                        ),
                        rank + 1,
                    )
                )

        processed = user_end

        if (
            processed % 10_000 == 0
            or processed == total_users
        ):

            print(
                f"Processed users: "
                f"{processed:,}/"
                f"{total_users:,}"
            )

    recommendations = pd.DataFrame(
        recommendation_rows,
        columns=[
            "user_index",
            "item_index",
            "rank",
        ],
    )

    # --------------------------------------------------------
    # Validation checks.
    # --------------------------------------------------------

    expected_rows = (
        total_users * TOP_K
    )

    if len(recommendations) != (
        expected_rows
    ):
        raise RuntimeError(
            f"Expected {expected_rows:,} "
            f"recommendation rows, got "
            f"{len(recommendations):,}"
        )

    if recommendations.duplicated(
        ["user_index", "rank"]
    ).any():

        raise RuntimeError(
            "Duplicate user/rank rows detected"
        )

    if recommendations[
        "item_index"
    ].min() < 0:

        raise RuntimeError(
            "Negative item index detected"
        )

    if recommendations[
        "item_index"
    ].max() >= NUM_ITEMS:

        raise RuntimeError(
            "Out-of-range item index detected"
        )

    # Verify no training-seen item was recommended.
    user_position = {
        int(user_id): position
        for position, user_id
        in enumerate(global_user_ids)
    }

    for user_id, group in recommendations.groupby(
        "user_index",
        sort=False,
    ):

        position = user_position[
            int(user_id)
        ]

        history = seen_items[
            position
        ]

        recommended_items = set(
            group[
                "item_index"
            ].astype(int)
        )

        overlap = (
            recommended_items
            & history
        )

        if overlap:

            raise RuntimeError(
                f"Training-seen items recommended "
                f"to user {user_id}: "
                f"{list(overlap)[:10]}"
            )

    print()
    print(
        f"Recommendation rows: "
        f"{len(recommendations):,}"
    )

    print(
        f"Recommendation users: "
        f"{recommendations['user_index'].nunique():,}"
    )

    print(
        f"Unique recommended items: "
        f"{recommendations['item_index'].nunique():,}"
    )

    print(
        "[PASS] Recommendation validation"
    )

    return recommendations


# ============================================================
# MAIN
# ============================================================

def main():

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    start_time = time.time()

    print("=" * 80)
    print("HYBRID V3-A VALIDATION RECOMMENDATION GENERATION")
    print("=" * 80)

    print(f"Device: {DEVICE}")
    print(f"Training users: {NUM_USERS:,}")
    print(f"Training items: {NUM_ITEMS:,}")
    print(f"Top-K: {TOP_K}")

    # --------------------------------------------------------
    # Files
    # --------------------------------------------------------

    check_required_files()

    # --------------------------------------------------------
    # Validation data
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("LOADING VALIDATION DATA")
    print("=" * 80)

    val_df = pd.read_parquet(
        VAL_FILE
    )

    required_columns = {
        "user_index",
        "item_index",
    }

    missing = (
        required_columns
        - set(val_df.columns)
    )

    if missing:
        raise RuntimeError(
            f"Missing validation columns: "
            f"{sorted(missing)}"
        )

    print(
        f"Validation interactions: "
        f"{len(val_df):,}"
    )

    # --------------------------------------------------------
    # Mappings
    # --------------------------------------------------------

    (
        global_to_train_user,
        global_to_train_item,
    ) = build_global_to_train_mappings()

    # --------------------------------------------------------
    # Warm-start validation
    # --------------------------------------------------------

    warm_start = (
        build_warm_start_validation_population(
            val_df,
            global_to_train_user,
            global_to_train_item,
        )
    )

    # --------------------------------------------------------
    # Training histories
    # --------------------------------------------------------

    histories = (
        build_training_histories()
    )

    # --------------------------------------------------------
    # Validation sequences
    # --------------------------------------------------------

    (
        global_user_ids,
        train_user_ids,
        sequences,
        seen_items,
    ) = build_validation_sequences(
        warm_start,
        global_to_train_user,
        histories,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_model()

    # --------------------------------------------------------
    # LightGCN embeddings
    # --------------------------------------------------------

    (
        graph_users,
        graph_items,
    ) = load_lightgcn_embeddings()

    # --------------------------------------------------------
    # SASRec validation representation
    # --------------------------------------------------------

    sasrec_users = (
        build_sasrec_user_representations(
            model,
            sequences,
        )
    )

    # --------------------------------------------------------
    # Gather validation users from LightGCN
    # --------------------------------------------------------

    train_user_tensor = (
        torch.from_numpy(
            train_user_ids
        ).long()
    )

    validation_graph_users = (
        graph_users[
            train_user_tensor
        ]
    )

    if validation_graph_users.shape != (
        len(global_user_ids),
        EMBEDDING_DIM,
    ):

        raise RuntimeError(
            "Unexpected validation "
            "LightGCN user shape: "
            f"{tuple(validation_graph_users.shape)}"
        )

    # --------------------------------------------------------
    # V3-A user representation
    # --------------------------------------------------------

    hybrid_users = (
        build_v3a_user_representations(
            model,
            validation_graph_users,
            sasrec_users,
        )
    )

    # --------------------------------------------------------
    # V3-A item representation
    # --------------------------------------------------------

    hybrid_items = (
        build_v3a_item_representations(
            model,
            graph_items,
        )
    )

    # --------------------------------------------------------
    # Recommendations
    # --------------------------------------------------------

    recommendations = (
        generate_recommendations(
            global_user_ids,
            hybrid_users,
            hybrid_items,
            seen_items,
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("SAVING V3-A VALIDATION RECOMMENDATIONS")
    print("=" * 80)

    recommendations.to_parquet(
        OUTPUT_RECOMMENDATIONS_FILE,
        index=False,
    )

    elapsed = (
        time.time()
        - start_time
    )

    metadata = {
        "model": "Hybrid V3-A",
        "architecture": (
            "Adaptive graph-sequential hybrid"
        ),
        "training_negative_strategy": (
            "random negatives"
        ),
        "lightgcn_hard_negatives": False,
        "sasrec_hard_negatives": False,
        "num_training_users": NUM_USERS,
        "num_training_items": NUM_ITEMS,
        "embedding_dim": EMBEDDING_DIM,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "validation_protocol": (
            "warm-start user + warm-start item"
        ),
        "warm_start_interactions": int(
            len(warm_start)
        ),
        "warm_start_users": int(
            len(global_user_ids)
        ),
        "top_k": TOP_K,
        "recommendation_rows": int(
            len(recommendations)
        ),
        "unique_recommended_items": int(
            recommendations[
                "item_index"
            ].nunique()
        ),
        "trainable_parameters": 41_280,
        "total_parameters": 59_504_832,
        "lightgcn_frozen": True,
        "sasrec_frozen": True,
        "training_seen_items_excluded": True,
        "generation_time_seconds": float(
            elapsed
        ),
    }

    OUTPUT_METADATA_FILE.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Saved: "
        f"{OUTPUT_RECOMMENDATIONS_FILE}"
    )

    print(
        f"Saved: "
        f"{OUTPUT_METADATA_FILE}"
    )

    print()
    print("=" * 80)
    print("V3-A VALIDATION RECOMMENDATION GENERATION COMPLETE")
    print("=" * 80)

    print(
        f"Warm-start users: "
        f"{len(global_user_ids):,}"
    )

    print(
        f"Recommendation rows: "
        f"{len(recommendations):,}"
    )

    print(
        f"Unique recommended items: "
        f"{recommendations['item_index'].nunique():,}"
    )

    print(
        f"Total time: "
        f"{elapsed:.2f}s "
        f"({elapsed / 60:.2f} min)"
    )

    print()
    print(
        "V3-A VALIDATION RECOMMENDATION "
        "GENERATION PASSED"
    )


if __name__ == "__main__":
    main()