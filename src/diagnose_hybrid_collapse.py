from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hybrid_model import HybridLightGCNSASRec


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

TRAIN_PATH = DATA_DIR / "transactions_train.parquet"
VAL_PATH = DATA_DIR / "transactions_val.parquet"

TRAIN_USER_MAPPING_PATH = (
    DATA_DIR / "train_user_mapping.parquet"
)

TRAIN_ITEM_MAPPING_PATH = (
    DATA_DIR / "train_item_mapping.parquet"
)

HYBRID_MODEL_PATH = (
    OUTPUT_DIR / "hybrid_model.pt"
)

LIGHTGCN_USER_EMBEDDINGS_PATH = (
    OUTPUT_DIR / "lightgcn_user_embeddings.npy"
)

LIGHTGCN_ITEM_EMBEDDINGS_PATH = (
    OUTPUT_DIR / "lightgcn_item_embeddings.npy"
)

DEVICE = torch.device("cpu")

SEED = 42

# ------------------------------------------------------------
# Global/final dataset vocabulary
# ------------------------------------------------------------

GLOBAL_NUM_USERS = 797_782
GLOBAL_NUM_ITEMS = 79_026

# ------------------------------------------------------------
# Training-specific model vocabulary
# ------------------------------------------------------------

NUM_USERS = 778_421
NUM_ITEMS = 74_541

EMBEDDING_DIM = 64
MAX_SEQUENCE_LENGTH = 50

NUM_DIAGNOSTIC_USERS = 1_000
TOP_K = 20


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def check(condition, message):
    if not condition:
        raise AssertionError(message)

    print(f"[PASS] {message}")


def extract_last_valid_sasrec_representation(
    sasrec_output,
    sequences,
):
    """
    Extract the representation at the last non-padding
    position for each sequence.

    SASRec:
        padding ID = 0
        real item IDs = 1..NUM_ITEMS

    Parameters
    ----------
    sasrec_output : torch.Tensor
        Shape [batch_size, max_sequence_length, embedding_dim]

    sequences : torch.Tensor
        Shape [batch_size, max_sequence_length]

    Returns
    -------
    torch.Tensor
        Shape [batch_size, embedding_dim]
    """

    lengths = (
        sequences != 0
    ).sum(dim=1)

    check(
        torch.all(lengths > 0).item(),
        "All diagnostic sequences contain at least one real item",
    )

    last_positions = lengths - 1

    batch_indices = torch.arange(
        sasrec_output.shape[0],
        device=sasrec_output.device,
    )

    representations = sasrec_output[
        batch_indices,
        last_positions,
    ]

    return representations


def build_left_padded_sequence(
    item_ids,
    max_length,
):
    """
    Convert a training-item-ID history into a left-padded
    SASRec input sequence.

    Training item IDs:
        0 .. NUM_ITEMS-1

    SASRec item IDs:
        1 .. NUM_ITEMS

    Padding:
        0
    """

    if len(item_ids) == 0:
        raise ValueError(
            "Cannot build a sequence from an empty history."
        )

    item_ids = np.asarray(
        item_ids,
        dtype=np.int32,
    )

    check(
        item_ids.min() >= 0,
        "History contains no negative training item IDs",
    )

    check(
        item_ids.max() < NUM_ITEMS,
        "History training item IDs fit training vocabulary",
    )

    # Convert training item IDs to SASRec IDs.
    sasrec_ids = item_ids + 1

    # Keep only the most recent MAX_SEQUENCE_LENGTH items.
    sasrec_ids = sasrec_ids[-max_length:]

    sequence = np.zeros(
        max_length,
        dtype=np.int32,
    )

    sequence[-len(sasrec_ids):] = sasrec_ids

    return sequence


# ============================================================
# MAIN
# ============================================================

def main():

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ========================================================
    # 1. PATH VALIDATION
    # ========================================================

    print_section("1. PATH VALIDATION")

    required_files = [
        TRAIN_PATH,
        VAL_PATH,
        TRAIN_USER_MAPPING_PATH,
        TRAIN_ITEM_MAPPING_PATH,
        HYBRID_MODEL_PATH,
        LIGHTGCN_USER_EMBEDDINGS_PATH,
        LIGHTGCN_ITEM_EMBEDDINGS_PATH,
    ]

    for path in required_files:

        check(
            path.exists(),
            f"Required file exists: {path.name}",
        )


    # ========================================================
    # 2. LOAD DATA
    # ========================================================

    print_section("2. LOAD DATA")

    train = pd.read_parquet(
        TRAIN_PATH
    )

    val = pd.read_parquet(
        VAL_PATH
    )

    train_user_mapping = pd.read_parquet(
        TRAIN_USER_MAPPING_PATH
    )

    train_item_mapping = pd.read_parquet(
        TRAIN_ITEM_MAPPING_PATH
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
        f"Training users in mapping: "
        f"{len(train_user_mapping):,}"
    )

    print(
        f"Training items in mapping: "
        f"{len(train_item_mapping):,}"
    )


    # ========================================================
    # 3. GLOBAL ID VALIDATION
    # ========================================================

    print_section("3. GLOBAL ID VALIDATION")

    print(
        "Training global user_index range:",
        f"{train['user_index'].min()} -> "
        f"{train['user_index'].max()}",
    )

    print(
        "Training global item_index range:",
        f"{train['item_index'].min()} -> "
        f"{train['item_index'].max()}",
    )

    print(
        "Validation global user_index range:",
        f"{val['user_index'].min()} -> "
        f"{val['user_index'].max()}",
    )

    print(
        "Validation global item_index range:",
        f"{val['item_index'].min()} -> "
        f"{val['item_index'].max()}",
    )

    check(
        train["user_index"].min() >= 0,
        "Training global user IDs are non-negative",
    )

    check(
        train["user_index"].max() < GLOBAL_NUM_USERS,
        "Training global user IDs fit global vocabulary",
    )

    check(
        train["item_index"].min() >= 0,
        "Training global item IDs are non-negative",
    )

    check(
        train["item_index"].max() < GLOBAL_NUM_ITEMS,
        "Training global item IDs fit global vocabulary",
    )

    check(
        val["user_index"].min() >= 0,
        "Validation global user IDs are non-negative",
    )

    check(
        val["user_index"].max() < GLOBAL_NUM_USERS,
        "Validation global user IDs fit global vocabulary",
    )

    check(
        val["item_index"].min() >= 0,
        "Validation global item IDs are non-negative",
    )

    check(
        val["item_index"].max() < GLOBAL_NUM_ITEMS,
        "Validation global item IDs fit global vocabulary",
    )


    # ========================================================
    # 4. TRAINING MAPPING VALIDATION
    # ========================================================

    print_section(
        "4. TRAINING-SPECIFIC MAPPING VALIDATION"
    )

    print(
        "User mapping columns:",
        list(train_user_mapping.columns),
    )

    print(
        "Item mapping columns:",
        list(train_item_mapping.columns),
    )

    check(
        "user_index" in train_user_mapping.columns,
        "User mapping contains global user_index",
    )

    check(
        "train_user_id" in train_user_mapping.columns,
        "User mapping contains train_user_id",
    )

    check(
        "item_index" in train_item_mapping.columns,
        "Item mapping contains global item_index",
    )

    check(
        "train_item_id" in train_item_mapping.columns,
        "Item mapping contains train_item_id",
    )

    check(
        len(train_user_mapping) == NUM_USERS,
        "Training user mapping row count is correct",
    )

    check(
        len(train_item_mapping) == NUM_ITEMS,
        "Training item mapping row count is correct",
    )

    check(
        train_user_mapping["user_index"].is_unique,
        "Global user IDs in training mapping are unique",
    )

    check(
        train_user_mapping["train_user_id"].is_unique,
        "Training user IDs are unique",
    )

    check(
        train_item_mapping["item_index"].is_unique,
        "Global item IDs in training mapping are unique",
    )

    check(
        train_item_mapping["train_item_id"].is_unique,
        "Training item IDs are unique",
    )

    check(
        train_user_mapping["train_user_id"].min() == 0,
        "Training user IDs start at 0",
    )

    check(
        train_user_mapping["train_user_id"].max()
        == NUM_USERS - 1,
        "Training user IDs end at NUM_USERS - 1",
    )

    check(
        train_item_mapping["train_item_id"].min() == 0,
        "Training item IDs start at 0",
    )

    check(
        train_item_mapping["train_item_id"].max()
        == NUM_ITEMS - 1,
        "Training item IDs end at NUM_ITEMS - 1",
    )


    # ========================================================
    # 5. BUILD GLOBAL -> TRAINING LOOKUP ARRAYS
    # ========================================================

    print_section(
        "5. GLOBAL -> TRAINING ID CONVERSION"
    )

    global_to_train_user = np.full(
        GLOBAL_NUM_USERS,
        -1,
        dtype=np.int32,
    )

    global_to_train_item = np.full(
        GLOBAL_NUM_ITEMS,
        -1,
        dtype=np.int32,
    )

    global_user_ids = (
        train_user_mapping["user_index"]
        .to_numpy(dtype=np.int64)
    )

    train_user_ids = (
        train_user_mapping["train_user_id"]
        .to_numpy(dtype=np.int64)
    )

    global_item_ids = (
        train_item_mapping["item_index"]
        .to_numpy(dtype=np.int64)
    )

    train_item_ids = (
        train_item_mapping["train_item_id"]
        .to_numpy(dtype=np.int64)
    )

    check(
        global_user_ids.min() >= 0,
        "Mapped global user IDs are non-negative",
    )

    check(
        global_user_ids.max() < GLOBAL_NUM_USERS,
        "Mapped global user IDs fit global vocabulary",
    )

    check(
        global_item_ids.min() >= 0,
        "Mapped global item IDs are non-negative",
    )

    check(
        global_item_ids.max() < GLOBAL_NUM_ITEMS,
        "Mapped global item IDs fit global vocabulary",
    )

    global_to_train_user[
        global_user_ids
    ] = train_user_ids

    global_to_train_item[
        global_item_ids
    ] = train_item_ids

    check(
        np.all(
            global_to_train_user[
                global_user_ids
            ]
            == train_user_ids
        ),
        "Global user -> training user conversion is correct",
    )

    check(
        np.all(
            global_to_train_item[
                global_item_ids
            ]
            == train_item_ids
        ),
        "Global item -> training item conversion is correct",
    )

    print(
        "Training users represented in global vocabulary:",
        np.sum(global_to_train_user >= 0),
    )

    print(
        "Training items represented in global vocabulary:",
        np.sum(global_to_train_item >= 0),
    )


    # ========================================================
    # 6. TRAINING ID COVERAGE
    # ========================================================

    print_section(
        "6. TRAINING ID COVERAGE"
    )

    train_global_users = (
        train["user_index"]
        .to_numpy(dtype=np.int64)
    )

    train_global_items = (
        train["item_index"]
        .to_numpy(dtype=np.int64)
    )

    train_specific_users = (
        global_to_train_user[
            train_global_users
        ]
    )

    train_specific_items = (
        global_to_train_item[
            train_global_items
        ]
    )

    check(
        np.all(train_specific_users >= 0),
        "Every training interaction user maps to a training user ID",
    )

    check(
        np.all(train_specific_items >= 0),
        "Every training interaction item maps to a training item ID",
    )

    check(
        train_specific_users.max() < NUM_USERS,
        "Mapped training user IDs fit model vocabulary",
    )

    check(
        train_specific_items.max() < NUM_ITEMS,
        "Mapped training item IDs fit model vocabulary",
    )

    print(
        "Mapped training user range:",
        f"{train_specific_users.min()} -> "
        f"{train_specific_users.max()}",
    )

    print(
        "Mapped training item range:",
        f"{train_specific_items.min()} -> "
        f"{train_specific_items.max()}",
    )


    # ========================================================
    # 7. LOAD LIGHTGCN EMBEDDINGS
    # ========================================================

    print_section(
        "7. LOAD LIGHTGCN EMBEDDINGS"
    )

    lightgcn_user_embeddings = np.load(
        LIGHTGCN_USER_EMBEDDINGS_PATH,
        mmap_mode="r",
    )

    lightgcn_item_embeddings = np.load(
        LIGHTGCN_ITEM_EMBEDDINGS_PATH,
        mmap_mode="r",
    )

    print(
        "LightGCN user embedding shape:",
        lightgcn_user_embeddings.shape,
    )

    print(
        "LightGCN item embedding shape:",
        lightgcn_item_embeddings.shape,
    )

    check(
        lightgcn_user_embeddings.shape
        == (
            NUM_USERS,
            EMBEDDING_DIM,
        ),
        "LightGCN user embedding shape is correct",
    )

    check(
        lightgcn_item_embeddings.shape
        == (
            NUM_ITEMS,
            EMBEDDING_DIM,
        ),
        "LightGCN item embedding shape is correct",
    )

    check(
        np.isfinite(
            lightgcn_user_embeddings
        ).all(),
        "LightGCN user embeddings are finite",
    )

    check(
        np.isfinite(
            lightgcn_item_embeddings
        ).all(),
        "LightGCN item embeddings are finite",
    )


    # ========================================================
    # 8. LOAD HYBRID MODEL
    # ========================================================

    print_section(
        "8. LOAD HYBRID MODEL"
    )

    hybrid_checkpoint = torch.load(
        HYBRID_MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    print(
        "Hybrid checkpoint keys:",
        list(hybrid_checkpoint.keys()),
    )

    model = HybridLightGCNSASRec(
        num_users=NUM_USERS,
        num_items=NUM_ITEMS,
        embedding_dim=EMBEDDING_DIM,
        max_seq_len=MAX_SEQUENCE_LENGTH,
        num_heads=2,
        num_layers=2,
        dropout=0.2,
    )

    model.load_state_dict(
        hybrid_checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(DEVICE)

    # Freeze pretrained encoders.
    model.freeze_pretrained_encoders()

    # IMPORTANT:
    # The entire hybrid model is put into evaluation mode.
    #
    # This disables Dropout in the user fusion network.
    #
    # Without this, two mathematically identical representation
    # calculations can receive different random dropout masks,
    # producing a false API/manual mismatch.
    model.eval()

    # Explicitly keep encoders in eval mode as well.
    model.lightgcn.eval()
    model.sasrec.eval()

    check(
        model.lightgcn.num_users == NUM_USERS,
        "Hybrid LightGCN user vocabulary is correct",
    )

    check(
        model.lightgcn.num_items == NUM_ITEMS,
        "Hybrid LightGCN item vocabulary is correct",
    )

    check(
        model.sasrec.item_embedding.num_embeddings
        == NUM_ITEMS + 1,
        "SASRec item embedding includes padding row",
    )

    check(
        model.sasrec.position_embedding.num_embeddings
        == MAX_SEQUENCE_LENGTH,
        "SASRec maximum sequence length is 50",
    )

    print(
        "Hybrid model loaded successfully."
    )


    # ========================================================
    # 9. PARAMETER AND FREEZING VALIDATION
    # ========================================================

    print_section(
        "9. PARAMETER AND FREEZING VALIDATION"
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Total parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    check(
        trainable_parameters == 16_512,
        "Only the 16,512-parameter user fusion network is trainable",
    )

    lightgcn_trainable = any(
        parameter.requires_grad
        for parameter in model.lightgcn.parameters()
    )

    sasrec_trainable = any(
        parameter.requires_grad
        for parameter in model.sasrec.parameters()
    )

    check(
        not lightgcn_trainable,
        "LightGCN parameters are frozen",
    )

    check(
        not sasrec_trainable,
        "SASRec parameters are frozen",
    )


    # ========================================================
    # 10. VALIDATION WARM-START ANALYSIS
    # ========================================================

    print_section(
        "10. VALIDATION WARM-START ANALYSIS"
    )

    train_user_set = set(
        train["user_index"].unique()
    )

    train_item_set = set(
        train["item_index"].unique()
    )

    val_user_known_mask = (
        val["user_index"].isin(
            train_user_set
        )
    )

    val_item_known_mask = (
        val["item_index"].isin(
            train_item_set
        )
    )

    warm_start_mask = (
        val_user_known_mask
        & val_item_known_mask
    )

    warm_val = val.loc[
        warm_start_mask
    ].copy()

    print(
        f"Validation interactions: "
        f"{len(val):,}"
    )

    print(
        f"User-known validation interactions: "
        f"{val_user_known_mask.sum():,}"
    )

    print(
        f"Item-known validation interactions: "
        f"{val_item_known_mask.sum():,}"
    )

    print(
        f"Warm-start validation interactions: "
        f"{len(warm_val):,}"
    )

    warm_users_global = (
        warm_val["user_index"]
        .drop_duplicates()
        .to_numpy(dtype=np.int64)
    )

    check(
        len(warm_users_global) > 0,
        "Warm-start validation users exist",
    )

    check(
        np.all(
            global_to_train_user[
                warm_users_global
            ] >= 0
        ),
        "Every warm-start validation user maps to a training user ID",
    )

    warm_train_user_ids = (
        global_to_train_user[
            warm_users_global
        ]
    )

    check(
        warm_train_user_ids.min() >= 0,
        "Warm-start training user IDs are non-negative",
    )

    check(
        warm_train_user_ids.max() < NUM_USERS,
        "Warm-start training user IDs fit model vocabulary",
    )

    print(
        f"Warm-start users: "
        f"{len(warm_users_global):,}"
    )

    print(
        "Warm-start global user range:",
        f"{warm_users_global.min()} -> "
        f"{warm_users_global.max()}",
    )

    print(
        "Warm-start training user range:",
        f"{warm_train_user_ids.min()} -> "
        f"{warm_train_user_ids.max()}",
    )


    # ========================================================
    # 11. BUILD TRAINING HISTORIES
    # ========================================================

    print_section(
        "11. BUILD TRAINING HISTORIES"
    )

    # --------------------------------------------------------
    # IMPORTANT ID RULE
    # --------------------------------------------------------
    #
    # transactions_train.parquet:
    #
    #     user_index = GLOBAL USER ID
    #     item_index = GLOBAL ITEM ID
    #
    # Hybrid model:
    #
    #     user_id = TRAIN USER ID
    #     item_id = TRAIN ITEM ID
    #
    # SASRec:
    #
    #     padding = 0
    #     item IDs = train_item_id + 1
    #
    # Therefore:
    #
    # global user_index
    #       ↓
    # train_user_id
    #
    # global item_index
    #       ↓
    # train_item_id
    #       ↓
    # + 1
    #       ↓
    # SASRec item ID
    #
    # item_index is used as a deterministic same-day tie-breaker.
    # --------------------------------------------------------

    train_sorted = train.sort_values(
        [
            "user_index",
            "t_dat",
            "item_index",
        ]
    )

    history_by_global_user = {}

    for global_user_id, group in train_sorted.groupby(
        "user_index",
        sort=False,
    ):

        global_history = (
            group["item_index"]
            .to_numpy(dtype=np.int64)
        )

        training_history = (
            global_to_train_item[
                global_history
            ]
        )

        check(
            np.all(training_history >= 0),
            "Training history contains only training-vocabulary items",
        )

        history_by_global_user[
            int(global_user_id)
        ] = training_history.astype(
            np.int32,
            copy=False,
        )

    diagnostic_users = warm_users_global[
        :NUM_DIAGNOSTIC_USERS
    ]

    diagnostic_train_user_ids = (
        global_to_train_user[
            diagnostic_users
        ]
    )

    sequences = []

    sequence_lengths = []

    for global_user_id in diagnostic_users:

        history = history_by_global_user[
            int(global_user_id)
        ]

        sequence_lengths.append(
            min(
                len(history),
                MAX_SEQUENCE_LENGTH,
            )
        )

        sequence = build_left_padded_sequence(
            history,
            MAX_SEQUENCE_LENGTH,
        )

        sequences.append(
            sequence
        )

    sequences = np.stack(
        sequences
    ).astype(
        np.int32,
        copy=False,
    )

    sequence_lengths = np.asarray(
        sequence_lengths,
        dtype=np.int32,
    )

    print(
        "Diagnostic sequence shape:",
        sequences.shape,
    )

    print(
        "Minimum sequence length:",
        sequence_lengths.min(),
    )

    print(
        "Maximum sequence length:",
        sequence_lengths.max(),
    )

    print(
        "Mean sequence length:",
        sequence_lengths.mean(),
    )

    check(
        sequences.shape
        == (
            len(diagnostic_users),
            MAX_SEQUENCE_LENGTH,
        ),
        "Diagnostic sequence shape is correct",
    )

    check(
        sequences.min() >= 0,
        "SASRec sequence IDs are non-negative",
    )

    check(
        sequences.max() <= NUM_ITEMS,
        "SASRec sequence IDs fit 1..NUM_ITEMS",
    )


    # ========================================================
    # 12. SASRec PADDING VALIDATION
    # ========================================================

    print_section(
        "12. SASREC PADDING VALIDATION"
    )

    for row_index in range(
        min(
            10,
            len(sequences),
        )
    ):

        sequence = sequences[
            row_index
        ]

        non_zero_positions = np.where(
            sequence != 0
        )[0]

        check(
            len(non_zero_positions) > 0,
            f"Diagnostic sequence {row_index} has a real item",
        )

        first_real_position = (
            non_zero_positions[0]
        )

        if first_real_position > 0:

            check(
                np.all(
                    sequence[
                        :first_real_position
                    ]
                    == 0
                ),
                f"Sequence {row_index} uses left padding",
            )

        check(
            np.all(
                sequence[
                    first_real_position:
                ]
                != 0
            ),
            f"Sequence {row_index} contains no internal padding",
        )


    # ========================================================
    # 13. SASRec REPRESENTATION CONSISTENCY
    # ========================================================

    print_section(
        "13. SASREC REPRESENTATION CONSISTENCY"
    )

    sequence_tensor = (
        torch.from_numpy(
            sequences.copy()
        )
        .long()
        .to(DEVICE)
    )

    diagnostic_train_user_tensor = (
        torch.from_numpy(
            diagnostic_train_user_ids.copy()
        )
        .long()
        .to(DEVICE)
    )

    # Copy the mmap array before converting to a tensor.
    # This avoids the NumPy non-writable-array warning.
    full_lightgcn_users_tensor = (
        torch.from_numpy(
            np.asarray(
                lightgcn_user_embeddings
            ).copy()
        )
        .float()
        .to(DEVICE)
    )

    with torch.no_grad():

        # --------------------------------------------
        # Direct SASRec forward pass
        # --------------------------------------------

        direct_sasrec_output = model.sasrec(
            sequence_tensor
        )

        direct_sasrec_representation = (
            extract_last_valid_sasrec_representation(
                direct_sasrec_output,
                sequence_tensor,
            )
        )

        # --------------------------------------------
        # Hybrid API representation
        # --------------------------------------------

        api_hybrid_user = (
            model.get_user_representation(
                user_ids=diagnostic_train_user_tensor,
                sequences=sequence_tensor,
                lightgcn_user_embeddings=(
                    full_lightgcn_users_tensor
                ),
            )
        )

    check(
        direct_sasrec_output.shape
        == (
            len(diagnostic_users),
            MAX_SEQUENCE_LENGTH,
            EMBEDDING_DIM,
        ),
        "Direct SASRec output shape is correct",
    )

    check(
        direct_sasrec_representation.shape
        == (
            len(diagnostic_users),
            EMBEDDING_DIM,
        ),
        "Direct SASRec representation shape is correct",
    )

    check(
        torch.isfinite(
            direct_sasrec_representation
        ).all().item(),
        "Direct SASRec representations are finite",
    )


    # ========================================================
    # 14. MANUAL HYBRID USER RECONSTRUCTION
    # ========================================================

    print_section(
        "14. HYBRID USER REPRESENTATION RECONSTRUCTION"
    )

    with torch.no_grad():

        diagnostic_lightgcn_users = (
            full_lightgcn_users_tensor[
                diagnostic_train_user_tensor
            ]
        )

        manual_concat = torch.cat(
            [
                diagnostic_lightgcn_users,
                direct_sasrec_representation,
            ],
            dim=1,
        )

        manual_hybrid_user = (
            model.user_fusion(
                manual_concat
            )
        )

    check(
        manual_concat.shape
        == (
            len(diagnostic_users),
            128,
        ),
        "Manual hybrid user input is 128-dimensional",
    )

    check(
        manual_hybrid_user.shape
        == (
            len(diagnostic_users),
            128,
        ),
        "Manual hybrid user representation is 128-dimensional",
    )

    check(
        torch.isfinite(
            manual_hybrid_user
        ).all().item(),
        "Manual hybrid user representations are finite",
    )

    representation_difference = (
        api_hybrid_user
        - manual_hybrid_user
    ).abs()

    max_representation_difference = (
        representation_difference.max().item()
    )

    mean_representation_difference = (
        representation_difference.mean().item()
    )

    print()
    print(
        "API vs manual hybrid representation:"
    )

    print(
        f"Maximum absolute difference: "
        f"{max_representation_difference:.12f}"
    )

    print(
        f"Mean absolute difference: "
        f"{mean_representation_difference:.12f}"
    )

    check(
        max_representation_difference <= 1e-6,
        "Hybrid API representation matches manual reconstruction",
    )


    # ========================================================
    # 15. REPRESENTATION SCALE DIAGNOSTICS
    # ========================================================

    print_section(
        "15. REPRESENTATION SCALE DIAGNOSTICS"
    )

    with torch.no_grad():

        lightgcn_user_norms = (
            torch.linalg.norm(
                diagnostic_lightgcn_users,
                dim=1,
            )
        )

        sasrec_user_norms = (
            torch.linalg.norm(
                direct_sasrec_representation,
                dim=1,
            )
        )

        hybrid_user_norms = (
            torch.linalg.norm(
                manual_hybrid_user,
                dim=1,
            )
        )

        # Inspect the first 10,000 items.
        item_sample_size = min(
            10_000,
            NUM_ITEMS,
        )

        lightgcn_item_sample = (
            torch.from_numpy(
                np.asarray(
                    lightgcn_item_embeddings[
                        :item_sample_size
                    ]
                ).copy()
            )
            .float()
            .to(DEVICE)
        )

        sasrec_item_embeddings = (
            model.sasrec
            .item_embedding
            .weight[
                1:
            ]
        )

        sasrec_item_sample = (
            sasrec_item_embeddings[
                :item_sample_size
            ]
        )

        hybrid_item_sample = torch.cat(
            [
                lightgcn_item_sample,
                sasrec_item_sample,
            ],
            dim=1,
        )

        lightgcn_item_norms = (
            torch.linalg.norm(
                lightgcn_item_sample,
                dim=1,
            )
        )

        sasrec_item_norms = (
            torch.linalg.norm(
                sasrec_item_sample,
                dim=1,
            )
        )

        hybrid_item_norms = (
            torch.linalg.norm(
                hybrid_item_sample,
                dim=1,
            )
        )

    print()
    print("USER NORMS")

    print(
        f"LightGCN user mean norm: "
        f"{lightgcn_user_norms.mean().item():.8f}"
    )

    print(
        f"SASRec user mean norm: "
        f"{sasrec_user_norms.mean().item():.8f}"
    )

    print(
        f"Hybrid user mean norm: "
        f"{hybrid_user_norms.mean().item():.8f}"
    )

    print(
        f"LightGCN user median norm: "
        f"{lightgcn_user_norms.median().item():.8f}"
    )

    print(
        f"SASRec user median norm: "
        f"{sasrec_user_norms.median().item():.8f}"
    )

    print(
        f"Hybrid user median norm: "
        f"{hybrid_user_norms.median().item():.8f}"
    )

    print(
        f"LightGCN user std norm: "
        f"{lightgcn_user_norms.std().item():.8f}"
    )

    print(
        f"SASRec user std norm: "
        f"{sasrec_user_norms.std().item():.8f}"
    )

    print(
        f"Hybrid user std norm: "
        f"{hybrid_user_norms.std().item():.8f}"
    )

    print()
    print("ITEM NORMS")

    print(
        f"LightGCN item mean norm: "
        f"{lightgcn_item_norms.mean().item():.8f}"
    )

    print(
        f"SASRec item mean norm: "
        f"{sasrec_item_norms.mean().item():.8f}"
    )

    print(
        f"Hybrid item mean norm: "
        f"{hybrid_item_norms.mean().item():.8f}"
    )

    print(
        f"LightGCN item median norm: "
        f"{lightgcn_item_norms.median().item():.8f}"
    )

    print(
        f"SASRec item median norm: "
        f"{sasrec_item_norms.median().item():.8f}"
    )

    print(
        f"Hybrid item median norm: "
        f"{hybrid_item_norms.median().item():.8f}"
    )


    # ========================================================
    # 16. REPRESENTATION VARIANCE
    # ========================================================

    print_section(
        "16. REPRESENTATION VARIANCE"
    )

    lightgcn_user_variance = (
        diagnostic_lightgcn_users
        .var(
            dim=0,
            unbiased=False,
        )
        .mean()
        .item()
    )

    sasrec_user_variance = (
        direct_sasrec_representation
        .var(
            dim=0,
            unbiased=False,
        )
        .mean()
        .item()
    )

    hybrid_user_variance = (
        manual_hybrid_user
        .var(
            dim=0,
            unbiased=False,
        )
        .mean()
        .item()
    )

    lightgcn_item_variance = (
        lightgcn_item_sample
        .var(
            dim=0,
            unbiased=False,
        )
        .mean()
        .item()
    )

    sasrec_item_variance = (
        sasrec_item_sample
        .var(
            dim=0,
            unbiased=False,
        )
        .mean()
        .item()
    )

    hybrid_item_variance = (
        hybrid_item_sample
        .var(
            dim=0,
            unbiased=False,
        )
        .mean()
        .item()
    )

    print(
        f"LightGCN user variance: "
        f"{lightgcn_user_variance:.10f}"
    )

    print(
        f"SASRec user variance: "
        f"{sasrec_user_variance:.10f}"
    )

    print(
        f"Hybrid user variance: "
        f"{hybrid_user_variance:.10f}"
    )

    print(
        f"LightGCN item variance: "
        f"{lightgcn_item_variance:.10f}"
    )

    print(
        f"SASRec item variance: "
        f"{sasrec_item_variance:.10f}"
    )

    print(
        f"Hybrid item variance: "
        f"{hybrid_item_variance:.10f}"
    )


    # ========================================================
    # 17. HYBRID USER COSINE SIMILARITY
    # ========================================================

    print_section(
        "17. HYBRID USER COSINE SIMILARITY"
    )

    normalized_users = (
        torch.nn.functional.normalize(
            manual_hybrid_user,
            p=2,
            dim=1,
        )
    )

    cosine_matrix = (
        normalized_users
        @ normalized_users.T
    )

    upper_triangle_indices = (
        torch.triu_indices(
            cosine_matrix.shape[0],
            cosine_matrix.shape[1],
            offset=1,
        )
    )

    pairwise_cosines = cosine_matrix[
        upper_triangle_indices[0],
        upper_triangle_indices[1],
    ]

    print(
        f"Pairwise hybrid user cosine mean: "
        f"{pairwise_cosines.mean().item():.8f}"
    )

    print(
        f"Pairwise hybrid user cosine std: "
        f"{pairwise_cosines.std().item():.8f}"
    )

    print(
        f"Pairwise hybrid user cosine min: "
        f"{pairwise_cosines.min().item():.8f}"
    )

    print(
        f"Pairwise hybrid user cosine max: "
        f"{pairwise_cosines.max().item():.8f}"
    )


    # ========================================================
    # 18. LIGHTGCN VS SASRec USER ALIGNMENT
    # ========================================================

    print_section(
        "18. LIGHTGCN VS SASREC USER ALIGNMENT"
    )

    normalized_lightgcn_users = (
        torch.nn.functional.normalize(
            diagnostic_lightgcn_users,
            p=2,
            dim=1,
        )
    )

    normalized_sasrec_users = (
        torch.nn.functional.normalize(
            direct_sasrec_representation,
            p=2,
            dim=1,
        )
    )

    component_cosines = (
        normalized_lightgcn_users
        * normalized_sasrec_users
    ).sum(dim=1)

    print(
        f"LightGCN/SASRec user cosine mean: "
        f"{component_cosines.mean().item():.8f}"
    )

    print(
        f"LightGCN/SASRec user cosine std: "
        f"{component_cosines.std().item():.8f}"
    )

    print(
        f"LightGCN/SASRec user cosine min: "
        f"{component_cosines.min().item():.8f}"
    )

    print(
        f"LightGCN/SASRec user cosine max: "
        f"{component_cosines.max().item():.8f}"
    )


    # ========================================================
    # 19. FUSION WEIGHT DIAGNOSTICS
    # ========================================================

    print_section(
        "19. FUSION WEIGHT DIAGNOSTICS"
    )

    fusion_weight = (
        model.user_fusion[0]
        .weight
        .detach()
    )

    fusion_bias = (
        model.user_fusion[0]
        .bias
        .detach()
    )

    print(
        "Fusion weight shape:",
        tuple(fusion_weight.shape),
    )

    print(
        f"Fusion weight min: "
        f"{fusion_weight.min().item():.8f}"
    )

    print(
        f"Fusion weight max: "
        f"{fusion_weight.max().item():.8f}"
    )

    print(
        f"Fusion weight mean: "
        f"{fusion_weight.mean().item():.8f}"
    )

    print(
        f"Fusion weight std: "
        f"{fusion_weight.std().item():.8f}"
    )

    print(
        f"Fusion bias min: "
        f"{fusion_bias.min().item():.8f}"
    )

    print(
        f"Fusion bias max: "
        f"{fusion_bias.max().item():.8f}"
    )

    print(
        f"Fusion bias mean: "
        f"{fusion_bias.mean().item():.8f}"
    )

    print(
        f"Fusion bias std: "
        f"{fusion_bias.std().item():.8f}"
    )


    # ========================================================
    # 20. FULL CATALOGUE ITEM REPRESENTATIONS
    # ========================================================

    print_section(
        "20. FULL CATALOGUE ITEM REPRESENTATIONS"
    )

    full_lightgcn_items = (
        torch.from_numpy(
            np.asarray(
                lightgcn_item_embeddings
            ).copy()
        )
        .float()
        .to(DEVICE)
    )

    all_item_ids = torch.arange(
        NUM_ITEMS,
        dtype=torch.long,
        device=DEVICE,
    )

    with torch.no_grad():

        full_hybrid_items = (
            model.get_item_representation(
                item_ids=all_item_ids,
                lightgcn_item_embeddings=(
                    full_lightgcn_items
                ),
            )
        )

    check(
        full_hybrid_items.shape
        == (
            NUM_ITEMS,
            128,
        ),
        "Full hybrid item representation shape is 128D",
    )

    check(
        torch.isfinite(
            full_hybrid_items
        ).all().item(),
        "Full hybrid item representations are finite",
    )

    print(
        "Full hybrid item representation shape:",
        tuple(
            full_hybrid_items.shape
        ),
    )


    # ========================================================
    # 21. FULL CATALOGUE SCORING
    # ========================================================

    print_section(
        "21. FULL CATALOGUE SCORING"
    )

    with torch.no_grad():

        scores = (
            manual_hybrid_user
            @ full_hybrid_items.T
        )

    check(
        scores.shape
        == (
            len(diagnostic_users),
            NUM_ITEMS,
        ),
        "Full-catalogue score matrix shape is correct",
    )

    check(
        torch.isfinite(
            scores
        ).all().item(),
        "Full-catalogue scores are finite",
    )

    print(
        "Score matrix shape:",
        tuple(scores.shape),
    )


    # ========================================================
    # 22. INSPECT ONE USER
    # ========================================================

    print_section(
        "22. INSPECTION USER SCORE DISTRIBUTION"
    )

    inspection_scores = scores[0]

    print(
        f"Global user ID: "
        f"{diagnostic_users[0]}"
    )

    print(
        f"Training user ID: "
        f"{diagnostic_train_user_ids[0]}"
    )

    print(
        f"Score mean: "
        f"{inspection_scores.mean().item():.8f}"
    )

    print(
        f"Score std: "
        f"{inspection_scores.std().item():.8f}"
    )

    print(
        f"Score min: "
        f"{inspection_scores.min().item():.8f}"
    )

    print(
        f"Score median: "
        f"{inspection_scores.median().item():.8f}"
    )

    print(
        f"Score max: "
        f"{inspection_scores.max().item():.8f}"
    )


    # ========================================================
    # 23. TOP-K RECOMMENDATION CONCENTRATION
    # ========================================================

    print_section(
        "23. RECOMMENDATION CONCENTRATION"
    )

    top_scores, top_items = torch.topk(
        scores,
        k=TOP_K,
        dim=1,
    )

    top_items_np = (
        top_items
        .cpu()
        .numpy()
    )

    top_scores_np = (
        top_scores
        .cpu()
        .numpy()
    )

    check(
        top_items_np.shape
        == (
            len(diagnostic_users),
            TOP_K,
        ),
        "Top-K recommendation shape is correct",
    )

    check(
        np.all(
            top_items_np >= 0
        ),
        "Recommended item IDs are non-negative",
    )

    check(
        np.all(
            top_items_np < NUM_ITEMS
        ),
        "Recommended item IDs fit training item vocabulary",
    )

    # --------------------------------------------------------
    # Inspection user top-20
    # --------------------------------------------------------

    print(
        "\nTop 20 items for inspection user:"
    )

    for rank in range(TOP_K):

        item_id = int(
            top_items_np[
                0,
                rank,
            ]
        )

        score_value = float(
            top_scores_np[
                0,
                rank,
            ]
        )

        print(
            f"Rank {rank + 1:02d}: "
            f"train_item_id={item_id:<6d} "
            f"score={score_value:.8f}"
        )


    # --------------------------------------------------------
    # Unique recommended items
    # --------------------------------------------------------

    flattened_items = (
        top_items_np.reshape(-1)
    )

    unique_items = np.unique(
        flattened_items
    )

    recommendation_coverage = (
        len(unique_items)
        / NUM_ITEMS
    )

    print()

    print(
        f"Diagnostic users: "
        f"{len(diagnostic_users):,}"
    )

    print(
        f"Total recommendations: "
        f"{len(flattened_items):,}"
    )

    print(
        f"Unique recommended items: "
        f"{len(unique_items):,}"
    )

    print(
        f"Recommendation coverage: "
        f"{recommendation_coverage:.8f}"
    )


    # ========================================================
    # 24. RANK-WISE UNIQUE ITEMS
    # ========================================================

    print_section(
        "24. RANK-WISE UNIQUE RECOMMENDATIONS"
    )

    for rank in range(TOP_K):

        unique_at_rank = np.unique(
            top_items_np[
                :,
                rank,
            ]
        )

        print(
            f"Rank {rank + 1:02d}: "
            f"{len(unique_at_rank):,} unique items"
        )


    # ========================================================
    # 25. TOP RECOMMENDED ITEM FREQUENCIES
    # ========================================================

    print_section(
        "25. TOP RECOMMENDED ITEM FREQUENCIES"
    )

    item_frequency = {}

    for item_id in flattened_items:

        item_id = int(
            item_id
        )

        item_frequency[item_id] = (
            item_frequency.get(
                item_id,
                0,
            )
            + 1
        )

    sorted_frequency = sorted(
        item_frequency.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    print(
        "Top 30 recommended training item IDs:"
    )

    for item_id, frequency in (
        sorted_frequency[:30]
    ):

        percentage = (
            frequency
            / len(flattened_items)
            * 100
        )

        print(
            f"item {item_id:<6d} "
            f"frequency={frequency:<5d} "
            f"share={percentage:.4f}%"
        )


    # ========================================================
    # 26. TOP-N CONCENTRATION
    # ========================================================

    print_section(
        "26. TOP-N ITEM CONCENTRATION"
    )

    frequencies = np.asarray(
        [
            frequency
            for _, frequency
            in sorted_frequency
        ],
        dtype=np.int64,
    )

    total_recommendations = (
        len(flattened_items)
    )

    for n in [
        10,
        20,
        50,
        100,
    ]:

        top_n_count = min(
            n,
            len(frequencies),
        )

        share = (
            frequencies[
                :top_n_count
            ].sum()
            / total_recommendations
        )

        print(
            f"Top {n:<3d} items account for "
            f"{share * 100:.4f}% of all recommendations"
        )


    # ========================================================
    # 27. PAIRWISE TOP-K JACCARD
    # ========================================================

    print_section(
        "27. TOP-K LIST SIMILARITY"
    )

    recommendation_sets = [
        set(
            row.tolist()
        )
        for row in top_items_np
    ]

    jaccard_values = []

    number_of_sets = len(
        recommendation_sets
    )

    for i in range(
        number_of_sets
    ):

        set_i = (
            recommendation_sets[i]
        )

        for j in range(
            i + 1,
            number_of_sets,
        ):

            set_j = (
                recommendation_sets[j]
            )

            intersection = len(
                set_i & set_j
            )

            union = len(
                set_i | set_j
            )

            if union > 0:

                jaccard_values.append(
                    intersection
                    / union
                )

    jaccard_values = np.asarray(
        jaccard_values,
        dtype=np.float32,
    )

    print(
        f"Average pairwise top-{TOP_K} Jaccard: "
        f"{jaccard_values.mean():.8f}"
    )

    print(
        f"Top-{TOP_K} Jaccard std: "
        f"{jaccard_values.std():.8f}"
    )

    print(
        f"Top-{TOP_K} Jaccard min: "
        f"{jaccard_values.min():.8f}"
    )

    print(
        f"Top-{TOP_K} Jaccard max: "
        f"{jaccard_values.max():.8f}"
    )


    # ========================================================
    # 28. ID SEMANTIC CONSISTENCY CHECK
    # ========================================================

    print_section(
        "28. ID SEMANTIC CONSISTENCY CHECK"
    )

    sample_global_user = int(
        diagnostic_users[0]
    )

    sample_train_user = int(
        diagnostic_train_user_ids[0]
    )

    sample_global_history = (
        train.loc[
            train["user_index"]
            == sample_global_user,
            "item_index",
        ]
        .to_numpy(dtype=np.int64)
    )

    sample_train_history = (
        global_to_train_item[
            sample_global_history
        ]
    )

    check(
        np.all(
            sample_train_history >= 0
        ),
        "Sample user history converts from global to training item IDs",
    )

    sample_sasrec_history = (
        sample_train_history + 1
    )

    check(
        sample_sasrec_history.min() >= 1,
        "Converted SASRec history starts at ID 1 or higher",
    )

    check(
        sample_sasrec_history.max()
        <= NUM_ITEMS,
        "Converted SASRec history fits SASRec item vocabulary",
    )

    print(
        f"Sample global user ID: "
        f"{sample_global_user}"
    )

    print(
        f"Sample training user ID: "
        f"{sample_train_user}"
    )

    print(
        "Sample global item history:",
        sample_global_history[
            :20
        ].tolist(),
    )

    print(
        "Sample training item history:",
        sample_train_history[
            :20
        ].tolist(),
    )

    print(
        "Sample SASRec item history:",
        sample_sasrec_history[
            :20
        ].tolist(),
    )


    # ========================================================
    # 29. FINAL SUMMARY
    # ========================================================

    print_section(
        "29. DIAGNOSTIC SUMMARY"
    )

    print(
        "ID SYSTEMS"
    )

    print(
        f"Global users:             "
        f"{GLOBAL_NUM_USERS:,}"
    )

    print(
        f"Global items:             "
        f"{GLOBAL_NUM_ITEMS:,}"
    )

    print(
        f"Training users:           "
        f"{NUM_USERS:,}"
    )

    print(
        f"Training items:           "
        f"{NUM_ITEMS:,}"
    )

    print()

    print(
        "Global -> training mapping:"
    )

    print(
        "global user_index -> train_user_id"
    )

    print(
        "global item_index -> train_item_id -> SASRec ID (+1)"
    )

    print()

    print(
        "WARM-START VALIDATION"
    )

    print(
        f"Warm-start users:         "
        f"{len(warm_users_global):,}"
    )

    print(
        f"Diagnostic users:         "
        f"{len(diagnostic_users):,}"
    )

    print()

    print(
        "REPRESENTATIONS"
    )

    print(
        f"LightGCN user dimension:  "
        f"{EMBEDDING_DIM}"
    )

    print(
        f"SASRec user dimension:    "
        f"{EMBEDDING_DIM}"
    )

    print(
        "Hybrid user dimension:    128"
    )

    print(
        "Hybrid item dimension:    128"
    )

    print()

    print(
        "RECOMMENDATION CONCENTRATION"
    )

    print(
        f"Unique recommended items: "
        f"{len(unique_items):,}"
    )

    print(
        f"Coverage: "
        f"{recommendation_coverage:.8f}"
    )

    print(
        f"Average top-{TOP_K} Jaccard: "
        f"{jaccard_values.mean():.8f}"
    )

    print()

    print(
        "REPRESENTATION CONSISTENCY"
    )

    print(
        f"API/manual max difference: "
        f"{max_representation_difference:.12f}"
    )

    print(
        f"API/manual mean difference: "
        f"{mean_representation_difference:.12f}"
    )

    print()

    print(
        "MODEL MODE"
    )

    print(
        f"Hybrid model training mode: "
        f"{model.training}"
    )

    print(
        f"LightGCN training mode: "
        f"{model.lightgcn.training}"
    )

    print(
        f"SASRec training mode: "
        f"{model.sasrec.training}"
    )

    check(
        not model.training,
        "Hybrid model is in evaluation mode",
    )

    check(
        not model.lightgcn.training,
        "LightGCN is in evaluation mode",
    )

    check(
        not model.sasrec.training,
        "SASRec is in evaluation mode",
    )

    print()

    print(
        "HYBRID COLLAPSE DIAGNOSTIC COMPLETED"
    )

    print(
        "No model weights were modified."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()