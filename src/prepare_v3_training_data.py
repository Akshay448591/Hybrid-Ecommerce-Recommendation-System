"""
prepare_v3_training_data.py

Prepare training data for Hybrid V3:

    V3-A: Adaptive gate + random negatives
    V3-B: Projection + hard negatives
    V3:   Adaptive gate + hard negatives

Training examples follow strict next-item semantics:

    prefix history -> positive target

Important:
- Repeated user-item interactions are preserved.
- A positive target IS allowed to appear earlier in the prefix.
- Negative items must not appear anywhere in the user's complete
  training history.
- SASRec uses padding ID 0, so actual item IDs are shifted by +1
  inside stored sequences.
- LightGCN and SASRec hard negatives are generated from the
  training-item vocabulary only.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

TRAIN_FILE = DATA_DIR / "transactions_train_model_ids.parquet"

LIGHTGCN_MODEL_FILE = OUTPUT_DIR / "lightgcn_model.pt"
LIGHTGCN_USER_EMBEDDINGS_FILE = OUTPUT_DIR / "lightgcn_user_embeddings.npy"
LIGHTGCN_ITEM_EMBEDDINGS_FILE = OUTPUT_DIR / "lightgcn_item_embeddings.npy"

SASREC_MODEL_FILE = OUTPUT_DIR / "sasrec_model.pt"

OUTPUT_USERS = OUTPUT_DIR / "v3_train_users.npy"
OUTPUT_POSITIVE = OUTPUT_DIR / "v3_train_positive_items.npy"
OUTPUT_RANDOM_NEGATIVE = OUTPUT_DIR / "v3_train_random_negative_items.npy"
OUTPUT_LIGHTGCN_NEGATIVE = OUTPUT_DIR / "v3_train_lightgcn_hard_negative_items.npy"
OUTPUT_SASREC_NEGATIVE = OUTPUT_DIR / "v3_train_sasrec_hard_negative_items.npy"
OUTPUT_SEQUENCES = OUTPUT_DIR / "v3_train_sequences.npy"
OUTPUT_METADATA = OUTPUT_DIR / "v3_training_data_metadata.json"

# Benchmark first with 50_000.
# After validation passes, change to 2_000_000.
NUM_EXAMPLES = 2_000_000

HARD_NEGATIVE_TOP_K = 100

MAX_SEQUENCE_LENGTH = 50

EMBEDDING_DIM = 64

SASREC_REP_BATCH_SIZE = 512
HARD_NEGATIVE_QUERY_BATCH_SIZE = 2048

SEED = 42

NUM_USERS = 778_421
NUM_ITEMS = 74_541

DEVICE = torch.device("cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# UTILITY
# ============================================================

def print_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def ensure_output_directory() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_numpy_copy(path: Path) -> np.ndarray:
    """
    Load an ndarray and explicitly copy it so that the resulting
    array is writable.

    This avoids PyTorch warnings when arrays originate from mmap.
    """
    return np.array(np.load(path), copy=True)


# ============================================================
# OPTIONAL FAISS BACKEND
# ============================================================

def get_faiss():
    try:
        import faiss

        print("FAISS detected. Using exact inner-product search.")
        return faiss

    except ImportError:
        print("FAISS not installed.")
        print("Falling back to batched PyTorch exact search.")
        return None


# ============================================================
# LOAD TRAINING DATA
# ============================================================

def load_training_data() -> pd.DataFrame:
    print_header("LOADING TRAINING DATA")

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training file not found:\n{TRAIN_FILE}"
        )

    df = pd.read_parquet(TRAIN_FILE)

    required_columns = {
        "user_index",
        "item_index",
        "train_user_id",
        "train_item_id",
        "t_dat",
    }

    missing = required_columns.difference(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing required columns: {sorted(missing)}"
        )

    print(f"Training interactions: {len(df):,}")
    print(
        f"Training users: "
        f"{df['train_user_id'].nunique():,}"
    )
    print(
        f"Training items: "
        f"{df['train_item_id'].nunique():,}"
    )

    return df


# ============================================================
# BUILD USER HISTORIES
# ============================================================

def build_user_histories(
    df: pd.DataFrame,
) -> Dict[int, List[int]]:
    """
    Build chronological training histories.

    Important:
    - train_user_id is the model user ID.
    - train_item_id is the model item ID.
    - item_index is used as deterministic same-day tie-breaker.

    Repeated purchases are intentionally preserved.
    """

    print_header("BUILDING TRAINING HISTORIES")

    work = df[
        [
            "train_user_id",
            "train_item_id",
            "t_dat",
            "item_index",
        ]
    ].copy()

    work["t_dat"] = pd.to_datetime(work["t_dat"])

    # Frozen deterministic chronology.
    #
    # H&M data has date resolution rather than timestamp resolution.
    # Therefore item_index is retained as deterministic same-day
    # ordering, consistent with the project's frozen preprocessing.
    work = work.sort_values(
        [
            "train_user_id",
            "t_dat",
            "item_index",
        ],
        kind="mergesort",
    )

    histories: Dict[int, List[int]] = {}

    for user_id, group in work.groupby(
        "train_user_id",
        sort=False,
    ):
        histories[int(user_id)] = (
            group["train_item_id"]
            .astype(np.int32)
            .tolist()
        )

    print(f"Histories: {len(histories):,}")

    lengths = np.fromiter(
        (len(v) for v in histories.values()),
        dtype=np.int32,
    )

    print(f"Minimum history length: {lengths.min():,}")
    print(f"Maximum history length: {lengths.max():,}")
    print(f"Mean history length: {lengths.mean():.2f}")

    return histories


# ============================================================
# COUNT POSSIBLE NEXT-ITEM EXAMPLES
# ============================================================

def count_possible_examples(
    histories: Dict[int, List[int]],
) -> int:
    """
    Every user with N interactions produces N-1 next-item examples.
    """

    total = 0

    for history in histories.values():
        if len(history) >= 2:
            total += len(history) - 1

    return total


# ============================================================
# RESERVOIR SAMPLE EXAMPLE LOCATIONS
# ============================================================

def reservoir_sample_example_locations(
    histories: Dict[int, List[int]],
    num_examples: int,
    seed: int,
) -> List[Tuple[int, int]]:
    """
    Uniformly sample (user_id, target_position) pairs from the
    complete set of possible next-item training examples.

    target_position points to the positive item.

    Prefix is:

        history[:target_position]

    Therefore the positive target is strictly after the prefix.

    Repeated positive items in the prefix are valid.
    """

    print_header("SAMPLING NEXT-ITEM TRAINING EXAMPLES")

    rng = random.Random(seed)

    total_possible = count_possible_examples(histories)

    print(
        f"Possible next-item examples: "
        f"{total_possible:,}"
    )

    target_count = min(
        num_examples,
        total_possible,
    )

    reservoir: List[Tuple[int, int]] = []

    seen = 0

    for user_id, history in histories.items():

        if len(history) < 2:
            continue

        # Position 1 -> first target.
        #
        # Position 0 cannot be used because it has no preceding
        # interaction.
        for target_position in range(1, len(history)):

            candidate = (
                int(user_id),
                int(target_position),
            )

            seen += 1

            if len(reservoir) < target_count:
                reservoir.append(candidate)

            else:
                replacement_index = rng.randint(
                    0,
                    seen - 1,
                )

                if replacement_index < target_count:
                    reservoir[replacement_index] = candidate

    print(f"Selected examples: {len(reservoir):,}")

    return reservoir


# ============================================================
# NEGATIVE SAMPLING
# ============================================================

def sample_random_negative(
    positive_item: int,
    user_history_set: set,
    rng: np.random.Generator,
) -> int:
    """
    Sample a random training-vocabulary item that:

    1. is not the positive item
    2. has never appeared in the user's complete training history
    """

    while True:

        negative = int(
            rng.integers(
                0,
                NUM_ITEMS,
            )
        )

        if negative == positive_item:
            continue

        if negative in user_history_set:
            continue

        return negative


# ============================================================
# BUILD SEQUENCE / POSITIVE / RANDOM NEGATIVE ARRAYS
# ============================================================

def build_training_examples(
    histories: Dict[int, List[int]],
    example_locations: List[Tuple[int, int]],
):
    """
    Build:

        users
        positives
        random negatives
        padded SASRec sequences

    SASRec convention:

        padding ID = 0
        actual train_item_id = train_item_id + 1

    Example:

        actual history:
        [12, 45, 81]

        stored sequence:
        [0, 0, ..., 13, 46, 82]

    Important:
    The target item is NOT inserted into the sequence.

    A target item is allowed to have appeared earlier in the
    history because repeated purchases are valid.
    """

    print_header("BUILDING TRAINING EXAMPLES")

    rng = np.random.default_rng(SEED)

    num_examples = len(example_locations)

    users = np.empty(
        num_examples,
        dtype=np.int32,
    )

    positives = np.empty(
        num_examples,
        dtype=np.int32,
    )

    random_negatives = np.empty(
        num_examples,
        dtype=np.int32,
    )

    sequences = np.zeros(
        (
            num_examples,
            MAX_SEQUENCE_LENGTH,
        ),
        dtype=np.int32,
    )

    for index, (user_id, target_position) in enumerate(
        example_locations
    ):

        history = histories[user_id]

        # Strict chronological prefix.
        prefix = history[:target_position]

        if len(prefix) == 0:
            raise RuntimeError(
                "Internal error: empty prefix encountered."
            )

        # Positive target.
        positive_item = int(
            history[target_position]
        )

        # Keep most recent MAX_SEQUENCE_LENGTH items.
        prefix = prefix[-MAX_SEQUENCE_LENGTH:]

        # Left padding.
        sequence_start = (
            MAX_SEQUENCE_LENGTH - len(prefix)
        )

        # +1 because SASRec reserves 0 for padding.
        sequences[
            index,
            sequence_start:,
        ] = (
            np.asarray(
                prefix,
                dtype=np.int32,
            )
            + 1
        )

        users[index] = user_id
        positives[index] = positive_item

        user_history_set = set(history)

        random_negatives[index] = sample_random_negative(
            positive_item=positive_item,
            user_history_set=user_history_set,
            rng=rng,
        )

    print(f"Users shape: {users.shape}")
    print(f"Positive shape: {positives.shape}")
    print(f"Random negative shape: {random_negatives.shape}")
    print(f"Sequence shape: {sequences.shape}")

    sequence_lengths = np.count_nonzero(
        sequences,
        axis=1,
    )

    print(
        f"Sequence length min: "
        f"{sequence_lengths.min()}"
    )

    print(
        f"Sequence length max: "
        f"{sequence_lengths.max()}"
    )

    print(
        f"Sequence length mean: "
        f"{sequence_lengths.mean():.2f}"
    )

    return (
        users,
        positives,
        random_negatives,
        sequences,
    )


# ============================================================
# LOAD LIGHTGCN EMBEDDINGS
# ============================================================

def load_lightgcn_embeddings():
    print_header("LOADING LIGHTGCN EMBEDDINGS")

    if not LIGHTGCN_USER_EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            LIGHTGCN_USER_EMBEDDINGS_FILE
        )

    if not LIGHTGCN_ITEM_EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            LIGHTGCN_ITEM_EMBEDDINGS_FILE
        )

    user_embeddings = load_numpy_copy(
        LIGHTGCN_USER_EMBEDDINGS_FILE
    )

    item_embeddings = load_numpy_copy(
        LIGHTGCN_ITEM_EMBEDDINGS_FILE
    )

    if user_embeddings.shape != (
        NUM_USERS,
        EMBEDDING_DIM,
    ):
        raise RuntimeError(
            "Unexpected LightGCN user embedding shape: "
            f"{user_embeddings.shape}"
        )

    if item_embeddings.shape != (
        NUM_ITEMS,
        EMBEDDING_DIM,
    ):
        raise RuntimeError(
            "Unexpected LightGCN item embedding shape: "
            f"{item_embeddings.shape}"
        )

    if not np.isfinite(user_embeddings).all():
        raise RuntimeError(
            "LightGCN user embeddings contain "
            "non-finite values."
        )

    if not np.isfinite(item_embeddings).all():
        raise RuntimeError(
            "LightGCN item embeddings contain "
            "non-finite values."
        )

    print(
        f"User embeddings: "
        f"{user_embeddings.shape}"
    )

    print(
        f"Item embeddings: "
        f"{item_embeddings.shape}"
    )

    return (
        user_embeddings.astype(
            np.float32,
            copy=False,
        ),
        item_embeddings.astype(
            np.float32,
            copy=False,
        ),
    )


# ============================================================
# LIGHTGCN HARD NEGATIVE SEARCH
# ============================================================

def normalize_embeddings(
    embeddings: np.ndarray,
) -> np.ndarray:

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    norms = np.maximum(
        norms,
        1e-12,
    )

    return embeddings / norms


def search_top_k_faiss(
    queries: np.ndarray,
    candidates: np.ndarray,
    top_k: int,
    faiss_module,
) -> np.ndarray:
    """
    Exact cosine-equivalent search using inner product after
    L2 normalization.
    """

    index = faiss_module.IndexFlatIP(
        candidates.shape[1]
    )

    index.add(
        candidates.astype(
            np.float32,
            copy=False,
        )
    )

    _, indices = index.search(
        queries.astype(
            np.float32,
            copy=False,
        ),
        top_k,
    )

    return indices


def search_top_k_torch(
    queries: np.ndarray,
    candidates: np.ndarray,
    top_k: int,
    batch_size: int,
) -> np.ndarray:
    """
    Exact batched PyTorch inner-product search.

    Used only when FAISS is unavailable.
    """

    query_tensor = torch.from_numpy(
        queries
    ).to(
        DEVICE,
        dtype=torch.float32,
    )

    candidate_tensor = torch.from_numpy(
        candidates
    ).to(
        DEVICE,
        dtype=torch.float32,
    )

    outputs = []

    for start in range(
        0,
        len(query_tensor),
        batch_size,
    ):

        end = min(
            start + batch_size,
            len(query_tensor),
        )

        scores = (
            query_tensor[start:end]
            @ candidate_tensor.T
        )

        _, indices = torch.topk(
            scores,
            k=top_k,
            dim=1,
        )

        outputs.append(
            indices.cpu().numpy()
        )

    return np.concatenate(
        outputs,
        axis=0,
    )


def choose_valid_hard_negative(
    candidates: np.ndarray,
    positive_item: int,
    history_set: set,
    fallback: int,
) -> int:
    """
    Choose the first hard-negative candidate that:

    - is not positive
    - is not in complete training history

    If all top-K candidates are invalid, use the supplied fallback.
    """

    for candidate in candidates:

        candidate = int(candidate)

        if candidate == positive_item:
            continue

        if candidate in history_set:
            continue

        return candidate

    return int(fallback)


def generate_lightgcn_hard_negatives(
    users: np.ndarray,
    positives: np.ndarray,
    histories: Dict[int, List[int]],
    lightgcn_user_embeddings: np.ndarray,
    lightgcn_item_embeddings: np.ndarray,
    faiss_module,
) -> np.ndarray:
    """
    Generate LightGCN hard negatives.

    The LightGCN user representation is identical for every
    training example belonging to the same user.

    Therefore search is performed once per unique sampled user,
    rather than once per training example.
    """

    print_header(
        "GENERATING LIGHTGCN HARD NEGATIVES"
    )

    normalized_users = normalize_embeddings(
        lightgcn_user_embeddings
    )

    normalized_items = normalize_embeddings(
        lightgcn_item_embeddings
    )

    unique_users = np.unique(users)

    print(
        "Unique users requiring LightGCN search: "
        f"{len(unique_users):,}"
    )

    unique_user_embeddings = normalized_users[
        unique_users
    ]

    if faiss_module is not None:

        candidate_matrix = search_top_k_faiss(
            queries=unique_user_embeddings,
            candidates=normalized_items,
            top_k=HARD_NEGATIVE_TOP_K,
            faiss_module=faiss_module,
        )

    else:

        candidate_matrix = search_top_k_torch(
            queries=unique_user_embeddings,
            candidates=normalized_items,
            top_k=HARD_NEGATIVE_TOP_K,
            batch_size=HARD_NEGATIVE_QUERY_BATCH_SIZE,
        )

    user_candidates = {
        int(user_id): candidate_matrix[index]
        for index, user_id in enumerate(unique_users)
    }

    result = np.empty(
        len(users),
        dtype=np.int32,
    )

    rng = np.random.default_rng(
        SEED + 100
    )

    for index in range(len(users)):

        user_id = int(users[index])
        positive = int(positives[index])

        history_set = set(
            histories[user_id]
        )

        candidates = user_candidates[user_id]

        fallback = sample_random_negative(
            positive_item=positive,
            user_history_set=history_set,
            rng=rng,
        )

        result[index] = choose_valid_hard_negative(
            candidates=candidates,
            positive_item=positive,
            history_set=history_set,
            fallback=fallback,
        )

    if np.any(
        result == positives
    ):
        raise RuntimeError(
            "LightGCN hard-negative generation produced "
            "positive items."
        )

    print(
        "LightGCN hard-negative generation PASSED"
    )

    return result


# ============================================================
# SASREC LOADING
# ============================================================

def load_sasrec_model():
    """
    Import and load the project's SASRec implementation.

    The current V3 hybrid model file contains the compatible
    SASRec implementation, so use it when available.
    """

    print_header("LOADING SASREC MODEL")

    if not SASREC_MODEL_FILE.exists():
        raise FileNotFoundError(
            SASREC_MODEL_FILE
        )

    # Import the project's V3 model implementation.
    from hybrid_model import SASRec

    model = SASRec(
        num_items=NUM_ITEMS,
        embedding_dim=EMBEDDING_DIM,
        max_seq_len=MAX_SEQUENCE_LENGTH,
        num_heads=2,
        num_layers=2,
        dropout=0.2,
    )

    checkpoint = torch.load(
        SASREC_MODEL_FILE,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint[
                "model_state_dict"
            ]
        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    # Handle older checkpoint naming.
    converted_state_dict = {}

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith(
            "transformer_encoder.layers."
        ):
            new_key = new_key.replace(
                "transformer_encoder.layers.",
                "transformer_layers.",
                1,
            )

        elif new_key.startswith(
            "layer_norm."
        ):
            new_key = new_key.replace(
                "layer_norm.",
                "final_layer_norm.",
                1,
            )

        converted_state_dict[new_key] = value

    missing, unexpected = model.load_state_dict(
        converted_state_dict,
        strict=False,
    )

    if missing or unexpected:

        raise RuntimeError(
            "SASRec checkpoint loading failed.\n"
            f"Missing keys: {missing}\n"
            f"Unexpected keys: {unexpected}"
        )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    print(
        f"SASRec parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    print(
        "SASRec frozen: "
        f"{all(not p.requires_grad for p in model.parameters())}"
    )

    return model


# ============================================================
# SASREC PREFIX REPRESENTATIONS
# ============================================================

@torch.no_grad()
def generate_sasrec_prefix_representations(
    sasrec_model,
    sequences: np.ndarray,
) -> np.ndarray:
    """
    Convert padded SASRec sequences into the final 64-D
    sequence representation.

    The model receives:
        actual_item_id + 1

    Padding remains:
        0
    """

    print_header(
        "GENERATING SASREC PREFIX REPRESENTATIONS"
    )

    sasrec_model.eval()

    sequence_tensor = torch.from_numpy(
        np.asarray(
            sequences,
            dtype=np.int64,
        )
    )

    outputs = []

    total = len(sequence_tensor)

    start_time = time.time()

    for start in range(
        0,
        total,
        SASREC_REP_BATCH_SIZE,
    ):

        end = min(
            start + SASREC_REP_BATCH_SIZE,
            total,
        )

        batch = sequence_tensor[
            start:end
        ].to(DEVICE)

        # The V3 SASRec implementation exposes
        # get_sequence_representation().
        if hasattr(
            sasrec_model,
            "get_sequence_representation",
        ):

            representation = (
                sasrec_model.get_sequence_representation(
                    batch
                )
            )

        else:

            # Compatibility fallback.
            output = sasrec_model(
                batch
            )

            mask = batch != 0

            lengths = mask.sum(
                dim=1
            ).clamp_min(1)

            last_positions = (
                lengths - 1
            )

            batch_indices = torch.arange(
                len(batch),
                device=batch.device,
            )

            representation = output[
                batch_indices,
                last_positions,
            ]

        if representation.ndim != 2:
            raise RuntimeError(
                "Unexpected SASRec representation "
                f"shape: {representation.shape}"
            )

        outputs.append(
            representation.cpu().numpy()
        )

        processed = end

        elapsed = time.time() - start_time

        if processed % (
            SASREC_REP_BATCH_SIZE * 10
        ) == 0 or processed == total:

            rate = (
                processed / elapsed
                if elapsed > 0
                else 0
            )

            print(
                f"SASRec representations "
                f"{processed:,}/{total:,} "
                f"at ~{rate:.1f} examples/sec"
            )

    representations = np.concatenate(
        outputs,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    if representations.shape != (
        len(sequences),
        EMBEDDING_DIM,
    ):
        raise RuntimeError(
            "Unexpected final SASRec representation "
            f"shape: {representations.shape}"
        )

    if not np.isfinite(
        representations
    ).all():

        raise RuntimeError(
            "SASRec representations contain "
            "non-finite values."
        )

    elapsed = time.time() - start_time

    print(
        f"SASRec representation generation: "
        f"{elapsed:.2f}s"
    )

    print(
        f"Representation shape: "
        f"{representations.shape}"
    )

    return representations


# ============================================================
# SASREC HARD NEGATIVES
# ============================================================

def generate_sasrec_hard_negatives(
    users: np.ndarray,
    positives: np.ndarray,
    sequences: np.ndarray,
    histories: Dict[int, List[int]],
    sasrec_model,
    faiss_module,
) -> np.ndarray:
    """
    Generate SASRec hard negatives using each example's
    chronological prefix representation.
    """

    print_header(
        "GENERATING SASREC HARD NEGATIVES"
    )

    prefix_representations = (
        generate_sasrec_prefix_representations(
            sasrec_model=sasrec_model,
            sequences=sequences,
        )
    )

    normalized_prefixes = normalize_embeddings(
        prefix_representations
    )

    # SASRec item embedding row 0 is padding.
    #
    # Actual train item IDs correspond to:
    #   embedding row = train_item_id + 1
    #
    # Therefore remove row 0.
    item_embedding = (
        sasrec_model.item_embedding.weight
        .detach()
        .cpu()
        .numpy()
    )

    if item_embedding.shape[0] != (
        NUM_ITEMS + 1
    ):
        raise RuntimeError(
            "Unexpected SASRec item embedding shape: "
            f"{item_embedding.shape}"
        )

    sasrec_items = item_embedding[1:]

    sasrec_items = sasrec_items.astype(
        np.float32,
        copy=False,
    )

    normalized_items = normalize_embeddings(
        sasrec_items
    )

    if faiss_module is not None:

        candidate_matrix = search_top_k_faiss(
            queries=normalized_prefixes,
            candidates=normalized_items,
            top_k=HARD_NEGATIVE_TOP_K,
            faiss_module=faiss_module,
        )

    else:

        candidate_matrix = search_top_k_torch(
            queries=normalized_prefixes,
            candidates=normalized_items,
            top_k=HARD_NEGATIVE_TOP_K,
            batch_size=HARD_NEGATIVE_QUERY_BATCH_SIZE,
        )

    result = np.empty(
        len(users),
        dtype=np.int32,
    )

    rng = np.random.default_rng(
        SEED + 200
    )

    for index in range(len(users)):

        user_id = int(users[index])
        positive = int(positives[index])

        history_set = set(
            histories[user_id]
        )

        candidates = candidate_matrix[index]

        fallback = sample_random_negative(
            positive_item=positive,
            user_history_set=history_set,
            rng=rng,
        )

        result[index] = choose_valid_hard_negative(
            candidates=candidates,
            positive_item=positive,
            history_set=history_set,
            fallback=fallback,
        )

    if np.any(
        result == positives
    ):
        raise RuntimeError(
            "SASRec hard-negative generation produced "
            "positive items."
        )

    print(
        "SASRec hard-negative generation PASSED"
    )

    return result


# ============================================================
# VALIDATION
# ============================================================

def validate_training_data(
    users: np.ndarray,
    positives: np.ndarray,
    random_negatives: np.ndarray,
    lightgcn_negatives: np.ndarray,
    sasrec_negatives: np.ndarray,
    sequences: np.ndarray,
    histories: Dict[int, List[int]],
) -> None:
    """
    Validate generated V3 training data.

    IMPORTANT:

    We intentionally DO NOT enforce:

        positive not in prefix

    because repeated purchases are valid.

    Example:

        history = [12, 45, 81, 45]

        prefix = [12, 45, 81]
        target = 45

    This is a legitimate next-item training example.

    The actual invariant is:

        prefix = history[:target_position]

    and target_position > 0.

    The generator itself constructs prefixes using exactly this
    rule. This validator therefore focuses on the invariants that
    can be checked from the stored arrays.
    """

    print_header(
        "VALIDATING V3 TRAINING DATA"
    )

    failures: List[str] = []

    num_examples = len(users)

    # --------------------------------------------------------
    # Shape checks
    # --------------------------------------------------------

    expected_1d_arrays = {
        "users": users,
        "positives": positives,
        "random_negatives": random_negatives,
        "lightgcn_negatives": lightgcn_negatives,
        "sasrec_negatives": sasrec_negatives,
    }

    for name, array in expected_1d_arrays.items():

        if array.ndim != 1:
            failures.append(
                f"{name} is not 1D: "
                f"{array.shape}"
            )

        if len(array) != num_examples:
            failures.append(
                f"{name} length mismatch: "
                f"{len(array)} != {num_examples}"
            )

    if sequences.ndim != 2:

        failures.append(
            "sequences is not 2D: "
            f"{sequences.shape}"
        )

    else:

        if sequences.shape != (
            num_examples,
            MAX_SEQUENCE_LENGTH,
        ):

            failures.append(
                "Unexpected sequence shape: "
                f"{sequences.shape}"
            )

    # --------------------------------------------------------
    # User range
    # --------------------------------------------------------

    if len(users) > 0:

        if users.min() < 0:
            failures.append(
                "Negative user ID detected."
            )

        if users.max() >= NUM_USERS:
            failures.append(
                "User ID exceeds training vocabulary."
            )

    # --------------------------------------------------------
    # Item ranges
    # --------------------------------------------------------

    item_arrays = {
        "positives": positives,
        "random_negatives": random_negatives,
        "lightgcn_negatives": lightgcn_negatives,
        "sasrec_negatives": sasrec_negatives,
    }

    for name, array in item_arrays.items():

        if len(array) == 0:
            continue

        if array.min() < 0:
            failures.append(
                f"{name} contains negative item IDs."
            )

        if array.max() >= NUM_ITEMS:
            failures.append(
                f"{name} exceeds training item vocabulary."
            )

    # --------------------------------------------------------
    # Sequence range
    #
    # 0 = padding
    # 1..NUM_ITEMS = actual items + 1
    # --------------------------------------------------------

    if sequences.size > 0:

        if sequences.min() < 0:
            failures.append(
                "Sequence contains negative IDs."
            )

        if sequences.max() > NUM_ITEMS:
            failures.append(
                "Sequence contains item IDs above "
                "NUM_ITEMS."
            )

    # --------------------------------------------------------
    # Positive != negative
    # --------------------------------------------------------

    negative_arrays = {
        "random": random_negatives,
        "LightGCN": lightgcn_negatives,
        "SASRec": sasrec_negatives,
    }

    for name, negatives in negative_arrays.items():

        bad_indices = np.where(
            negatives == positives
        )[0]

        if len(bad_indices) > 0:

            failures.append(
                f"{name} negative equals positive "
                f"for {len(bad_indices):,} examples."
            )

    # --------------------------------------------------------
    # Sequence padding checks
    # --------------------------------------------------------

    for index in range(
        min(num_examples, 100_000)
    ):

        sequence = sequences[index]

        zero_positions = np.where(
            sequence == 0
        )[0]

        if len(zero_positions) == 0:
            # Full 50-item sequence.
            pass

        else:

            first_nonzero = np.argmax(
                sequence != 0
            )

            # If the sequence contains any nonzero value,
            # all zeros must be before it.
            nonzero_positions = np.where(
                sequence != 0
            )[0]

            if len(nonzero_positions) > 0:

                expected_start = (
                    MAX_SEQUENCE_LENGTH
                    - len(nonzero_positions)
                )

                if first_nonzero != expected_start:

                    failures.append(
                        "Internal/right padding violation "
                        f"at example {index}."
                    )
                    break

                if np.any(
                    sequence[
                        expected_start:
                    ] == 0
                ):

                    failures.append(
                        "Zero found inside sequence "
                        f"at example {index}."
                    )
                    break

    # --------------------------------------------------------
    # Negative history checks
    #
    # A valid negative must not appear anywhere in the
    # user's complete training history.
    # --------------------------------------------------------

    max_history_checks = min(
        num_examples,
        100_000,
    )

    for index in range(
        max_history_checks
    ):

        user_id = int(
            users[index]
        )

        history_set = set(
            histories[user_id]
        )

        negatives = (
            random_negatives[index],
            lightgcn_negatives[index],
            sasrec_negatives[index],
        )

        for negative in negatives:

            negative = int(negative)

            if negative in history_set:

                failures.append(
                    "Negative item appears in complete "
                    f"user history at example {index}: "
                    f"user={user_id}, "
                    f"negative={negative}"
                )

                break

        if failures:
            break

    # --------------------------------------------------------
    # Important repeated-purchase rule
    # --------------------------------------------------------

    print(
        "Positive-in-prefix validation: ALLOWED"
    )

    print(
        "Reason: repeated purchases/interactions are "
        "preserved by the project methodology."
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    if failures:

        print()
        print("VALIDATION FAILURES:")
        print("-" * 80)

        for failure in failures[:50]:
            print(f"[FAIL] {failure}")

        if len(failures) > 50:
            print(
                f"... and {len(failures) - 50:,} more."
            )

        raise RuntimeError(
            "V3 training data validation failed with "
            f"{len(failures)} failures."
        )

    print()
    print(
        "[PASS] Shapes are correct."
    )
    print(
        "[PASS] User IDs are valid."
    )
    print(
        "[PASS] Item IDs are valid."
    )
    print(
        "[PASS] Positive/negative IDs are valid."
    )
    print(
        "[PASS] Negative items differ from positives."
    )
    print(
        "[PASS] Negative items are absent from complete "
        "user histories."
    )
    print(
        "[PASS] Left padding is valid."
    )
    print(
        "[PASS] No invalid internal padding detected."
    )
    print(
        "[PASS] Repeated positive items in prefixes "
        "are allowed."
    )

    print()
    print(
        "V3 TRAINING DATA VALIDATION PASSED"
    )


# ============================================================
# SAVE DATA
# ============================================================

def save_training_data(
    users: np.ndarray,
    positives: np.ndarray,
    random_negatives: np.ndarray,
    lightgcn_negatives: np.ndarray,
    sasrec_negatives: np.ndarray,
    sequences: np.ndarray,
    metadata: dict,
) -> None:

    print_header(
        "SAVING V3 TRAINING DATA"
    )

    np.save(
        OUTPUT_USERS,
        users,
    )

    np.save(
        OUTPUT_POSITIVE,
        positives,
    )

    np.save(
        OUTPUT_RANDOM_NEGATIVE,
        random_negatives,
    )

    np.save(
        OUTPUT_LIGHTGCN_NEGATIVE,
        lightgcn_negatives,
    )

    np.save(
        OUTPUT_SASREC_NEGATIVE,
        sasrec_negatives,
    )

    np.save(
        OUTPUT_SEQUENCES,
        sequences,
    )

    with open(
        OUTPUT_METADATA,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(
        f"Saved: {OUTPUT_USERS.name}"
    )
    print(
        f"Saved: {OUTPUT_POSITIVE.name}"
    )
    print(
        f"Saved: {OUTPUT_RANDOM_NEGATIVE.name}"
    )
    print(
        f"Saved: {OUTPUT_LIGHTGCN_NEGATIVE.name}"
    )
    print(
        f"Saved: {OUTPUT_SASREC_NEGATIVE.name}"
    )
    print(
        f"Saved: {OUTPUT_SEQUENCES.name}"
    )
    print(
        f"Saved: {OUTPUT_METADATA.name}"
    )


# ============================================================
# FINAL OUTPUT VALIDATION
# ============================================================

def validate_saved_files() -> None:

    print_header(
        "VALIDATING SAVED FILES"
    )

    paths = [
        OUTPUT_USERS,
        OUTPUT_POSITIVE,
        OUTPUT_RANDOM_NEGATIVE,
        OUTPUT_LIGHTGCN_NEGATIVE,
        OUTPUT_SASREC_NEGATIVE,
        OUTPUT_SEQUENCES,
        OUTPUT_METADATA,
    ]

    for path in paths:

        if not path.exists():
            raise RuntimeError(
                f"Expected output file does not exist: "
                f"{path}"
            )

        print(
            f"[PASS] {path.name}"
        )

    users = np.load(
        OUTPUT_USERS
    )

    positives = np.load(
        OUTPUT_POSITIVE
    )

    random_negatives = np.load(
        OUTPUT_RANDOM_NEGATIVE
    )

    lightgcn_negatives = np.load(
        OUTPUT_LIGHTGCN_NEGATIVE
    )

    sasrec_negatives = np.load(
        OUTPUT_SASREC_NEGATIVE
    )

    sequences = np.load(
        OUTPUT_SEQUENCES
    )

    if len(users) != NUM_EXAMPLES:
        raise RuntimeError(
            "Saved user count does not match "
            "NUM_EXAMPLES."
        )

    if positives.shape != users.shape:
        raise RuntimeError(
            "Saved positive shape mismatch."
        )

    if random_negatives.shape != users.shape:
        raise RuntimeError(
            "Saved random negative shape mismatch."
        )

    if lightgcn_negatives.shape != users.shape:
        raise RuntimeError(
            "Saved LightGCN negative shape mismatch."
        )

    if sasrec_negatives.shape != users.shape:
        raise RuntimeError(
            "Saved SASRec negative shape mismatch."
        )

    if sequences.shape != (
        NUM_EXAMPLES,
        MAX_SEQUENCE_LENGTH,
    ):
        raise RuntimeError(
            "Saved sequence shape mismatch."
        )

    print(
        "[PASS] Saved array shapes."
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    total_start = time.time()

    print_header(
        "V3 TRAINING DATA PREPARATION"
    )

    print(
        f"Configured examples: "
        f"{NUM_EXAMPLES:,}"
    )

    print(
        f"Hard-negative Top-K: "
        f"{HARD_NEGATIVE_TOP_K}"
    )

    print(
        f"Maximum sequence length: "
        f"{MAX_SEQUENCE_LENGTH}"
    )

    print(
        f"Training users: "
        f"{NUM_USERS:,}"
    )

    print(
        f"Training items: "
        f"{NUM_ITEMS:,}"
    )

    print(
        f"Seed: {SEED}"
    )

    ensure_output_directory()

    set_seed(SEED)

    # --------------------------------------------------------
    # 1. Training data
    # --------------------------------------------------------

    train_df = load_training_data()

    # --------------------------------------------------------
    # 2. Histories
    # --------------------------------------------------------

    histories = build_user_histories(
        train_df
    )

    # --------------------------------------------------------
    # 3. Sample next-item examples
    # --------------------------------------------------------

    example_locations = (
        reservoir_sample_example_locations(
            histories=histories,
            num_examples=NUM_EXAMPLES,
            seed=SEED,
        )
    )

    # --------------------------------------------------------
    # 4. Build sequences / positive / random negative
    # --------------------------------------------------------

    (
        users,
        positives,
        random_negatives,
        sequences,
    ) = build_training_examples(
        histories=histories,
        example_locations=example_locations,
    )

    # --------------------------------------------------------
    # 5. Load LightGCN embeddings
    # --------------------------------------------------------

    (
        lightgcn_user_embeddings,
        lightgcn_item_embeddings,
    ) = load_lightgcn_embeddings()

    # --------------------------------------------------------
    # 6. FAISS backend
    # --------------------------------------------------------

    faiss_module = get_faiss()

    # --------------------------------------------------------
    # 7. LightGCN hard negatives
    # --------------------------------------------------------

    lightgcn_start = time.time()

    lightgcn_negatives = (
        generate_lightgcn_hard_negatives(
            users=users,
            positives=positives,
            histories=histories,
            lightgcn_user_embeddings=(
                lightgcn_user_embeddings
            ),
            lightgcn_item_embeddings=(
                lightgcn_item_embeddings
            ),
            faiss_module=faiss_module,
        )
    )

    print(
        "LightGCN hard-negative generation time: "
        f"{time.time() - lightgcn_start:.2f}s"
    )

    # --------------------------------------------------------
    # 8. Load SASRec
    # --------------------------------------------------------

    sasrec_model = load_sasrec_model()

    # --------------------------------------------------------
    # 9. SASRec hard negatives
    # --------------------------------------------------------

    sasrec_start = time.time()

    sasrec_negatives = (
        generate_sasrec_hard_negatives(
            users=users,
            positives=positives,
            sequences=sequences,
            histories=histories,
            sasrec_model=sasrec_model,
            faiss_module=faiss_module,
        )
    )

    print(
        "SASRec hard-negative generation time: "
        f"{time.time() - sasrec_start:.2f}s"
    )

    # --------------------------------------------------------
    # 10. Validate generated data
    # --------------------------------------------------------

    validate_training_data(
        users=users,
        positives=positives,
        random_negatives=random_negatives,
        lightgcn_negatives=lightgcn_negatives,
        sasrec_negatives=sasrec_negatives,
        sequences=sequences,
        histories=histories,
    )

    # --------------------------------------------------------
    # 11. Metadata
    # --------------------------------------------------------

    sequence_lengths = np.count_nonzero(
        sequences,
        axis=1,
    )

    metadata = {
        "project": "Hybrid Ecommerce Recommendation Model",
        "experiment": "V3",
        "purpose": (
            "Training data for adaptive graph-sequential "
            "hybrid recommendation model."
        ),
        "seed": SEED,
        "num_training_examples": int(
            len(users)
        ),
        "possible_next_item_examples": int(
            count_possible_examples(histories)
        ),
        "num_training_users": NUM_USERS,
        "num_training_items": NUM_ITEMS,
        "embedding_dim": EMBEDDING_DIM,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "hard_negative_top_k": HARD_NEGATIVE_TOP_K,
        "padding_id": 0,
        "actual_item_id_offset": 1,
        "negative_sampling": {
            "random": True,
            "exclude_positive": True,
            "exclude_complete_user_history": True,
        },
        "hard_negative_sampling": {
            "lightgcn": True,
            "sasrec": True,
            "backend": (
                "FAISS_IndexFlatIP"
                if faiss_module is not None
                else "PyTorch_exact_inner_product"
            ),
            "top_k": HARD_NEGATIVE_TOP_K,
        },
        "sequence_construction": {
            "chronological": True,
            "prefix_before_target": True,
            "truncate_to_recent_items": MAX_SEQUENCE_LENGTH,
            "left_padding": True,
            "same_day_tiebreaker": "item_index",
            "repeated_interactions_preserved": True,
            "positive_allowed_in_prefix": True,
        },
        "sequence_statistics": {
            "min": int(
                sequence_lengths.min()
            ),
            "max": int(
                sequence_lengths.max()
            ),
            "mean": float(
                sequence_lengths.mean()
            ),
        },
        "validation": {
            "positive_in_prefix_is_not_invalid": True,
            "negative_items_must_not_appear_in_complete_history": True,
        },
        "outputs": {
            "users": str(
                OUTPUT_USERS
            ),
            "positive_items": str(
                OUTPUT_POSITIVE
            ),
            "random_negative_items": str(
                OUTPUT_RANDOM_NEGATIVE
            ),
            "lightgcn_hard_negative_items": str(
                OUTPUT_LIGHTGCN_NEGATIVE
            ),
            "sasrec_hard_negative_items": str(
                OUTPUT_SASREC_NEGATIVE
            ),
            "sequences": str(
                OUTPUT_SEQUENCES
            ),
        },
    }

    # --------------------------------------------------------
    # 12. Save
    # --------------------------------------------------------

    save_training_data(
        users=users,
        positives=positives,
        random_negatives=random_negatives,
        lightgcn_negatives=lightgcn_negatives,
        sasrec_negatives=sasrec_negatives,
        sequences=sequences,
        metadata=metadata,
    )

    # --------------------------------------------------------
    # 13. Reload validation
    # --------------------------------------------------------

    validate_saved_files()

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    total_elapsed = (
        time.time() - total_start
    )

    print_header(
        "V3 TRAINING DATA PREPARATION COMPLETE"
    )

    print(
        f"Examples generated: "
        f"{len(users):,}"
    )

    print(
        f"Sequence shape: "
        f"{sequences.shape}"
    )

    print(
        f"Random negatives: "
        f"{len(random_negatives):,}"
    )

    print(
        f"LightGCN hard negatives: "
        f"{len(lightgcn_negatives):,}"
    )

    print(
        f"SASRec hard negatives: "
        f"{len(sasrec_negatives):,}"
    )

    print(
        f"Total time: "
        f"{total_elapsed:.2f}s "
        f"({total_elapsed / 60:.2f} min)"
    )

    print()
    print(
        "V3 TRAINING DATA PREPARATION PASSED"
    )


if __name__ == "__main__":
    main()