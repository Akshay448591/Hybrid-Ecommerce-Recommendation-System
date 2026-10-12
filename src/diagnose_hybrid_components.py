"""
Component-level diagnostic for Hybrid LightGCN + SASRec.

Purpose
-------
Determine whether recommendation concentration originates from:
1. LightGCN alone
2. SASRec alone
3. Hybrid v1 fusion

The diagnostic uses the same global -> training ID mapping and the same
warm-start validation-user protocol used by the hybrid validation pipeline.

No model is trained or modified.

Outputs
-------
- Component representation norms
- Score distributions
- LightGCN-only recommendation concentration
- SASRec-only recommendation concentration
- Hybrid v1 recommendation concentration
- Top-K item concentration
- Top-20 Jaccard similarity
- Pairwise score correlations
- Component-vs-hybrid score relationships
- Final diagnostic conclusion
"""

from pathlib import Path
from collections import Counter
import json
import random
import time

import numpy as np
import pandas as pd
import torch


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

TRANSACTIONS_TRAIN = DATA_DIR / "transactions_train.parquet"
TRANSACTIONS_VAL = DATA_DIR / "transactions_val.parquet"

TRAIN_USER_MAPPING = DATA_DIR / "train_user_mapping.parquet"
TRAIN_ITEM_MAPPING = DATA_DIR / "train_item_mapping.parquet"

LIGHTGCN_MODEL_PATH = OUTPUT_DIR / "lightgcn_model.pt"
LIGHTGCN_USER_EMBEDDINGS_PATH = OUTPUT_DIR / "lightgcn_user_embeddings.npy"
LIGHTGCN_ITEM_EMBEDDINGS_PATH = OUTPUT_DIR / "lightgcn_item_embeddings.npy"

SASREC_MODEL_PATH = OUTPUT_DIR / "sasrec_model.pt"

HYBRID_MODEL_PATH = OUTPUT_DIR / "hybrid_model.pt"

OUTPUT_JSON = OUTPUT_DIR / "hybrid_component_diagnostic.json"


# Global/final vocabulary
GLOBAL_NUM_USERS = 797_782
GLOBAL_NUM_ITEMS = 79_026

# Training vocabulary
NUM_USERS = 778_421
NUM_ITEMS = 74_541

EMBEDDING_DIM = 64
MAX_SEQUENCE_LENGTH = 50

TOP_K = 20
NUM_DIAGNOSTIC_USERS = 1_000

BATCH_SIZE = 256

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")

print("=" * 80)
print("HYBRID COMPONENT-LEVEL DIAGNOSTIC")
print("=" * 80)

print(f"Project root: {PROJECT_ROOT}")
print(f"Device: {DEVICE}")
print(f"Diagnostic users: {NUM_DIAGNOSTIC_USERS}")
print(f"Top-K: {TOP_K}")
print()


# ============================================================
# IMPORT HYBRID MODEL
# ============================================================

import sys

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hybrid_model import HybridLightGCNSASRec


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def check_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    print(f"[PASS] File exists: {path.name}")


def load_numpy(path):
    print(f"Loading: {path.name}")

    array = np.load(path, mmap_mode="r")

    print(
        f"  shape={array.shape}, "
        f"dtype={array.dtype}"
    )

    return array


def describe_tensor(name, tensor):
    tensor = tensor.detach().cpu()

    values = tensor.reshape(-1)

    print(f"\n{name}")
    print(f"  shape: {tuple(tensor.shape)}")
    print(f"  mean:  {values.mean().item():.8f}")
    print(f"  std:   {values.std().item():.8f}")
    print(f"  min:   {values.min().item():.8f}")
    print(f"  max:   {values.max().item():.8f}")


def describe_norms(name, embeddings):
    norms = torch.linalg.norm(embeddings, dim=1)

    print(f"\n{name}")
    print(f"  mean norm:   {norms.mean().item():.8f}")
    print(f"  median norm: {norms.median().item():.8f}")
    print(f"  std norm:    {norms.std().item():.8f}")
    print(f"  min norm:    {norms.min().item():.8f}")
    print(f"  max norm:    {norms.max().item():.8f}")

    return norms


def topk_concentration(recommendations, top_k_values=(10, 20, 50, 100)):
    """
    recommendations:
        numpy array [num_users, K]

    Returns percentage of all recommendation slots occupied
    by the top-N most frequently recommended items.
    """

    flat = recommendations.reshape(-1)

    counts = Counter(flat.tolist())

    total = len(flat)

    result = {}

    for k in top_k_values:
        top_items = counts.most_common(k)

        top_count = sum(count for _, count in top_items)

        percentage = top_count / total

        result[k] = percentage

    return result


def unique_coverage(recommendations, catalogue_size):
    unique_items = np.unique(recommendations)

    coverage = len(unique_items) / catalogue_size

    return len(unique_items), coverage


def average_pairwise_jaccard(recommendations):
    """
    Average pairwise Jaccard similarity between recommendation lists.
    """

    sets = [
        set(row.tolist())
        for row in recommendations
    ]

    values = []

    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            intersection = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j])

            if union == 0:
                continue

            values.append(intersection / union)

    if not values:
        return 0.0, 0.0, 0.0

    values = np.asarray(values, dtype=np.float64)

    return (
        float(values.mean()),
        float(values.std()),
        float(values.min()),
    )


def top_item_frequency(recommendations, top_n=30):
    flat = recommendations.reshape(-1)

    counts = Counter(flat.tolist())

    return counts.most_common(top_n)


def print_concentration_report(name, recommendations):
    unique_count, coverage = unique_coverage(
        recommendations,
        NUM_ITEMS
    )

    concentration = topk_concentration(
        recommendations
    )

    mean_jaccard, std_jaccard, min_jaccard = (
        average_pairwise_jaccard(recommendations)
    )

    print("\n" + "-" * 80)
    print(f"{name}")
    print("-" * 80)

    print(
        f"Recommendation matrix: "
        f"{recommendations.shape}"
    )

    print(
        f"Unique recommended items: "
        f"{unique_count:,} / {NUM_ITEMS:,}"
    )

    print(
        f"Catalogue coverage: "
        f"{coverage:.8f} "
        f"({coverage * 100:.4f}%)"
    )

    print(
        f"Average pairwise Top-{TOP_K} Jaccard: "
        f"{mean_jaccard:.8f}"
    )

    print(
        f"Jaccard std: "
        f"{std_jaccard:.8f}"
    )

    print(
        f"Jaccard min: "
        f"{min_jaccard:.8f}"
    )

    print("\nRecommendation concentration:")

    for k, value in concentration.items():
        print(
            f"  Top {k:>3}: "
            f"{value * 100:.2f}%"
        )

    print("\nTop 20 recommended items:")

    top_items = top_item_frequency(
        recommendations,
        top_n=20
    )

    total = recommendations.size

    for rank, (item_id, count) in enumerate(
        top_items,
        start=1
    ):
        percentage = count / total * 100

        print(
            f"  {rank:>2}. "
            f"item {item_id:<6} "
            f"{count:>6} "
            f"({percentage:.3f}%)"
        )

    return {
        "shape": list(recommendations.shape),
        "unique_items": int(unique_count),
        "coverage": float(coverage),
        "average_top20_jaccard": float(mean_jaccard),
        "top20_jaccard_std": float(std_jaccard),
        "top20_jaccard_min": float(min_jaccard),
        "top_concentration": {
            str(k): float(v)
            for k, v in concentration.items()
        },
        "top_items": [
            {
                "item_id": int(item_id),
                "count": int(count),
                "percentage": float(
                    count / total * 100
                ),
            }
            for item_id, count in top_items
        ],
    }


def safe_correlation(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    a_std = a.std()
    b_std = b.std()

    if a_std == 0 or b_std == 0:
        return 0.0

    return float(np.corrcoef(a, b)[0, 1])


# ============================================================
# VERIFY REQUIRED FILES
# ============================================================

print("Checking required files...")

required_files = [
    TRANSACTIONS_TRAIN,
    TRANSACTIONS_VAL,
    TRAIN_USER_MAPPING,
    TRAIN_ITEM_MAPPING,
    LIGHTGCN_MODEL_PATH,
    LIGHTGCN_USER_EMBEDDINGS_PATH,
    LIGHTGCN_ITEM_EMBEDDINGS_PATH,
    SASREC_MODEL_PATH,
    HYBRID_MODEL_PATH,
]

for path in required_files:
    check_file(path)

print()


# ============================================================
# LOAD MAPPINGS
# ============================================================

print("=" * 80)
print("LOADING GLOBAL -> TRAINING ID MAPPINGS")
print("=" * 80)

user_mapping = pd.read_parquet(
    TRAIN_USER_MAPPING
)

item_mapping = pd.read_parquet(
    TRAIN_ITEM_MAPPING
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
    raise ValueError(
        "train_user_mapping.parquet does not "
        "contain the required columns."
    )

if not required_item_columns.issubset(
    item_mapping.columns
):
    raise ValueError(
        "train_item_mapping.parquet does not "
        "contain the required columns."
    )

print(
    f"User mapping rows: "
    f"{len(user_mapping):,}"
)

print(
    f"Item mapping rows: "
    f"{len(item_mapping):,}"
)

assert len(user_mapping) == NUM_USERS
assert len(item_mapping) == NUM_ITEMS

global_to_train_user = np.full(
    GLOBAL_NUM_USERS,
    -1,
    dtype=np.int32
)

global_to_train_user[
    user_mapping["user_index"].to_numpy(
        dtype=np.int32
    )
] = user_mapping["train_user_id"].to_numpy(
    dtype=np.int32
)

global_to_train_item = np.full(
    GLOBAL_NUM_ITEMS,
    -1,
    dtype=np.int32
)

global_to_train_item[
    item_mapping["item_index"].to_numpy(
        dtype=np.int32
    )
] = item_mapping["train_item_id"].to_numpy(
    dtype=np.int32
)

print("[PASS] Global -> training user mapping created")
print("[PASS] Global -> training item mapping created")

print()


# ============================================================
# LOAD TRANSACTIONS
# ============================================================

print("=" * 80)
print("LOADING TRANSACTIONS")
print("=" * 80)

train = pd.read_parquet(
    TRANSACTIONS_TRAIN,
    columns=[
        "user_index",
        "item_index",
        "t_dat",
    ]
)

val = pd.read_parquet(
    TRANSACTIONS_VAL,
    columns=[
        "user_index",
        "item_index",
        "t_dat",
    ]
)

print(
    f"Training rows: "
    f"{len(train):,}"
)

print(
    f"Validation rows: "
    f"{len(val):,}"
)

# Ensure chronological ordering is deterministic.
train = train.sort_values(
    ["user_index", "t_dat", "item_index"]
).reset_index(drop=True)

val = val.sort_values(
    ["user_index", "t_dat", "item_index"]
).reset_index(drop=True)

print("[PASS] Training chronological ordering prepared")
print("[PASS] Validation chronological ordering prepared")

print()


# ============================================================
# IDENTIFY WARM-START VALIDATION USERS
# ============================================================

print("=" * 80)
print("BUILDING WARM-START VALIDATION USERS")
print("=" * 80)

train_users_global = set(
    train["user_index"].unique().tolist()
)

train_items_global = set(
    train["item_index"].unique().tolist()
)

val_users_global = set(
    val["user_index"].unique().tolist()
)

warm_start_users_global = sorted(
    val_users_global.intersection(
        train_users_global
    )
)

print(
    f"Validation users: "
    f"{len(val_users_global):,}"
)

print(
    f"Warm-start validation users: "
    f"{len(warm_start_users_global):,}"
)

if len(warm_start_users_global) < NUM_DIAGNOSTIC_USERS:
    raise ValueError(
        "Not enough warm-start users for diagnostic."
    )

# Deterministic sample.
rng = np.random.default_rng(SEED)

diagnostic_users_global = rng.choice(
    np.asarray(
        warm_start_users_global,
        dtype=np.int32
    ),
    size=NUM_DIAGNOSTIC_USERS,
    replace=False
)

diagnostic_users_global = np.sort(
    diagnostic_users_global
)

diagnostic_users_train = (
    global_to_train_user[
        diagnostic_users_global
    ]
)

if np.any(diagnostic_users_train < 0):
    raise ValueError(
        "Diagnostic user mapping contains "
        "unseen users."
    )

print(
    f"Diagnostic users selected: "
    f"{len(diagnostic_users_global):,}"
)

print(
    "[PASS] All diagnostic users have "
    "training-model IDs."
)

print()


# ============================================================
# BUILD USER HISTORIES
# ============================================================

print("=" * 80)
print("BUILDING SASREC HISTORIES")
print("=" * 80)

train_user_groups = train.groupby(
    "user_index",
    sort=False
)

user_histories = {}

for global_user_id in diagnostic_users_global:

    if global_user_id not in train_user_groups.groups:
        raise ValueError(
            f"User {global_user_id} missing from training."
        )

    rows = train_user_groups.get_group(
        global_user_id
    )

    global_item_history = (
        rows["item_index"]
        .to_numpy(dtype=np.int32)
        .tolist()
    )

    train_item_history = []

    for global_item_id in global_item_history:

        train_item_id = global_to_train_item[
            global_item_id
        ]

        if train_item_id < 0:
            raise ValueError(
                "Training history contains an "
                "item absent from training vocabulary."
            )

        train_item_history.append(
            int(train_item_id)
        )

    # SASRec convention:
    # 0 = padding
    # actual item IDs = train_item_id + 1
    sasrec_history = [
        item_id + 1
        for item_id in train_item_history
    ]

    sasrec_history = sasrec_history[
        -MAX_SEQUENCE_LENGTH:
    ]

    if len(sasrec_history) == 0:
        raise ValueError(
            "Empty SASRec history."
        )

    padded = (
        [0] *
        (
            MAX_SEQUENCE_LENGTH
            - len(sasrec_history)
        )
        +
        sasrec_history
    )

    user_histories[
        int(global_user_id)
    ] = np.asarray(
        padded,
        dtype=np.int64
    )

print(
    f"Histories created: "
    f"{len(user_histories):,}"
)

print(
    "[PASS] SASRec history construction complete."
)

print()


# ============================================================
# LOAD PRECOMPUTED LIGHTGCN EMBEDDINGS
# ============================================================

print("=" * 80)
print("LOADING LIGHTGCN EMBEDDINGS")
print("=" * 80)

lightgcn_user_np = load_numpy(
    LIGHTGCN_USER_EMBEDDINGS_PATH
)

lightgcn_item_np = load_numpy(
    LIGHTGCN_ITEM_EMBEDDINGS_PATH
)

assert lightgcn_user_np.shape == (
    NUM_USERS,
    EMBEDDING_DIM
)

assert lightgcn_item_np.shape == (
    NUM_ITEMS,
    EMBEDDING_DIM
)

print("[PASS] LightGCN user embedding shape")
print("[PASS] LightGCN item embedding shape")

lightgcn_user_embeddings = torch.from_numpy(
    np.asarray(
        lightgcn_user_np,
        dtype=np.float32
    )
)

lightgcn_item_embeddings = torch.from_numpy(
    np.asarray(
        lightgcn_item_np,
        dtype=np.float32
    )
)

print()


# ============================================================
# LOAD HYBRID MODEL
# ============================================================

print("=" * 80)
print("LOADING PRETRAINED MODELS")
print("=" * 80)

hybrid_checkpoint = torch.load(
    HYBRID_MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
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
    hybrid_checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()

# Explicitly keep pretrained encoders in evaluation mode.
model.lightgcn.eval()
model.sasrec.eval()

print(
    "[PASS] Hybrid checkpoint loaded."
)

print(
    "[PASS] Hybrid model set to eval mode."
)

print(
    "[PASS] LightGCN set to eval mode."
)

print(
    "[PASS] SASRec set to eval mode."
)

print()


# ============================================================
# LOAD SASREC ITEM EMBEDDINGS
# ============================================================

print("=" * 80)
print("PREPARING SASREC ITEM EMBEDDINGS")
print("=" * 80)

with torch.no_grad():

    sasrec_item_embeddings = (
        model.sasrec.item_embedding.weight[
            1:
        ]
        .detach()
        .cpu()
    )

assert sasrec_item_embeddings.shape == (
    NUM_ITEMS,
    EMBEDDING_DIM
)

print(
    f"SASRec item embeddings: "
    f"{tuple(sasrec_item_embeddings.shape)}"
)

print(
    "[PASS] SASRec item vocabulary matches "
    "training vocabulary."
)

print()


# ============================================================
# COMPONENT REPRESENTATION DIAGNOSTICS
# ============================================================

print("=" * 80)
print("REPRESENTATION SCALE DIAGNOSTICS")
print("=" * 80)

lightgcn_user_norms = describe_norms(
    "LightGCN user embeddings",
    lightgcn_user_embeddings
)

lightgcn_item_norms = describe_norms(
    "LightGCN item embeddings",
    lightgcn_item_embeddings
)

sasrec_item_norms = describe_norms(
    "SASRec item embeddings",
    sasrec_item_embeddings
)

print()


# ============================================================
# BUILD SASREC USER REPRESENTATIONS
# ============================================================

print("=" * 80)
print("BUILDING SASREC USER REPRESENTATIONS")
print("=" * 80)

sasrec_user_representations = []

with torch.no_grad():

    for global_user_id in diagnostic_users_global:

        train_user_id = int(
            global_to_train_user[
                global_user_id
            ]
        )

        sequence = torch.tensor(
            user_histories[
                int(global_user_id)
            ],
            dtype=torch.long,
            device=DEVICE
        ).unsqueeze(0)

        user_ids = torch.tensor(
            [train_user_id],
            dtype=torch.long,
            device=DEVICE
        )

        # We use SASRec's internal sequence representation
        # directly through the hybrid model API.
        representation = (
            model.get_user_representation(
                user_ids=user_ids,
                sequences=sequence,
                lightgcn_user_embeddings=None,
            )
        )

        # The returned representation is hybrid.
        # To obtain pure SASRec representation, reproduce
        # the SASRec sequence encoder directly.

        encoded = model.sasrec(
            sequence
        )

        valid_mask = (
            sequence != 0
        )

        lengths = valid_mask.sum(
            dim=1
        )

        last_positions = (
            lengths - 1
        )

        sasrec_rep = encoded[
            torch.arange(
                encoded.size(0),
                device=DEVICE
            ),
            last_positions
        ]

        sasrec_user_representations.append(
            sasrec_rep.squeeze(0)
            .detach()
            .cpu()
        )

sasrec_user_embeddings = torch.stack(
    sasrec_user_representations,
    dim=0
)

assert sasrec_user_embeddings.shape == (
    NUM_DIAGNOSTIC_USERS,
    EMBEDDING_DIM
)

print(
    f"SASRec user representations: "
    f"{tuple(sasrec_user_embeddings.shape)}"
)

print(
    "[PASS] SASRec user representation extraction."
)

sasrec_user_norms = describe_norms(
    "SASRec user representations",
    sasrec_user_embeddings
)

print()


# ============================================================
# BUILD HYBRID USER REPRESENTATIONS
# ============================================================

print("=" * 80)
print("BUILDING HYBRID USER REPRESENTATIONS")
print("=" * 80)

hybrid_user_representations = []

with torch.no_grad():

    for index, global_user_id in enumerate(
        diagnostic_users_global
    ):

        train_user_id = int(
            global_to_train_user[
                global_user_id
            ]
        )

        sequence = torch.tensor(
            user_histories[
                int(global_user_id)
            ],
            dtype=torch.long,
            device=DEVICE
        ).unsqueeze(0)

        user_ids = torch.tensor(
            [train_user_id],
            dtype=torch.long,
            device=DEVICE
        )

        hybrid_user = model.get_user_representation(
            user_ids=user_ids,
            sequences=sequence,
            lightgcn_user_embeddings=lightgcn_user_embeddings,
        )

        hybrid_user_representations.append(
            hybrid_user.squeeze(0)
            .detach()
            .cpu()
        )

hybrid_user_embeddings = torch.stack(
    hybrid_user_representations,
    dim=0
)

assert hybrid_user_embeddings.shape == (
    NUM_DIAGNOSTIC_USERS,
    128
)

print(
    f"Hybrid user representations: "
    f"{tuple(hybrid_user_embeddings.shape)}"
)

print(
    "[PASS] Hybrid user representation extraction."
)

hybrid_user_norms = describe_norms(
    "Hybrid user representations",
    hybrid_user_embeddings
)

print()


# ============================================================
# BUILD HYBRID ITEM REPRESENTATIONS
# ============================================================

print("=" * 80)
print("BUILDING ITEM REPRESENTATIONS")
print("=" * 80)

with torch.no_grad():

    hybrid_item_embeddings = torch.cat(
        [
            lightgcn_item_embeddings,
            sasrec_item_embeddings,
        ],
        dim=1
    )

assert hybrid_item_embeddings.shape == (
    NUM_ITEMS,
    128
)

print(
    f"Hybrid item embeddings: "
    f"{tuple(hybrid_item_embeddings.shape)}"
)

print(
    "[PASS] Hybrid item representation shape."
)

hybrid_item_norms = describe_norms(
    "Hybrid item representations",
    hybrid_item_embeddings
)

print()


# ============================================================
# SCORE ALL THREE COMPONENTS
# ============================================================

print("=" * 80)
print("COMPUTING COMPONENT SCORES")
print("=" * 80)

num_users = NUM_DIAGNOSTIC_USERS

lightgcn_scores = torch.empty(
    (num_users, NUM_ITEMS),
    dtype=torch.float32
)

sasrec_scores = torch.empty(
    (num_users, NUM_ITEMS),
    dtype=torch.float32
)

hybrid_scores = torch.empty(
    (num_users, NUM_ITEMS),
    dtype=torch.float32
)

start_time = time.time()

with torch.no_grad():

    for start in range(
        0,
        num_users,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            num_users
        )

        # ----------------------------------------------------
        # LightGCN
        # ----------------------------------------------------

        user_ids_train = torch.from_numpy(
            global_to_train_user[
                diagnostic_users_global[
                    start:end
                ]
            ]
        ).long()

        batch_lgcn_users = (
            lightgcn_user_embeddings[
                user_ids_train
            ]
        )

        lgcn_scores = (
            batch_lgcn_users
            @ lightgcn_item_embeddings.T
        )

        lightgcn_scores[
            start:end
        ] = lgcn_scores

        # ----------------------------------------------------
        # SASRec
        # ----------------------------------------------------

        batch_sasrec_users = (
            sasrec_user_embeddings[
                start:end
            ]
        )

        sasrec_scores[
            start:end
        ] = (
            batch_sasrec_users
            @ sasrec_item_embeddings.T
        )

        # ----------------------------------------------------
        # Hybrid
        # ----------------------------------------------------

        batch_hybrid_users = (
            hybrid_user_embeddings[
                start:end
            ]
        )

        hybrid_scores[
            start:end
        ] = (
            batch_hybrid_users
            @ hybrid_item_embeddings.T
        )

elapsed = time.time() - start_time

print(
    f"Score computation time: "
    f"{elapsed:.2f}s"
)

print(
    f"Score matrix shape: "
    f"{tuple(hybrid_scores.shape)}"
)

assert lightgcn_scores.shape == (
    NUM_DIAGNOSTIC_USERS,
    NUM_ITEMS
)

assert sasrec_scores.shape == (
    NUM_DIAGNOSTIC_USERS,
    NUM_ITEMS
)

assert hybrid_scores.shape == (
    NUM_DIAGNOSTIC_USERS,
    NUM_ITEMS
)

assert torch.isfinite(
    lightgcn_scores
).all()

assert torch.isfinite(
    sasrec_scores
).all()

assert torch.isfinite(
    hybrid_scores
).all()

print(
    "[PASS] All component score matrices finite."
)

print()


# ============================================================
# SCORE DISTRIBUTIONS
# ============================================================

print("=" * 80)
print("SCORE DISTRIBUTIONS")
print("=" * 80)

describe_tensor(
    "LightGCN scores",
    lightgcn_scores
)

describe_tensor(
    "SASRec scores",
    sasrec_scores
)

describe_tensor(
    "Hybrid scores",
    hybrid_scores
)

print()


# ============================================================
# SCORE CORRELATIONS
# ============================================================

print("=" * 80)
print("SCORE CORRELATIONS")
print("=" * 80)

lgcn_flat = lightgcn_scores.numpy().reshape(-1)
sasrec_flat = sasrec_scores.numpy().reshape(-1)
hybrid_flat = hybrid_scores.numpy().reshape(-1)

corr_lgcn_sasrec = safe_correlation(
    lgcn_flat,
    sasrec_flat
)

corr_lgcn_hybrid = safe_correlation(
    lgcn_flat,
    hybrid_flat
)

corr_sasrec_hybrid = safe_correlation(
    sasrec_flat,
    hybrid_flat
)

print(
    f"LightGCN vs SASRec: "
    f"{corr_lgcn_sasrec:.8f}"
)

print(
    f"LightGCN vs Hybrid: "
    f"{corr_lgcn_hybrid:.8f}"
)

print(
    f"SASRec vs Hybrid: "
    f"{corr_sasrec_hybrid:.8f}"
)

print()


# ============================================================
# TOP-K RECOMMENDATIONS
# ============================================================

print("=" * 80)
print("GENERATING TOP-K RECOMMENDATIONS")
print("=" * 80)

lightgcn_topk = torch.topk(
    lightgcn_scores,
    k=TOP_K,
    dim=1
).indices.numpy()

sasrec_topk = torch.topk(
    sasrec_scores,
    k=TOP_K,
    dim=1
).indices.numpy()

hybrid_topk = torch.topk(
    hybrid_scores,
    k=TOP_K,
    dim=1
).indices.numpy()

print(
    f"LightGCN Top-{TOP_K}: "
    f"{lightgcn_topk.shape}"
)

print(
    f"SASRec Top-{TOP_K}: "
    f"{sasrec_topk.shape}"
)

print(
    f"Hybrid Top-{TOP_K}: "
    f"{hybrid_topk.shape}"
)

print(
    "[PASS] Top-K recommendations generated."
)

print()


# ============================================================
# CONCENTRATION REPORTS
# ============================================================

lightgcn_report = print_concentration_report(
    "LIGHTGCN ONLY",
    lightgcn_topk
)

sasrec_report = print_concentration_report(
    "SASREC ONLY",
    sasrec_topk
)

hybrid_report = print_concentration_report(
    "HYBRID V1",
    hybrid_topk
)


# ============================================================
# COMPONENT CONTRIBUTION ANALYSIS
# ============================================================

print("=" * 80)
print("COMPONENT CONTRIBUTION ANALYSIS")
print("=" * 80)

# Hybrid score is:
#
# [hybrid user] dot [LightGCN item || SASRec item]
#
# Split the hybrid user into its two 64-D halves.
#
# The fusion network transforms the concatenated user input,
# so these are NOT pure additive LightGCN/SASRec scores.
# They are useful only as diagnostic projections.

hybrid_user_first = (
    hybrid_user_embeddings[:, :64]
)

hybrid_user_second = (
    hybrid_user_embeddings[:, 64:]
)

component_score_1 = (
    hybrid_user_first
    @ lightgcn_item_embeddings.T
)

component_score_2 = (
    hybrid_user_second
    @ sasrec_item_embeddings.T
)

component_score_1_flat = (
    component_score_1.numpy().reshape(-1)
)

component_score_2_flat = (
    component_score_2.numpy().reshape(-1)
)

print(
    "Hybrid user first-half × LightGCN item:"
)

describe_tensor(
    "Hybrid first component score",
    component_score_1
)

print(
    "\nHybrid user second-half × SASRec item:"
)

describe_tensor(
    "Hybrid second component score",
    component_score_2
)

print(
    "\nCorrelation with final hybrid score:"
)

print(
    f"  First component vs Hybrid: "
    f"{safe_correlation(component_score_1_flat, hybrid_flat):.8f}"
)

print(
    f"  Second component vs Hybrid: "
    f"{safe_correlation(component_score_2_flat, hybrid_flat):.8f}"
)

print()


# ============================================================
# TOP ITEM OVERLAP
# ============================================================

print("=" * 80)
print("TOP-ITEM OVERLAP")
print("=" * 80)

lgcn_top20_global = lightgcn_topk
sasrec_top20_global = sasrec_topk
hybrid_top20_global = hybrid_topk

lgcn_set = set(
    lgcn_top20_global.reshape(-1).tolist()
)

sasrec_set = set(
    sasrec_top20_global.reshape(-1).tolist()
)

hybrid_set = set(
    hybrid_top20_global.reshape(-1).tolist()
)

def jaccard(a, b):
    union = a | b

    if len(union) == 0:
        return 0.0

    return len(a & b) / len(union)


print(
    f"Global unique LightGCN Top-{TOP_K}: "
    f"{len(lgcn_set):,}"
)

print(
    f"Global unique SASRec Top-{TOP_K}: "
    f"{len(sasrec_set):,}"
)

print(
    f"Global unique Hybrid Top-{TOP_K}: "
    f"{len(hybrid_set):,}"
)

print(
    f"LightGCN vs SASRec Jaccard: "
    f"{jaccard(lgcn_set, sasrec_set):.8f}"
)

print(
    f"LightGCN vs Hybrid Jaccard: "
    f"{jaccard(lgcn_set, hybrid_set):.8f}"
)

print(
    f"SASRec vs Hybrid Jaccard: "
    f"{jaccard(sasrec_set, hybrid_set):.8f}"
)

print()


# ============================================================
# INSPECT ONE USER
# ============================================================

print("=" * 80)
print("SINGLE USER SCORE INSPECTION")
print("=" * 80)

inspection_index = 0

inspection_global_user = int(
    diagnostic_users_global[
        inspection_index
    ]
)

inspection_train_user = int(
    global_to_train_user[
        inspection_global_user
    ]
)

print(
    f"Global user ID: "
    f"{inspection_global_user}"
)

print(
    f"Training user ID: "
    f"{inspection_train_user}"
)

for name, score_matrix, recommendations in [
    (
        "LightGCN",
        lightgcn_scores,
        lightgcn_topk
    ),
    (
        "SASRec",
        sasrec_scores,
        sasrec_topk
    ),
    (
        "Hybrid",
        hybrid_scores,
        hybrid_topk
    ),
]:

    user_scores = score_matrix[
        inspection_index
    ]

    user_top_items = recommendations[
        inspection_index
    ]

    print("\n" + name)

    print(
        f"  score mean: "
        f"{user_scores.mean().item():.8f}"
    )

    print(
        f"  score std: "
        f"{user_scores.std().item():.8f}"
    )

    print(
        f"  score min: "
        f"{user_scores.min().item():.8f}"
    )

    print(
        f"  score median: "
        f"{user_scores.median().item():.8f}"
    )

    print(
        f"  score max: "
        f"{user_scores.max().item():.8f}"
    )

    print(
        f"  Top-{TOP_K} items:"
    )

    for rank, item_id in enumerate(
        user_top_items,
        start=1
    ):

        score = user_scores[
            item_id
        ].item()

        print(
            f"    {rank:>2}. "
            f"train_item_id={int(item_id):<6} "
            f"score={score:.8f}"
        )

print()


# ============================================================
# REPRESENTATION VARIANCE
# ============================================================

print("=" * 80)
print("REPRESENTATION VARIANCE")
print("=" * 80)

def mean_feature_variance(embeddings):
    return float(
        embeddings.var(
            dim=0,
            unbiased=False
        ).mean().item()
    )

representation_variance = {
    "lightgcn_user": mean_feature_variance(
        lightgcn_user_embeddings[
            torch.from_numpy(
                diagnostic_users_global
            ).long()
            * 0
        ]
    ),
    "sasrec_user": mean_feature_variance(
        sasrec_user_embeddings
    ),
    "hybrid_user": mean_feature_variance(
        hybrid_user_embeddings
    ),
    "lightgcn_item": mean_feature_variance(
        lightgcn_item_embeddings
    ),
    "sasrec_item": mean_feature_variance(
        sasrec_item_embeddings
    ),
    "hybrid_item": mean_feature_variance(
        hybrid_item_embeddings
    ),
}

# Correct LightGCN diagnostic-user selection.
diagnostic_train_user_ids = torch.from_numpy(
    diagnostic_users_train
).long()

representation_variance[
    "lightgcn_user"
] = mean_feature_variance(
    lightgcn_user_embeddings[
        diagnostic_train_user_ids
    ]
)

for key, value in representation_variance.items():

    print(
        f"{key:<20}: "
        f"{value:.10f}"
    )

print()


# ============================================================
# FINAL DIAGNOSTIC INTERPRETATION
# ============================================================

print("=" * 80)
print("DIAGNOSTIC SUMMARY")
print("=" * 80)

lgcn_coverage = lightgcn_report[
    "coverage"
]

sasrec_coverage = sasrec_report[
    "coverage"
]

hybrid_coverage = hybrid_report[
    "coverage"
]

lgcn_top100 = (
    lightgcn_report[
        "top_concentration"
    ]["100"]
)

sasrec_top100 = (
    sasrec_report[
        "top_concentration"
    ]["100"]
)

hybrid_top100 = (
    hybrid_report[
        "top_concentration"
    ]["100"]
)

print(
    "\nCatalogue coverage:"
)

print(
    f"  LightGCN : "
    f"{lgcn_coverage * 100:.4f}%"
)

print(
    f"  SASRec   : "
    f"{sasrec_coverage * 100:.4f}%"
)

print(
    f"  Hybrid   : "
    f"{hybrid_coverage * 100:.4f}%"
)

print(
    "\nTop-100 recommendation concentration:"
)

print(
    f"  LightGCN : "
    f"{lgcn_top100 * 100:.2f}%"
)

print(
    f"  SASRec   : "
    f"{sasrec_top100 * 100:.2f}%"
)

print(
    f"  Hybrid   : "
    f"{hybrid_top100 * 100:.2f}%"
)

print(
    "\nInterpretation:"
)

if (
    hybrid_coverage < lgcn_coverage * 0.25
    and hybrid_coverage < sasrec_coverage * 0.25
):
    print(
        "  [DIAGNOSIS] Hybrid v1 introduces "
        "substantial recommendation concentration "
        "relative to both individual components."
    )

elif hybrid_coverage < lgcn_coverage:
    print(
        "  [DIAGNOSIS] Hybrid v1 is more concentrated "
        "than LightGCN."
    )

elif hybrid_coverage < sasrec_coverage:
    print(
        "  [DIAGNOSIS] Hybrid v1 is more concentrated "
        "than SASRec."
    )

else:
    print(
        "  [DIAGNOSIS] Hybrid v1 does not show "
        "additional catalogue concentration "
        "relative to both components."
    )

if (
    sasrec_coverage < lgcn_coverage * 0.25
):
    print(
        "  [NOTE] SASRec itself shows strong "
        "catalogue concentration."
    )

if (
    lgcn_coverage > hybrid_coverage * 2
):
    print(
        "  [NOTE] LightGCN has materially broader "
        "catalogue coverage than Hybrid v1."
    )

if (
    abs(
        corr_sasrec_hybrid
    )
    > abs(
        corr_lgcn_hybrid
    )
):
    print(
        "  [NOTE] Hybrid scores are more correlated "
        "with SASRec scores than LightGCN scores."
    )

else:
    print(
        "  [NOTE] Hybrid scores are more correlated "
        "with LightGCN scores than SASRec scores."
    )

print()


# ============================================================
# SAVE DIAGNOSTIC RESULTS
# ============================================================

print("=" * 80)
print("SAVING DIAGNOSTIC RESULTS")
print("=" * 80)

results = {
    "configuration": {
        "seed": SEED,
        "diagnostic_users": NUM_DIAGNOSTIC_USERS,
        "top_k": TOP_K,
        "global_num_users": GLOBAL_NUM_USERS,
        "global_num_items": GLOBAL_NUM_ITEMS,
        "training_num_users": NUM_USERS,
        "training_num_items": NUM_ITEMS,
        "embedding_dim": EMBEDDING_DIM,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
    },

    "representation_norms": {
        "lightgcn_user": {
            "mean": float(
                lightgcn_user_norms.mean()
            ),
            "median": float(
                lightgcn_user_norms.median()
            ),
            "std": float(
                lightgcn_user_norms.std()
            ),
        },
        "sasrec_user": {
            "mean": float(
                sasrec_user_norms.mean()
            ),
            "median": float(
                sasrec_user_norms.median()
            ),
            "std": float(
                sasrec_user_norms.std()
            ),
        },
        "hybrid_user": {
            "mean": float(
                hybrid_user_norms.mean()
            ),
            "median": float(
                hybrid_user_norms.median()
            ),
            "std": float(
                hybrid_user_norms.std()
            ),
        },
        "lightgcn_item": {
            "mean": float(
                lightgcn_item_norms.mean()
            ),
            "median": float(
                lightgcn_item_norms.median()
            ),
            "std": float(
                lightgcn_item_norms.std()
            ),
        },
        "sasrec_item": {
            "mean": float(
                sasrec_item_norms.mean()
            ),
            "median": float(
                sasrec_item_norms.median()
            ),
            "std": float(
                sasrec_item_norms.std()
            ),
        },
        "hybrid_item": {
            "mean": float(
                hybrid_item_norms.mean()
            ),
            "median": float(
                hybrid_item_norms.median()
            ),
            "std": float(
                hybrid_item_norms.std()
            ),
        },
    },

    "score_correlations": {
        "lightgcn_vs_sasrec": corr_lgcn_sasrec,
        "lightgcn_vs_hybrid": corr_lgcn_hybrid,
        "sasrec_vs_hybrid": corr_sasrec_hybrid,
    },

    "lightgcn": lightgcn_report,
    "sasrec": sasrec_report,
    "hybrid": hybrid_report,

    "representation_variance":
        representation_variance,

    "top20_global_jaccard": {
        "lightgcn_vs_sasrec": float(
            jaccard(
                lgcn_set,
                sasrec_set
            )
        ),
        "lightgcn_vs_hybrid": float(
            jaccard(
                lgcn_set,
                hybrid_set
            )
        ),
        "sasrec_vs_hybrid": float(
            jaccard(
                sasrec_set,
                hybrid_set
            )
        ),
    },

    "component_contribution": {
        "first_component_vs_hybrid": safe_correlation(
            component_score_1_flat,
            hybrid_flat
        ),
        "second_component_vs_hybrid": safe_correlation(
            component_score_2_flat,
            hybrid_flat
        ),
    },
}

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )

print(
    f"Saved: {OUTPUT_JSON}"
)

print()

print("=" * 80)
print("HYBRID COMPONENT DIAGNOSTIC COMPLETE")
print("=" * 80)