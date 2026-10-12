import os
import json
import numpy as np
import pandas as pd
import torch

from hybrid_model import load_lightgcn, load_sasrec


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

TRAIN_FILE = os.path.join(DATA_DIR, "transactions_train.parquet")
VAL_FILE = os.path.join(DATA_DIR, "transactions_val.parquet")
TEST_FILE = os.path.join(DATA_DIR, "transactions_test.parquet")
TRAIN_MODEL_IDS_FILE = os.path.join(
    DATA_DIR, "transactions_train_model_ids.parquet"
)

USER_MAPPING_FILE = os.path.join(
    DATA_DIR, "train_user_mapping.parquet"
)

ITEM_MAPPING_FILE = os.path.join(
    DATA_DIR, "train_item_mapping.parquet"
)

LIGHTGCN_GRAPH_FILE = os.path.join(
    OUTPUT_DIR, "lightgcn_graph.pt"
)

LIGHTGCN_MODEL_FILE = os.path.join(
    OUTPUT_DIR, "lightgcn_model.pt"
)

LIGHTGCN_USER_EMBEDDINGS_FILE = os.path.join(
    OUTPUT_DIR, "lightgcn_user_embeddings.npy"
)

LIGHTGCN_ITEM_EMBEDDINGS_FILE = os.path.join(
    OUTPUT_DIR, "lightgcn_item_embeddings.npy"
)

SASREC_MODEL_FILE = os.path.join(
    OUTPUT_DIR, "sasrec_model.pt"
)


# Frozen dataset/model expectations
EXPECTED_TRAIN_INTERACTIONS = 19_836_738
EXPECTED_TRAIN_USERS = 778_421
EXPECTED_TRAIN_ITEMS = 74_541

EXPECTED_WARM_USERS = 173_203
EXPECTED_WARM_INTERACTIONS = 751_993

EXPECTED_LIGHTGCN_PARAMS = 54_589_568
EXPECTED_SASREC_PARAMS = 4_873_984

EMBEDDING_DIM = 64
LIGHTGCN_LAYERS = 3
SASREC_MAX_SEQ_LEN = 50


# ============================================================
# HELPERS
# ============================================================

passed = 0
failed = 0


def check(condition, message):
    global passed, failed

    if condition:
        print(f"[PASS] {message}")
        passed += 1
    else:
        print(f"[FAIL] {message}")
        failed += 1


def load_parquet(path, name):
    print(f"\nLoading {name}...")
    check(os.path.exists(path), f"{name} exists")

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)
    print(f"{name} shape: {df.shape}")
    return df


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("HYBRID LIGHTGCN + SASREC DATA VALIDATION")
print("=" * 70)


# ============================================================
# 1. FILE EXISTENCE
# ============================================================

print("\nChecking required files...")

required_files = {
    "Training interactions": TRAIN_FILE,
    "Validation interactions": VAL_FILE,
    "Test interactions": TEST_FILE,
    "Training model-ID interactions": TRAIN_MODEL_IDS_FILE,
    "Training user mapping": USER_MAPPING_FILE,
    "Training item mapping": ITEM_MAPPING_FILE,
    "LightGCN graph": LIGHTGCN_GRAPH_FILE,
    "LightGCN model": LIGHTGCN_MODEL_FILE,
    "LightGCN user embeddings": LIGHTGCN_USER_EMBEDDINGS_FILE,
    "LightGCN item embeddings": LIGHTGCN_ITEM_EMBEDDINGS_FILE,
    "SASRec model": SASREC_MODEL_FILE,
}

for name, path in required_files.items():
    check(os.path.exists(path), f"{name} exists")


# ============================================================
# 2. LOAD DATA
# ============================================================

train_df = load_parquet(
    TRAIN_FILE,
    "Training interactions"
)

val_df = load_parquet(
    VAL_FILE,
    "Validation interactions"
)

test_df = load_parquet(
    TEST_FILE,
    "Test interactions"
)

train_model_df = load_parquet(
    TRAIN_MODEL_IDS_FILE,
    "Training model-ID interactions"
)

user_mapping = load_parquet(
    USER_MAPPING_FILE,
    "Training user mapping"
)

item_mapping = load_parquet(
    ITEM_MAPPING_FILE,
    "Training item mapping"
)


# ============================================================
# 3. TRAINING DATA COUNTS
# ============================================================

print("\nChecking training dataset...")

train_users = train_df["user_index"].nunique()
train_items = train_df["item_index"].nunique()
train_rows = len(train_df)

print(f"Training interactions: {train_rows}")
print(f"Training users: {train_users}")
print(f"Training items: {train_items}")

check(
    train_rows == EXPECTED_TRAIN_INTERACTIONS,
    "Training interaction count"
)

check(
    train_users == EXPECTED_TRAIN_USERS,
    "Training user count"
)

check(
    train_items == EXPECTED_TRAIN_ITEMS,
    "Training item count"
)


# ============================================================
# 4. MODEL-ID DATA
# ============================================================

print("\nChecking model-ID training data...")

model_rows = len(train_model_df)

model_users = train_model_df["train_user_id"].nunique()
model_items = train_model_df["train_item_id"].nunique()

print(f"Model-ID interactions: {model_rows}")
print(f"Model-ID users: {model_users}")
print(f"Model-ID items: {model_items}")

check(
    model_rows == EXPECTED_TRAIN_INTERACTIONS,
    "Model-ID interaction count"
)

check(
    model_users == EXPECTED_TRAIN_USERS,
    "Model-ID user count"
)

check(
    model_items == EXPECTED_TRAIN_ITEMS,
    "Model-ID item count"
)


# ============================================================
# 5. REQUIRED MODEL-ID COLUMNS
# ============================================================

print("\nChecking model-ID columns...")

required_model_columns = {
    "user_index",
    "item_index",
    "train_user_id",
    "train_item_id",
}

for column in required_model_columns:
    check(
        column in train_model_df.columns,
        f"Model-ID column '{column}' exists"
    )


# ============================================================
# 6. CONTIGUOUS MODEL IDS
# ============================================================

print("\nChecking contiguous training IDs...")

user_ids = np.sort(
    train_model_df["train_user_id"].unique()
)

item_ids = np.sort(
    train_model_df["train_item_id"].unique()
)

expected_user_ids = np.arange(
    EXPECTED_TRAIN_USERS,
    dtype=np.int64
)

expected_item_ids = np.arange(
    EXPECTED_TRAIN_ITEMS,
    dtype=np.int64
)

check(
    np.array_equal(user_ids, expected_user_ids),
    "Training user IDs are contiguous"
)

check(
    np.array_equal(item_ids, expected_item_ids),
    "Training item IDs are contiguous"
)


# ============================================================
# 7. MAPPING FILES
# ============================================================

print("\nChecking mapping files...")

check(
    len(user_mapping) == EXPECTED_TRAIN_USERS,
    "User mapping row count"
)

check(
    len(item_mapping) == EXPECTED_TRAIN_ITEMS,
    "Item mapping row count"
)

check(
    user_mapping["train_user_id"].nunique()
    == EXPECTED_TRAIN_USERS,
    "User mapping IDs unique"
)

check(
    item_mapping["train_item_id"].nunique()
    == EXPECTED_TRAIN_ITEMS,
    "Item mapping IDs unique"
)


# ============================================================
# 8. MODEL-ID RANGE
# ============================================================

print("\nChecking model-ID ranges...")

check(
    train_model_df["train_user_id"].min() == 0,
    "Minimum train_user_id = 0"
)

check(
    train_model_df["train_user_id"].max()
    == EXPECTED_TRAIN_USERS - 1,
    "Maximum train_user_id correct"
)

check(
    train_model_df["train_item_id"].min() == 0,
    "Minimum train_item_id = 0"
)

check(
    train_model_df["train_item_id"].max()
    == EXPECTED_TRAIN_ITEMS - 1,
    "Maximum train_item_id correct"
)


# ============================================================
# 9. LIGHTGCN GRAPH
# ============================================================

print("\nLoading LightGCN graph...")

graph = torch.load(
    LIGHTGCN_GRAPH_FILE,
    map_location="cpu"
)

if isinstance(graph, dict):
    adjacency = graph.get("adjacency", graph.get("graph"))

    if adjacency is None:
        # Try common single-key dictionary format
        tensor_values = [
            value for value in graph.values()
            if torch.is_tensor(value)
        ]

        if len(tensor_values) == 1:
            adjacency = tensor_values[0]
else:
    adjacency = graph

check(
    adjacency is not None,
    "LightGCN adjacency loaded"
)

expected_nodes = (
    EXPECTED_TRAIN_USERS +
    EXPECTED_TRAIN_ITEMS
)

if adjacency is not None:

    print(f"Adjacency shape: {tuple(adjacency.shape)}")

    check(
        tuple(adjacency.shape)
        == (expected_nodes, expected_nodes),
        "LightGCN graph shape"
    )

    check(
        adjacency.is_sparse,
        "LightGCN graph is sparse"
    )


# ============================================================
# 10. LIGHTGCN MODEL
# ============================================================

print("\nLoading LightGCN model...")

lightgcn = load_lightgcn(
    LIGHTGCN_MODEL_FILE,
    num_users=EXPECTED_TRAIN_USERS,
    num_items=EXPECTED_TRAIN_ITEMS,
    embedding_dim=EMBEDDING_DIM,
    num_layers=LIGHTGCN_LAYERS,
)

lightgcn_params = sum(
    p.numel()
    for p in lightgcn.parameters()
)

print(f"LightGCN parameters: {lightgcn_params:,}")

check(
    lightgcn_params == EXPECTED_LIGHTGCN_PARAMS,
    "LightGCN parameter count"
)

check(
    lightgcn.embedding.weight.shape
    == (
        expected_nodes,
        EMBEDDING_DIM
    ),
    "LightGCN embedding shape"
)


# ============================================================
# 11. LIGHTGCN PRECOMPUTED EMBEDDINGS
# ============================================================

print("\nLoading LightGCN precomputed embeddings...")

user_embeddings = np.load(
    LIGHTGCN_USER_EMBEDDINGS_FILE
)

item_embeddings = np.load(
    LIGHTGCN_ITEM_EMBEDDINGS_FILE
)

print(
    f"User embeddings: {user_embeddings.shape}"
)

print(
    f"Item embeddings: {item_embeddings.shape}"
)

check(
    user_embeddings.shape
    == (
        EXPECTED_TRAIN_USERS,
        EMBEDDING_DIM
    ),
    "LightGCN user embedding shape"
)

check(
    item_embeddings.shape
    == (
        EXPECTED_TRAIN_ITEMS,
        EMBEDDING_DIM
    ),
    "LightGCN item embedding shape"
)

check(
    user_embeddings.dtype == np.float32,
    "LightGCN user embeddings are float32"
)

check(
    item_embeddings.dtype == np.float32,
    "LightGCN item embeddings are float32"
)

check(
    np.isfinite(user_embeddings).all(),
    "LightGCN user embeddings finite"
)

check(
    np.isfinite(item_embeddings).all(),
    "LightGCN item embeddings finite"
)


# ============================================================
# 12. SASREC MODEL
# ============================================================

print("\nLoading SASRec model...")

sasrec = load_sasrec(
    SASREC_MODEL_FILE
)

sasrec_params = sum(
    p.numel()
    for p in sasrec.parameters()
)

print(f"SASRec parameters: {sasrec_params:,}")

check(
    sasrec_params == EXPECTED_SASREC_PARAMS,
    "SASRec parameter count"
)

check(
    sasrec.embedding_dim == EMBEDDING_DIM,
    "SASRec embedding dimension"
)

check(
    sasrec.max_seq_len == SASREC_MAX_SEQ_LEN,
    "SASRec maximum sequence length"
)


# ============================================================
# 13. SASREC ITEM EMBEDDING
# ============================================================

print("\nChecking SASRec item embeddings...")

sasrec_item_embedding = sasrec.item_embedding.weight

print(
    f"SASRec item embedding shape: "
    f"{tuple(sasrec_item_embedding.shape)}"
)

check(
    tuple(sasrec_item_embedding.shape)
    == (
        EXPECTED_TRAIN_ITEMS + 1,
        EMBEDDING_DIM
    ),
    "SASRec item embedding shape including padding"
)

padding_norm = torch.norm(
    sasrec_item_embedding[0]
).item()

print(
    f"SASRec padding embedding norm: "
    f"{padding_norm:.10f}"
)

check(
    padding_norm == 0.0,
    "SASRec padding embedding is zero"
)


# ============================================================
# 14. SASREC NUMERICAL VALIDITY
# ============================================================

print("\nChecking SASRec numerical validity...")

all_sasrec_finite = True

for name, parameter in sasrec.named_parameters():
    if not torch.isfinite(parameter).all():
        print(
            f"[FAIL] Non-finite SASRec parameter: {name}"
        )
        all_sasrec_finite = False

check(
    all_sasrec_finite,
    "All SASRec parameters finite"
)


# ============================================================
# 15. MODEL VOCABULARY ALIGNMENT
# ============================================================

print("\nChecking LightGCN/SASRec vocabulary alignment...")

check(
    EXPECTED_TRAIN_USERS == EXPECTED_TRAIN_USERS,
    "User vocabulary aligned"
)

check(
    EXPECTED_TRAIN_ITEMS == EXPECTED_TRAIN_ITEMS,
    "Item vocabulary aligned"
)

check(
    user_embeddings.shape[0]
    == sasrec.num_users,
    "User count compatible with SASRec"
)

check(
    item_embeddings.shape[0]
    == sasrec.num_items,
    "Item count compatible with SASRec"
)


# ============================================================
# 16. VALIDATION WARM-START POPULATION
# ============================================================

print("\nBuilding warm-start validation population...")

train_user_set = set(
    train_df["user_index"].unique()
)

train_item_set = set(
    train_df["item_index"].unique()
)

val_warm = val_df[
    val_df["user_index"].isin(train_user_set)
    &
    val_df["item_index"].isin(train_item_set)
].copy()

warm_users = val_warm["user_index"].nunique()
warm_interactions = len(val_warm)

print(
    f"Warm-start validation users: {warm_users}"
)

print(
    f"Warm-start validation interactions: "
    f"{warm_interactions}"
)

check(
    warm_users == EXPECTED_WARM_USERS,
    "Warm-start validation user count"
)

check(
    warm_interactions == EXPECTED_WARM_INTERACTIONS,
    "Warm-start validation interaction count"
)


# ============================================================
# 17. TRAINING HISTORY COMPATIBILITY
# ============================================================

print("\nChecking SASRec history compatibility...")

train_history_users = (
    train_model_df["train_user_id"].nunique()
)

check(
    train_history_users
    == EXPECTED_TRAIN_USERS,
    "Every training user has model-ID history"
)

check(
    len(train_model_df)
    == EXPECTED_TRAIN_INTERACTIONS,
    "Training history interaction count"
)


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("HYBRID DATA VALIDATION SUMMARY")
print("=" * 70)

print(
    f"Training interactions : "
    f"{EXPECTED_TRAIN_INTERACTIONS:,}"
)

print(
    f"Training users        : "
    f"{EXPECTED_TRAIN_USERS:,}"
)

print(
    f"Training items        : "
    f"{EXPECTED_TRAIN_ITEMS:,}"
)

print(
    f"Warm-start users      : "
    f"{EXPECTED_WARM_USERS:,}"
)

print(
    f"Warm-start interactions: "
    f"{EXPECTED_WARM_INTERACTIONS:,}"
)

print(
    f"LightGCN parameters   : "
    f"{lightgcn_params:,}"
)

print(
    f"SASRec parameters     : "
    f"{sasrec_params:,}"
)

print(
    f"Embedding dimension   : "
    f"{EMBEDDING_DIM}"
)

print(
    f"LightGCN layers       : "
    f"{LIGHTGCN_LAYERS}"
)

print(
    f"SASRec max sequence   : "
    f"{SASREC_MAX_SEQ_LEN}"
)

print()
print(f"Checks passed: {passed}")
print(f"Checks failed: {failed}")

if failed == 0:
    print()
    print("=" * 70)
    print("HYBRID DATA VALIDATION COMPLETE")
    print("ALL HYBRID DATA CHECKS PASSED")
    print("=" * 70)
else:
    print()
    print("=" * 70)
    print("HYBRID DATA VALIDATION FAILED")
    print("FIX THE FAILED CHECKS BEFORE CONTINUING")
    print("=" * 70)

    raise SystemExit(1)