import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

# V3 training data
V3_USERS_FILE = OUTPUT_DIR / "v3_train_users.npy"
V3_POSITIVE_FILE = OUTPUT_DIR / "v3_train_positive_items.npy"
V3_RANDOM_NEGATIVE_FILE = OUTPUT_DIR / "v3_train_random_negative_items.npy"
V3_SEQUENCE_FILE = OUTPUT_DIR / "v3_train_sequences.npy"
V3_METADATA_FILE = OUTPUT_DIR / "v3_training_data_metadata.json"

# Pretrained encoders
LIGHTGCN_MODEL_FILE = OUTPUT_DIR / "lightgcn_model.pt"
LIGHTGCN_USER_EMBEDDINGS_FILE = OUTPUT_DIR / "lightgcn_user_embeddings.npy"
LIGHTGCN_ITEM_EMBEDDINGS_FILE = OUTPUT_DIR / "lightgcn_item_embeddings.npy"
SASREC_MODEL_FILE = OUTPUT_DIR / "sasrec_model.pt"

# Output
SASREC_REPRESENTATIONS_FILE = (
    OUTPUT_DIR / "v3a_train_sasrec_representations.npy"
)

SASREC_REP_METADATA_FILE = (
    OUTPUT_DIR / "v3a_sasrec_representation_metadata.json"
)

MODEL_FILE = OUTPUT_DIR / "hybrid_v3a_model.pt"
METRICS_FILE = OUTPUT_DIR / "hybrid_v3a_training_metrics.json"


# =============================================================================
# CONFIGURATION
# =============================================================================

NUM_USERS = 778_421
NUM_ITEMS = 74_541

EMBEDDING_DIM = 64
MAX_SEQUENCE_LENGTH = 50

NUM_EXAMPLES = 2_000_000

EPOCHS = 5
BATCH_SIZE = 4_096

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0

SEED = 42

SASREC_REP_BATCH_SIZE = 512

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# UTILITY
# =============================================================================

def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_numpy_copy(path):
    """
    Load a NumPy array as a writable contiguous array.

    Copying is intentional because torch.from_numpy() should not receive
    read-only mmap arrays.
    """
    return np.array(np.load(path, mmap_mode="r"), copy=True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# =============================================================================
# IMPORT MODEL
# =============================================================================

import sys

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hybrid_model import HybridLightGCNSASRecV3


# =============================================================================
# CHECK FILES
# =============================================================================

def check_required_files():

    print_header("CHECKING REQUIRED FILES")

    required_files = [
        V3_USERS_FILE,
        V3_POSITIVE_FILE,
        V3_RANDOM_NEGATIVE_FILE,
        V3_SEQUENCE_FILE,
        V3_METADATA_FILE,
        LIGHTGCN_MODEL_FILE,
        LIGHTGCN_USER_EMBEDDINGS_FILE,
        LIGHTGCN_ITEM_EMBEDDINGS_FILE,
        SASREC_MODEL_FILE,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

        print(f"[PASS] {path.name}")


# =============================================================================
# LOAD V3 TRAINING DATA
# =============================================================================

def load_v3_training_data():

    print_header("LOADING V3-A TRAINING DATA")

    users = load_numpy_copy(V3_USERS_FILE)
    positive_items = load_numpy_copy(V3_POSITIVE_FILE)
    random_negative_items = load_numpy_copy(
        V3_RANDOM_NEGATIVE_FILE
    )
    sequences = load_numpy_copy(V3_SEQUENCE_FILE)

    metadata = json.loads(
        V3_METADATA_FILE.read_text(encoding="utf-8")
    )

    print(f"Users shape: {users.shape}")
    print(f"Positive shape: {positive_items.shape}")
    print(f"Random negative shape: {random_negative_items.shape}")
    print(f"Sequence shape: {sequences.shape}")

    if len(users) != NUM_EXAMPLES:
        raise ValueError(
            f"Expected {NUM_EXAMPLES} examples, "
            f"found {len(users)}"
        )

    if positive_items.shape != (NUM_EXAMPLES,):
        raise ValueError("Positive item shape is incorrect.")

    if random_negative_items.shape != (NUM_EXAMPLES,):
        raise ValueError("Random negative shape is incorrect.")

    if sequences.shape != (
        NUM_EXAMPLES,
        MAX_SEQUENCE_LENGTH,
    ):
        raise ValueError("Sequence shape is incorrect.")

    # -------------------------------------------------------------------------
    # Range checks
    # -------------------------------------------------------------------------

    if users.min() < 0 or users.max() >= NUM_USERS:
        raise ValueError("User IDs are outside training vocabulary.")

    if (
        positive_items.min() < 0
        or positive_items.max() >= NUM_ITEMS
    ):
        raise ValueError(
            "Positive item IDs are outside training vocabulary."
        )

    if (
        random_negative_items.min() < 0
        or random_negative_items.max() >= NUM_ITEMS
    ):
        raise ValueError(
            "Random negative item IDs are outside training vocabulary."
        )

    if np.any(positive_items == random_negative_items):
        raise ValueError(
            "At least one random negative equals its positive item."
        )

    # Padding convention:
    # 0 = padding
    # Actual SASRec item IDs = train_item_id + 1
    if sequences.min() < 0:
        raise ValueError("Sequence contains negative values.")

    if sequences.max() > NUM_ITEMS:
        raise ValueError(
            "Sequence contains invalid SASRec item IDs."
        )

    print("[PASS] Training data shapes")
    print("[PASS] User ID ranges")
    print("[PASS] Positive item ranges")
    print("[PASS] Random negative item ranges")
    print("[PASS] Positive != random negative")
    print("[PASS] Sequence ranges")
    print(
        f"[PASS] Loaded {NUM_EXAMPLES:,} V3-A examples"
    )

    metadata_examples = metadata.get(
        "num_examples",
        "unknown",
    )

    if isinstance(metadata_examples, (int, float)):
        print(
            f"Metadata examples: "
            f"{int(metadata_examples):,}"
        )
    else:
        print(
            f"Metadata examples: "
            f"{metadata_examples}"
        )

        return (
            users,
            positive_items,
            random_negative_items,
            sequences,
            metadata,
        )


# =============================================================================
# LOAD LIGHTGCN EMBEDDINGS
# =============================================================================

def load_lightgcn_embeddings():

    print_header("LOADING LIGHTGCN EMBEDDINGS")

    user_embeddings = load_numpy_copy(
        LIGHTGCN_USER_EMBEDDINGS_FILE
    )

    item_embeddings = load_numpy_copy(
        LIGHTGCN_ITEM_EMBEDDINGS_FILE
    )

    expected_user_shape = (
        NUM_USERS,
        EMBEDDING_DIM,
    )

    expected_item_shape = (
        NUM_ITEMS,
        EMBEDDING_DIM,
    )

    print(
        f"User embeddings: {user_embeddings.shape}"
    )
    print(
        f"Item embeddings: {item_embeddings.shape}"
    )

    if user_embeddings.shape != expected_user_shape:
        raise ValueError(
            f"Unexpected LightGCN user embedding shape: "
            f"{user_embeddings.shape}"
        )

    if item_embeddings.shape != expected_item_shape:
        raise ValueError(
            f"Unexpected LightGCN item embedding shape: "
            f"{item_embeddings.shape}"
        )

    if not np.isfinite(user_embeddings).all():
        raise ValueError(
            "LightGCN user embeddings contain NaN/Inf."
        )

    if not np.isfinite(item_embeddings).all():
        raise ValueError(
            "LightGCN item embeddings contain NaN/Inf."
        )

    print("[PASS] LightGCN user embeddings")
    print("[PASS] LightGCN item embeddings")
    print("[PASS] LightGCN embeddings are finite")

    return user_embeddings, item_embeddings


# =============================================================================
# LOAD SASREC
# =============================================================================

def remap_sasrec_checkpoint_keys(state_dict):

    """
    Convert older checkpoint naming to the current V3 SASRec naming.

    Older:
        transformer_encoder.layers.X.*
        layer_norm.*

    Current:
        transformer_layers.X.*
        final_layer_norm.*
    """

    converted = {}

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

        elif new_key.startswith("layer_norm."):
            new_key = new_key.replace(
                "layer_norm.",
                "final_layer_norm.",
                1,
            )

        converted[new_key] = value

    return converted


def load_sasrec_model():

    print_header("LOADING SASREC MODEL")

    checkpoint = torch.load(
        SASREC_MODEL_FILE,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    state_dict = remap_sasrec_checkpoint_keys(
        state_dict
    )

    # Import the SASRec class from the current hybrid model.
    from hybrid_model import SASRec

    sasrec = SASRec(
        num_items=NUM_ITEMS,
        embedding_dim=EMBEDDING_DIM,
        max_seq_len=MAX_SEQUENCE_LENGTH,
        num_heads=2,
        num_layers=2,
        dropout=0.2,
    )

    result = sasrec.load_state_dict(
        state_dict,
        strict=False,
    )

    if result.missing_keys:
        raise RuntimeError(
            "SASRec checkpoint missing keys:\n"
            + "\n".join(result.missing_keys)
        )

    if result.unexpected_keys:
        raise RuntimeError(
            "SASRec checkpoint has unexpected keys:\n"
            + "\n".join(result.unexpected_keys)
        )

    sasrec.eval()

    for parameter in sasrec.parameters():
        parameter.requires_grad = False

    parameter_count = count_parameters(sasrec)

    print(
        f"SASRec parameters: {parameter_count:,}"
    )

    print(
        f"SASRec training mode: {sasrec.training}"
    )

    print(
        "[PASS] SASRec checkpoint loaded strictly"
    )

    print(
        "[PASS] SASRec frozen"
    )

    print(
        "[PASS] SASRec evaluation mode"
    )

    if parameter_count != 4_873_984:
        raise ValueError(
            f"Unexpected SASRec parameter count: "
            f"{parameter_count}"
        )

    return sasrec


# =============================================================================
# PRECOMPUTE SASREC PREFIX REPRESENTATIONS
# =============================================================================

def precompute_sasrec_representations(
    sasrec,
    sequences,
):

    print_header(
        "PRECOMPUTING V3-A SASREC PREFIX REPRESENTATIONS"
    )

    if SASREC_REPRESENTATIONS_FILE.exists():

        print(
            "Existing V3-A SASRec representation file found."
        )
        print(
            f"Loading: "
            f"{SASREC_REPRESENTATIONS_FILE.name}"
        )

        representations = np.load(
            SASREC_REPRESENTATIONS_FILE,
            mmap_mode="r",
        )

        expected_shape = (
            NUM_EXAMPLES,
            EMBEDDING_DIM,
        )

        if representations.shape != expected_shape:
            raise ValueError(
                "Existing SASRec representation shape "
                "does not match current V3-A dataset."
            )

        if not np.isfinite(
            representations[:10000]
        ).all():
            raise ValueError(
                "Existing SASRec representations contain "
                "NaN/Inf."
            )

        print(
            f"Representation shape: "
            f"{representations.shape}"
        )

        print(
            "[PASS] Existing representations are compatible."
        )

        return representations

    print(
        "No existing representation file found."
    )

    print(
        "Generating SASRec representations once..."
    )

    sasrec = sasrec.to(DEVICE)
    sasrec.eval()

    # -------------------------------------------------------------------------
    # Allocate memory-mapped output directly on disk.
    #
    # 2,000,000 x 64 x float32 ≈ 512 MB.
    # -------------------------------------------------------------------------

    representations = np.lib.format.open_memmap(
        SASREC_REPRESENTATIONS_FILE,
        mode="w+",
        dtype=np.float32,
        shape=(
            NUM_EXAMPLES,
            EMBEDDING_DIM,
        ),
    )

    total_start = time.time()

    with torch.no_grad():

        for start in range(
            0,
            NUM_EXAMPLES,
            SASREC_REP_BATCH_SIZE,
        ):

            end = min(
                start + SASREC_REP_BATCH_SIZE,
                NUM_EXAMPLES,
            )

            batch_sequences = torch.from_numpy(
                sequences[start:end]
            ).long().to(DEVICE)

            # Current V3 SASRec implementation exposes
            # get_sequence_representation().
            if hasattr(
                sasrec,
                "get_sequence_representation",
            ):

                batch_representation = (
                    sasrec.get_sequence_representation(
                        batch_sequences
                    )
                )

            else:

                # Fallback for compatible SASRec
                # implementations.
                outputs = sasrec(
                    batch_sequences
                )

                if outputs.ndim != 3:
                    raise RuntimeError(
                        "Unexpected SASRec output shape: "
                        f"{outputs.shape}"
                    )

                non_padding = (
                    batch_sequences != 0
                ).sum(dim=1)

                last_positions = torch.clamp(
                    non_padding - 1,
                    min=0,
                )

                batch_indices = torch.arange(
                    end - start,
                    device=DEVICE,
                )

                batch_representation = outputs[
                    batch_indices,
                    last_positions,
                ]

            batch_representation = (
                batch_representation
                .detach()
                .cpu()
                .float()
                .numpy()
            )

            if batch_representation.shape != (
                end - start,
                EMBEDDING_DIM,
            ):
                raise RuntimeError(
                    "Unexpected SASRec representation "
                    f"shape: {batch_representation.shape}"
                )

            representations[
                start:end
            ] = batch_representation

            if (
                (start // SASREC_REP_BATCH_SIZE) % 10 == 0
                or end == NUM_EXAMPLES
            ):

                elapsed = time.time() - total_start

                rate = (
                    end / elapsed
                    if elapsed > 0
                    else 0
                )

                print(
                    f"SASRec representations "
                    f"{end:,}/{NUM_EXAMPLES:,} "
                    f"at ~{rate:.1f} examples/sec"
                )

    representations.flush()

    total_time = time.time() - total_start

    print(
        f"SASRec representation generation: "
        f"{total_time:.2f}s"
    )

    print(
        f"Representation shape: "
        f"{representations.shape}"
    )

    # -------------------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------------------

    sample = np.array(
        representations[:10_000],
        copy=True,
    )

    if not np.isfinite(sample).all():
        raise ValueError(
            "SASRec representations contain NaN/Inf."
        )

    print(
        f"Representation mean: "
        f"{sample.mean():.6f}"
    )

    print(
        f"Representation std: "
        f"{sample.std():.6f}"
    )

    print(
        "[PASS] SASRec representations finite"
    )

    metadata = {
        "num_examples": NUM_EXAMPLES,
        "embedding_dim": EMBEDDING_DIM,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "source_sequence_file": str(
            V3_SEQUENCE_FILE
        ),
        "source_training_data": "V3-A",
        "generation_time_seconds": total_time,
        "device": str(DEVICE),
    }

    SASREC_REP_METADATA_FILE.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Saved: "
        f"{SASREC_REPRESENTATIONS_FILE.name}"
    )

    print(
        f"Saved: "
        f"{SASREC_REP_METADATA_FILE.name}"
    )

    return representations


# =============================================================================
# LOAD PRETRAINED LIGHTGCN INTO V3 MODEL
# =============================================================================

def load_lightgcn_checkpoint(model):

    print_header(
        "LOADING LIGHTGCN CHECKPOINT"
    )

    checkpoint = torch.load(
        LIGHTGCN_MODEL_FILE,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    result = model.lightgcn.load_state_dict(
        state_dict,
        strict=True,
    )

    if result.missing_keys:
        raise RuntimeError(
            f"LightGCN missing keys: "
            f"{result.missing_keys}"
        )

    if result.unexpected_keys:
        raise RuntimeError(
            f"LightGCN unexpected keys: "
            f"{result.unexpected_keys}"
        )

    print(
        "[PASS] LightGCN checkpoint loaded"
    )


# =============================================================================
# LOAD PRETRAINED SASREC INTO V3 MODEL
# =============================================================================

def load_sasrec_checkpoint(model):

    print_header(
        "LOADING SASREC CHECKPOINT INTO V3 MODEL"
    )

    checkpoint = torch.load(
        SASREC_MODEL_FILE,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    state_dict = remap_sasrec_checkpoint_keys(
        state_dict
    )

    result = model.sasrec.load_state_dict(
        state_dict,
        strict=False,
    )

    if result.missing_keys:
        raise RuntimeError(
            "SASRec missing keys:\n"
            + "\n".join(result.missing_keys)
        )

    if result.unexpected_keys:
        raise RuntimeError(
            "SASRec unexpected keys:\n"
            + "\n".join(result.unexpected_keys)
        )

    print(
        "[PASS] SASRec checkpoint loaded"
    )


# =============================================================================
# BUILD V3-A MODEL
# =============================================================================

def build_model():

    print_header("BUILDING V3-A MODEL")

    model = HybridLightGCNSASRecV3(
        num_users=NUM_USERS,
        num_items=NUM_ITEMS,
        embedding_dim=EMBEDDING_DIM,
        max_seq_len=MAX_SEQUENCE_LENGTH,
        num_heads=2,
        num_layers=2,
        dropout=0.2,
    )

    print(
        f"Total parameters before loading: "
        f"{count_parameters(model):,}"
    )

    expected_total_parameters = 59_504_832

    if count_parameters(model) != expected_total_parameters:
        raise ValueError(
            "Unexpected V3 total parameter count: "
            f"{count_parameters(model):,}"
        )

    load_lightgcn_checkpoint(model)
    load_sasrec_checkpoint(model)

    # Freeze pretrained encoders.
    model.freeze_pretrained_encoders()

    # -------------------------------------------------------------------------
    # Explicitly keep pretrained encoders in eval mode.
    # This prevents SASRec dropout from being active during fusion training.
    # -------------------------------------------------------------------------

    model.lightgcn.eval()
    model.sasrec.eval()

    # User fusion / gate / item fusion remain trainable.
    model.user_gate.train()
    model.user_fusion.train()
    model.item_fusion.train()
    model.fusion_dropout.train()

    trainable_parameters = count_trainable_parameters(
        model
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    expected_trainable = 41_280

    if trainable_parameters != expected_trainable:
        raise ValueError(
            f"Expected {expected_trainable:,} trainable "
            f"parameters, found {trainable_parameters:,}"
        )

    # -------------------------------------------------------------------------
    # Verify encoder freezing
    # -------------------------------------------------------------------------

    lightgcn_trainable = any(
        p.requires_grad
        for p in model.lightgcn.parameters()
    )

    sasrec_trainable = any(
        p.requires_grad
        for p in model.sasrec.parameters()
    )

    if lightgcn_trainable:
        raise RuntimeError(
            "LightGCN contains trainable parameters."
        )

    if sasrec_trainable:
        raise RuntimeError(
            "SASRec contains trainable parameters."
        )

    print(
        "[PASS] LightGCN frozen"
    )

    print(
        "[PASS] SASRec frozen"
    )

    print(
        "[PASS] V3-A trainable parameter count"
    )

    return model


# =============================================================================
# PREPARE FROZEN ITEM COMPONENT REPRESENTATIONS
# =============================================================================

def prepare_item_component_representations(
    model,
    lightgcn_item_embeddings,
):
    """
    Prepare the frozen item-side components used by V3-A training.

    IMPORTANT:
        The final item_fusion layer must NOT be applied here under
        torch.no_grad(). item_fusion is trainable, so applying it here
        would detach its output from the computation graph and cause:

            item_fusion.* received no gradient

        Instead, this function only prepares the frozen normalized
        LightGCN and SASRec item embeddings. item_fusion is applied
        inside every training batch with gradients enabled.
    """

    print_header(
        "PREPARING V3-A ITEM COMPONENT REPRESENTATIONS"
    )

    lightgcn_items = torch.from_numpy(
        lightgcn_item_embeddings
    ).float().to(DEVICE)

    if lightgcn_items.shape != (
        NUM_ITEMS,
        EMBEDDING_DIM,
    ):
        raise RuntimeError(
            "Unexpected LightGCN item embedding shape: "
            f"{tuple(lightgcn_items.shape)}"
        )

    with torch.no_grad():
        normalized_graph_items = (
            torch.nn.functional.normalize(
                lightgcn_items,
                p=2,
                dim=1,
            )
        )

        # Row 0 is SASRec padding. Remove it so model item IDs
        # 0..NUM_ITEMS-1 align directly with the training item IDs.
        sasrec_items = (
            model.sasrec.item_embedding.weight[1:]
        )

        if sasrec_items.shape != (
            NUM_ITEMS,
            EMBEDDING_DIM,
        ):
            raise RuntimeError(
                "Unexpected SASRec item embedding shape: "
                f"{tuple(sasrec_items.shape)}"
            )

        normalized_sequence_items = (
            torch.nn.functional.normalize(
                sasrec_items,
                p=2,
                dim=1,
            )
        )

        # Copy the frozen component representations so they are
        # independent of the model's trainable item_fusion graph.
        normalized_graph_items = (
            normalized_graph_items.detach()
        )

        normalized_sequence_items = (
            normalized_sequence_items.detach()
        )

    if not torch.isfinite(
        normalized_graph_items
    ).all():
        raise RuntimeError(
            "Normalized LightGCN item representations contain NaN/Inf."
        )

    if not torch.isfinite(
        normalized_sequence_items
    ).all():
        raise RuntimeError(
            "Normalized SASRec item representations contain NaN/Inf."
        )

    print(
        f"Normalized LightGCN item shape: "
        f"{tuple(normalized_graph_items.shape)}"
    )

    print(
        f"Normalized SASRec item shape: "
        f"{tuple(normalized_sequence_items.shape)}"
    )

    print(
        "[PASS] Frozen item component representations finite"
    )

    return (
        normalized_graph_items,
        normalized_sequence_items,
    )


# =============================================================================
# TRAINING

# =============================================================================

# =============================================================================
# INDEXED TRAINING DATASET
# =============================================================================

class IndexedTensorDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        users,
        positive_items,
        random_negative_items,
    ):
        self.users = users
        self.positive_items = positive_items
        self.random_negative_items = (
            random_negative_items
        )

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):

        return (
            index,
            self.users[index],
            self.positive_items[index],
            self.random_negative_items[index],
        )


# =============================================================================
# ACTUAL V3-A TRAINING
# =============================================================================

def run_training(
    model,
    users,
    positive_items,
    random_negative_items,
    sasrec_representations,
    lightgcn_user_embeddings,
    normalized_graph_items,
    normalized_sequence_items,
):

    print_header(
        "RUNNING V3-A TRAINING"
    )

    dataset = IndexedTensorDataset(
        torch.from_numpy(
            users.astype(np.int64)
        ),
        torch.from_numpy(
            positive_items.astype(np.int64)
        ),
        torch.from_numpy(
            random_negative_items.astype(np.int64)
        ),
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )

    optimizer = torch.optim.Adam(
        model.trainable_parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    lightgcn_users = torch.from_numpy(
        lightgcn_user_embeddings
    ).float().to(DEVICE)

    sasrec_representations = np.asarray(
        sasrec_representations
    )


    training_history = []

    total_training_start = time.time()

    for epoch in range(1, EPOCHS + 1):

        epoch_start = time.time()

        # ---------------------------------------------------------------------
        # Mode control
        # ---------------------------------------------------------------------

        model.lightgcn.eval()
        model.sasrec.eval()

        model.user_gate.train()
        model.user_fusion.train()
        model.item_fusion.train()
        model.fusion_dropout.train()

        epoch_loss = 0.0
        epoch_examples = 0

        gate_sum = 0.0
        gate_count = 0

        for (
            batch_indices,
            batch_users,
            batch_positive,
            batch_random_negative,
        ) in dataloader:

            batch_indices_np = (
                batch_indices.numpy()
            )

            batch_users = batch_users.to(
                DEVICE,
                non_blocking=True,
            )

            batch_positive = batch_positive.to(
                DEVICE,
                non_blocking=True,
            )

            batch_random_negative = (
                batch_random_negative.to(
                    DEVICE,
                    non_blocking=True,
                )
            )

            # -------------------------------------------------------------
            # LightGCN user representation
            # -------------------------------------------------------------

            graph_user = lightgcn_users[
                batch_users
            ]

            # -------------------------------------------------------------
            # Precomputed SASRec sequence representation
            # -------------------------------------------------------------

            sequence_user = torch.from_numpy(
                np.array(
                    sasrec_representations[
                        batch_indices_np
                    ],
                    copy=True,
                )
            ).float().to(DEVICE)

            # -------------------------------------------------------------
            # V3 L2 normalization
            # -------------------------------------------------------------

            normalized_graph_user = (
                torch.nn.functional.normalize(
                    graph_user,
                    p=2,
                    dim=1,
                )
            )

            normalized_sequence_user = (
                torch.nn.functional.normalize(
                    sequence_user,
                    p=2,
                    dim=1,
                )
            )

            # -------------------------------------------------------------
            # Adaptive graph/sequence gate
            #
            # gate = sigmoid(W[g;s] + b)
            #
            # adaptive_graph = gate * graph
            # adaptive_sequence = (1-gate) * sequence
            # -------------------------------------------------------------

            combined_user = torch.cat(
                [
                    normalized_graph_user,
                    normalized_sequence_user,
                ],
                dim=1,
            )

            gate = torch.sigmoid(
                model.user_gate(
                    combined_user
                )
            )

            adaptive_graph = (
                gate * normalized_graph_user
            )

            adaptive_sequence = (
                (1.0 - gate)
                * normalized_sequence_user
            )

            adaptive_user = torch.cat(
                [
                    adaptive_graph,
                    adaptive_sequence,
                ],
                dim=1,
            )

            hybrid_user = model.user_fusion(
                adaptive_user
            )

            hybrid_user = model.fusion_dropout(
                hybrid_user
            )

            # -------------------------------------------------------------
            # V3-A item representations
            #
            # The LightGCN/SASRec item components are frozen.
            # item_fusion is TRAINABLE and is therefore applied here
            # with gradients enabled.
            # -------------------------------------------------------------

            positive_graph_items = normalized_graph_items[
                batch_positive
            ]

            positive_sequence_items = normalized_sequence_items[
                batch_positive
            ]

            negative_graph_items = normalized_graph_items[
                batch_random_negative
            ]

            negative_sequence_items = normalized_sequence_items[
                batch_random_negative
            ]

            positive_item_input = torch.cat(
                [
                    positive_graph_items,
                    positive_sequence_items,
                ],
                dim=1,
            )

            negative_item_input = torch.cat(
                [
                    negative_graph_items,
                    negative_sequence_items,
                ],
                dim=1,
            )

            positive_items = model.item_fusion(
                positive_item_input
            )

            negative_items = model.item_fusion(
                negative_item_input
            )

            current_batch_size = batch_users.shape[0]
            if positive_items.shape != (
                current_batch_size,
                EMBEDDING_DIM * 2,
            ):
                raise RuntimeError(
                    "Unexpected positive item representation shape: "
                    f"{tuple(positive_items.shape)}"
                )

            if negative_items.shape != (
                current_batch_size,
                EMBEDDING_DIM * 2,
            ):
                raise RuntimeError(
                    "Unexpected negative item representation shape: "
                    f"{tuple(negative_items.shape)}"
                )

            # -------------------------------------------------------------
            # Scores
            # -------------------------------------------------------------

            positive_scores = (
                hybrid_user
                * positive_items
            ).sum(dim=1)

            negative_scores = (
                hybrid_user
                * negative_items
            ).sum(dim=1)

            # -------------------------------------------------------------
            # BPR loss
            # -------------------------------------------------------------

            loss = -torch.nn.functional.logsigmoid(
                positive_scores
                - negative_scores
            ).mean()

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite V3-A loss at epoch "
                    f"{epoch}."
                )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            # -------------------------------------------------------------
            # Validate gradients
            # -------------------------------------------------------------

            trainable_gradient_names = []

            for name, parameter in model.named_parameters():

                if parameter.requires_grad:

                    if parameter.grad is None:
                        raise RuntimeError(
                            f"Trainable parameter "
                            f"{name} received no gradient."
                        )

                    if not torch.isfinite(
                        parameter.grad
                    ).all():
                        raise RuntimeError(
                            f"Non-finite gradient in "
                            f"{name}."
                        )

                    trainable_gradient_names.append(name)

            # Explicitly verify the three V3-A trainable components.
            required_gradient_prefixes = [
                "user_gate.",
                "user_fusion.",
                "item_fusion.",
            ]

            for prefix in required_gradient_prefixes:
                if not any(
                    name.startswith(prefix)
                    for name in trainable_gradient_names
                ):
                    raise RuntimeError(
                        f"No gradient found for required "
                        f"trainable module: {prefix}"
                    )

            # Validate frozen encoders remain gradient-free.
            for name, parameter in model.lightgcn.named_parameters():
                if parameter.grad is not None:
                    raise RuntimeError(
                        "Frozen LightGCN parameter received a gradient: "
                        f"{name}"
                    )

            for name, parameter in model.sasrec.named_parameters():
                if parameter.grad is not None:
                    raise RuntimeError(
                        "Frozen SASRec parameter received a gradient: "
                        f"{name}"
                    )

            optimizer.step()

            # -------------------------------------------------------------
            # Statistics
            # -------------------------------------------------------------

            epoch_loss += (
                loss.item()
                * batch_users.size(0)
            )

            epoch_examples += (
                batch_users.size(0)
            )

            gate_sum += gate.detach().mean().item() * (
                batch_users.size(0)
            )

            gate_count += (
                batch_users.size(0)
            )

        epoch_loss /= epoch_examples

        mean_gate = (
            gate_sum / gate_count
            if gate_count > 0
            else 0.0
        )

        epoch_time = (
            time.time()
            - epoch_start
        )

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Loss: {epoch_loss:.8f} | "
            f"Mean gate: {mean_gate:.6f} | "
            f"Time: {epoch_time:.2f}s"
        )

        training_history.append(
            {
                "epoch": epoch,
                "loss": epoch_loss,
                "mean_gate": mean_gate,
                "epoch_time_seconds": epoch_time,
                "examples": epoch_examples,
            }
        )

    total_training_time = (
        time.time()
        - total_training_start
    )

    print(
        f"\nTotal V3-A training time: "
        f"{total_training_time:.2f}s "
        f"({total_training_time / 60:.2f} min)"
    )

    return training_history, total_training_time


# =============================================================================
# SAVE MODEL
# =============================================================================

def save_model(
    model,
    training_history,
    total_training_time,
):

    print_header(
        "SAVING V3-A MODEL"
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),

        "num_users": NUM_USERS,
        "num_items": NUM_ITEMS,

        "embedding_dim": EMBEDDING_DIM,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,

        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,

        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,

        "seed": SEED,

        "num_training_examples": NUM_EXAMPLES,

        "negative_strategy": "random",

        "item_fusion_training": (
            "item_fusion applied per batch with gradients enabled; "
            "frozen normalized LightGCN/SASRec item components "
            "are prepared once"
        ),

        "architecture": (
            "V3-A adaptive graph-sequential gate "
            "with separate user/item projections"
        ),

        "trainable_parameters": count_trainable_parameters(
            model
        ),

        "total_parameters": count_parameters(
            model
        ),

        "training_history": training_history,

        "total_training_time_seconds": (
            total_training_time
        ),

        "device": str(DEVICE),
    }

    torch.save(
        checkpoint,
        MODEL_FILE,
    )

    print(
        f"Saved: {MODEL_FILE}"
    )

    metrics = {
        "model": "Hybrid V3-A",
        "architecture": (
            "adaptive graph-sequential gate "
            "with separate projections"
        ),
        "negative_strategy": "random",

        "item_fusion_training": (
            "item_fusion applied per batch with gradients enabled; "
            "frozen normalized LightGCN/SASRec item components "
            "are prepared once"
        ),

        "num_users": NUM_USERS,
        "num_items": NUM_ITEMS,

        "embedding_dim": EMBEDDING_DIM,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,

        "num_training_examples": NUM_EXAMPLES,

        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,

        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,

        "trainable_parameters": (
            count_trainable_parameters(model)
        ),

        "total_parameters": (
            count_parameters(model)
        ),

        "seed": SEED,

        "total_training_time_seconds": (
            total_training_time
        ),

        "training_history": training_history,

        "device": str(DEVICE),
    }

    METRICS_FILE.write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Saved: {METRICS_FILE}"
    )


# =============================================================================
# POST-TRAINING VALIDATION
# =============================================================================

def validate_trained_model(model):

    print_header(
        "VALIDATING TRAINED V3-A MODEL"
    )

    # -------------------------------------------------------------------------
    # Parameter counts
    # -------------------------------------------------------------------------

    trainable = count_trainable_parameters(
        model
    )

    total = count_parameters(
        model
    )

    print(
        f"Trainable parameters: "
        f"{trainable:,}"
    )

    print(
        f"Total parameters: "
        f"{total:,}"
    )

    if trainable != 41_280:
        raise RuntimeError(
            f"Unexpected trainable parameter count: "
            f"{trainable}"
        )

    if total != 59_504_832:
        raise RuntimeError(
            f"Unexpected total parameter count: "
            f"{total}"
        )

    print(
        "[PASS] Trainable parameter count"
    )

    print(
        "[PASS] Total parameter count"
    )

    # -------------------------------------------------------------------------
    # Verify frozen encoders
    # -------------------------------------------------------------------------

    if any(
        p.requires_grad
        for p in model.lightgcn.parameters()
    ):
        raise RuntimeError(
            "LightGCN is not fully frozen."
        )

    if any(
        p.requires_grad
        for p in model.sasrec.parameters()
    ):
        raise RuntimeError(
            "SASRec is not fully frozen."
        )

    print(
        "[PASS] LightGCN remains frozen"
    )

    print(
        "[PASS] SASRec remains frozen"
    )

    # -------------------------------------------------------------------------
    # Verify modes
    # -------------------------------------------------------------------------

    if model.lightgcn.training:
        raise RuntimeError(
            "LightGCN must remain in eval mode."
        )

    if model.sasrec.training:
        raise RuntimeError(
            "SASRec must remain in eval mode."
        )

    print(
        "[PASS] LightGCN eval mode"
    )

    print(
        "[PASS] SASRec eval mode"
    )

    # -------------------------------------------------------------------------
    # Check trainable modules
    # -------------------------------------------------------------------------

    trainable_names = [
        name
        for name, parameter
        in model.named_parameters()
        if parameter.requires_grad
    ]

    print(
        "\nTrainable parameters:"
    )

    for name in trainable_names:
        print(
            f"  {name}"
        )

    expected_prefixes = [
        "user_gate.",
        "user_fusion.",
        "item_fusion.",
    ]

    for name in trainable_names:

        if not any(
            name.startswith(prefix)
            for prefix in expected_prefixes
        ):
            raise RuntimeError(
                "Unexpected trainable parameter: "
                f"{name}"
            )

    print(
        "[PASS] Only V3-A fusion/gate modules are trainable"
    )

    print(
        "\nV3-A MODEL VALIDATION PASSED"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print_header(
        "HYBRID V3-A TRAINING"
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Configured examples: "
        f"{NUM_EXAMPLES:,}"
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    print(
        f"Batch size: {BATCH_SIZE:,}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        f"Random negatives: ENABLED"
    )

    print(
        f"LightGCN hard negatives: NOT USED"
    )

    print(
        f"SASRec hard negatives: NOT USED"
    )

    print(
        "Item fusion: TRAINABLE PER BATCH"
    )

    print(
        f"SASRec hard negatives: NOT USED"
    )

    set_seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    check_required_files()

    (
        users,
        positive_items,
        random_negative_items,
        sequences,
        metadata,
    ) = load_v3_training_data()

    lightgcn_user_embeddings, lightgcn_item_embeddings = (
        load_lightgcn_embeddings()
    )

    # -------------------------------------------------------------------------
    # Build model and load pretrained encoders.
    # -------------------------------------------------------------------------

    model = build_model()

    # -------------------------------------------------------------------------
    # SASRec used separately for one-time prefix representation generation.
    # -------------------------------------------------------------------------

    sasrec_for_precompute = load_sasrec_model()

    sasrec_representations = (
        precompute_sasrec_representations(
            sasrec_for_precompute,
            sequences,
        )
    )

    del sasrec_for_precompute

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Move main hybrid model to device.
    # -------------------------------------------------------------------------

    model = model.to(DEVICE)

    model.lightgcn.eval()
    model.sasrec.eval()

    # -------------------------------------------------------------------------
    # Prepare frozen item components.
    #
    # IMPORTANT:
    # The final item_fusion layer remains trainable and is therefore
    # applied inside each training batch, not precomputed here.
    # -------------------------------------------------------------------------

    (
        normalized_graph_items,
        normalized_sequence_items,
    ) = prepare_item_component_representations(
        model,
        lightgcn_item_embeddings,
    )

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    training_history, total_training_time = (
        run_training(
            model=model,
            users=users,
            positive_items=positive_items,
            random_negative_items=random_negative_items,
            sasrec_representations=sasrec_representations,
            lightgcn_user_embeddings=lightgcn_user_embeddings,
            normalized_graph_items=normalized_graph_items,
            normalized_sequence_items=normalized_sequence_items,
        )
    )

    # -------------------------------------------------------------------------
    # Final validation
    # -------------------------------------------------------------------------

    validate_trained_model(
        model
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    save_model(
        model=model,
        training_history=training_history,
        total_training_time=total_training_time,
    )

    print_header(
        "V3-A TRAINING COMPLETE"
    )

    print(
        f"Model: {MODEL_FILE.name}"
    )

    print(
        f"Metrics: {METRICS_FILE.name}"
    )

    print(
        f"Training examples: "
        f"{NUM_EXAMPLES:,}"
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    print(
        f"Trainable parameters: "
        f"{count_trainable_parameters(model):,}"
    )

    print(
        f"Total parameters: "
        f"{count_parameters(model):,}"
    )

    print(
        f"Total training time: "
        f"{total_training_time / 60:.2f} min"
    )

    print(
        "\nV3-A TRAINING PASSED"
    )


if __name__ == "__main__":
    main()