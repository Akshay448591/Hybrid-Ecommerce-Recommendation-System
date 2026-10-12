"""
Train Hybrid LightGCN + SASRec Recommendation Model - V2

Hybrid V2 architecture:
    LightGCN user -> L2 normalize ─┐
                                   ├─ concat -> user projection
    SASRec user   -> L2 normalize ─┘

    LightGCN item -> L2 normalize ─┐
                                   ├─ concat -> item projection
    SASRec item   -> L2 normalize ─┘

Only the two projection layers are trained.

Pretrained LightGCN and SASRec encoders remain frozen.

Training data:
    2,000,000 pre-generated next-item examples.

The exact same examples and SASRec prefix representations
used by Hybrid V1 are reused for controlled comparison.
"""

from pathlib import Path
import json
import random
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from hybrid_model import HybridLightGCNSASRec


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"


# ============================================================
# FILES
# ============================================================

TRAIN_USERS_FILE = (
    OUTPUT_DIR / "hybrid_train_users.npy"
)

TRAIN_POSITIVE_FILE = (
    OUTPUT_DIR / "hybrid_train_positive_items.npy"
)

TRAIN_NEGATIVE_FILE = (
    OUTPUT_DIR / "hybrid_train_negative_items.npy"
)

SASREC_REPRESENTATIONS_FILE = (
    OUTPUT_DIR /
    "hybrid_train_sasrec_representations.npy"
)

LIGHTGCN_USER_EMBEDDINGS_FILE = (
    OUTPUT_DIR /
    "lightgcn_user_embeddings.npy"
)

LIGHTGCN_ITEM_EMBEDDINGS_FILE = (
    OUTPUT_DIR /
    "lightgcn_item_embeddings.npy"
)

LIGHTGCN_MODEL_FILE = (
    OUTPUT_DIR /
    "lightgcn_model.pt"
)

SASREC_MODEL_FILE = (
    OUTPUT_DIR /
    "sasrec_model.pt"
)

OUTPUT_MODEL_FILE = (
    OUTPUT_DIR /
    "hybrid_v2_model.pt"
)

OUTPUT_METRICS_FILE = (
    OUTPUT_DIR /
    "hybrid_v2_training_metrics.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

SANITY_MODE = False

SANITY_EXAMPLES = 5_000
SANITY_EPOCHS = 2

FULL_EPOCHS = 5

BATCH_SIZE = 4_096

LEARNING_RATE = 0.001

WEIGHT_DECAY = 0.0

SEED = 42

NUM_USERS = 778_421
NUM_ITEMS = 74_541

EMBEDDING_DIM = 64
MAX_SEQUENCE_LENGTH = 50

NUM_HEADS = 2
NUM_LAYERS = 2
DROPOUT = 0.2

EXPECTED_TRAINABLE_PARAMETERS = 33_024


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    """
    Set all relevant random seeds.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATASET
# ============================================================

class HybridTrainingDataset(Dataset):
    """
    Dataset backed by pre-generated NumPy arrays.

    Arrays:
        user IDs
        positive item IDs
        negative item IDs
        SASRec prefix representations
    """

    def __init__(
        self,
        user_ids,
        positive_items,
        negative_items,
        sasrec_representations,
        num_examples=None,
    ):

        self.user_ids = user_ids
        self.positive_items = positive_items
        self.negative_items = negative_items
        self.sasrec_representations = (
            sasrec_representations
        )

        total_examples = len(
            self.user_ids
        )

        if num_examples is None:
            num_examples = total_examples

        if num_examples > total_examples:
            raise ValueError(
                f"Requested {num_examples:,} examples "
                f"but only {total_examples:,} exist."
            )

        self.num_examples = num_examples

    def __len__(self):

        return self.num_examples

    def __getitem__(self, index):

        return (
            int(self.user_ids[index]),
            int(self.positive_items[index]),
            int(self.negative_items[index]),
            self.sasrec_representations[index],
        )


# ============================================================
# VALIDATE FILES
# ============================================================

def validate_required_files():

    print("=" * 80)
    print("CHECKING REQUIRED FILES")
    print("=" * 80)

    required_files = [
        TRAIN_USERS_FILE,
        TRAIN_POSITIVE_FILE,
        TRAIN_NEGATIVE_FILE,
        SASREC_REPRESENTATIONS_FILE,
        LIGHTGCN_USER_EMBEDDINGS_FILE,
        LIGHTGCN_ITEM_EMBEDDINGS_FILE,
        LIGHTGCN_MODEL_FILE,
        SASREC_MODEL_FILE,
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n"
                f"{file_path}"
            )

        print(
            f"[FOUND] {file_path.name}"
        )

    print()


# ============================================================
# LOAD ARRAYS
# ============================================================

def load_training_arrays():

    print("=" * 80)
    print("LOADING HYBRID TRAINING ARRAYS")
    print("=" * 80)

    user_ids = np.load(
        TRAIN_USERS_FILE,
        mmap_mode="r",
    )

    positive_items = np.load(
        TRAIN_POSITIVE_FILE,
        mmap_mode="r",
    )

    negative_items = np.load(
        TRAIN_NEGATIVE_FILE,
        mmap_mode="r",
    )

    sasrec_representations = np.load(
        SASREC_REPRESENTATIONS_FILE,
        mmap_mode="r",
    )

    lightgcn_user_embeddings = np.load(
        LIGHTGCN_USER_EMBEDDINGS_FILE,
        mmap_mode="r",
    )

    lightgcn_item_embeddings = np.load(
        LIGHTGCN_ITEM_EMBEDDINGS_FILE,
        mmap_mode="r",
    )

    print(
        f"User IDs shape: "
        f"{user_ids.shape}"
    )

    print(
        f"Positive items shape: "
        f"{positive_items.shape}"
    )

    print(
        f"Negative items shape: "
        f"{negative_items.shape}"
    )

    print(
        f"SASRec representations shape: "
        f"{sasrec_representations.shape}"
    )

    print(
        f"LightGCN user embeddings shape: "
        f"{lightgcn_user_embeddings.shape}"
    )

    print(
        f"LightGCN item embeddings shape: "
        f"{lightgcn_item_embeddings.shape}"
    )

    # --------------------------------------------------------
    # Shape validation
    # --------------------------------------------------------

    num_examples = len(user_ids)

    assert len(
        positive_items
    ) == num_examples

    assert len(
        negative_items
    ) == num_examples

    assert (
        sasrec_representations.shape[0]
        == num_examples
    )

    assert (
        sasrec_representations.shape[1]
        == EMBEDDING_DIM
    )

    assert (
        lightgcn_user_embeddings.shape
        == (
            NUM_USERS,
            EMBEDDING_DIM,
        )
    )

    assert (
        lightgcn_item_embeddings.shape
        == (
            NUM_ITEMS,
            EMBEDDING_DIM,
        )
    )

    print(
        "[PASS] Training array shapes"
    )

    # --------------------------------------------------------
    # Range validation
    # --------------------------------------------------------

    if num_examples > 0:

        sample_size = min(
            100_000,
            num_examples,
        )

        rng = np.random.default_rng(
            SEED
        )

        sample_indices = (
            rng.choice(
                num_examples,
                size=sample_size,
                replace=False,
            )
        )

        sampled_users = np.asarray(
            user_ids[sample_indices]
        )

        sampled_positive = np.asarray(
            positive_items[
                sample_indices
            ]
        )

        sampled_negative = np.asarray(
            negative_items[
                sample_indices
            ]
        )

        assert (
            sampled_users.min()
            >= 0
        )

        assert (
            sampled_users.max()
            < NUM_USERS
        )

        assert (
            sampled_positive.min()
            >= 0
        )

        assert (
            sampled_positive.max()
            < NUM_ITEMS
        )

        assert (
            sampled_negative.min()
            >= 0
        )

        assert (
            sampled_negative.max()
            < NUM_ITEMS
        )

        assert np.all(
            sampled_positive
            != sampled_negative
        )

        print(
            "[PASS] Sampled ID ranges"
        )

        print(
            "[PASS] Positive != negative"
        )

    # --------------------------------------------------------
    # Finite validation
    # --------------------------------------------------------

    assert np.isfinite(
        np.asarray(
            sasrec_representations[
                :min(10_000, num_examples)
            ]
        )
    ).all()

    assert np.isfinite(
        np.asarray(
            lightgcn_user_embeddings[
                :min(10_000, NUM_USERS)
            ]
        )
    ).all()

    assert np.isfinite(
        np.asarray(
            lightgcn_item_embeddings[
                :min(10_000, NUM_ITEMS)
            ]
        )
    ).all()

    print(
        "[PASS] Embedding values finite"
    )

    print()

    return (
        user_ids,
        positive_items,
        negative_items,
        sasrec_representations,
        lightgcn_user_embeddings,
        lightgcn_item_embeddings,
    )


# ============================================================
# LOAD MODEL
# ============================================================

def create_model():

    print("=" * 80)
    print("CREATING HYBRID V2 MODEL")
    print("=" * 80)

    model = HybridLightGCNSASRec(
        num_users=NUM_USERS,
        num_items=NUM_ITEMS,
        embedding_dim=EMBEDDING_DIM,
        max_seq_len=MAX_SEQUENCE_LENGTH,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )

    print(
        f"Users: {NUM_USERS:,}"
    )

    print(
        f"Items: {NUM_ITEMS:,}"
    )

    print(
        f"Embedding dimension: "
        f"{EMBEDDING_DIM}"
    )

    print(
        f"Sequence length: "
        f"{MAX_SEQUENCE_LENGTH}"
    )

    print(
        f"Transformer heads: "
        f"{NUM_HEADS}"
    )

    print(
        f"Transformer layers: "
        f"{NUM_LAYERS}"
    )

    # ============================================================
    # LOAD PRETRAINED LIGHTGCN
    # ============================================================

    print()
    print("-" * 80)
    print("LOADING PRETRAINED LIGHTGCN")
    print("-" * 80)

    lightgcn_checkpoint = torch.load(
        LIGHTGCN_MODEL_FILE,
        map_location="cpu",
        weights_only=False,
    )

    if (
        isinstance(lightgcn_checkpoint, dict)
        and "model_state_dict" in lightgcn_checkpoint
    ):
        lightgcn_state_dict = (
            lightgcn_checkpoint["model_state_dict"]
        )
    else:
        lightgcn_state_dict = lightgcn_checkpoint

    model.lightgcn.load_state_dict(
        lightgcn_state_dict
    )

    print(
        "[PASS] Pretrained LightGCN loaded"
    )

    # ============================================================
    # LOAD PRETRAINED SASREC
    # ============================================================

    print()
    print("-" * 80)
    print("LOADING PRETRAINED SASREC")
    print("-" * 80)

    sasrec_checkpoint = torch.load(
        SASREC_MODEL_FILE,
        map_location="cpu",
        weights_only=False,
    )

    if (
        isinstance(sasrec_checkpoint, dict)
        and "model_state_dict" in sasrec_checkpoint
    ):
        sasrec_state_dict = (
            sasrec_checkpoint["model_state_dict"]
        )
    else:
        sasrec_state_dict = sasrec_checkpoint

    # ------------------------------------------------------------
    # IMPORTANT:
    #
    # The original SASRec checkpoint uses:
    #
    #   transformer_encoder.layers.*
    #   layer_norm.*
    #
    # The currently imported Hybrid SASRec implementation uses:
    #
    #   transformer_layers.*
    #   final_layer_norm.*
    #
    # The tensor structures are compatible; only the registered
    # module names differ.
    #
    # Remap the checkpoint keys so the pretrained weights load
    # into the current Hybrid SASRec implementation.
    # ------------------------------------------------------------

    compatible_sasrec_state_dict = {}

    for key, value in sasrec_state_dict.items():

        new_key = key

        if key.startswith(
            "transformer_encoder.layers."
        ):
            new_key = key.replace(
                "transformer_encoder.layers.",
                "transformer_layers.",
                1,
            )

        elif key.startswith(
            "layer_norm."
        ):
            new_key = key.replace(
                "layer_norm.",
                "final_layer_norm.",
                1,
            )

        compatible_sasrec_state_dict[
            new_key
        ] = value

    # ------------------------------------------------------------
    # Verify the expected conversion happened.
    # ------------------------------------------------------------

    original_transformer_keys = [
        key
        for key in sasrec_state_dict.keys()
        if key.startswith(
            "transformer_encoder.layers."
        )
    ]

    converted_transformer_keys = [
        key
        for key in compatible_sasrec_state_dict.keys()
        if key.startswith(
            "transformer_layers."
        )
    ]

    assert len(
        original_transformer_keys
    ) == len(
        converted_transformer_keys
    ), (
        "SASRec transformer checkpoint-key "
        "conversion failed."
    )

    assert (
        "layer_norm.weight"
        in sasrec_state_dict
    )

    assert (
        "layer_norm.bias"
        in sasrec_state_dict
    )

    assert (
        "final_layer_norm.weight"
        in compatible_sasrec_state_dict
    )

    assert (
        "final_layer_norm.bias"
        in compatible_sasrec_state_dict
    )

    print(
        "[PASS] SASRec checkpoint keys "
        "converted for Hybrid SASRec"
    )

    # ------------------------------------------------------------
    # Load converted pretrained SASRec weights.
    # ------------------------------------------------------------

    missing_keys, unexpected_keys = (
        model.sasrec.load_state_dict(
            compatible_sasrec_state_dict,
            strict=False,
        )
    )

    # No missing or unexpected keys should remain.
    assert not missing_keys, (
        "Unexpected missing SASRec keys:\n"
        + "\n".join(missing_keys)
    )

    assert not unexpected_keys, (
        "Unexpected SASRec checkpoint keys:\n"
        + "\n".join(unexpected_keys)
    )

    print(
        "[PASS] Pretrained SASRec loaded"
    )

    # ============================================================
    # FREEZE PRETRAINED ENCODERS
    # ============================================================

    model.freeze_pretrained_encoders()

    print(
        "[PASS] LightGCN frozen"
    )

    print(
        "[PASS] SASRec frozen"
    )

    # ============================================================
    # ENCODER EVALUATION MODE
    # ============================================================

    assert (
        model.lightgcn.training
        is False
    )

    assert (
        model.sasrec.training
        is False
    )

    print(
        "[PASS] Frozen encoders in eval mode"
    )

    # ============================================================
    # TRAINABLE PARAMETER VALIDATION
    # ============================================================

    trainable_parameters = (
        model.trainable_parameters()
    )

    trainable_count = sum(
        parameter.numel()
        for parameter in trainable_parameters
    )

    print(
        f"Trainable parameters: "
        f"{trainable_count:,}"
    )

    assert (
        trainable_count
        == EXPECTED_TRAINABLE_PARAMETERS
    )

    print(
        "[PASS] Trainable parameter count = "
        f"{EXPECTED_TRAINABLE_PARAMETERS:,}"
    )

    trainable_names = [
        name
        for name, parameter
        in model.named_parameters()
        if parameter.requires_grad
    ]

    expected_names = {
        "user_fusion.0.weight",
        "user_fusion.0.bias",
        "item_fusion.0.weight",
        "item_fusion.0.bias",
    }

    assert (
        set(trainable_names)
        == expected_names
    )

    print(
        "[PASS] Only V2 fusion layers are trainable"
    )

    print()

    return model

# ============================================================
# VALIDATE PRECOMPUTED REPRESENTATIONS
# ============================================================

def validate_representations(
    model,
    lightgcn_user_embeddings,
    lightgcn_item_embeddings,
    device,
):

    print("=" * 80)
    print("VALIDATING PRECOMPUTED REPRESENTATIONS")
    print("=" * 80)

    # --------------------------------------------------------
    # Important:
    # np.memmap is read-only. Copy before converting to
    # torch tensors to avoid the non-writable NumPy warning.
    # --------------------------------------------------------

    sample_users = torch.from_numpy(
        np.array(
            lightgcn_user_embeddings[
                :4
            ],
            copy=True,
        )
    ).float().to(device)

    sample_items = torch.from_numpy(
        np.array(
            lightgcn_item_embeddings[
                :4
            ],
            copy=True,
        )
    ).float().to(device)

    assert sample_users.shape == (
        4,
        EMBEDDING_DIM,
    )

    assert sample_items.shape == (
        4,
        EMBEDDING_DIM,
    )

    assert torch.isfinite(
        sample_users
    ).all()

    assert torch.isfinite(
        sample_items
    ).all()

    print(
        "[PASS] LightGCN user embeddings"
    )

    print(
        "[PASS] LightGCN item embeddings"
    )

    # --------------------------------------------------------
    # Test component normalization
    # --------------------------------------------------------

    normalized_users = (
        model.normalize_component(
            sample_users
        )
    )

    normalized_items = (
        model.normalize_component(
            sample_items
        )
    )

    user_norms = torch.norm(
        normalized_users,
        dim=1,
    )

    item_norms = torch.norm(
        normalized_items,
        dim=1,
    )

    assert torch.allclose(
        user_norms,
        torch.ones_like(
            user_norms
        ),
        atol=1e-5,
    )

    assert torch.allclose(
        item_norms,
        torch.ones_like(
            item_norms
        ),
        atol=1e-5,
    )

    print(
        "[PASS] L2 normalization"
    )

    print()

    return


# ============================================================
# TRAINING
# ============================================================

def train():

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print("HYBRID V2 TRAINING")
    print("=" * 80)

    print(
        f"Device: {device}"
    )

    print(
        f"Seed: {SEED}"
    )

    print(
        f"Batch size: {BATCH_SIZE:,}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        f"Weight decay: {WEIGHT_DECAY}"
    )

    print()

    # --------------------------------------------------------
    # Files
    # --------------------------------------------------------

    validate_required_files()

    # --------------------------------------------------------
    # Load arrays
    # --------------------------------------------------------

    (
        user_ids,
        positive_items,
        negative_items,
        sasrec_representations,
        lightgcn_user_embeddings,
        lightgcn_item_embeddings,
    ) = load_training_arrays()

    total_available_examples = (
        len(user_ids)
    )

    if SANITY_MODE:

        num_examples = min(
            SANITY_EXAMPLES,
            total_available_examples,
        )

        num_epochs = SANITY_EPOCHS

    else:

        num_examples = (
            total_available_examples
        )

        num_epochs = FULL_EPOCHS

    print(
        f"Training examples available: "
        f"{total_available_examples:,}"
    )

    print(
        f"Training examples used: "
        f"{num_examples:,}"
    )

    print(
        f"Epochs: {num_epochs}"
    )

    print()

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = create_model()

    model = model.to(device)

    # --------------------------------------------------------
    # Keep pretrained encoders frozen and in eval mode
    # --------------------------------------------------------

    model.lightgcn.eval()
    model.sasrec.eval()

    # --------------------------------------------------------
    # Validate representations
    # --------------------------------------------------------

    validate_representations(
        model=model,
        lightgcn_user_embeddings=(
            lightgcn_user_embeddings
        ),
        lightgcn_item_embeddings=(
            lightgcn_item_embeddings
        ),
        device=device,
    )

    # --------------------------------------------------------
    # Convert LightGCN embeddings once.
    #
    # np.array(..., copy=True) intentionally removes the
    # read-only memmap property.
    # --------------------------------------------------------

    print(
        "=" * 80
    )

    print(
        "PREPARING LIGHTGCN EMBEDDINGS"
    )

    print(
        "=" * 80
    )

    lightgcn_user_tensor = (
        torch.from_numpy(
            np.array(
                lightgcn_user_embeddings,
                dtype=np.float32,
                copy=True,
            )
        )
        .float()
        .to(device)
    )

    lightgcn_item_tensor = (
        torch.from_numpy(
            np.array(
                lightgcn_item_embeddings,
                dtype=np.float32,
                copy=True,
            )
        )
        .float()
        .to(device)
    )

    print(
        f"LightGCN user tensor: "
        f"{tuple(lightgcn_user_tensor.shape)}"
    )

    print(
        f"LightGCN item tensor: "
        f"{tuple(lightgcn_item_tensor.shape)}"
    )

    print(
        "[PASS] LightGCN tensors prepared"
    )

    print()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = HybridTrainingDataset(
        user_ids=user_ids,
        positive_items=positive_items,
        negative_items=negative_items,
        sasrec_representations=(
            sasrec_representations
        ),
        num_examples=num_examples,
    )

    print(
        f"Dataset size: "
        f"{len(dataset):,}"
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    print(
        f"Batches per epoch: "
        f"{len(data_loader):,}"
    )

    print()

    # --------------------------------------------------------
    # Optimizer
    #
    # IMPORTANT:
    # only user_fusion + item_fusion parameters are passed.
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.trainable_parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print(
        "[PASS] Optimizer configured with "
        "trainable fusion parameters only"
    )

    print()

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    training_history = []

    total_training_start = (
        time.perf_counter()
    )

    # ========================================================
    # EPOCH LOOP
    # ========================================================

    for epoch in range(
        1,
        num_epochs + 1,
    ):

        epoch_start = (
            time.perf_counter()
        )

        # ----------------------------------------------------
        # Critical:
        # model.train() enables dropout in the trainable
        # user_fusion layer.
        #
        # Immediately restore frozen encoder eval modes.
        # ----------------------------------------------------

        model.train()

        model.lightgcn.eval()
        model.sasrec.eval()

        total_loss = 0.0
        total_examples = 0

        first_batch_checked = False

        # ----------------------------------------------------
        # Batches
        # ----------------------------------------------------

        for batch_number, batch in enumerate(
            data_loader,
            start=1,
        ):

            (
                batch_user_ids,
                batch_positive_items,
                batch_negative_items,
                batch_sasrec_representations,
            ) = batch

            batch_user_ids = (
                batch_user_ids
                .long()
                .to(device)
            )

            batch_positive_items = (
                batch_positive_items
                .long()
                .to(device)
            )

            batch_negative_items = (
                batch_negative_items
                .long()
                .to(device)
            )

            batch_sasrec_representations = (
                batch_sasrec_representations
                .float()
                .to(device)
            )

            # ------------------------------------------------
            # Validate dimensions on first batch
            # ------------------------------------------------

            if not first_batch_checked:

                assert (
                    batch_sasrec_representations.shape[
                        1
                    ]
                    == EMBEDDING_DIM
                )

                assert (
                    batch_user_ids.min()
                    >= 0
                )

                assert (
                    batch_user_ids.max()
                    < NUM_USERS
                )

                assert (
                    batch_positive_items.min()
                    >= 0
                )

                assert (
                    batch_positive_items.max()
                    < NUM_ITEMS
                )

                assert (
                    batch_negative_items.min()
                    >= 0
                )

                assert (
                    batch_negative_items.max()
                    < NUM_ITEMS
                )

                assert torch.all(
                    batch_positive_items
                    !=
                    batch_negative_items
                )

                first_batch_checked = True

                print(
                    f"Epoch {epoch}: "
                    f"first batch validation PASS"
                )

            # ------------------------------------------------
            # LightGCN user representation
            # ------------------------------------------------

            batch_lightgcn_users = (
                lightgcn_user_tensor[
                    batch_user_ids
                ]
            )

            # ------------------------------------------------
            # Normalize LightGCN user
            # ------------------------------------------------

            batch_lightgcn_users = (
                model.normalize_component(
                    batch_lightgcn_users
                )
            )

            # ------------------------------------------------
            # SASRec representation
            #
            # Already precomputed from the prefix that occurs
            # strictly before the positive target.
            # ------------------------------------------------

            batch_sasrec_users = (
                batch_sasrec_representations
            )

            batch_sasrec_users = (
                model.normalize_component(
                    batch_sasrec_users
                )
            )

            # ------------------------------------------------
            # Hybrid user
            # ------------------------------------------------

            combined_users = torch.cat(
                [
                    batch_lightgcn_users,
                    batch_sasrec_users,
                ],
                dim=1,
            )

            hybrid_users = (
                model.user_fusion(
                    combined_users
                )
            )

            # ------------------------------------------------
            # LightGCN positive items
            # ------------------------------------------------

            batch_lightgcn_positive = (
                lightgcn_item_tensor[
                    batch_positive_items
                ]
            )

            batch_lightgcn_positive = (
                model.normalize_component(
                    batch_lightgcn_positive
                )
            )

            # ------------------------------------------------
            # LightGCN negative items
            # ------------------------------------------------

            batch_lightgcn_negative = (
                lightgcn_item_tensor[
                    batch_negative_items
                ]
            )

            batch_lightgcn_negative = (
                model.normalize_component(
                    batch_lightgcn_negative
                )
            )

            # ------------------------------------------------
            # SASRec positive items
            #
            # +1 because:
            # training item 0 -> SASRec ID 1
            # training item 1 -> SASRec ID 2
            # ...
            # ------------------------------------------------

            batch_positive_sasrec_ids = (
                batch_positive_items + 1
            )

            batch_negative_sasrec_ids = (
                batch_negative_items + 1
            )

            batch_sasrec_positive = (
                model.sasrec.item_embedding(
                    batch_positive_sasrec_ids
                )
            )

            batch_sasrec_negative = (
                model.sasrec.item_embedding(
                    batch_negative_sasrec_ids
                )
            )

            # ------------------------------------------------
            # Normalize SASRec item representations
            # ------------------------------------------------

            batch_sasrec_positive = (
                model.normalize_component(
                    batch_sasrec_positive
                )
            )

            batch_sasrec_negative = (
                model.normalize_component(
                    batch_sasrec_negative
                )
            )

            # ------------------------------------------------
            # Hybrid positive item
            # ------------------------------------------------

            combined_positive_items = (
                torch.cat(
                    [
                        batch_lightgcn_positive,
                        batch_sasrec_positive,
                    ],
                    dim=1,
                )
            )

            hybrid_positive_items = (
                model.item_fusion(
                    combined_positive_items
                )
            )

            # ------------------------------------------------
            # Hybrid negative item
            # ------------------------------------------------

            combined_negative_items = (
                torch.cat(
                    [
                        batch_lightgcn_negative,
                        batch_sasrec_negative,
                    ],
                    dim=1,
                )
            )

            hybrid_negative_items = (
                model.item_fusion(
                    combined_negative_items
                )
            )

            # ------------------------------------------------
            # Scores
            # ------------------------------------------------

            positive_scores = (
                hybrid_users
                *
                hybrid_positive_items
            ).sum(dim=1)

            negative_scores = (
                hybrid_users
                *
                hybrid_negative_items
            ).sum(dim=1)

            # ------------------------------------------------
            # BPR loss
            # ------------------------------------------------

            loss = model.bpr_loss(
                positive_scores,
                negative_scores,
            )

            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            # ------------------------------------------------
            # First-batch gradient validation
            # ------------------------------------------------

            if batch_number == 1:

                user_grad = (
                    model.user_fusion[
                        0
                    ].weight.grad
                )

                item_grad = (
                    model.item_fusion[
                        0
                    ].weight.grad
                )

                assert (
                    user_grad is not None
                )

                assert (
                    item_grad is not None
                )

                assert torch.isfinite(
                    user_grad
                ).all()

                assert torch.isfinite(
                    item_grad
                ).all()

                # Frozen encoders must not receive gradients.
                lightgcn_has_grad = any(
                    parameter.grad
                    is not None
                    for parameter
                    in model.lightgcn.parameters()
                )

                sasrec_has_grad = any(
                    parameter.grad
                    is not None
                    for parameter
                    in model.sasrec.parameters()
                )

                assert (
                    not lightgcn_has_grad
                )

                assert (
                    not sasrec_has_grad
                )

                print(
                    "[PASS] Fusion gradients finite"
                )

                print(
                    "[PASS] Frozen encoder gradients absent"
                )

            # ------------------------------------------------
            # Optimizer update
            # ------------------------------------------------

            optimizer.step()

            batch_size_actual = (
                batch_user_ids.size(0)
            )

            total_loss += (
                loss.item()
                *
                batch_size_actual
            )

            total_examples += (
                batch_size_actual
            )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                batch_number == 1
                or
                batch_number % 50 == 0
                or
                batch_number
                == len(data_loader)
            ):

                running_average = (
                    total_loss
                    /
                    total_examples
                )

                print(
                    f"Epoch {epoch} | "
                    f"Batch "
                    f"{batch_number:,}/"
                    f"{len(data_loader):,} | "
                    f"Loss "
                    f"{running_average:.6f}"
                )

        # ----------------------------------------------------
        # Epoch metrics
        # ----------------------------------------------------

        epoch_loss = (
            total_loss
            /
            total_examples
        )

        epoch_time = (
            time.perf_counter()
            -
            epoch_start
        )

        assert np.isfinite(
            epoch_loss
        )

        # ----------------------------------------------------
        # Representation sanity after epoch
        # ----------------------------------------------------

        with torch.no_grad():

            sample_hybrid_user = (
                hybrid_users[:128]
            )

            sample_hybrid_positive = (
                hybrid_positive_items[
                    :128
                ]
            )

            assert torch.isfinite(
                sample_hybrid_user
            ).all()

            assert torch.isfinite(
                sample_hybrid_positive
            ).all()

        # ----------------------------------------------------
        # Store metrics
        # ----------------------------------------------------

        epoch_record = {
            "epoch": epoch,
            "loss": float(
                epoch_loss
            ),
            "examples": int(
                total_examples
            ),
            "epoch_time_seconds": float(
                epoch_time
            ),
        }

        training_history.append(
            epoch_record
        )

        print()
        print(
            f"Epoch {epoch} complete"
        )

        print(
            f"Average BPR loss: "
            f"{epoch_loss:.8f}"
        )

        print(
            f"Examples: "
            f"{total_examples:,}"
        )

        print(
            f"Time: "
            f"{epoch_time:.2f}s"
        )

        print()

    # ========================================================
    # TRAINING COMPLETE
    # ========================================================

    total_training_time = (
        time.perf_counter()
        -
        total_training_start
    )

    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    print(
        f"Total examples: "
        f"{num_examples:,}"
    )

    print(
        f"Epochs: "
        f"{num_epochs}"
    )

    print(
        f"Total training time: "
        f"{total_training_time:.2f}s"
    )

    print(
        f"Total training time: "
        f"{total_training_time / 60:.2f} min"
    )

    print(
        f"Final loss: "
        f"{training_history[-1]['loss']:.8f}"
    )

    print()

    # ========================================================
    # FINAL PARAMETER VALIDATION
    # ========================================================

    print(
        "=" * 80
    )

    print(
        "FINAL PARAMETER VALIDATION"
    )

    print(
        "=" * 80
    )

    trainable_count = (
        model.num_trainable_parameters()
    )

    assert (
        trainable_count
        == EXPECTED_TRAINABLE_PARAMETERS
    )

    print(
        f"[PASS] Trainable parameters: "
        f"{trainable_count:,}"
    )

    # Check pretrained parameters have no gradients.
    for parameter in (
        model.lightgcn.parameters()
    ):
        assert parameter.grad is None

    for parameter in (
        model.sasrec.parameters()
    ):
        assert parameter.grad is None

    print(
        "[PASS] LightGCN gradients absent"
    )

    print(
        "[PASS] SASRec gradients absent"
    )

    # Check fusion parameters exist.
    for parameter in (
        model.user_fusion.parameters()
    ):
        assert parameter.requires_grad

    for parameter in (
        model.item_fusion.parameters()
    ):
        assert parameter.requires_grad

    print(
        "[PASS] User fusion trainable"
    )

    print(
        "[PASS] Item fusion trainable"
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    checkpoint = {
        "model_state_dict": (
            model.state_dict()
        ),
        "num_users": NUM_USERS,
        "num_items": NUM_ITEMS,
        "embedding_dim": EMBEDDING_DIM,
        "max_sequence_length": (
            MAX_SEQUENCE_LENGTH
        ),
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "epochs": num_epochs,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "sanity_mode": SANITY_MODE,
        "num_training_examples": (
            num_examples
        ),
        "trainable_parameters": (
            trainable_count
        ),
        "architecture": (
            "Hybrid V2: "
            "L2-normalized LightGCN + "
            "L2-normalized SASRec components "
            "with separate user/item projections"
        ),
        "training_history": (
            training_history
        ),
        "total_training_time_seconds": (
            total_training_time
        ),
    }

    torch.save(
        checkpoint,
        OUTPUT_MODEL_FILE,
    )

    print()
    print(
        f"Saved model:"
    )

    print(
        OUTPUT_MODEL_FILE
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics = {
        "model": "Hybrid LightGCN + SASRec V2",
        "seed": SEED,
        "device": str(device),
        "num_users": NUM_USERS,
        "num_items": NUM_ITEMS,
        "embedding_dim": EMBEDDING_DIM,
        "max_sequence_length": (
            MAX_SEQUENCE_LENGTH
        ),
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "epochs": num_epochs,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "num_training_examples": (
            num_examples
        ),
        "trainable_parameters": (
            trainable_count
        ),
        "pretrained_encoders_frozen": True,
        "user_component_normalization": (
            "L2"
        ),
        "item_component_normalization": (
            "L2"
        ),
        "user_projection": (
            "Linear(128,128) + ReLU + Dropout"
        ),
        "item_projection": (
            "Linear(128,128) + ReLU"
        ),
        "loss": "BPR",
        "training_history": (
            training_history
        ),
        "total_training_time_seconds": (
            total_training_time
        ),
        "final_loss": (
            training_history[-1]["loss"]
        ),
    }

    with open(
        OUTPUT_METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    print(
        f"Saved metrics:"
    )

    print(
        OUTPUT_METRICS_FILE
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print()
    print("=" * 80)
    print("HYBRID V2 TRAINING PASSED")
    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train()