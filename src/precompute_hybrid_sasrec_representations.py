# ============================================================
# precompute_hybrid_sasrec_representations.py
# ============================================================
#
# Purpose:
#   Precompute frozen SASRec representations for the 2M
#   leakage-free hybrid training prefixes.
#
# Input:
#   output/hybrid_train_sequences.npy
#   output/sasrec_model.pt
#
# Output:
#   output/hybrid_train_sasrec_representations.npy
#   output/hybrid_sasrec_representation_metadata.json
#
# Representation:
#   For each prefix sequence, extract the SASRec hidden
#   representation at the LAST NON-PADDING POSITION.
#
# Important:
#   - SASRec is frozen.
#   - SASRec is evaluated in eval() mode.
#   - Padding ID = 0.
#   - Real item IDs = 1..74541.
# ============================================================

from pathlib import Path
import json
import time

import numpy as np
import torch


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

SEQUENCE_FILE = OUTPUT_DIR / "hybrid_train_sequences.npy"
SASREC_MODEL_FILE = OUTPUT_DIR / "sasrec_model.pt"

REPRESENTATION_FILE = (
    OUTPUT_DIR /
    "hybrid_train_sasrec_representations.npy"
)

METADATA_FILE = (
    OUTPUT_DIR /
    "hybrid_sasrec_representation_metadata.json"
)

BATCH_SIZE = 512

EXPECTED_EXAMPLES = 2_000_000
EXPECTED_SEQUENCE_LENGTH = 50
EXPECTED_EMBEDDING_DIM = 64
EXPECTED_NUM_ITEMS = 74_541

PADDING_ID = 0

DEVICE = torch.device("cpu")


# ============================================================
# Helpers
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def validate_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


# ============================================================
# [1] Validate input files
# ============================================================

print_header("[1] Validating input files")

validate_file(SEQUENCE_FILE)
validate_file(SASREC_MODEL_FILE)

print(f"Sequence file: {SEQUENCE_FILE}")
print(f"SASRec model: {SASREC_MODEL_FILE}")

print("Input files: PASS")


# ============================================================
# [2] Load sequences
# ============================================================

print_header("[2] Loading hybrid training sequences")

sequences = np.load(
    SEQUENCE_FILE,
    mmap_mode="r",
)

print(f"Sequence shape: {sequences.shape}")
print(f"Sequence dtype: {sequences.dtype}")

assert sequences.shape == (
    EXPECTED_EXAMPLES,
    EXPECTED_SEQUENCE_LENGTH,
)

assert sequences.dtype == np.int32

print("Sequence shape: PASS")
print("Sequence dtype: PASS")


# ============================================================
# [3] Validate sequence IDs
# ============================================================

print_header("[3] Validating sequence IDs")

sample_min = int(sequences.min())
sample_max = int(sequences.max())

print(f"Minimum sequence ID: {sample_min}")
print(f"Maximum sequence ID: {sample_max}")

assert sample_min >= PADDING_ID
assert sample_max <= EXPECTED_NUM_ITEMS

print("Sequence ID range: PASS")


# ============================================================
# [4] Load SASRec model
# ============================================================

print_header("[4] Loading pretrained SASRec")

checkpoint = torch.load(
    SASREC_MODEL_FILE,
    map_location=DEVICE,
    weights_only=False,
)

print(
    f"Checkpoint type: {type(checkpoint).__name__}"
)


# ------------------------------------------------------------
# Import the SASRec implementation from hybrid_model.py
# ------------------------------------------------------------

import sys

sys.path.insert(
    0,
    str(BASE_DIR / "src"),
)

from hybrid_model import SASRec


# ------------------------------------------------------------
# Reconstruct architecture
# ------------------------------------------------------------

sasrec = SASRec(
    num_items=EXPECTED_NUM_ITEMS,
    embedding_dim=64,
    max_seq_len=EXPECTED_SEQUENCE_LENGTH,
    num_heads=2,
    num_layers=2,
    dropout=0.2,
)
# ------------------------------------------------------------
# Load state dict
# ------------------------------------------------------------

if "model_state_dict" in checkpoint:

    sasrec.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    sasrec.load_state_dict(
        checkpoint
    )


sasrec = sasrec.to(DEVICE)

# Critical:
# disable dropout
sasrec.eval()


# Critical:
# freeze every SASRec parameter

for parameter in sasrec.parameters():
    parameter.requires_grad = False


print(
    f"SASRec parameters: "
    f"{sum(p.numel() for p in sasrec.parameters()):,}"
)

print(
    "SASRec training mode:",
    sasrec.training,
)

assert sasrec.training is False

assert all(
    not p.requires_grad
    for p in sasrec.parameters()
)

print("SASRec frozen: PASS")
print("SASRec eval mode: PASS")


# ============================================================
# [5] Validate SASRec architecture
# ============================================================

print_header("[5] Validating SASRec architecture")

item_embedding_shape = (
    sasrec.item_embedding.weight.shape
)

position_embedding_shape = (
    sasrec.position_embedding.weight.shape
)

print(
    "Item embedding shape:",
    tuple(item_embedding_shape),
)

print(
    "Position embedding shape:",
    tuple(position_embedding_shape),
)

assert item_embedding_shape == (
    EXPECTED_NUM_ITEMS + 1,
    EXPECTED_EMBEDDING_DIM,
)

assert position_embedding_shape == (
    EXPECTED_SEQUENCE_LENGTH,
    EXPECTED_EMBEDDING_DIM,
)

print("SASRec architecture: PASS")


# ============================================================
# [6] Test one batch
# ============================================================

print_header("[6] Running single-batch representation test")

test_sequences = torch.from_numpy(
    np.asarray(
        sequences[:4],
        dtype=np.int64,
    )
).to(DEVICE)


with torch.no_grad():

    test_output = sasrec(
        test_sequences
    )


print(
    "SASRec output shape:",
    tuple(test_output.shape),
)

assert test_output.shape == (
    4,
    EXPECTED_SEQUENCE_LENGTH,
    EXPECTED_EMBEDDING_DIM,
)

assert torch.isfinite(
    test_output
).all()

print("SASRec forward pass: PASS")


# ============================================================
# [7] Extract last non-padding representation
# ============================================================

print_header(
    "[7] Testing last non-padding extraction"
)


def extract_last_representation(
    sequences_tensor,
    hidden_states,
):
    """
    Extract the hidden state at the last real
    item position for each sequence.
    """

    non_padding_mask = (
        sequences_tensor != PADDING_ID
    )

    sequence_lengths = (
        non_padding_mask.sum(dim=1)
    )

    assert torch.all(
        sequence_lengths >= 1
    )

    last_positions = (
        sequence_lengths - 1
    )

    batch_indices = torch.arange(
        sequences_tensor.size(0),
        device=sequences_tensor.device,
    )

    representations = hidden_states[
        batch_indices,
        last_positions,
    ]

    return representations


with torch.no_grad():

    test_representations = (
        extract_last_representation(
            test_sequences,
            test_output,
        )
    )


print(
    "Representation shape:",
    tuple(test_representations.shape)
)

assert test_representations.shape == (
    4,
    EXPECTED_EMBEDDING_DIM,
)

assert torch.isfinite(
    test_representations
).all()

print("Representation extraction: PASS")


# ============================================================
# [8] Allocate output
# ============================================================

print_header("[8] Allocating representation output")

num_examples = sequences.shape[0]

representations = np.empty(
    (
        num_examples,
        EXPECTED_EMBEDDING_DIM,
    ),
    dtype=np.float32,
)

print(
    f"Output shape: {representations.shape}"
)

print(
    f"Output dtype: {representations.dtype}"
)

expected_memory_mb = (
    num_examples *
    EXPECTED_EMBEDDING_DIM *
    4
    /
    (1024 ** 2)
)

print(
    f"Expected output size: "
    f"{expected_memory_mb:.2f} MB"
)


# ============================================================
# [9] Precompute representations
# ============================================================

print_header(
    "[9] Precomputing frozen SASRec representations"
)

start_time = time.time()

for start in range(
    0,
    num_examples,
    BATCH_SIZE,
):

    end = min(
        start + BATCH_SIZE,
        num_examples,
    )

    batch_numpy = np.asarray(
        sequences[start:end],
        dtype=np.int64,
    )

    batch_sequences = torch.from_numpy(
        batch_numpy
    ).to(DEVICE)

    with torch.no_grad():

        hidden_states = sasrec(
            batch_sequences
        )

        batch_representations = (
            extract_last_representation(
                batch_sequences,
                hidden_states,
            )
        )

    batch_representations = (
        batch_representations
        .cpu()
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
    )

    representations[
        start:end
    ] = batch_representations

    if (
        end % 10_000 == 0
        or end == num_examples
    ):

        elapsed = (
            time.time() -
            start_time
        )

        rate = (
            end / elapsed
            if elapsed > 0
            else 0
        )

        print(
            f"Processed "
            f"{end:,} / "
            f"{num_examples:,} "
            f"| "
            f"{rate:,.1f} examples/sec"
        )


total_time = (
    time.time() -
    start_time
)

print()
print(
    f"Total preprocessing time: "
    f"{total_time:.2f}s"
)

print(
    f"Total preprocessing time: "
    f"{total_time / 60:.2f} min"
)


# ============================================================
# [10] Validate representations
# ============================================================

print_header(
    "[10] Validating generated representations"
)

assert representations.shape == (
    EXPECTED_EXAMPLES,
    EXPECTED_EMBEDDING_DIM,
)

assert representations.dtype == np.float32

print("Shape: PASS")
print("Dtype: PASS")


finite_mask = np.isfinite(
    representations
)

assert finite_mask.all()

print("Finite values: PASS")


print(
    f"Minimum value: "
    f"{representations.min():.6f}"
)

print(
    f"Maximum value: "
    f"{representations.max():.6f}"
)

print(
    f"Mean value: "
    f"{representations.mean():.6f}"
)

print(
    f"Std value: "
    f"{representations.std():.6f}"
)


# ============================================================
# [11] Validate against direct computation
# ============================================================

print_header(
    "[11] Validating stored representations"
)

validation_indices = [
    0,
    1,
    100,
    999,
]

for index in validation_indices:

    sequence_numpy = np.asarray(
        sequences[index:index + 1],
        dtype=np.int64,
    )

    sequence_tensor = torch.from_numpy(
        sequence_numpy
    ).to(DEVICE)

    with torch.no_grad():

        hidden = sasrec(
            sequence_tensor
        )

        direct_representation = (
            extract_last_representation(
                sequence_tensor,
                hidden,
            )
        )

    direct_numpy = (
        direct_representation
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    stored_numpy = (
        representations[index:index + 1]
    )

    assert np.allclose(
        stored_numpy,
        direct_numpy,
        rtol=1e-5,
        atol=1e-6,
    )

print(
    "Stored vs direct representations: PASS"
)


# ============================================================
# [12] Save representation array
# ============================================================

print_header(
    "[12] Saving SASRec representations"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

np.save(
    REPRESENTATION_FILE,
    representations,
)

print(
    f"Saved: {REPRESENTATION_FILE}"
)


# ============================================================
# [13] Reload validation
# ============================================================

print_header(
    "[13] Reloading saved representations"
)

reloaded = np.load(
    REPRESENTATION_FILE,
)

assert reloaded.shape == representations.shape
assert reloaded.dtype == np.float32
assert np.array_equal(
    reloaded,
    representations,
)

print("Reload shape: PASS")
print("Reload dtype: PASS")
print("Exact equality after reload: PASS")


# ============================================================
# [14] Save metadata
# ============================================================

print_header(
    "[14] Saving metadata"
)

metadata = {

    "dataset": (
        "H&M Personalized Fashion "
        "Recommendations"
    ),

    "source_sequences": str(
        SEQUENCE_FILE
    ),

    "source_sasrec_model": str(
        SASREC_MODEL_FILE
    ),

    "num_examples": int(
        num_examples
    ),

    "sequence_length": int(
        EXPECTED_SEQUENCE_LENGTH
    ),

    "embedding_dimension": int(
        EXPECTED_EMBEDDING_DIM
    ),

    "num_training_items": int(
        EXPECTED_NUM_ITEMS
    ),

    "padding_id": int(
        PADDING_ID
    ),

    "batch_size": int(
        BATCH_SIZE
    ),

    "device": str(
        DEVICE
    ),

    "representation_definition": (
        "SASRec hidden representation at "
        "the last non-padding position"
    ),

    "sequence_semantics": (
        "Chronological prefix strictly before "
        "the positive target"
    ),

    "future_information_used": False,

    "target_information_used": False,

    "sasrec_frozen": True,

    "sasrec_eval_mode": True,

    "output_shape": [
        int(num_examples),
        int(EXPECTED_EMBEDDING_DIM),
    ],

    "output_dtype": "float32",

    "precomputation_time_seconds": (
        float(total_time)
    ),

    "validation": {

        "input_sequence_shape": True,

        "sasrec_architecture": True,

        "sasrec_frozen": True,

        "sasrec_eval_mode": True,

        "forward_pass": True,

        "representation_shape": True,

        "finite_values": True,

        "stored_vs_direct": True,

        "reload_exact_equality": True,
    },
}


with open(
    METADATA_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        indent=4,
    )


print(
    f"Saved: {METADATA_FILE}"
)


# ============================================================
# [15] Final result
# ============================================================

print_header(
    "[15] SASRec REPRESENTATION PRECOMPUTATION PASSED"
)

print(
    f"Examples: "
    f"{num_examples:,}"
)

print(
    f"Representation dimension: "
    f"{EXPECTED_EMBEDDING_DIM}"
)

print(
    f"Representation shape: "
    f"{representations.shape}"
)

print(
    f"Precomputation time: "
    f"{total_time / 60:.2f} min"
)

print()
print(
    "Frozen SASRec representations are ready "
    "for hybrid fusion training."
)

print("=" * 70)