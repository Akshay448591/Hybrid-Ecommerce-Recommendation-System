import os
import sys

import torch


# ============================================================
# Project path
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    CURRENT_DIR
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT,
    )


# ============================================================
# Imports
# ============================================================

from hybrid_model import (
    HybridLightGCNSASRecV3,
    count_parameters,
    count_trainable_parameters,
)


# ============================================================
# Configuration
# ============================================================

NUM_USERS = 778_421
NUM_ITEMS = 74_541

EMBEDDING_DIM = 64
MAX_SEQUENCE_LENGTH = 50

NUM_HEADS = 2
NUM_LAYERS = 2

DROPOUT = 0.2

EXPECTED_SASREC_PARAMETERS = 4_873_984

EXPECTED_TRAINABLE_PARAMETERS = 41_280

EXPECTED_TOTAL_PARAMETERS = 59_504_832

SEED = 42


# ============================================================
# Paths
# ============================================================

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "output",
)

LIGHTGCN_CHECKPOINT = os.path.join(
    OUTPUT_DIR,
    "lightgcn_model.pt",
)

SASREC_CHECKPOINT = os.path.join(
    OUTPUT_DIR,
    "sasrec_model.pt",
)

LIGHTGCN_USER_EMBEDDINGS = os.path.join(
    OUTPUT_DIR,
    "lightgcn_user_embeddings.npy",
)

LIGHTGCN_ITEM_EMBEDDINGS = os.path.join(
    OUTPUT_DIR,
    "lightgcn_item_embeddings.npy",
)


# ============================================================
# Test helper
# ============================================================

passed = 0
failed = 0


def check(condition, message):
    global passed
    global failed

    if condition:

        print(
            f"[PASS] {message}"
        )

        passed += 1

    else:

        print(
            f"[FAIL] {message}"
        )

        failed += 1

        raise AssertionError(
            f"[FAIL] {message}"
        )


# ============================================================
# Main
# ============================================================

def main():

    torch.manual_seed(SEED)

    print("=" * 70)
    print("HYBRID V3 MODEL SANITY TEST")
    print("=" * 70)

    # ========================================================
    # Required files
    # ========================================================

    print("\nChecking required files...")

    check(
        os.path.exists(
            LIGHTGCN_CHECKPOINT
        ),
        "LightGCN checkpoint exists",
    )

    check(
        os.path.exists(
            SASREC_CHECKPOINT
        ),
        "SASRec checkpoint exists",
    )

    check(
        os.path.exists(
            LIGHTGCN_USER_EMBEDDINGS
        ),
        "LightGCN user embeddings exist",
    )

    check(
        os.path.exists(
            LIGHTGCN_ITEM_EMBEDDINGS
        ),
        "LightGCN item embeddings exist",
    )

    # ========================================================
    # Create model
    # ========================================================

    print("\nCreating V3 hybrid model...")

    model = HybridLightGCNSASRecV3(
        num_users=NUM_USERS,
        num_items=NUM_ITEMS,
        embedding_dim=EMBEDDING_DIM,
        max_seq_len=MAX_SEQUENCE_LENGTH,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )

    # ========================================================
    # Basic architecture
    # ========================================================

    print("\nChecking architecture...")

    check(
        model.num_users == NUM_USERS,
        f"Number of users = {model.num_users}",
    )

    check(
        model.num_items == NUM_ITEMS,
        f"Number of items = {model.num_items}",
    )

    check(
        model.embedding_dim == EMBEDDING_DIM,
        f"Embedding dimension = {model.embedding_dim}",
    )

    check(
        model.max_seq_len == MAX_SEQUENCE_LENGTH,
        f"Maximum sequence length = {model.max_seq_len}",
    )

    check(
        model.num_heads == NUM_HEADS,
        f"Number of SASRec heads = {model.num_heads}",
    )

    check(
        model.num_layers == NUM_LAYERS,
        f"Number of SASRec layers = {model.num_layers}",
    )

    # ========================================================
    # Transformer layer independence
    # ========================================================

    print(
        "\nChecking Transformer layer independence..."
    )

    transformer_layer_0 = (
        model.sasrec.transformer_layers[0]
    )

    transformer_layer_1 = (
        model.sasrec.transformer_layers[1]
    )

    check(
        transformer_layer_0 is not transformer_layer_1,
        "SASRec Transformer layers are separate objects",
    )

    check(
        transformer_layer_0.self_attn.in_proj_weight
        is not
        transformer_layer_1.self_attn.in_proj_weight,
        "SASRec Transformer layers do not share parameters",
    )

    # ========================================================
    # SASRec parameter count
    # ========================================================

    print("\nChecking SASRec parameter count...")

    sasrec_parameters = count_parameters(
        model.sasrec
    )

    print(
        f"SASRec parameters = "
        f"{sasrec_parameters:,}"
    )

    check(
        sasrec_parameters
        == EXPECTED_SASREC_PARAMETERS,
        (
            "SASRec parameter count = "
            f"{EXPECTED_SASREC_PARAMETERS:,}"
        ),
    )

    # ========================================================
    # V3 trainable parameter count
    # ========================================================

    print(
        "\nChecking V3 trainable parameters..."
    )

    model.freeze_pretrained_encoders()

    trainable_parameters = (
        count_trainable_parameters(model)
    )

    print(
        f"Trainable parameters = "
        f"{trainable_parameters:,}"
    )

    check(
        trainable_parameters
        == EXPECTED_TRAINABLE_PARAMETERS,
        (
            "V3 trainable parameter count = "
            f"{EXPECTED_TRAINABLE_PARAMETERS:,}"
        ),
    )

    # ========================================================
    # Total parameter count
    # ========================================================

    total_parameters = count_parameters(
        model
    )

    print(
        f"Total parameters = "
        f"{total_parameters:,}"
    )

    check(
        total_parameters
        == EXPECTED_TOTAL_PARAMETERS,
        (
            "V3 total parameter count = "
            f"{EXPECTED_TOTAL_PARAMETERS:,}"
        ),
    )

    # ========================================================
    # Expected trainable names
    # ========================================================

    print(
        "\nChecking trainable parameter names..."
    )

    trainable_names = [
        name
        for name, parameter
        in model.named_parameters()
        if parameter.requires_grad
    ]

    expected_trainable_names = {
        "user_gate.0.weight",
        "user_gate.0.bias",
        "user_fusion.0.weight",
        "user_fusion.0.bias",
        "item_fusion.0.weight",
        "item_fusion.0.bias",
    }

    check(
        set(trainable_names)
        == expected_trainable_names,
        "Only V3 gate/fusion parameters are trainable",
    )

    # ========================================================
    # Pretrained encoder frozen
    # ========================================================

    print(
        "\nChecking pretrained encoder freezing..."
    )

    lightgcn_trainable = [
        name
        for name, parameter
        in model.lightgcn.named_parameters()
        if parameter.requires_grad
    ]

    sasrec_trainable = [
        name
        for name, parameter
        in model.sasrec.named_parameters()
        if parameter.requires_grad
    ]

    check(
        len(lightgcn_trainable) == 0,
        "LightGCN is frozen",
    )

    check(
        len(sasrec_trainable) == 0,
        "SASRec is frozen",
    )

    # ========================================================
    # Encoder eval mode
    # ========================================================

    check(
        not model.lightgcn.training,
        "LightGCN is in evaluation mode",
    )

    check(
        not model.sasrec.training,
        "SASRec is in evaluation mode",
    )

    # ========================================================
    # Load pretrained checkpoints
    # ========================================================

    print(
        "\nLoading pretrained checkpoints..."
    )

    lightgcn_checkpoint = torch.load(
        LIGHTGCN_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" in lightgcn_checkpoint:
        lightgcn_state_dict = (
            lightgcn_checkpoint[
                "model_state_dict"
            ]
        )
    else:
        lightgcn_state_dict = (
            lightgcn_checkpoint
        )

    model.lightgcn.load_state_dict(
        lightgcn_state_dict,
        strict=True,
    )

    sasrec_checkpoint = torch.load(
        SASREC_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" in sasrec_checkpoint:
        sasrec_state_dict = (
            sasrec_checkpoint[
                "model_state_dict"
            ]
        )
    else:
        sasrec_state_dict = (
            sasrec_checkpoint
        )

    # --------------------------------------------------------
    # Convert old SASRec checkpoint names
    # --------------------------------------------------------

    converted_sasrec_state_dict = {}

    for key, value in sasrec_state_dict.items():

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

        converted_sasrec_state_dict[
            new_key
        ] = value

    model.sasrec.load_state_dict(
        converted_sasrec_state_dict,
        strict=True,
    )

    print(
        "[PASS] LightGCN checkpoint loaded"
    )

    print(
        "[PASS] SASRec checkpoint loaded"
    )

    # ========================================================
    # Confirm checkpoint did not reactivate training
    # ========================================================

    model.freeze_pretrained_encoders()

    # ========================================================
    # Dummy inputs
    # ========================================================

    print(
        "\nCreating dummy inputs..."
    )

    batch_size = 4

    user_ids = torch.tensor(
        [0, 1, 2, 3],
        dtype=torch.long,
    )

    # --------------------------------------------------------
    # Item IDs are zero-based training IDs.
    # SASRec internally converts them to +1.
    # --------------------------------------------------------

    positive_item_ids = torch.tensor(
        [10, 20, 30, 40],
        dtype=torch.long,
    )

    negative_item_ids = torch.tensor(
        [100, 200, 300, 400],
        dtype=torch.long,
    )

    # --------------------------------------------------------
    # Non-empty left-padded sequences.
    # 0 = padding.
    # --------------------------------------------------------

    sequences = torch.tensor(
        [
            [0, 0, 0, 0, 10],
            [0, 0, 20, 21, 22],
            [0, 30, 31, 32, 33],
            [40, 41, 42, 43, 44],
        ],
        dtype=torch.long,
    )

    # ========================================================
    # Dummy LightGCN embeddings
    # ========================================================

    lightgcn_user_embeddings = torch.randn(
        NUM_USERS,
        EMBEDDING_DIM,
    )

    lightgcn_item_embeddings = torch.randn(
        NUM_ITEMS,
        EMBEDDING_DIM,
    )

    # ========================================================
    # SASRec raw representation
    # ========================================================

    print(
        "\nTesting raw SASRec sequence representation..."
    )

    with torch.no_grad():

        sequence_representations = (
            model.sasrec.get_sequence_representation(
                sequences
            )
        )

    check(
        sequence_representations.shape
        == (
            batch_size,
            EMBEDDING_DIM,
        ),
        (
            "Raw SASRec sequence representation "
            f"shape = {list(sequence_representations.shape)}"
        ),
    )

    check(
        torch.isfinite(
            sequence_representations
        ).all(),
        "Raw SASRec representations are finite",
    )

    raw_sequence_norms = (
        torch.linalg.vector_norm(
            sequence_representations,
            dim=1,
        )
    )

    print(
        "Raw SASRec norms:",
        raw_sequence_norms.tolist(),
    )

    check(
        torch.all(
            raw_sequence_norms > 0.0
        ),
        "Raw SASRec representations are non-zero",
    )

    # ========================================================
    # User representation
    # ========================================================

    print(
        "\nTesting adaptive user representation..."
    )

    model.eval()

    with torch.no_grad():

        (
            hybrid_user,
            gate,
            graph_user,
            sequence_user,
        ) = model.get_user_representation(
            user_ids=user_ids,
            sequences=sequences,
            lightgcn_user_embeddings=(
                lightgcn_user_embeddings
            ),
        )

    # --------------------------------------------------------
    # Shapes
    # --------------------------------------------------------

    check(
        graph_user.shape
        == (
            batch_size,
            EMBEDDING_DIM,
        ),
        (
            "Normalized graph user shape = "
            f"{list(graph_user.shape)}"
        ),
    )

    check(
        sequence_user.shape
        == (
            batch_size,
            EMBEDDING_DIM,
        ),
        (
            "Normalized sequence user shape = "
            f"{list(sequence_user.shape)}"
        ),
    )

    check(
        gate.shape
        == (
            batch_size,
            EMBEDDING_DIM,
        ),
        (
            "Adaptive gate shape = "
            f"{list(gate.shape)}"
        ),
    )

    check(
        hybrid_user.shape
        == (
            batch_size,
            EMBEDDING_DIM * 2,
        ),
        (
            "Hybrid user shape = "
            f"{list(hybrid_user.shape)}"
        ),
    )

    # --------------------------------------------------------
    # Finite
    # --------------------------------------------------------

    check(
        torch.isfinite(
            hybrid_user
        ).all(),
        "Hybrid user representations are finite",
    )

    check(
        torch.isfinite(
            gate
        ).all(),
        "Gate values are finite",
    )

    # --------------------------------------------------------
    # Gate range
    # --------------------------------------------------------

    check(
        torch.all(gate >= 0.0)
        and torch.all(gate <= 1.0),
        "Gate values are within [0,1]",
    )

    # ========================================================
    # Normalization verification
    # ========================================================

    print(
        "\nChecking L2 normalization..."
    )

    graph_norms = (
        torch.linalg.vector_norm(
            graph_user,
            dim=1,
        )
    )

    sequence_norms = (
        torch.linalg.vector_norm(
            sequence_user,
            dim=1,
        )
    )

    print(
        "Normalized LightGCN user norms:",
        graph_norms.tolist(),
    )

    print(
        "Normalized SASRec user norms:",
        sequence_norms.tolist(),
    )

    check(
        torch.allclose(
            graph_norms,
            torch.ones_like(graph_norms),
            atol=1e-5,
        ),
        "LightGCN user vectors are L2 normalized",
    )

    check(
        torch.allclose(
            sequence_norms,
            torch.ones_like(sequence_norms),
            atol=1e-5,
        ),
        "SASRec user vectors are L2 normalized",
    )

    # ========================================================
    # Direct normalization function test
    # ========================================================

    normalized_direct = model.l2_normalize(
        sequence_representations
    )

    normalized_direct_norms = (
        torch.linalg.vector_norm(
            normalized_direct,
            dim=1,
        )
    )

    check(
        torch.allclose(
            normalized_direct_norms,
            torch.ones_like(
                normalized_direct_norms
            ),
            atol=1e-5,
        ),
        "V3 L2 normalization produces unit SASRec vectors",
    )

    # ========================================================
    # Item representation
    # ========================================================

    print(
        "\nTesting item representation..."
    )

    test_item_ids = torch.tensor(
        [10, 20, 30, 40],
        dtype=torch.long,
    )

    with torch.no_grad():

        hybrid_items = (
            model.get_item_representation(
                item_ids=test_item_ids,
                lightgcn_item_embeddings=(
                    lightgcn_item_embeddings
                ),
            )
        )

    check(
        hybrid_items.shape
        == (
            batch_size,
            EMBEDDING_DIM * 2,
        ),
        (
            "Hybrid item shape = "
            f"{list(hybrid_items.shape)}"
        ),
    )

    check(
        torch.isfinite(
            hybrid_items
        ).all(),
        "Hybrid item representations are finite",
    )

    # ========================================================
    # Pair scoring
    # ========================================================

    print(
        "\nTesting pairwise scoring..."
    )

    with torch.no_grad():

        positive_scores = model.score_pairs(
            hybrid_user=hybrid_user,
            item_ids=positive_item_ids,
            lightgcn_item_embeddings=(
                lightgcn_item_embeddings
            ),
        )

        negative_scores = model.score_pairs(
            hybrid_user=hybrid_user,
            item_ids=negative_item_ids,
            lightgcn_item_embeddings=(
                lightgcn_item_embeddings
            ),
        )

    check(
        positive_scores.shape
        == (batch_size,),
        (
            "Positive score shape = "
            f"{list(positive_scores.shape)}"
        ),
    )

    check(
        negative_scores.shape
        == (batch_size,),
        (
            "Negative score shape = "
            f"{list(negative_scores.shape)}"
        ),
    )

    check(
        torch.isfinite(
            positive_scores
        ).all(),
        "Positive scores are finite",
    )

    check(
        torch.isfinite(
            negative_scores
        ).all(),
        "Negative scores are finite",
    )

    # ========================================================
    # Full catalogue scoring
    # ========================================================

    print(
        "\nTesting full-catalogue scoring..."
    )

    with torch.no_grad():

        all_scores = model.score_all_items(
            hybrid_user=hybrid_user,
            lightgcn_item_embeddings=(
                lightgcn_item_embeddings
            ),
        )

    check(
        all_scores.shape
        == (
            batch_size,
            NUM_ITEMS,
        ),
        (
            "Full-catalogue score shape = "
            f"{list(all_scores.shape)}"
        ),
    )

    check(
        torch.isfinite(
            all_scores
        ).all(),
        "Full-catalogue scores are finite",
    )

    # ========================================================
    # BPR loss
    # ========================================================

    print(
        "\nTesting BPR loss..."
    )

    loss = model.bpr_loss(
        positive_scores,
        negative_scores,
    )

    print(
        f"BPR loss = {loss.item():.8f}"
    )

    check(
        torch.isfinite(loss),
        "BPR loss is finite",
    )

    check(
        loss.item() > 0.0,
        "BPR loss is positive",
    )

    # ========================================================
    # Full forward
    # ========================================================

    print(
        "\nTesting full forward pass..."
    )

    with torch.no_grad():

        output = model(
            user_ids=user_ids,
            sequences=sequences,
            positive_item_ids=positive_item_ids,
            negative_item_ids=negative_item_ids,
            lightgcn_user_embeddings=(
                lightgcn_user_embeddings
            ),
            lightgcn_item_embeddings=(
                lightgcn_item_embeddings
            ),
        )

    expected_output_keys = {
        "loss",
        "positive_scores",
        "negative_scores",
        "hybrid_user",
        "gate",
        "graph_user",
        "sequence_user",
    }

    check(
        set(output.keys())
        == expected_output_keys,
        "Forward output keys are correct",
    )

    check(
        torch.isfinite(
            output["loss"]
        ),
        "Forward loss is finite",
    )

    check(
        output["positive_scores"].shape
        == (batch_size,),
        "Forward positive score shape is correct",
    )

    check(
        output["negative_scores"].shape
        == (batch_size,),
        "Forward negative score shape is correct",
    )

    check(
        output["hybrid_user"].shape
        == (
            batch_size,
            EMBEDDING_DIM * 2,
        ),
        "Forward hybrid user shape is correct",
    )

    # ========================================================
    # Forward BPR consistency
    # ========================================================

    independent_loss = model.bpr_loss(
        output["positive_scores"],
        output["negative_scores"],
    )

    check(
        torch.allclose(
            output["loss"],
            independent_loss,
            atol=1e-6,
        ),
        "Forward loss matches independent BPR loss",
    )

    # ========================================================
    # Forward representation consistency
    # ========================================================

    direct_user = output[
        "hybrid_user"
    ]

    check(
        torch.allclose(
            direct_user,
            hybrid_user,
            atol=1e-6,
        ),
        "Forward hybrid user matches direct representation",
    )

    # ========================================================
    # Gradient flow
    # ========================================================

    print(
        "\nTesting gradient flow..."
    )

    model.train()

    # Keep pretrained encoders in eval mode even while
    # fusion layers are trained.
    model.lightgcn.eval()
    model.sasrec.eval()

    output = model(
        user_ids=user_ids,
        sequences=sequences,
        positive_item_ids=positive_item_ids,
        negative_item_ids=negative_item_ids,
        lightgcn_user_embeddings=(
            lightgcn_user_embeddings
        ),
        lightgcn_item_embeddings=(
            lightgcn_item_embeddings
        ),
    )

    loss = output["loss"]

    model.zero_grad()

    loss.backward()

    # --------------------------------------------------------
    # Fusion gradients
    # --------------------------------------------------------

    trainable_gradient_names = []

    for name, parameter in model.named_parameters():

        if parameter.requires_grad:

            check(
                parameter.grad is not None,
                f"Trainable parameter has gradient: {name}",
            )

            check(
                torch.isfinite(
                    parameter.grad
                ).all(),
                f"Trainable gradient is finite: {name}",
            )

            trainable_gradient_names.append(
                name
            )

    # --------------------------------------------------------
    # Frozen gradients
    # --------------------------------------------------------

    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:

            check(
                parameter.grad is None,
                f"Frozen parameter has no gradient: {name}",
            )

    # ========================================================
    # Gate statistics
    # ========================================================

    print(
        "\nChecking gate statistics..."
    )

    with torch.no_grad():

        model.eval()

        (
            _,
            gate,
            _,
            _,
        ) = model.get_user_representation(
            user_ids=user_ids,
            sequences=sequences,
            lightgcn_user_embeddings=(
                lightgcn_user_embeddings
            ),
        )

    gate_mean = gate.mean().item()
    gate_min = gate.min().item()
    gate_max = gate.max().item()

    print(
        f"Gate mean = {gate_mean:.6f}"
    )

    print(
        f"Gate min  = {gate_min:.6f}"
    )

    print(
        f"Gate max  = {gate_max:.6f}"
    )

    check(
        gate_min >= 0.0,
        "Gate minimum >= 0",
    )

    check(
        gate_max <= 1.0,
        "Gate maximum <= 1",
    )

    # ========================================================
    # Final summary
    # ========================================================

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    print(
        f"Passed: {passed}"
    )

    print(
        f"Failed: {failed}"
    )

    print(
        f"SASRec parameters: "
        f"{sasrec_parameters:,}"
    )

    print(
        f"Trainable V3 parameters: "
        f"{trainable_parameters:,}"
    )

    print(
        f"Total V3 parameters: "
        f"{total_parameters:,}"
    )

    print("=" * 70)

    if failed == 0:

        print(
            "HYBRID V3 MODEL SANITY TEST PASSED"
        )

    else:

        print(
            "HYBRID V3 MODEL SANITY TEST FAILED"
        )

        raise SystemExit(1)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()