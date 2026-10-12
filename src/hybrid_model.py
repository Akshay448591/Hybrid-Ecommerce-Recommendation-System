import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Global configuration
# ============================================================

EMBEDDING_DIM = 64
MAX_SEQUENCE_LENGTH = 50
NUM_HEADS = 2
NUM_LAYERS = 2
DROPOUT = 0.2

LIGHTGCN_NUM_LAYERS = 3


# ============================================================
# SASRec
# ============================================================

class SASRec(nn.Module):
    """
    SASRec encoder used by the pretrained sequential model.

    Important:
    - Item ID 0 is reserved for padding.
    - Actual training item IDs are shifted by +1.
    - Transformer layers are independent.
    - Padded positions are explicitly zeroed after every layer
      to prevent NaN propagation with padding masks.
    """

    def __init__(
        self,
        num_items,
        embedding_dim=EMBEDDING_DIM,
        max_seq_len=MAX_SEQUENCE_LENGTH,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ):
        super().__init__()

        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # --------------------------------------------------------
        # Item embedding
        #
        # 0 = padding
        # 1..num_items = actual item IDs + 1
        # --------------------------------------------------------

        self.item_embedding = nn.Embedding(
            num_items + 1,
            embedding_dim,
            padding_idx=0,
        )

        # --------------------------------------------------------
        # Positional embedding
        # --------------------------------------------------------

        self.position_embedding = nn.Embedding(
            max_seq_len,
            embedding_dim,
        )

        # --------------------------------------------------------
        # Transformer layers
        #
        # IMPORTANT:
        # Create a NEW layer for every Transformer block.
        # Do NOT reuse the same encoder_layer object.
        # --------------------------------------------------------

        self.transformer_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=num_heads,
                    dim_feedforward=embedding_dim * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=False,
                )
                for _ in range(num_layers)
            ]
        )

        # --------------------------------------------------------
        # Final layer normalization
        # --------------------------------------------------------

        self.final_layer_norm = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    # ========================================================
    # Parameter initialization
    # ========================================================

    def _reset_parameters(self):
        nn.init.normal_(
            self.item_embedding.weight,
            mean=0.0,
            std=0.02,
        )

        nn.init.normal_(
            self.position_embedding.weight,
            mean=0.0,
            std=0.02,
        )

        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    # ========================================================
    # Causal mask
    # ========================================================

    def _build_causal_mask(self, seq_len, device):
        """
        Prevent position t from attending to future positions.
        """

        mask = torch.full(
            (seq_len, seq_len),
            float("-inf"),
            device=device,
        )

        mask = torch.triu(
            mask,
            diagonal=1,
        )

        return mask

    # ========================================================
    # Forward
    # ========================================================

    def forward(self, sequences):
        """
        sequences:
            [batch_size, sequence_length]

        Returns:
            [batch_size, sequence_length, embedding_dim]
        """

        batch_size, seq_len = sequences.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"max_seq_len={self.max_seq_len}"
            )

        device = sequences.device

        # --------------------------------------------------------
        # Padding mask
        #
        # True  = padding
        # False = real token
        # --------------------------------------------------------

        padding_mask = sequences.eq(0)

        # --------------------------------------------------------
        # Item embeddings
        # --------------------------------------------------------

        item_embeddings = self.item_embedding(sequences)

        # --------------------------------------------------------
        # Position embeddings
        # --------------------------------------------------------

        positions = torch.arange(
            seq_len,
            device=device,
        ).unsqueeze(0)

        position_embeddings = self.position_embedding(
            positions
        )

        # --------------------------------------------------------
        # Combine embeddings
        # --------------------------------------------------------

        x = item_embeddings + position_embeddings

        x = self.dropout(x)

        # Explicitly zero padded positions.
        x = x.masked_fill(
            padding_mask.unsqueeze(-1),
            0.0,
        )

        # --------------------------------------------------------
        # Causal mask
        #
        # Boolean mask:
        # True  = blocked
        # False = allowed
        #
        # This now has the SAME type as padding_mask.
        # --------------------------------------------------------

        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=device,
            ),
            diagonal=1,
        )

        # --------------------------------------------------------
        # Transformer
        # --------------------------------------------------------

        for transformer_layer in self.transformer_layers:

            x = transformer_layer(
                x,
                src_mask=causal_mask,
                src_key_padding_mask=padding_mask,
            )

            # Explicitly zero padded positions.
            x = x.masked_fill(
                padding_mask.unsqueeze(-1),
                0.0,
            )

        # --------------------------------------------------------
        # Final normalization
        # --------------------------------------------------------

        x = self.final_layer_norm(x)

        # Restore exact zero at padding positions.
        x = x.masked_fill(
            padding_mask.unsqueeze(-1),
            0.0,
        )

        return x

    # ========================================================
    # Sequence representation
    # ========================================================

    def get_sequence_representation(self, sequences):
    

        hidden = self.forward(sequences)

        batch_size, seq_len = sequences.shape

    # --------------------------------------------------------
    # Identify real-item positions.
    #
    # True  = real item
    # False = padding
    # --------------------------------------------------------

        non_padding = sequences.ne(0)

    # --------------------------------------------------------
    # Position indices:
    #
    # [0, 1, 2, 3, 4]
    # --------------------------------------------------------

        positions = torch.arange(
            seq_len,
            device=sequences.device,
        ).unsqueeze(0)

    # --------------------------------------------------------
    # Keep positions containing real items.
    #
    # Padding positions become -1.
    #
    # Example:
    #
    # [0, 0, 0, 0, 10]
    #
    # -> [-1, -1, -1, -1, 4]
    # --------------------------------------------------------

        real_positions = torch.where(
            non_padding,
            positions,
            torch.full_like(
                positions,
                -1,
            ),
        )

    # --------------------------------------------------------
    # Last real-item position.
    # --------------------------------------------------------

        last_indices = real_positions.max(
            dim=1
        ).values

    # --------------------------------------------------------
    # Safety check.
    #
    # Actual training prefixes are guaranteed to contain
    # at least one real item.
    # --------------------------------------------------------

        if torch.any(last_indices < 0):
            raise ValueError(
                "SASRec received an empty sequence containing "
                "only padding tokens."
            )

    # --------------------------------------------------------
    # Batch indices.
    # --------------------------------------------------------

        batch_indices = torch.arange(
            batch_size,
            device=sequences.device,
        )

    # --------------------------------------------------------
    # Extract representation of the last real item.
    # --------------------------------------------------------

        representation = hidden[
            batch_indices,
            last_indices,
        ]

        return representation

    # ========================================================
    # Score all items
    # ========================================================

    def score_all_items(self, sequence_representation):
        """
        Score every real item using the SASRec item embeddings.

        sequence_representation:
            [batch_size, embedding_dim]

        Returns:
            [batch_size, num_items]
        """

        item_embeddings = self.item_embedding.weight[1:]

        return torch.matmul(
            sequence_representation,
            item_embeddings.transpose(0, 1),
        )


# ============================================================
# LightGCN
# ============================================================

class LightGCN(nn.Module):
    """
    LightGCN encoder.

    The model contains:
    - user embeddings
    - item embeddings

    Propagation is performed using the externally supplied
    normalized graph adjacency matrix.
    """

    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim=EMBEDDING_DIM,
        num_layers=LIGHTGCN_NUM_LAYERS,
    ):
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.num_nodes = num_users + num_items

        self.embedding = nn.Embedding(
            self.num_nodes,
            embedding_dim,
        )

        nn.init.normal_(
            self.embedding.weight,
            mean=0.0,
            std=0.01,
        )

    # ========================================================
    # Propagation
    # ========================================================

    def propagate(self, adjacency):
        """
        LightGCN propagation.

        Returns:
            Final averaged node representations.
        """

        all_embeddings = self.embedding.weight

        embeddings = [
            all_embeddings
        ]

        x = all_embeddings

        for _ in range(self.num_layers):

            x = torch.sparse.mm(
                adjacency,
                x,
            )

            embeddings.append(x)

        final_embeddings = torch.stack(
            embeddings,
            dim=0,
        ).mean(dim=0)

        return final_embeddings

    # ========================================================
    # User embeddings
    # ========================================================

    def get_user_embeddings(self, adjacency):
        all_embeddings = self.propagate(adjacency)

        return all_embeddings[
            :self.num_users
        ]

    # ========================================================
    # Item embeddings
    # ========================================================

    def get_item_embeddings(self, adjacency):
        all_embeddings = self.propagate(adjacency)

        return all_embeddings[
            self.num_users:
        ]

    # ========================================================
    # Pair scoring
    # ========================================================

    def score_pairs(
        self,
        user_embeddings,
        item_embeddings,
    ):
        return (
            user_embeddings *
            item_embeddings
        ).sum(dim=1)


# ============================================================
# Hybrid V3
# ============================================================

class HybridLightGCNSASRecV3(nn.Module):
    """
    V3 Adaptive Graph-Sequential Hybrid.

    User representation:

        LightGCN user 64
                    |
                 L2 norm
                    |
                    ├──────────────┐
                    |              |
        SASRec user 64             |
                    |              |
                 L2 norm           |
                    |              |
                    └── concat ────┘
                         128
                          |
                    user_gate
                    Linear 128->64
                       Sigmoid
                          |
                         gate
                          |
        graph * gate     seq * (1-gate)
              |                |
              └──────┬─────────┘
                     |
                  concat
                    128
                     |
               Linear 128->128
                     |
                    ReLU
                     |
                  Dropout

    Item representation:

        LightGCN item 64
                    |
                 L2 norm
                    |
                    ├──────────────┐
                    |              |
        SASRec item 64             |
                    |              |
                 L2 norm           |
                    |              |
                    └── concat ────┘
                         128
                          |
                    Linear 128->128
                          |
                         ReLU

    Score:

        user_representation · item_representation

    Trainable parameters:
        user_gate      = 8,256
        user_fusion    = 16,512
        item_fusion    = 16,512

        Total          = 41,280
    """

    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim=EMBEDDING_DIM,
        max_seq_len=MAX_SEQUENCE_LENGTH,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ):
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # ----------------------------------------------------
        # Pretrained encoders
        # ----------------------------------------------------

        self.lightgcn = LightGCN(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            num_layers=LIGHTGCN_NUM_LAYERS,
        )

        self.sasrec = SASRec(
            num_items=num_items,
            embedding_dim=embedding_dim,
            max_seq_len=max_seq_len,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        # ----------------------------------------------------
        # Adaptive user gate
        #
        # Input:
        #   normalized graph + normalized sequence
        #   = 128
        #
        # Output:
        #   64 gates
        # ----------------------------------------------------

        self.user_gate = nn.Sequential(
            nn.Linear(
                embedding_dim * 2,
                embedding_dim,
            ),
            nn.Sigmoid(),
        )

        # ----------------------------------------------------
        # User fusion
        # ----------------------------------------------------

        self.user_fusion = nn.Sequential(
            nn.Linear(
                embedding_dim * 2,
                embedding_dim * 2,
            ),
            nn.ReLU(),
        )

        # ----------------------------------------------------
        # Item fusion
        # ----------------------------------------------------

        self.item_fusion = nn.Sequential(
            nn.Linear(
                embedding_dim * 2,
                embedding_dim * 2,
            ),
            nn.ReLU(),
        )

        # ----------------------------------------------------
        # Dropout
        # ----------------------------------------------------

        self.fusion_dropout = nn.Dropout(
            dropout
        )

    # ========================================================
    # Freeze pretrained encoders
    # ========================================================

    def freeze_pretrained_encoders(self):
        """
        Freeze LightGCN and SASRec parameters.
        """

        for parameter in self.lightgcn.parameters():
            parameter.requires_grad = False

        for parameter in self.sasrec.parameters():
            parameter.requires_grad = False

        self.lightgcn.eval()
        self.sasrec.eval()

    # ========================================================
    # Trainable parameters
    # ========================================================

    def trainable_parameters(self):
        return [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        ]

    # ========================================================
    # L2 normalization
    # ========================================================

    @staticmethod
    def l2_normalize(x):
        return F.normalize(
            x,
            p=2,
            dim=1,
            eps=1e-12,
        )

    # ========================================================
    # User representation
    # ========================================================

    def get_user_representation(
        self,
        user_ids,
        sequences,
        lightgcn_user_embeddings=None,
    ):
        """
        Returns:

            hybrid_user:
                [batch, 128]

            gate:
                [batch, 64]

            normalized_graph_user:
                [batch, 64]

            normalized_sequence_user:
                [batch, 64]
        """

        # ----------------------------------------------------
        # LightGCN user representation
        # ----------------------------------------------------

        if lightgcn_user_embeddings is not None:

            graph_user = lightgcn_user_embeddings[
                user_ids
            ]

        else:

            raise ValueError(
                "lightgcn_user_embeddings must be supplied "
                "for V3 hybrid training/evaluation."
            )

        # ----------------------------------------------------
        # SASRec sequence representation
        # ----------------------------------------------------

        sequence_user = self.sasrec.get_sequence_representation(
            sequences
        )

        # ----------------------------------------------------
        # Normalize both components independently
        # ----------------------------------------------------

        normalized_graph_user = self.l2_normalize(
            graph_user
        )

        normalized_sequence_user = self.l2_normalize(
            sequence_user
        )

        # ----------------------------------------------------
        # Concatenate
        # ----------------------------------------------------

        combined_user = torch.cat(
            [
                normalized_graph_user,
                normalized_sequence_user,
            ],
            dim=1,
        )

        # ----------------------------------------------------
        # Adaptive gate
        # ----------------------------------------------------

        gate = self.user_gate(
            combined_user
        )

        # ----------------------------------------------------
        # Adaptive component weighting
        #
        # gate -> graph contribution
        # 1-gate -> sequential contribution
        # ----------------------------------------------------

        adaptive_graph = (
            gate *
            normalized_graph_user
        )

        adaptive_sequence = (
            (1.0 - gate) *
            normalized_sequence_user
        )

        # ----------------------------------------------------
        # Recombine
        # ----------------------------------------------------

        adaptive_user = torch.cat(
            [
                adaptive_graph,
                adaptive_sequence,
            ],
            dim=1,
        )

        # ----------------------------------------------------
        # User projection
        # ----------------------------------------------------

        hybrid_user = self.user_fusion(
            adaptive_user
        )

        # ----------------------------------------------------
        # Dropout is active only during training mode.
        # ----------------------------------------------------

        hybrid_user = self.fusion_dropout(
            hybrid_user
        )

        return (
            hybrid_user,
            gate,
            normalized_graph_user,
            normalized_sequence_user,
        )

    # ========================================================
    # Item representation
    # ========================================================

    def get_item_representation(
        self,
        item_ids,
        lightgcn_item_embeddings=None,
    ):
        """
        Returns:
            [batch, 128]
        """

        if lightgcn_item_embeddings is None:
            raise ValueError(
                "lightgcn_item_embeddings must be supplied "
                "for V3 hybrid training/evaluation."
            )

        # ----------------------------------------------------
        # LightGCN item
        # ----------------------------------------------------

        graph_item = lightgcn_item_embeddings[
            item_ids
        ]

        # ----------------------------------------------------
        # SASRec item
        #
        # SASRec item 0 is padding.
        # Therefore actual item ID i maps to embedding i+1.
        # ----------------------------------------------------

        sequence_item = self.sasrec.item_embedding.weight[
            item_ids + 1
        ]

        # ----------------------------------------------------
        # Independent L2 normalization
        # ----------------------------------------------------

        normalized_graph_item = self.l2_normalize(
            graph_item
        )

        normalized_sequence_item = self.l2_normalize(
            sequence_item
        )

        # ----------------------------------------------------
        # Concatenate
        # ----------------------------------------------------

        combined_item = torch.cat(
            [
                normalized_graph_item,
                normalized_sequence_item,
            ],
            dim=1,
        )

        # ----------------------------------------------------
        # Item projection
        # ----------------------------------------------------

        hybrid_item = self.item_fusion(
            combined_item
        )

        return hybrid_item

    # ========================================================
    # Pair scoring
    # ========================================================

    def score_pairs(
        self,
        hybrid_user,
        item_ids,
        lightgcn_item_embeddings=None,
    ):
        hybrid_item = self.get_item_representation(
            item_ids=item_ids,
            lightgcn_item_embeddings=lightgcn_item_embeddings,
        )

        return (
            hybrid_user *
            hybrid_item
        ).sum(dim=1)

    # ========================================================
    # Score complete catalogue
    # ========================================================

    def score_all_items(
        self,
        hybrid_user,
        lightgcn_item_embeddings=None,
    ):
        """
        hybrid_user:
            [batch, 128]

        Returns:
            [batch, num_items]
        """

        if lightgcn_item_embeddings is None:
            raise ValueError(
                "lightgcn_item_embeddings must be supplied."
            )

        # ----------------------------------------------------
        # All LightGCN items
        # ----------------------------------------------------

        graph_items = lightgcn_item_embeddings

        # ----------------------------------------------------
        # All SASRec items
        #
        # Exclude padding row 0.
        # ----------------------------------------------------

        sequence_items = self.sasrec.item_embedding.weight[
            1:
        ]

        # ----------------------------------------------------
        # Normalize independently
        # ----------------------------------------------------

        normalized_graph_items = self.l2_normalize(
            graph_items
        )

        normalized_sequence_items = self.l2_normalize(
            sequence_items
        )

        # ----------------------------------------------------
        # Concatenate
        # ----------------------------------------------------

        combined_items = torch.cat(
            [
                normalized_graph_items,
                normalized_sequence_items,
            ],
            dim=1,
        )

        # ----------------------------------------------------
        # Project
        # ----------------------------------------------------

        hybrid_items = self.item_fusion(
            combined_items
        )

        # ----------------------------------------------------
        # Dot product
        # ----------------------------------------------------

        return torch.matmul(
            hybrid_user,
            hybrid_items.transpose(0, 1),
        )

    # ========================================================
    # BPR loss
    # ========================================================

    @staticmethod
    def bpr_loss(
        positive_scores,
        negative_scores,
    ):
        return -torch.mean(
            F.logsigmoid(
                positive_scores -
                negative_scores
            )
        )

    # ========================================================
    # Full forward
    # ========================================================

    def forward(
        self,
        user_ids,
        sequences,
        positive_item_ids,
        negative_item_ids,
        lightgcn_user_embeddings=None,
        lightgcn_item_embeddings=None,
    ):
        (
            hybrid_user,
            gate,
            graph_user,
            sequence_user,
        ) = self.get_user_representation(
            user_ids=user_ids,
            sequences=sequences,
            lightgcn_user_embeddings=lightgcn_user_embeddings,
        )

        # ----------------------------------------------------
        # Positive scores
        # ----------------------------------------------------

        positive_scores = self.score_pairs(
            hybrid_user=hybrid_user,
            item_ids=positive_item_ids,
            lightgcn_item_embeddings=lightgcn_item_embeddings,
        )

        # ----------------------------------------------------
        # Negative scores
        # ----------------------------------------------------

        negative_scores = self.score_pairs(
            hybrid_user=hybrid_user,
            item_ids=negative_item_ids,
            lightgcn_item_embeddings=lightgcn_item_embeddings,
        )

        # ----------------------------------------------------
        # BPR
        # ----------------------------------------------------

        loss = self.bpr_loss(
            positive_scores,
            negative_scores,
        )

        return {
            "loss": loss,
            "positive_scores": positive_scores,
            "negative_scores": negative_scores,
            "hybrid_user": hybrid_user,
            "gate": gate,
            "graph_user": graph_user,
            "sequence_user": sequence_user,
        }


# ============================================================
# Checkpoint key compatibility
# ============================================================

def convert_sasrec_checkpoint_keys(
    checkpoint_state_dict,
):
    """
    Converts the older SASRec checkpoint naming convention
    to the current V3 naming convention.

    Old:
        transformer_encoder.layers.X.*
        layer_norm.*

    New:
        transformer_layers.X.*
        final_layer_norm.*
    """

    converted = OrderedDict()

    for key, value in checkpoint_state_dict.items():

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

        converted[new_key] = value

    return converted


# ============================================================
# Load LightGCN
# ============================================================

def load_lightgcn(
    checkpoint_path,
    num_users,
    num_items,
    embedding_dim=EMBEDDING_DIM,
    num_layers=LIGHTGCN_NUM_LAYERS,
):
    model = LightGCN(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint[
            "model_state_dict"
        ]
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    return model


# ============================================================
# Load SASRec
# ============================================================

def load_sasrec(
    checkpoint_path,
    num_items=None,
    embedding_dim=EMBEDDING_DIM,
    max_seq_len=MAX_SEQUENCE_LENGTH,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    # --------------------------------------------------------
    # Determine state dict
    # --------------------------------------------------------

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint[
            "model_state_dict"
        ]
    else:
        state_dict = checkpoint

    # --------------------------------------------------------
    # Infer number of items from checkpoint when possible.
    #
    # item_embedding shape:
    #     [num_items + 1, embedding_dim]
    # --------------------------------------------------------

    if num_items is None:

        item_embedding_key = "item_embedding.weight"

        if item_embedding_key not in state_dict:
            raise KeyError(
                "Could not find item_embedding.weight "
                "in SASRec checkpoint."
            )

        num_items = (
            state_dict[
                item_embedding_key
            ].shape[0]
            - 1
        )

    # --------------------------------------------------------
    # Infer max sequence length
    # --------------------------------------------------------

    position_embedding_key = (
        "position_embedding.weight"
    )

    if position_embedding_key in state_dict:

        max_seq_len = state_dict[
            position_embedding_key
        ].shape[0]

    # --------------------------------------------------------
    # Infer embedding dimension
    # --------------------------------------------------------

    embedding_dim = state_dict[
        "item_embedding.weight"
    ].shape[1]

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = SASRec(
        num_items=num_items,
        embedding_dim=embedding_dim,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )

    # --------------------------------------------------------
    # Convert old checkpoint names.
    # --------------------------------------------------------

    converted_state_dict = (
        convert_sasrec_checkpoint_keys(
            state_dict
        )
    )

    # --------------------------------------------------------
    # Strict load.
    #
    # We intentionally do not silently ignore keys.
    # --------------------------------------------------------

    model.load_state_dict(
        converted_state_dict,
        strict=True,
    )

    return model


# ============================================================
# Load both pretrained encoders
# ============================================================

def load_pretrained_encoders(
    lightgcn_checkpoint_path,
    sasrec_checkpoint_path,
    num_users,
    num_items,
):
    lightgcn = load_lightgcn(
        checkpoint_path=lightgcn_checkpoint_path,
        num_users=num_users,
        num_items=num_items,
        embedding_dim=EMBEDDING_DIM,
        num_layers=LIGHTGCN_NUM_LAYERS,
    )

    sasrec = load_sasrec(
        checkpoint_path=sasrec_checkpoint_path,
        num_items=num_items,
        embedding_dim=EMBEDDING_DIM,
        max_seq_len=MAX_SEQUENCE_LENGTH,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )

    return lightgcn, sasrec


# ============================================================
# Parameter utilities
# ============================================================

def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def count_trainable_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )