import os
import math
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Global configuration matching training
EMBEDDING_DIM = 64
HYBRID_DIM = 128
MAX_SEQUENCE_LENGTH = 50
NUM_HEADS = 2
NUM_LAYERS = 2
DROPOUT = 0.2


class LightweightSASRec(nn.Module):
    """
    SASRec self-attention encoder reconstructed for inference.
    """
    def __init__(self, num_items=74541, embedding_dim=EMBEDDING_DIM, max_seq_len=MAX_SEQUENCE_LENGTH, num_heads=NUM_HEADS, num_layers=NUM_LAYERS):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=embedding_dim * 4,
                dropout=DROPOUT,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            for _ in range(num_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        seq_len = sequences.shape[1]
        padding_mask = sequences.eq(0)
        
        item_embeds = self.item_embedding(sequences)
        positions = torch.arange(seq_len, device=sequences.device).unsqueeze(0)
        pos_embeds = self.position_embedding(positions)
        
        x = item_embeds + pos_embeds
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=sequences.device),
            diagonal=1
        )
        
        for layer in self.transformer_layers:
            x = layer(x, src_mask=causal_mask, src_key_padding_mask=padding_mask)
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            
        x = self.final_layer_norm(x)
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return x

    def get_sequence_representation(self, sequences: torch.Tensor) -> torch.Tensor:
        hidden = self.forward(sequences)
        batch_size, seq_len = sequences.shape
        non_padding = sequences.ne(0)
        positions = torch.arange(seq_len, device=sequences.device).unsqueeze(0)
        real_positions = torch.where(non_padding, positions, torch.full_like(positions, -1))
        last_indices = real_positions.max(dim=1).values
        
        if torch.any(last_indices < 0):
            # Fallback for empty sequence
            return torch.zeros((batch_size, self.embedding_dim), device=sequences.device)
            
        batch_indices = torch.arange(batch_size, device=sequences.device)
        return hidden[batch_indices, last_indices]


class HybridV2Engine:
    """
    Hybrid V2 Recommendation Engine.
    Executes LightGCN collaborative filtering + SASRec sequential self-attention
    with L2-normalized component concatenation and trained projection heads.
    """
    def __init__(self, data_path: Optional[Path] = None):
        if data_path is None:
            data_path = Path(__file__).resolve().parent.parent / "data" / "hybrid_v2_core.pt"
            
        self.data_path = data_path
        self._load_core()

    def _load_core(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"Model core weights not found at {self.data_path}")
            
        core = torch.load(self.data_path, map_location="cpu")
        
        self.catalog_item_ids: List[int] = core["catalog_item_ids"]
        self.item_id_to_idx: Dict[int, int] = {iid: idx for idx, iid in enumerate(self.catalog_item_ids)}
        
        # User & Item Fusion heads
        self.user_fusion_w = core["user_fusion"]["weight"]  # [128, 128]
        self.user_fusion_b = core["user_fusion"]["bias"]    # [128]
        self.item_fusion_w = core["item_fusion"]["weight"]  # [128, 128]
        self.item_fusion_b = core["item_fusion"]["bias"]    # [128]
        
        # Embeddings
        self.lightgcn_item_embeddings = core["lightgcn_item_embeddings"]  # [N_items, 64]
        self.sasrec_item_embeddings = core["sasrec_item_embeddings"]      # [N_items, 64]
        self.lightgcn_user_embeddings = core["lightgcn_user_embeddings"]  # Dict[user_id -> [64]]
        
        # Build SASRec transformer
        self.sasrec = LightweightSASRec(num_items=74541)
        # Load state dict
        sasrec_state = core["sasrec_state"]
        
        # Match keys
        current_sd = self.sasrec.state_dict()
        filtered_sd = {}
        for k, v in sasrec_state.items():
            if k in current_sd:
                filtered_sd[k] = v
            elif k.startswith("transformer_layers.") and k in current_sd:
                filtered_sd[k] = v
        self.sasrec.load_state_dict(filtered_sd, strict=False)
        self.sasrec.eval()
        
        # Precompute normalized item representations
        self._precompute_hybrid_items()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1, eps=1e-12)

    def _precompute_hybrid_items(self):
        with torch.no_grad():
            self.norm_lightgcn_items = self._normalize(self.lightgcn_item_embeddings)
            self.norm_sasrec_items = self._normalize(self.sasrec_item_embeddings)
            
            combined_items = torch.cat([self.norm_lightgcn_items, self.norm_sasrec_items], dim=1) # [N, 128]
            # Item projection: Linear(128, 128) + ReLU
            self.hybrid_item_representations = F.relu(
                F.linear(combined_items, self.item_fusion_w, self.item_fusion_b)
            )

    def get_user_graph_embedding(self, user_id: int) -> torch.Tensor:
        if user_id in self.lightgcn_user_embeddings:
            return self.lightgcn_user_embeddings[user_id]
        # Fallback for cold start / unknown user
        return torch.zeros(EMBEDDING_DIM, dtype=torch.float32)

    def get_sequence_embedding(self, session_sequence: List[int]) -> torch.Tensor:
        """
        Encode recent session item IDs using SASRec transformer.
        Item IDs are mapped to 1..num_items (+1 shift).
        """
        if not session_sequence:
            # Default neutral sequence representation
            return torch.zeros(EMBEDDING_DIM, dtype=torch.float32)
            
        # Keep up to MAX_SEQUENCE_LENGTH items
        seq = session_sequence[-MAX_SEQUENCE_LENGTH:]
        # Shift items by +1 for SASRec embedding indexing
        shifted_seq = [iid + 1 for iid in seq]
        
        # Pad with zeros on left
        pad_len = MAX_SEQUENCE_LENGTH - len(shifted_seq)
        padded_seq = [0] * pad_len + shifted_seq
        
        seq_tensor = torch.tensor([padded_seq], dtype=torch.long)
        with torch.no_grad():
            seq_emb = self.sasrec.get_sequence_representation(seq_tensor)
        return seq_emb.squeeze(0)

    def predict(
        self,
        user_id: int = -1,
        session_sequence: Optional[List[int]] = None,
        top_k: int = 12,
        graph_weight: float = 1.0,
        seq_weight: float = 1.0,
        mode: str = "hybrid_v2"  # 'hybrid_v2', 'pure_lightgcn', 'pure_sasrec', 'unnormalized_v1'
    ) -> List[Dict[str, Any]]:
        """
        Calculates recommendations and detailed component contribution breakdown.
        """
        if session_sequence is None:
            session_sequence = []

        with torch.no_grad():
            # 1. User graph component
            raw_user_graph = self.get_user_graph_embedding(user_id)
            has_graph = torch.count_nonzero(raw_user_graph) > 0
            
            # 2. User sequential component
            raw_user_seq = self.get_sequence_embedding(session_sequence)
            has_seq = len(session_sequence) > 0
            
            # Normalize components
            norm_user_graph = self._normalize(raw_user_graph)
            norm_user_seq = self._normalize(raw_user_seq) if has_seq else torch.zeros_like(norm_user_graph)
            
            # Apply interactive weights if custom simulation
            weighted_graph = norm_user_graph * graph_weight
            weighted_seq = norm_user_seq * seq_weight
            
            # Combine
            if not has_graph and not has_seq:
                # Cold start default: average catalog affinity
                combined_user = torch.cat([norm_user_graph, self._normalize(self.norm_sasrec_items.mean(dim=0))], dim=0)
            elif not has_graph:
                # Only sequential history (e.g. guest or new user browsing)
                combined_user = torch.cat([weighted_seq, weighted_seq], dim=0)
            elif not has_seq:
                # Only graph history (no active session yet)
                combined_user = torch.cat([weighted_graph, weighted_graph], dim=0)
            else:
                combined_user = torch.cat([weighted_graph, weighted_seq], dim=0)
                
            # User projection: Linear(128, 128) + ReLU
            hybrid_user = F.relu(
                F.linear(combined_user, self.user_fusion_w, self.user_fusion_b)
            )
            
            # Component dot products for explainability:
            # 1. Graph affinity
            graph_scores = torch.mv(self.norm_lightgcn_items, norm_user_graph).numpy()
            # 2. Sequential affinity
            seq_scores = torch.mv(self.norm_sasrec_items, norm_user_seq).numpy() if has_seq else np.zeros_like(graph_scores)
            # 3. Cross terms
            cross_1 = torch.mv(self.norm_sasrec_items, norm_user_graph).numpy()
            cross_2 = torch.mv(self.norm_lightgcn_items, norm_user_seq).numpy() if has_seq else np.zeros_like(graph_scores)
            
            # Mode selection
            if mode == "pure_lightgcn":
                final_scores = graph_scores
            elif mode == "pure_sasrec":
                final_scores = seq_scores if has_seq else graph_scores
            elif mode == "unnormalized_v1":
                # Unnormalized dot product simulation that caused V1 catalogue collapse
                raw_dot = (raw_user_graph @ self.lightgcn_item_embeddings.T).numpy()
                final_scores = raw_dot * 0.1
            else:
                # Hybrid V2 projection score
                v2_scores = torch.mv(self.hybrid_item_representations, hybrid_user).numpy()
                final_scores = v2_scores

            # Rank items
            ranked_indices = np.argsort(-final_scores)
            
            results = []
            for rank, idx in enumerate(ranked_indices[:top_k], 1):
                item_id = self.catalog_item_ids[idx]
                score_val = float(final_scores[idx])
                
                g_score = float(graph_scores[idx])
                s_score = float(seq_scores[idx])
                c_score = float((cross_1[idx] + cross_2[idx]) / 2.0)
                
                # Compute relative contribution percentage
                pos_g = max(0.001, g_score + 1.0)
                pos_s = max(0.001, s_score + 1.0) if has_seq else 0.05
                pos_c = max(0.001, c_score + 1.0)
                tot = pos_g + pos_s + pos_c
                
                g_pct = round((pos_g / tot) * 100, 1)
                s_pct = round((pos_s / tot) * 100, 1)
                c_pct = round((pos_c / tot) * 100, 1)
                
                # Human readable explanation
                if has_seq and s_pct > 40:
                    explanation = f"Strongly influenced by your recent browsing in this session ({s_pct}% sequential attention)."
                elif has_graph and g_pct > 40:
                    explanation = f"Matches your long-term style profile and similar customer purchase graphs ({g_pct}% collaborative match)."
                else:
                    explanation = f"High hybrid cross-domain resonance between your history and active session."

                results.append({
                    "item_id": item_id,
                    "rank": rank,
                    "hybrid_score": round(score_val, 4),
                    "graph_score": round(g_score, 4),
                    "sequential_score": round(s_score, 4),
                    "cross_domain_score": round(c_score, 4),
                    "contributions": {
                        "graph_percent": g_pct,
                        "sequential_percent": s_pct,
                        "cross_percent": c_pct
                    },
                    "explanation": explanation
                })

            return results
