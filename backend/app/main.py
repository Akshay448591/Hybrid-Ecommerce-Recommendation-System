import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from app.hybrid_v2_engine import HybridV2Engine
    from app.catalog_service import CatalogService
except ImportError:
    from backend.app.hybrid_v2_engine import HybridV2Engine
    from backend.app.catalog_service import CatalogService

app = FastAPI(
    title="Hybrid LightGCN + SASRec V2 E-Commerce API",
    description="Production E-commerce recommendation engine and major project demonstration platform powered by the Hybrid V2 model.",
    version="2.0.0"
)

# Enable CORS for frontend and deployment domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
BASE_DIR = Path(__file__).resolve().parent.parent
engine = HybridV2Engine(BASE_DIR / "data" / "hybrid_v2_core.pt")
catalog_svc = CatalogService(
    data_path=BASE_DIR / "data" / "curated_catalog.json",
    personas_path=BASE_DIR / "data" / "sample_personas.json"
)

eval_summary_path = BASE_DIR / "data" / "evaluation_summary.json"
EVAL_SUMMARY = json.loads(eval_summary_path.read_text()) if eval_summary_path.exists() else {}


# Pydantic Models
class RecommendationRequest(BaseModel):
    user_id: int = Field(default=0, description="Active user ID (-1 for cold start/guest)")
    session_sequence: List[int] = Field(default_factory=list, description="Recent item IDs clicked or added to cart")
    top_k: int = Field(default=12, ge=1, le=50)
    graph_weight: float = Field(default=1.0, ge=0.0, le=3.0, description="Interactive graph component weight (alpha)")
    seq_weight: float = Field(default=1.0, ge=0.0, le=3.0, description="Interactive sequential component weight (beta)")
    mode: str = Field(default="hybrid_v2", description="Ablation mode: 'hybrid_v2', 'pure_lightgcn', 'pure_sasrec', 'unnormalized_v1'")


@app.get("/api/info")
def root():
    return {
        "project": "Hybrid Ecommerce Recommendation Model - V2",
        "model_architecture": "LightGCN (Graph Collaborative Filtering) + SASRec (Self-Attentive Transformer)",
        "version": "2.0.0",
        "status": "online",
        "endpoints": {
            "catalog": "/api/products",
            "personas": "/api/personas",
            "recommendations": "/api/recommendations (POST)",
            "architecture": "/api/demonstration/architecture",
            "benchmark_metrics": "/api/demonstration/metrics",
            "v1_vs_v2_comparison": "/api/demonstration/v1_vs_v2",
            "interactive_simulation": "/api/demonstration/simulate (POST)"
        }
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "device": "cpu",
        "trainable_parameters": 33024,
        "embedding_dimension": 64,
        "hybrid_dimension": 128,
        "total_catalog_products": len(catalog_svc.catalog)
    }


@app.get("/api/products")
def get_products(
    category: Optional[str] = None,
    department: Optional[str] = None,
    query: Optional[str] = None,
    sort_by: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0)
):
    return catalog_svc.get_all(
        category=category,
        department=department,
        query=query,
        sort_by=sort_by,
        limit=limit,
        offset=offset
    )


@app.get("/api/products/{product_id}")
def get_product(product_id: int):
    product = catalog_svc.get_by_id(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
        
    # Get 4 similar products using Hybrid item embedding similarity
    sim_recs = engine.predict(
        user_id=-1,
        session_sequence=[product_id],
        top_k=5,
        mode="hybrid_v2"
    )
    similar_items = []
    for r in sim_recs:
        if r["item_id"] != product_id and len(similar_items) < 4:
            p = catalog_svc.get_by_id(r["item_id"])
            if p:
                similar_items.append({**p, "similarity_score": r["hybrid_score"]})

    return {
        "product": product,
        "similar_products": similar_items
    }


@app.get("/api/personas")
def get_personas():
    personas = catalog_svc.get_personas()
    # Enrich with their product history objects
    enriched = []
    for p in personas:
        hist_items = [catalog_svc.get_by_id(iid) for iid in p.get("history_item_ids", []) if catalog_svc.get_by_id(iid)]
        enriched.append({
            **p,
            "history_products": hist_items
        })
    return enriched


@app.get("/api/personas/{user_id}")
def get_persona(user_id: int):
    p = catalog_svc.get_persona(user_id)
    if not p:
        raise HTTPException(status_code=404, detail="Persona not found")
    hist_items = [catalog_svc.get_by_id(iid) for iid in p.get("history_item_ids", []) if catalog_svc.get_by_id(iid)]
    return {**p, "history_products": hist_items}


@app.post("/api/recommendations")
def get_recommendations(req: RecommendationRequest):
    # Predict top recommendations
    recs = engine.predict(
        user_id=req.user_id,
        session_sequence=req.session_sequence,
        top_k=req.top_k,
        graph_weight=req.graph_weight,
        seq_weight=req.seq_weight,
        mode=req.mode
    )
    
    # Merge with product catalog details
    enriched_recommendations = []
    for r in recs:
        product = catalog_svc.get_by_id(r["item_id"])
        if product:
            enriched_recommendations.append({
                **product,
                "recommendation_metadata": {
                    "rank": r["rank"],
                    "hybrid_score": r["hybrid_score"],
                    "graph_score": r["graph_score"],
                    "sequential_score": r["sequential_score"],
                    "cross_domain_score": r["cross_domain_score"],
                    "contributions": r["contributions"],
                    "explanation": r["explanation"]
                }
            })

    # Persona metadata
    persona = catalog_svc.get_persona(req.user_id)

    return {
        "user_id": req.user_id,
        "persona": persona,
        "session_sequence_length": len(req.session_sequence),
        "mode": req.mode,
        "weights": {
            "graph_alpha": req.graph_weight,
            "sequential_beta": req.seq_weight
        },
        "count": len(enriched_recommendations),
        "recommendations": enriched_recommendations
    }


@app.post("/api/demonstration/simulate")
def simulate(req: RecommendationRequest):
    return get_recommendations(req)


@app.get("/api/demonstration/architecture")
def get_architecture():
    return {
        "model_name": "Hybrid LightGCN + SASRec V2",
        "description": "Adaptive dual-stream recommendation network fusing graph collaborative filtering and self-attentive sequential representations through L2 normalization and trained projection manifolds.",
        "components": {
            "lightgcn": {
                "name": "LightGCN Graph Collaborative Encoder",
                "embedding_dimension": 64,
                "propagation_layers": 3,
                "role": "Captures high-order collaborative signals and global user-item neighborhood structures from the bipartite interaction graph without non-linear activations."
            },
            "sasrec": {
                "name": "SASRec Sequential Transformer Encoder",
                "embedding_dimension": 64,
                "transformer_layers": 2,
                "attention_heads": 2,
                "max_sequence_length": 50,
                "activation": "GELU",
                "role": "Models fine-grained temporal dynamics and recent intra-session transitions using multi-head self-attention with causal triangular masking."
            },
            "fusion_engine": {
                "normalization": "L2 Component Normalization: e_norm = e / ||e||_2",
                "concatenation_dimension": 128,
                "user_projection": "Linear(128, 128) + ReLU + Dropout(0.2)",
                "item_projection": "Linear(128, 128) + ReLU",
                "trainable_parameters": 33024,
                "frozen_encoders": True,
                "loss": "Pairwise Ranking Optimization Loss"
            }
        },
        "mathematical_formulation": {
            "user_representation": "h_u = ReLU( W_u · [ ||e^G_u||_2  ||  ||e^S_u||_2 ] + b_u )",
            "item_representation": "h_i = ReLU( W_i · [ ||e^G_i||_2  ||  ||e^S_i||_2 ] + b_i )",
            "hybrid_affinity_score": "Score(u, i) = h_u · h_i",
            "ranking_objective": "L_rank = - sum( ln( sigma( Score(u, i) - Score(u, j) ) ) )"
        }
    }


@app.get("/api/demonstration/metrics")
def get_benchmark_metrics():
    return {
        "protocol": {
            "dataset": "H&M Personalized Fashion Dataset (891,329 validation interactions)",
            "warm_start_eval_users": 173203,
            "training_users": 778421,
            "catalogue_items": 74541,
            "eval_cutoff": "Top 10 and Top 20"
        },
        "metrics": EVAL_SUMMARY.get("validation_metrics", {}),
        "efficiency": EVAL_SUMMARY.get("efficiency", {}),
        "training_history": EVAL_SUMMARY.get("training_history", []),
        "total_training_time_seconds": EVAL_SUMMARY.get("total_training_time", 365.78)
    }


@app.get("/api/demonstration/v1_vs_v2")
def get_v1_vs_v2():
    return {
        "title": "Resolving Recommendation Collapse: V1 vs V2 Empirical Breakthrough",
        "diagnostic_summary": "In Hybrid V1, unnormalized component fusion caused extreme recommendation concentration where only 149 unique items dominated the entire catalog. Hybrid V2 resolved this with independent L2 normalization and learned projection manifolds, increasing catalog coverage by +832% while dramatically lowering list redundancy.",
        "comparative_data": EVAL_SUMMARY.get("v1_vs_v2", {}),
        "score_statistics": EVAL_SUMMARY.get("score_statistics", {}),
        "component_contributions": EVAL_SUMMARY.get("component_contribution", {}),
        "correlations": EVAL_SUMMARY.get("score_correlations", {})
    }


# Mount frontend static directory (React production build or static root)
FRONTEND_DIR = BASE_DIR.parent / "frontend"
FRONTEND_DIST = FRONTEND_DIR / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
elif FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

