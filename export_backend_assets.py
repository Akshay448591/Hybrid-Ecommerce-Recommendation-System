import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "backend" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("Loading validation recommendations and transactions...")
recs_df = pd.read_parquet(ROOT / "output" / "hybrid_v2_validation_recommendations.parquet")
val_tx = pd.read_parquet(ROOT / "data" / "transactions_val.parquet")

# Find top recommended items
top_item_counts = recs_df["train_item_id"].value_counts()
top_item_ids = top_item_counts.head(100).index.tolist()

# Also get items that appeared in validation transactions
tx_item_ids = val_tx["item_index"].value_counts().head(50).index.tolist()

# Combine unique candidate items
candidate_item_ids = []
for i in top_item_ids + tx_item_ids:
    if i not in candidate_item_ids and len(candidate_item_ids) < 72:
        candidate_item_ids.append(int(i))

print(f"Selected {len(candidate_item_ids)} distinct candidate item IDs from model.")

# Fashion categories & rich product templates
CATEGORIES = [
    {
        "category": "Outerwear & Coats",
        "department": "Outerwear",
        "items": [
            ("Oversized Wool Trench Coat", "Tailored double-breasted coat in rich brushed wool blend with wide lapels and storm flap.", 249.00, "https://images.unsplash.com/photo-1591047139829-d91aecb6caea?auto=format&fit=crop&w=800&q=80"),
            ("Minimalist Quilted Puffer Jacket", "Ultra-lightweight duck down insulation with matte water-repellent shell and fleece-lined pockets.", 189.00, "https://images.unsplash.com/photo-1544441893-675973e31985?auto=format&fit=crop&w=800&q=80"),
            ("Classic Camel Overcoat", "Timeless single-breasted silhouette cut from Italian wool cashmere blend with satin interior lining.", 289.00, "https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?auto=format&fit=crop&w=800&q=80"),
            ("Cropped Aviator Shearling Jacket", "Plush faux-shearling collar and buckle cuffs paired with structured vegan leather shell.", 195.00, "https://images.unsplash.com/photo-1551028719-00167b16eac5?auto=format&fit=crop&w=800&q=80"),
            ("Technical Hooded Parka", "Seam-sealed stormproof shell featuring thermal insulation, two-way zipper, and adjustable storm visor.", 219.00, "https://images.unsplash.com/photo-1548883354-7622d03aca27?auto=format&fit=crop&w=800&q=80"),
            ("Brushed Herringbone Wool Blazer", "Structured shoulder construction with notch lapel and horn buttons in timeless charcoal herringbone.", 175.00, "https://images.unsplash.com/photo-1507679799987-c73779587ccf?auto=format&fit=crop&w=800&q=80"),
        ]
    },
    {
        "category": "Tops & Knitwear",
        "department": "Tops",
        "items": [
            ("Pure Mongolian Cashmere Sweater", "Featherlight 2-ply pure cashmere knit with ribbed crew neckline and relaxed drape.", 165.00, "https://images.unsplash.com/photo-1620799140408-edc6dcb6d633?auto=format&fit=crop&w=800&q=80"),
            ("Relaxed Poplin Oxford Shirt", "Pre-washed organic cotton poplin with crisp point collar and mother-of-pearl buttons.", 85.00, "https://images.unsplash.com/photo-1596755094514-f87e34085b2c?auto=format&fit=crop&w=800&q=80"),
            ("Chunky Fisherman Ribbed Turtleneck", "Heavyweight lambswool rib-knit sweater crafted with raglan sleeves and substantial rolled neck.", 145.00, "https://images.unsplash.com/photo-1576566588028-4147f3842f27?auto=format&fit=crop&w=800&q=80"),
            ("Structured Heavyweight Tee", "Clean 280 GSM combed cotton crewneck t-shirt with boxy fit and reinforced binding.", 45.00, "https://images.unsplash.com/photo-1521572267360-ee0c2909d518?auto=format&fit=crop&w=800&q=80"),
            ("Merino Wool Polo Cardigan", "Fine-gauge extra-fine merino wool with mother-of-pearl button placket and ribbed hem.", 125.00, "https://images.unsplash.com/photo-1583743814966-8936f5b7be1a?auto=format&fit=crop&w=800&q=80"),
            ("Silk Blend Relaxed Button-Down", "Fluid silk-viscose blend drape with camp collar and subtle sand-washed sheen.", 115.00, "https://images.unsplash.com/photo-1602810318383-e386cc2a3ccf?auto=format&fit=crop&w=800&q=80"),
        ]
    },
    {
        "category": "Bottoms & Denim",
        "department": "Bottoms",
        "items": [
            ("Selvedge Straight Leg Denim", "Authentic 13.5oz Japanese selvedge denim in raw indigo wash with clean copper rivets.", 135.00, "https://images.unsplash.com/photo-1541099649105-f69ad21f3246?auto=format&fit=crop&w=800&q=80"),
            ("Pleated Wool Flannel Trousers", "Double reverse pleats with straight leg profile in Italian wool blend with coin pocket.", 155.00, "https://images.unsplash.com/photo-1624378439575-d8705ad7ae80?auto=format&fit=crop&w=800&q=80"),
            ("Relaxed Linen Drawstring Pants", "Breathable 100% Belgian flax linen with elasticated drawstring waistband for effortless ease.", 95.00, "https://images.unsplash.com/photo-1506629082955-511b1aa562c8?auto=format&fit=crop&w=800&q=80"),
            ("High-Rise Wide Leg Jeans", "Vintage stonewash denim featuring flattering high rise, relaxed thigh, and clean hem.", 110.00, "https://images.unsplash.com/photo-1582552938357-32b906df40cb?auto=format&fit=crop&w=800&q=80"),
            ("Structured Cotton Chinos", "Mid-weight stretch cotton twill with tapered leg and stain-resistant treatment.", 88.00, "https://images.unsplash.com/photo-1473966968600-fa801b869a1a?auto=format&fit=crop&w=800&q=80"),
            ("Tailored Wool Blend Shorts", "Sophisticated knee-length shorts in crisp tropical wool with belt loops and slash pockets.", 85.00, "https://images.unsplash.com/photo-1591195853828-11db59a44f6b?auto=format&fit=crop&w=800&q=80"),
        ]
    },
    {
        "category": "Dresses & Jumpsuits",
        "department": "Dresses",
        "items": [
            ("Bias-Cut Silk Slip Dress", "Gracefully draped 100% mulberry silk with delicate spaghetti straps and fluid midi hem.", 185.00, "https://images.unsplash.com/photo-1595777457583-95e059d581b8?auto=format&fit=crop&w=800&q=80"),
            ("Pleated Floral Chiffon Midi Dress", "Cascading accordion pleats with subtle botanical watercolor print and blouson sleeves.", 160.00, "https://images.unsplash.com/photo-1572804013309-59a88b7e92f1?auto=format&fit=crop&w=800&q=80"),
            ("Sculpted Ribbed Knit Maxi Dress", "Body-contouring compact rib knit with elegant mock turtleneck and walking slit.", 130.00, "https://images.unsplash.com/photo-1568252542512-9fe8fe9c87bb?auto=format&fit=crop&w=800&q=80"),
            ("Tailored Belted Utility Jumpsuit", "Crisp cotton twill jumpsuit featuring D-ring waist belt, flap pockets, and notched collar.", 168.00, "https://images.unsplash.com/photo-1502716119720-b23a93e5fe1b?auto=format&fit=crop&w=800&q=80"),
            ("Linen Tiered Bohemian Dress", "Airy airy garment-washed linen with ruffled tiers, keyhole neckline, and side pockets.", 140.00, "https://images.unsplash.com/photo-1496747611176-843222e1e57c?auto=format&fit=crop&w=800&q=80"),
            ("Asymmetrical Evening Gown", "Floor-sweeping satin crepe with one-shoulder neckline and thigh-high side slit.", 265.00, "https://images.unsplash.com/photo-1518895949257-7621c3c786d7?auto=format&fit=crop&w=800&q=80"),
        ]
    },
    {
        "category": "Footwear & Shoes",
        "department": "Footwear",
        "items": [
            ("Minimalist Calfskin Sneaker", "Italian nappa leather upper with vulcanized rubber sole and cushioned memory foam insole.", 145.00, "https://images.unsplash.com/photo-1560769629-975ec94e6a86?auto=format&fit=crop&w=800&q=80"),
            ("Chelsea Boot in Suede", "Handcrafted Portuguese suede with Goodyear welt, elastic side gussets, and pull tab.", 195.00, "https://images.unsplash.com/photo-1638247025967-b4e38f787b76?auto=format&fit=crop&w=800&q=80"),
            ("Lug-Sole Leather Loafers", "Polished spazzolato leather with chunky commando lug sole and penny keeper slot.", 175.00, "https://images.unsplash.com/photo-1614252235316-8c857d38b5f4?auto=format&fit=crop&w=800&q=80"),
            ("Strappy Heeled Sandal", "Graceful barely-there nappa straps with 70mm block heel and squared open toe.", 135.00, "https://images.unsplash.com/photo-1543163521-1bf539c55dd2?auto=format&fit=crop&w=800&q=80"),
            ("Performance Knit Running Shoe", "Engineered breathable mesh upper with responsive carbon-infused foam midsole.", 160.00, "https://images.unsplash.com/photo-1542291026-7eec264c27ff?auto=format&fit=crop&w=800&q=80"),
            ("Leather Combat Boots", "Durable full-grain leather with waxed laces, inside zipper, and treaded vibram sole.", 215.00, "https://images.unsplash.com/photo-1520639888713-7851133b1ed0?auto=format&fit=crop&w=800&q=80"),
        ]
    },
    {
        "category": "Accessories & Bags",
        "department": "Accessories",
        "items": [
            ("Structured Leather Crossbody Bag", "Full-grain Italian calf leather with accordion interior, magnetic clasp, and gold hardware.", 175.00, "https://images.unsplash.com/photo-1584917865442-de89df76afd3?auto=format&fit=crop&w=800&q=80"),
            ("Canvas & Leather Weekend Duffle", "Heavy duty 24oz water-resistant duck canvas with saddle leather straps and brass hardware.", 195.00, "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?auto=format&fit=crop&w=800&q=80"),
            ("Cashmere Wool Fringe Scarf", "Generously proportioned 100% cashmere scarf in heather grey with twisted fringe ends.", 85.00, "https://images.unsplash.com/photo-1608256246200-53e635b5b65f?auto=format&fit=crop&w=800&q=80"),
            ("Handcrafted Acetate Sunglasses", "Classic square silhouette cut from Italian Mazzucchelli acetate with polarized UV400 lenses.", 125.00, "https://images.unsplash.com/photo-1511499767150-a48a237f0083?auto=format&fit=crop&w=800&q=80"),
            ("Woven Leather Tote Bag", "Intricately hand-woven nappa leather basket tote with detachable zip pouch.", 220.00, "https://images.unsplash.com/photo-1590874103328-eac38a683ce7?auto=format&fit=crop&w=800&q=80"),
            ("Minimalist Automatic Watch", "Brushed 316L stainless steel case with sapphire crystal, clean Bauhaus dial, and Horween strap.", 240.00, "https://images.unsplash.com/photo-1524805444758-089113d48a6d?auto=format&fit=crop&w=800&q=80"),
        ]
    }
]

BRANDS = ["Aura Atelier", "Kite & Thread", "Nordic Studio", "Vanguard Denim", "Solstice Minimal", "Arc & Loom"]

# Build full catalog
catalog = []
item_idx = 0
for cat_obj in CATEGORIES:
    category = cat_obj["category"]
    department = cat_obj["department"]
    for title, desc, price, img_url in cat_obj["items"]:
        model_item_id = candidate_item_ids[item_idx % len(candidate_item_ids)]
        rating = round(4.3 + (item_idx % 7) * 0.1, 1)
        reviews_count = 24 + (item_idx * 17) % 350
        brand = BRANDS[item_idx % len(BRANDS)]
        
        catalog.append({
            "id": model_item_id,
            "title": title,
            "brand": brand,
            "category": category,
            "department": department,
            "price": price,
            "rating": rating,
            "reviews_count": reviews_count,
            "description": desc,
            "image": img_url,
            "in_stock": True,
            "tags": [department.lower(), brand.lower().replace(" ", "-"), "new-arrival" if item_idx % 2 == 0 else "bestseller"]
        })
        item_idx += 1

print(f"Catalog created with {len(catalog)} products.")
(DATA_DIR / "curated_catalog.json").write_text(json.dumps(catalog, indent=2))

# Personas
PERSONAS = [
    {
        "user_id": 0,
        "name": "Sophia Martinez",
        "tagline": "Minimalist Chic & Elevated Essentials",
        "description": "Prefers clean silhouettes, neutral palettes, fine cashmere knitwear, and tailored wool overcoats.",
        "avatar": "https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=300&q=80",
        "history_item_ids": [candidate_item_ids[0], candidate_item_ids[6], candidate_item_ids[12]],
        "preferred_category": "Tops & Knitwear"
    },
    {
        "user_id": 1,
        "name": "Marcus Vance",
        "tagline": "Contemporary Streetwear & Heavy Denim",
        "description": "Loves relaxed fit raw selvedge denim, heavy cotton t-shirts, aviator jackets, and lug-sole loafers.",
        "avatar": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?auto=format&fit=crop&w=300&q=80",
        "history_item_ids": [candidate_item_ids[1], candidate_item_ids[13], candidate_item_ids[25]],
        "preferred_category": "Bottoms & Denim"
    },
    {
        "user_id": 6,
        "name": "Elena Rostova",
        "tagline": "Luxury Eveningwear & Structured Elegance",
        "description": "Focuses on premium silk dresses, tailored blazers, leather bags, and statement evening pieces.",
        "avatar": "https://images.unsplash.com/photo-1517841905240-472988babdf9?auto=format&fit=crop&w=300&q=80",
        "history_item_ids": [candidate_item_ids[2], candidate_item_ids[18], candidate_item_ids[30]],
        "preferred_category": "Dresses & Jumpsuits"
    },
    {
        "user_id": 8,
        "name": "David Kim",
        "tagline": "Athletic Motion & Technical Outerwear",
        "description": "Values breathable technical parkas, performance knit footwear, and modern utility wear.",
        "avatar": "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?auto=format&fit=crop&w=300&q=80",
        "history_item_ids": [candidate_item_ids[4], candidate_item_ids[28], candidate_item_ids[16]],
        "preferred_category": "Outerwear & Coats"
    },
    {
        "user_id": 16,
        "name": "Chloe Bennett",
        "tagline": "Bohemian Artistry & Crafted Accessories",
        "description": "Appreciates hand-woven leather totes, fringe cashmere scarves, linen pieces, and acetate eyewear.",
        "avatar": "https://images.unsplash.com/photo-1524504388940-b1c1722653e1?auto=format&fit=crop&w=300&q=80",
        "history_item_ids": [candidate_item_ids[22], candidate_item_ids[32], candidate_item_ids[34]],
        "preferred_category": "Accessories & Bags"
    },
    {
        "user_id": -1,
        "name": "Guest Explorer",
        "tagline": "Cold-Start New Visitor (Session-Driven)",
        "description": "First-time visitor without interaction graph history; recommendations adapt live as items are clicked or added to cart via SASRec.",
        "avatar": "https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?auto=format&fit=crop&w=300&q=80",
        "history_item_ids": [],
        "preferred_category": "All"
    }
]

(DATA_DIR / "sample_personas.json").write_text(json.dumps(PERSONAS, indent=2))
print("Personas created.")

# Now extract model weights and embeddings
print("Extracting Hybrid V2 model tensors...")
ckpt = torch.load(ROOT / "output" / "hybrid_v2_model.pt", map_location="cpu")
sd = ckpt["model_state_dict"]

lightgcn_items_full = np.load(ROOT / "output" / "lightgcn_item_embeddings.npy", mmap_mode="r")
lightgcn_users_full = np.load(ROOT / "output" / "lightgcn_user_embeddings.npy", mmap_mode="r")

catalog_item_ids = [item["id"] for item in catalog]
lightgcn_item_subset = torch.from_numpy(lightgcn_items_full[catalog_item_ids].astype(np.float32))

persona_user_ids = [p["user_id"] for p in PERSONAS if p["user_id"] >= 0]
lightgcn_user_subset = {
    uid: torch.from_numpy(lightgcn_users_full[uid].astype(np.float32))
    for uid in persona_user_ids
}
# Cold start: mean user embedding or zeros
lightgcn_user_subset[-1] = torch.zeros(64, dtype=torch.float32)

# SASRec item embeddings for catalog items (+1 shift)
sasrec_full_emb = sd["sasrec.item_embedding.weight"]
sasrec_item_ids = torch.tensor([iid + 1 for iid in catalog_item_ids], dtype=torch.long)
sasrec_item_subset = sasrec_full_emb[sasrec_item_ids]

# Extract SASRec transformer state
sasrec_state = {
    k.replace("sasrec.", ""): v
    for k, v in sd.items()
    if k.startswith("sasrec.")
}

core_package = {
    "user_fusion": {
        "weight": sd["user_fusion.0.weight"],
        "bias": sd["user_fusion.0.bias"]
    },
    "item_fusion": {
        "weight": sd["item_fusion.0.weight"],
        "bias": sd["item_fusion.0.bias"]
    },
    "catalog_item_ids": catalog_item_ids,
    "lightgcn_item_embeddings": lightgcn_item_subset,
    "sasrec_item_embeddings": sasrec_item_subset,
    "lightgcn_user_embeddings": lightgcn_user_subset,
    "sasrec_state": sasrec_state,
    "embedding_dim": 64,
    "hybrid_dim": 128
}

torch.save(core_package, DATA_DIR / "hybrid_v2_core.pt")
print(f"Saved hybrid_v2_core.pt (size: {os.path.getsize(DATA_DIR / 'hybrid_v2_core.pt') / 1024 / 1024:.2f} MB)")

# Copy and structure evaluation metrics
val_metrics = json.loads((ROOT / "output" / "hybrid_v2_validation_metrics.json").read_text())
comp_diag = json.loads((ROOT / "output" / "hybrid_v2_component_diagnostic.json").read_text())
train_metrics = json.loads((ROOT / "output" / "hybrid_v2_training_metrics.json").read_text())

eval_summary = {
    "validation_metrics": val_metrics["metrics"],
    "efficiency": val_metrics["efficiency"],
    "architecture": val_metrics["architecture"],
    "training_history": train_metrics["training_history"],
    "total_training_time": train_metrics["total_training_time_seconds"],
    "v1_vs_v2": comp_diag["v1_vs_v2"],
    "score_statistics": comp_diag["score_statistics"],
    "component_contribution": comp_diag["component_contribution"],
    "score_correlations": comp_diag["score_correlations"],
    "diagnostic_flags": comp_diag["diagnostic_flags"]
}

(DATA_DIR / "evaluation_summary.json").write_text(json.dumps(eval_summary, indent=2))
print("evaluation_summary.json generated.")
print("All backend data successfully exported!")
