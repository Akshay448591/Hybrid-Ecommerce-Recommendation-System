# Hybrid V2 Recommendation Engine - Backend Service

Production-ready FastAPI service powering the **Hybrid LightGCN + SASRec V2 E-Commerce Recommendation System**.

## 🧠 Architecture Overview
- **Graph Collaborative Filtering (LightGCN)**: 64-dim embeddings capturing higher-order customer co-purchase graph topology across 778,421 users and 74,541 items.
- **Sequential Transformer (SASRec)**: 2-layer, 2-head self-attention network capturing fine-grained intra-session user intent from recent browsing sequences.
- **L2-Normalized Projection Fusion (Hybrid V2)**: Both LightGCN and SASRec user/item representations are L2 normalized, concatenated (128-dim), and passed through trained projection heads (`user_fusion` & `item_fusion`, 33,024 trainable parameters) trained with pairwise ranking optimization loss.
- **Explainable AI (XAI)**: Decomposes each recommendation into Graph Collaborative Score, Sequential Score, and Cross-Domain Interaction terms.

---

## 🚀 Running Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the development server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation will be available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## ☁️ Deploying to Render

### Method 1: Web Service (Python Native)
1. Push your project to GitHub.
2. Go to [Render Dashboard](https://dashboard.render.com/) -> **New** -> **Web Service**.
3. Select your repository.
4. Set:
   - **Root Directory**: `backend`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Click **Create Web Service**.

### Method 2: Docker Web Service
1. Select **Docker** runtime.
2. Render will automatically detect `backend/Dockerfile` and deploy.

---

## 📡 Key API Endpoints

- `GET /api/products`: Full catalog with category/department filtering and search.
- `GET /api/personas`: Sample user personas with interaction history.
- `POST /api/recommendations`: Real-time personalized recommendations with XAI decomposition.
- `GET /api/demonstration/architecture`: Mathematical formulations and layer dimensions.
- `GET /api/demonstration/metrics`: Benchmark evaluation metrics (P@k, R@k, HR@k, NDCG@k, MAP, Coverage, Diversity, Novelty).
- `GET /api/demonstration/v1_vs_v2`: Detailed comparison showing the resolution of catalogue collapse (+832% unique items).
- `POST /api/demonstration/simulate`: Interactive simulator for live tuning of graph weight $\alpha$ and sequential weight $\beta$.
