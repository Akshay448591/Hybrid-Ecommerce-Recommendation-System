# 🎓 Major Project Defense Guide: Hybrid LightGCN + SASRec V2 Recommendation System

## 1. Project Abstract & Motivation
Contemporary e-commerce recommendation platforms face three fundamental challenges:
1. **Cold Start & Sparsity**: Graph collaborative filtering models (such as LightGCN) perform well on warm-start user-item interaction topologies, but struggle when user history is scarce or when intent shifts rapidly within a single session.
2. **Session Drift**: Sequential models (such as SASRec) capture real-time intra-session transition dynamics using self-attention, but lack global collaborative regularisation from higher-order bipartite graphs.
3. **Catastrophic Recommendation Collapse in Early Hybrids (The V1 Flaw)**: Naive dual-stream hybrids suffer from **scale dominance**. In our Hybrid V1 experiments, the SASRec embedding magnitude variance (mean norm ~3.52) dominated LightGCN (mean norm ~0.20), causing extreme catalogue collapse where **only 149 unique items** were ever recommended across the entire customer base.

### The Solution: Hybrid V2 Architecture
**Hybrid V2** introduces **independent L2 component normalisation** onto the unit hypersphere prior to concatenation, followed by non-linear projection manifolds (`user_fusion` and `item_fusion`) trained under **Pairwise Ranking Loss**.

---

## 2. Mathematical Formulations

### 2.1 Graph Collaborative Stream (LightGCN)
LightGCN discards non-linear feature transformations and self-connections:
$$E^{(k+1)} = \left(D^{-\frac{1}{2}} A D^{-\frac{1}{2}}\right) E^{(k)}$$
$$E = \sum_{k=0}^{K} \alpha_k E^{(k)}, \quad \alpha_k = \frac{1}{K + 1}$$
- **Propagation Depth**: $K = 3$ layers
- **Latent Dimension**: $d = 64$

### 2.2 Sequential Self-Attention Stream (SASRec)
Given user interaction sequence $S = (s_1, s_2, \dots, s_t)$:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d}} + M\right) V$$
where $M$ is a causal upper-triangular mask ensuring position $i$ only attends to positions $j \le i$.
- **Transformer Encoder Blocks**: 2 layers
- **Attention Heads**: 2 heads
- **Activation Function**: GELU with LayerNorm and residual connections

### 2.3 Hybrid V2 L2 Normalisation & Projection Fusion
To eliminate scale disparity:
$$\tilde{e}^G_u = \frac{e^G_u}{\|e^G_u\|_2 + \epsilon}, \quad \tilde{e}^S_u = \frac{e^S_u}{\|e^S_u\|_2 + \epsilon}$$
$$\tilde{e}^G_i = \frac{e^G_i}{\|e^G_i\|_2 + \epsilon}, \quad \tilde{e}^S_i = \frac{e^S_i}{\|e^S_i\|_2 + \epsilon}$$

Concatenated vectors in $\mathbb{R}^{128}$:
$$c_u = [\tilde{e}^G_u \,\|\, \tilde{e}^S_u], \quad c_i = [\tilde{e}^G_i \,\|\, \tilde{e}^S_i]$$

Learned Projection Heads:
$$h_u = \text{ReLU}(W_u c_u + b_u), \quad h_i = \text{ReLU}(W_i c_i + b_i)$$
$$W_u, W_i \in \mathbb{R}^{128 \times 128}, \quad b_u, b_i \in \mathbb{R}^{128}$$

Hybrid Score:
$$\text{Score}(u, i) = h_u \cdot h_i$$

### 2.4 Objective Function (Pairwise Margin / Ranking Loss)
$$\mathcal{L}_{\text{Rank}} = - \sum_{(u, i, j) \in \mathcal{D}} \ln \sigma\left(\text{Score}(u, i) - \text{Score}(u, j)\right)$$
- **Trainable Parameters**: 33,024 (pretrained encoders remain frozen)
- **Training Samples**: 2,000,000 interaction triplets
- **Convergence**: Final Loss = 0.2832 (down from 0.3239 across 5 epochs)

---

## 3. Empirical Results: V1 vs V2 Breakthrough

Evaluated on **891,329 warm-start validation interactions** across **173,203 users** on the H&M Benchmark:

| Evaluation Dimension | Hybrid V1 (Unnormalised) | Hybrid V2 (L2 Normalised) | Absolute Change | Impact |
| :--- | :---: | :---: | :---: | :---: |
| **Unique Items (@20)** | 149 items | **1,389 items** | **+1,240 items** | **Recommendation Collapse Cured** |
| **Catalogue Coverage (@20)** | 0.199% | **1.863%** | **+1.664% (+832%)** | **8.3x Broader Item Discovery** |
| **List Similarity (Jaccard)** | 0.4548 | **0.1597** | **-0.2951 (-64.9%)** | **High User Personalisation** |
| **Top-10 Concentration** | 43.10% | **1.23%** | **-41.87%** | **Superstar Bias Eliminated** |
| **Precision@10** | — | **0.276%** | — | High Precision on Massive Catalogue |
| **Hit Rate@20** | — | **3.968%** | — | ~4 in 100 users buy in Top-20 |
| **Novelty@20** | — | **11.85 bits** | — | Promotes Long-Tail Discovery |
| **Diversity@20** | — | **0.0385** | — | Cosine Distance Diversity |

---

## 4. Full-Stack Implementation Structure

```
Hybrid Ecommerce Recommendation Model/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI service with CORS & endpoints
│   │   ├── hybrid_v2_engine.py  # PyTorch LightGCN + SASRec inference engine
│   │   └── catalog_service.py   # Product retrieval, filtering & persona data
│   ├── data/
│   │   ├── curated_catalog.json # E-commerce apparel & fashion catalogue
│   │   ├── sample_personas.json # Distinct persona profiles (Sophia, Marcus, etc.)
│   │   ├── hybrid_v2_core.pt    # Extracted lightweight V2 projection & transformer weights
│   │   └── evaluation_summary.json # Benchmark & diagnostic metrics
│   ├── requirements.txt         # Production dependencies for cloud hosting
│   ├── Dockerfile               # Container deployment configuration
│   └── README.md                # Backend deployment guide
├── frontend/
│   ├── index.html               # Semantic HTML5 single-page application
│   ├── src/                     # React component source code
│   │   ├── components/          # Modular UI components (Storefront, Recommendations, ModelTuningLab, XaiModal, CartDrawer, etc.)
│   │   ├── services/api.js      # REST API client
│   │   ├── App.jsx              # Main React state container & routing
│   │   └── main.jsx             # React DOM root entry
│   ├── dist/                    # Built React production bundle
│   ├── styles.css               # Luxury dark-theme design system
│   ├── package.json             # React dependencies & Vite build scripts
│   └── README.md                # Frontend documentation
├── server/                      # MERN Stack API Gateway (Node.js + Express + MongoDB)
│   ├── models/                  # Mongoose schemas (Product, User, Session)
│   ├── server.js                # Express REST Gateway
│   └── package.json             # Node dependencies
├── render.yaml                  # 1-Click Render Blueprint configuration
├── run_local.bat                # Windows Batch launcher
├── run_local.ps1                # PowerShell launcher
└── PROJECT_DEFENSE_GUIDE.md     # This comprehensive viva guide
```

---

## 5. Typical Major Project Viva / Defense Questions & Answers

### Q1: Why did you freeze the pretrained LightGCN and SASRec encoders during hybrid training?
**Answer**: Pretraining LightGCN on the entire bipartite graph allows it to capture global collaborative topologies, while pretraining SASRec models sequence dynamics. Freezing both encoders preserves their specialized inductive biases and prevents catastrophic forgetting. It also dramatically reduces trainable parameters to just **33,024 weights** in the projection heads, enabling training on 2,000,000 examples in under 6 minutes.

### Q2: What was the cause of recommendation collapse in Hybrid V1?
**Answer**: Variance and magnitude disparity. In V1 without normalisation, SASRec output vector magnitudes were over 17x larger than LightGCN vectors ($\|e^S\| \approx 3.52$ vs $\|e^G\| \approx 0.20$). Consequently, the projection gradients were almost entirely driven by SASRec, collapsing the recommendation output into only 149 popular items. Applying independent L2 normalisation ($\|e\|_2 = 1$) solved this immediately, expanding unique recommended items by +832% to 1,389.

### Q3: How does the system handle real-time session updates on the website?
**Answer**: Every time the user interacts with an item (clicking, viewing, or adding to cart), the item ID is appended to the user's active session sequence. SASRec's causal transformer processes this sequence through multi-head self-attention, generating an updated representation $e^S_u$. The hybrid engine normalises and projects this vector with the user's graph embedding, instantly re-ranking the recommendations in under 15ms.

### Q4: How is Explainability (XAI) calculated?
**Answer**: Because the hybrid representation fuses normalised components, we decompose the total score into:
1. **Graph Collaborative Affinity**: $\tilde{e}^G_u \cdot \tilde{e}^G_i$
2. **Sequential Attention Affinity**: $\tilde{e}^S_u \cdot \tilde{e}^S_i$
3. **Cross-Domain Interactions**: $\frac{1}{2}(\tilde{e}^G_u \cdot \tilde{e}^S_i + \tilde{e}^S_u \cdot \tilde{e}^G_i)$
The contribution percentages visualize the relative influence of long-term collaborative profile versus active session intent.

### Q5: How does the React & MERN stack integrate with the PyTorch AI Backend?
**Answer**: In enterprise environments, Deep Learning model computation and transactional web services are separated using a **Microservices Architecture**:
- **Frontend (React)**: High-performance single page application built with React 18, utilizing state hooks, modular component trees, and reactive updates for real-time recommendation feeds and parameter tuning.
- **MERN Gateway (Node.js/Express + MongoDB)**: Handles standard transactional e-commerce workflows (cart sessions, persistence, user profile accounts).
- **AI Engine (FastAPI + PyTorch)**: Asynchronous Python microservice hosting LightGCN and SASRec weights for low-latency matrix multiplication and tensor inference.

---

## 6. How to Run & Host

### To Run Locally:
Double-click `run_local.bat` or run:
```bash
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser at `http://localhost:8000`.

### To Run React Frontend in Development Mode:
```bash
cd frontend
npm run dev
```
Open `http://localhost:5173`.

### To Host on Render:
1. Push repository to GitHub.
2. In Render, select **Blueprints** -> Connect Repo.
3. Render reads `render.yaml` and deploys automatically.

