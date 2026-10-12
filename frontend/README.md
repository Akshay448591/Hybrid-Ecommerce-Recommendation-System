# HYBRID - E-Commerce & Major Project Defense Frontend (React & MERN Architecture)

Production-grade, high-performance responsive **React** web application for the **Hybrid LightGCN + SASRec V2 Recommendation System**.

## ✨ Features
1. **Curated Luxury Storefront**: Browse curated apparel, knitwear, footwear, denim, dresses, and accessories with instant category filtering and sorting.
2. **"For You" Hybrid V2 Feed**: Real-time personalized recommendation feed powered by the dual-stream Hybrid V2 neural network.
3. **Session-Aware Attention Tracker**: Every product clicked or added to the bag is captured live in the SASRec causal self-attention window, adapting recommendations dynamically.
4. **Explainable AI (XAI) Inspector**: Decomposes any recommended product into its Graph Collaborative Affinity (LightGCN), Sequential Attention Affinity (SASRec), and Cross-Domain Latent Synergy.
5. **Interactive Model Lab / Simulator**: Live sliders to tune Graph weight $\alpha$ and Sequential weight $\beta$, plus ablation mode testing (Hybrid V2 vs LightGCN Only vs SASRec Only).
6. **Shopping Bag & Checkout Simulator**: Live stateful shopping drawer capturing sequential interactions.

---

## 🛠️ Architecture: React + MERN & AI Microservice

This project is structured according to modern full-stack enterprise standards:
- **Frontend**: **React 18** with functional components, custom hooks, and Vite bundling (`frontend/src/`).
- **Styling**: Premium custom CSS design system with glassmorphism, glowing accents, and responsive design (`frontend/styles.css`).
- **MERN API Gateway (Optional)**: Node.js & Express server with MongoDB Mongoose schemas (`server/`) for persistent user session & catalog operations.
- **AI Recommendation Engine**: Python FastAPI microservice (`backend/`) executing PyTorch tensor operations for LightGCN graph embeddings and SASRec transformer inference.

---

## 🚀 Running Locally

### Option 1: Standard All-in-One Local Runner (Recommended)
FastAPI automatically serves the built React production bundle directly:
```bash
# Windows Command Prompt
run_local.bat

# Windows PowerShell
.\run_local.ps1
```
Then open: **`http://localhost:8000`**

### Option 2: React Development Server (with Hot Reload)
```bash
cd frontend
npm install
npm run dev
```
Open **`http://localhost:5173`** (Vite automatically proxies `/api` requests to `http://localhost:8000`).

### Option 3: Building Production Bundle
```bash
cd frontend
npm run build
```
Builds the optimized production bundle to `frontend/dist/`.
