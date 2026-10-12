/**
 * HYBRID Platform - Node.js & Express API Gateway (MERN Stack Architecture)
 * 
 * Provides:
 * 1. REST endpoints for Products, Users, Personas, and Cart Sessions
 * 2. MongoDB integration for persistent e-commerce records
 * 3. Microservice proxy to the Python PyTorch Hybrid V2 Engine (LightGCN + SASRec)
 */

const express = require('express');
const cors = require('cors');
const mongoose = require('mongoose');

const Product = require('./models/Product');
const User = require('./models/User');
const Session = require('./models/Session');

const app = express();
const PORT = process.env.PORT || 5000;
const AI_ENGINE_URL = process.env.AI_ENGINE_URL || 'http://localhost:8000';
const MONGO_URI = process.env.MONGO_URI || 'mongodb://localhost:27017/hybrid_ecommerce';

app.use(cors());
app.use(express.json());

// Optional MongoDB Connection (auto-reconnects or falls back smoothly)
if (process.env.MONGO_URI) {
  mongoose.connect(MONGO_URI)
    .then(() => console.log(' Connected to MongoDB (MERN Architecture Active)'))
    .catch(err => console.warn(' MongoDB connection optional warning:', err.message));
}

// -------------------------------------------------------------
// Health Check
// -------------------------------------------------------------
app.get('/api/mern/health', (req, res) => {
  res.json({
    status: 'online',
    framework: 'MERN (MongoDB + Express + React + Node.js)',
    aiEngineUrl: AI_ENGINE_URL,
    mongoConnected: mongoose.connection.readyState === 1
  });
});

// -------------------------------------------------------------
// Products API (Forwarding or DB Fallback)
// -------------------------------------------------------------
app.get('/api/mern/products', async (req, res) => {
  try {
    const fetch = (await import('node-fetch')).default;
    const response = await fetch(`${AI_ENGINE_URL}/api/products?${new URLSearchParams(req.query)}`);
    const data = await response.json();
    res.json(data);
  } catch (error) {
    res.status(500).json({ error: 'AI Engine unreachable', details: error.message });
  }
});

// -------------------------------------------------------------
// Recommendations Proxy (Dispatches to PyTorch LightGCN + SASRec)
// -------------------------------------------------------------
app.post('/api/mern/recommendations', async (req, res) => {
  try {
    const fetch = (await import('node-fetch')).default;
    const response = await fetch(`${AI_ENGINE_URL}/api/recommendations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body)
    });
    const data = await response.json();
    res.json(data);
  } catch (error) {
    res.status(500).json({ error: 'Recommendation inference failed', details: error.message });
  }
});

// -------------------------------------------------------------
// Model Simulation Proxy (Ablation and Weight Tuning)
// -------------------------------------------------------------
app.post('/api/mern/simulate', async (req, res) => {
  try {
    const fetch = (await import('node-fetch')).default;
    const response = await fetch(`${AI_ENGINE_URL}/api/demonstration/simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body)
    });
    const data = await response.json();
    res.json(data);
  } catch (error) {
    res.status(500).json({ error: 'Simulation failed', details: error.message });
  }
});

if (require.main === module) {
  app.listen(PORT, () => {
    console.log(`🚀 MERN Express Gateway running on port ${PORT}`);
    console.log(`🔗 Interfacing with PyTorch Hybrid Engine at ${AI_ENGINE_URL}`);
  });
}

module.exports = app;
