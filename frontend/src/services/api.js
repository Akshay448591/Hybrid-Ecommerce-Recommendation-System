/**
 * HYBRID Recommendation Engine - API Service Layer
 * Connects React frontend with the FastAPI PyTorch Hybrid V2 Engine.
 */

function detectApiBase() {
  if (typeof window === 'undefined') return 'http://localhost:8000';
  const urlParams = new URLSearchParams(window.location.search);
  if (urlParams.get('api')) {
    return urlParams.get('api');
  }
  // In dev with Vite proxy or when served directly from FastAPI port 8000
  if (window.location.port === '8000' || window.location.port === '5173') {
    return '';
  }
  return 'http://localhost:8000';
}

export const API_BASE = detectApiBase();

export async function checkApiHealth() {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return await res.json();
}

export async function fetchPersonas() {
  const res = await fetch(`${API_BASE}/api/personas`);
  if (!res.ok) throw new Error(`Failed to fetch personas: ${res.status}`);
  return await res.json();
}

export async function fetchProducts(params = { limit: 60 }) {
  const query = new URLSearchParams();
  if (params.category && params.category !== 'all') query.set('category', params.category);
  if (params.sort_by) query.set('sort_by', params.sort_by);
  if (params.limit) query.set('limit', params.limit);

  const url = `${API_BASE}/api/products${query.toString() ? `?${query.toString()}` : ''}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch products: ${res.status}`);
  return await res.json();
}

export async function fetchRecommendations({ userId = 0, sessionSequence = [], topK = 12, mode = 'hybrid_v2' }) {
  const res = await fetch(`${API_BASE}/api/recommendations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_id: userId,
      session_sequence: sessionSequence,
      top_k: topK,
      mode: mode
    })
  });
  if (!res.ok) throw new Error(`Failed to fetch recommendations: ${res.status}`);
  return await res.json();
}

export async function simulateRecommendations({
  userId = 0,
  sessionSequence = [],
  topK = 12,
  graphWeight = 1.0,
  seqWeight = 1.0,
  mode = 'hybrid_v2'
}) {
  const res = await fetch(`${API_BASE}/api/demonstration/simulate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_id: userId,
      session_sequence: sessionSequence,
      top_k: topK,
      graph_weight: graphWeight,
      seq_weight: seqWeight,
      mode: mode
    })
  });
  if (!res.ok) throw new Error(`Failed to simulate recommendations: ${res.status}`);
  return await res.json();
}

export async function fetchArchitecture() {
  const res = await fetch(`${API_BASE}/api/demonstration/architecture`);
  if (!res.ok) throw new Error(`Failed to fetch architecture: ${res.status}`);
  return await res.json();
}

export async function fetchBenchmarkMetrics() {
  const res = await fetch(`${API_BASE}/api/demonstration/metrics`);
  if (!res.ok) throw new Error(`Failed to fetch benchmark metrics: ${res.status}`);
  return await res.json();
}
