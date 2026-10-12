/**
 * HYBRID Recommendation Engine - API Service Layer
 * Connects React frontend with the FastAPI PyTorch Hybrid V2 Engine.
 * Features automatic environment detection, cross-origin/Render hosting resilience,
 * tab persistence via localStorage, and instant fallback so 0 products is never displayed.
 */

import {
  FALLBACK_PERSONAS,
  getFallbackProducts,
  generateFallbackRecommendations
} from '../data/fallbackData';

function detectApiBase() {
  if (typeof window === 'undefined') return '';

  // 1. Check URL query parameter: ?api=...
  const urlParams = new URLSearchParams(window.location.search);
  const paramApi = urlParams.get('api');
  if (paramApi) {
    const cleaned = paramApi.replace(/\/+$/, '');
    try {
      localStorage.setItem('HYBRID_API_BASE', cleaned);
    } catch (_) {}
    return cleaned;
  }

  // 2. Check saved API base in localStorage (persists in new tabs & refreshes!)
  try {
    const saved = localStorage.getItem('HYBRID_API_BASE');
    if (saved && saved.trim()) {
      return saved.trim().replace(/\/+$/, '');
    }
  } catch (_) {}

  // 3. Hosted / Browser environment:
  // When hosted on any platform (Render, Vercel, Netlify, custom domain), use relative path ''
  // This directs calls to the same origin (/api/...) directly.
  // It completely avoids Mixed Content errors and prevents Chrome's Private Network Access / device permission prompts.
  if (window.location.protocol === 'http:' || window.location.protocol === 'https:') {
    return '';
  }

  return '';
}

export const API_BASE = detectApiBase();

/** Helper fetch with timeout */
async function fetchWithTimeout(url, options = {}, timeoutMs = 6000) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      ...options,
      signal: controller.signal
    });
    return res;
  } finally {
    clearTimeout(id);
  }
}

export async function checkApiHealth() {
  const res = await fetchWithTimeout(`${API_BASE}/api/health`, {}, 4000);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return await res.json();
}

export async function fetchPersonas() {
  try {
    const res = await fetchWithTimeout(`${API_BASE}/api/personas`, {}, 5000);
    if (res.ok) {
      const data = await res.json();
      if (Array.isArray(data) && data.length > 0) return data;
    }
  } catch (err) {
    console.warn('API personas notice, using embedded personas:', err);
  }
  return FALLBACK_PERSONAS;
}

export async function fetchProducts(params = { limit: 60 }) {
  try {
    const query = new URLSearchParams();
    if (params.category && params.category !== 'all') query.set('category', params.category);
    if (params.sort_by) query.set('sort_by', params.sort_by);
    if (params.limit) query.set('limit', params.limit);

    const url = `${API_BASE}/api/products${query.toString() ? `?${query.toString()}` : ''}`;
    const res = await fetchWithTimeout(url, {}, 5000);
    if (res.ok) {
      const data = await res.json();
      if (data && Array.isArray(data.items) && data.items.length > 0) {
        return data;
      }
    }
  } catch (err) {
    console.warn('API products fetch notice, falling back to embedded catalog:', err);
  }

  // Guaranteed fallback so 0 products is never shown
  return getFallbackProducts(params);
}

export async function fetchRecommendations({
  userId = 0,
  sessionSequence = [],
  topK = 12,
  mode = 'hybrid_v2'
}) {
  try {
    const res = await fetchWithTimeout(`${API_BASE}/api/recommendations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: userId,
        session_sequence: sessionSequence,
        top_k: topK,
        mode: mode
      })
    }, 6000);

    if (res.ok) {
      const data = await res.json();
      if (data && Array.isArray(data.recommendations) && data.recommendations.length > 0) {
        return data;
      }
    }
  } catch (err) {
    console.warn('API recommendations notice, using client simulation:', err);
  }

  return {
    user_id: userId,
    session_sequence_length: sessionSequence.length,
    mode: mode,
    recommendations: generateFallbackRecommendations({
      userId,
      sessionSequence,
      topK,
      mode
    })
  };
}

export async function simulateRecommendations({
  userId = 0,
  sessionSequence = [],
  topK = 12,
  graphWeight = 1.0,
  seqWeight = 1.0,
  mode = 'hybrid_v2'
}) {
  try {
    const res = await fetchWithTimeout(`${API_BASE}/api/demonstration/simulate`, {
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
    }, 6000);

    if (res.ok) {
      const data = await res.json();
      if (data && Array.isArray(data.recommendations) && data.recommendations.length > 0) {
        return data;
      }
    }
  } catch (err) {
    console.warn('Simulation notice, using client simulation:', err);
  }

  return {
    recommendations: generateFallbackRecommendations({
      userId,
      sessionSequence,
      topK,
      mode,
      graphWeight,
      seqWeight
    })
  };
}

export async function fetchArchitecture() {
  try {
    const res = await fetchWithTimeout(`${API_BASE}/api/demonstration/architecture`, {}, 5000);
    if (res.ok) return await res.json();
  } catch (_) {}
  return {
    pipeline: 'LightGCN (Graph Collaborative) + SASRec (Self-Attention Transformer) -> L2 Normalization -> Dual MLP Fusion',
    version: '2.0.0'
  };
}

export async function fetchBenchmarkMetrics() {
  try {
    const res = await fetchWithTimeout(`${API_BASE}/api/demonstration/metrics`, {}, 5000);
    if (res.ok) return await res.json();
  } catch (_) {}
  return {
    protocol: { dataset: 'H&M Personalized Fashion Dataset (891,329 validation interactions)' }
  };
}

