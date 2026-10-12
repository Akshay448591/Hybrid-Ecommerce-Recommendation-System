import React from 'react';
import ProductCard from './ProductCard';

export default function ModelTuningLab({
  graphWeight,
  onChangeGraphWeight,
  seqWeight,
  onChangeSeqWeight,
  mode,
  onChangeMode,
  personas,
  selectedUserId,
  onChangePersona,
  onRunSimulation,
  simRecommendations,
  isLoading,
  onAddToCart,
  onInspect
}) {
  const modeDisplay = mode.replace('_', ' ').toUpperCase();

  return (
    <section className="tab-pane active" id="pane-simulator">
      <div className="doc-header">
        <span className="doc-badge">Interactive Controls</span>
        <h2 className="doc-title">Hybrid Model Tuning & Simulation</h2>
        <p className="doc-subtitle">
          Adjust the balance between Collaborative Graph topology and Sequential Attention in real-time to observe how the hybrid recommender adapts.
        </p>
      </div>

      {/* Controls Dashboard */}
      <div className="simulator-layout">
        <div className="sim-controls-panel">
          <h3 className="sim-panel-title">🎛️ Hyperparameter Controls</h3>

          {/* Slider 1: Graph Alpha */}
          <div className="sim-control-group">
            <div className="slider-header">
              <label htmlFor="sliderGraph">Graph Weight (α - Collaborative)</label>
              <span className="slider-val" id="valGraphWeight">{Number(graphWeight).toFixed(2)}</span>
            </div>
            <input
              type="range"
              id="sliderGraph"
              min="0"
              max="2.5"
              step="0.1"
              value={graphWeight}
              onChange={(e) => onChangeGraphWeight(parseFloat(e.target.value))}
            />
            <span className="slider-hint">Controls long-term collaborative topology match</span>
          </div>

          {/* Slider 2: Sequential Beta */}
          <div className="sim-control-group">
            <div className="slider-header">
              <label htmlFor="sliderSeq">Sequential Weight (β - Session)</label>
              <span className="slider-val" id="valSeqWeight">{Number(seqWeight).toFixed(2)}</span>
            </div>
            <input
              type="range"
              id="sliderSeq"
              min="0"
              max="2.5"
              step="0.1"
              value={seqWeight}
              onChange={(e) => onChangeSeqWeight(parseFloat(e.target.value))}
            />
            <span className="slider-hint">Controls sensitivity to recent session interactions</span>
          </div>

          {/* Model Stream Selector */}
          <div className="sim-control-group">
            <label className="group-label">Component Stream</label>
            <div className="ablation-buttons">
              <button
                className={`btn-ablation ${mode === 'hybrid_v2' ? 'active' : ''}`}
                data-mode="hybrid_v2"
                onClick={() => onChangeMode('hybrid_v2')}
              >
                <strong>Hybrid Model (Full)</strong>
                <span>Graph + Sequential</span>
              </button>
              <button
                className={`btn-ablation ${mode === 'pure_lightgcn' ? 'active' : ''}`}
                data-mode="pure_lightgcn"
                onClick={() => onChangeMode('pure_lightgcn')}
              >
                <strong>Graph Stream Only</strong>
                <span>LightGCN Only</span>
              </button>
              <button
                className={`btn-ablation ${mode === 'pure_sasrec' ? 'active' : ''}`}
                data-mode="pure_sasrec"
                onClick={() => onChangeMode('pure_sasrec')}
              >
                <strong>Sequential Stream Only</strong>
                <span>SASRec Only</span>
              </button>
            </div>
          </div>

          {/* Test Persona for Simulator */}
          <div className="sim-control-group">
            <label className="group-label">Test Customer Profile</label>
            <select
              className="sim-select"
              id="simPersonaSelect"
              value={selectedUserId}
              onChange={(e) => onChangePersona(parseInt(e.target.value, 10))}
            >
              {personas.map(p => (
                <option key={p.user_id} value={p.user_id}>
                  User #{p.user_id}: {p.name} ({p.tagline})
                </option>
              ))}
            </select>
          </div>

          <button
            className="btn btn-primary btn-block"
            onClick={onRunSimulation}
            disabled={isLoading}
          >
            {isLoading ? 'Computing Simulation...' : 'Run Simulation & Re-Rank'}
          </button>
        </div>

        {/* Simulator Results Output */}
        <div className="sim-results-panel">
          <div className="sim-results-head">
            <div className="sim-status-title">
              <h3>Live Re-Ranked Recommendations</h3>
              <span className="sim-badge" id="simModeBadge">
                Mode: {modeDisplay}
              </span>
            </div>
          </div>

          <div className="products-grid sim-products-grid" id="simProductsGrid">
            {isLoading ? (
              <div style={{ gridColumn: '1 / -1', textAlign: 'center', padding: '40px', color: 'var(--text-dim)' }}>
                <div className="spinner"></div>
                <p style={{ marginTop: '12px' }}>Simulating inference...</p>
              </div>
            ) : simRecommendations.length === 0 ? (
              <p style={{ gridColumn: '1 / -1', textAlign: 'center', color: 'var(--text-dim)' }}>
                No simulated items to display.
              </p>
            ) : (
              simRecommendations.map(item => (
                <ProductCard
                  key={`sim-${item.id}`}
                  item={item}
                  isRecommendation={true}
                  onAddToCart={onAddToCart}
                  onInspect={onInspect}
                />
              ))
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
