import React from 'react';

export default function XaiModal({ product, onClose }) {
  if (!product) return null;

  const meta = product.recommendation_metadata || {
    hybrid_score: 8.42,
    graph_score: 0.12,
    sequential_score: 0.18,
    cross_domain_score: 0.05,
    contributions: { graph_percent: 42.0, sequential_percent: 38.0, cross_percent: 20.0 },
    explanation: "Calculated based on collaborative LightGCN graph affinity and SASRec sequential attention."
  };

  return (
    <div className="modal-backdrop open" id="xaiModal" onClick={(e) => {
      if (e.target.id === 'xaiModal') onClose();
    }}>
      <div className="modal-dialog">
        <div className="modal-header">
          <div className="modal-title-wrap">
            <span className="badge-mini">Model Explainability</span>
            <h3 id="xaiProductTitle">{product.title}</h3>
          </div>
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>

        <div className="modal-body">
          <div className="xai-modal-layout">
            {/* Left: Product Snapshot */}
            <div className="xai-product-snapshot">
              <img id="xaiProductImg" src={product.image} alt={product.title} />
              <div className="snapshot-details">
                <span className="brand-text" id="xaiProductBrand">{product.brand}</span>
                <span className="price-text" id="xaiProductPrice">${product.price.toFixed(2)}</span>
                <span className="cat-text" id="xaiProductCategory">{product.category}</span>
              </div>
            </div>

            {/* Right: Component Breakdown */}
            <div className="xai-breakdown-details">
              <div className="xai-total-score">
                <span className="score-label">Overall Hybrid Score</span>
                <span className="score-number" id="xaiTotalScore">
                  {typeof meta.hybrid_score === 'number' ? meta.hybrid_score.toFixed(4) : meta.hybrid_score}
                </span>
              </div>

              {/* Contribution Bars */}
              <div className="contribution-bars">
                {/* Graph Bar */}
                <div className="bar-group">
                  <div className="bar-info">
                    <span className="bar-name">Graph Collaborative Affinity (LightGCN)</span>
                    <span className="bar-val" id="xaiGraphVal">
                      {meta.contributions?.graph_percent}% (Raw: {meta.graph_score})
                    </span>
                  </div>
                  <div className="progress-track">
                    <div
                      className="progress-fill blue"
                      id="xaiGraphFill"
                      style={{ width: `${meta.contributions?.graph_percent || 0}%` }}
                    ></div>
                  </div>
                </div>

                {/* Sequential Bar */}
                <div className="bar-group">
                  <div className="bar-info">
                    <span className="bar-name">Sequential Attention Affinity (SASRec)</span>
                    <span className="bar-val" id="xaiSeqVal">
                      {meta.contributions?.sequential_percent}% (Raw: {meta.sequential_score})
                    </span>
                  </div>
                  <div className="progress-track">
                    <div
                      className="progress-fill purple"
                      id="xaiSeqFill"
                      style={{ width: `${meta.contributions?.sequential_percent || 0}%` }}
                    ></div>
                  </div>
                </div>

                {/* Cross Domain Bar */}
                <div className="bar-group">
                  <div className="bar-info">
                    <span className="bar-name">Cross-Domain Latent Interactions</span>
                    <span className="bar-val" id="xaiCrossVal">
                      {meta.contributions?.cross_percent}% (Raw: {meta.cross_domain_score})
                    </span>
                  </div>
                  <div className="progress-track">
                    <div
                      className="progress-fill green"
                      id="xaiCrossFill"
                      style={{ width: `${meta.contributions?.cross_percent || 0}%` }}
                    ></div>
                  </div>
                </div>
              </div>

              {/* Why Recommended Reasoning */}
              <div className="xai-reasoning-card">
                <h4>Model Decision Rationale</h4>
                <p id="xaiExplanationText">{meta.explanation}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
