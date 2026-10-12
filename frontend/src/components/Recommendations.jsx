import React from 'react';
import ProductCard from './ProductCard';

export default function Recommendations({
  currentPersona,
  sessionLength,
  recommendations,
  isLoading,
  onRefresh,
  onAddToCart,
  onInspect
}) {
  return (
    <section className="tab-pane active" id="pane-recommendations">
      {/* Recommendations Header & Context */}
      <div className="recommendations-hero">
        <div className="recs-profile-card">
          <img
            src={currentPersona?.avatar || 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=200&q=80'}
            alt="Avatar"
            className="recs-avatar"
            id="recProfileAvatar"
          />
          <div className="recs-profile-meta">
            <div className="profile-title-row">
              <h2 id="recProfileName">{currentPersona?.name || 'Customer Profile'}</h2>
              <span className="persona-tag" id="recProfileTag">
                {currentPersona?.tagline ? currentPersona.tagline.split('&')[0] : 'Hybrid Recipient'}
              </span>
            </div>
            <p className="profile-desc" id="recProfileDesc">
              {currentPersona?.description || 'Prefers tailored fashion recommendations.'}
            </p>
            <div className="profile-signals">
              <span className="signal-badge">
                <span className="dot-green"></span> Graph Collaborative Profile: <strong>Active</strong>
              </span>
              <span className="signal-badge">
                <span className="dot-blue"></span> Active Session Length: <strong id="recSessionLen">{sessionLength} items</strong>
              </span>
            </div>
          </div>
          <div className="recs-actions-side">
            <button className="btn btn-outline" onClick={onRefresh} disabled={isLoading}>
              <span className="refresh-icon">🔄</span> Recompute Hybrid Scores
            </button>
          </div>
        </div>
      </div>

      {/* Live Recommendation Explanation Banner */}
      <div className="xai-summary-card">
        <div className="xai-icon">💡</div>
        <div className="xai-text">
          <h3>How The Hybrid Model Calculates This Feed</h3>
          <p id="xaiFeedSummary">
            The hybrid engine balances collaborative graph affinity (customer co-purchase patterns via LightGCN) with sequential self-attention representations (your active browsing session via SASRec) through trained projection heads.
          </p>
        </div>
      </div>

      {/* Recommendations Products Grid */}
      <div className="products-grid recommendations-grid" id="recommendationsGrid">
        {isLoading ? (
          <div style={{ gridColumn: '1 / -1', textAlign: 'center', padding: '40px', color: 'var(--text-dim)' }}>
            <div className="spinner"></div>
            <p style={{ marginTop: '12px' }}>
              Computing personalized recommendations for {currentPersona?.name || 'User'}...
            </p>
          </div>
        ) : recommendations.length === 0 ? (
          <p style={{ gridColumn: '1 / -1', textAlign: 'center', padding: '40px', color: 'var(--text-dim)' }}>
            No recommendations generated. Try refreshing or browsing catalog items.
          </p>
        ) : (
          recommendations.map(item => (
            <ProductCard
              key={`rec-${item.id}`}
              item={item}
              isRecommendation={true}
              onAddToCart={onAddToCart}
              onInspect={onInspect}
            />
          ))
        )}
      </div>
    </section>
  );
}
