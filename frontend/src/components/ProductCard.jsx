import React from 'react';

export default function ProductCard({
  item,
  isRecommendation = false,
  onAddToCart,
  onInspect
}) {
  const meta = item.recommendation_metadata;

  return (
    <div className="product-card" id={`product-${item.id}`}>
      <div className="product-media">
        <img
          src={item.image}
          alt={item.title}
          className="product-img"
          loading="lazy"
          onError={(e) => {
            e.currentTarget.onerror = null;
            e.currentTarget.src = 'https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?auto=format&fit=crop&w=800&q=80';
          }}
        />

        {isRecommendation && (
          <div className="product-badge-stack">
            <span className="badge-hybrid">
              Rank #{meta?.rank || 1}
            </span>
          </div>
        )}

        <div className="product-overlay-actions">
          <button
            className="btn btn-primary btn-sm btn-block"
            onClick={() => onAddToCart(item)}
          >
            Add to Bag & Track Session
          </button>
          <button
            className="btn btn-outline btn-sm btn-block"
            onClick={() => onInspect(item)}
          >
            {isRecommendation ? 'Inspect XAI Reasoning' : 'Interact with Item'}
          </button>
        </div>
      </div>

      <div className="product-info">
        <div className="product-brand-row">
          <span className="product-brand">{item.brand}</span>
          <span className="product-rating">★ {item.rating}</span>
        </div>

        <h3 className="product-title">{item.title}</h3>
        <p className="product-desc">{item.description}</p>

        {isRecommendation && meta && (
          <div
            className="rec-reasoning-pill"
            onClick={() => onInspect(item)}
            style={{ cursor: 'pointer' }}
            title="Click to view deep XAI breakdown"
          >
            <span className="rec-score-chip">Score: {meta.hybrid_score}</span>
            <span>{meta.explanation}</span>
          </div>
        )}

        <div className="product-footer">
          <span className="product-price">${item.price.toFixed(2)}</span>
          <button
            className="btn btn-outline btn-sm"
            onClick={() => onAddToCart(item)}
          >
            + Bag
          </button>
        </div>
      </div>
    </div>
  );
}
