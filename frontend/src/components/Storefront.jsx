import React from 'react';
import ProductCard from './ProductCard';

const CATEGORIES = [
  { id: 'all', label: 'All Products' },
  { id: 'Outerwear & Coats', label: 'Outerwear' },
  { id: 'Tops & Knitwear', label: 'Knitwear & Tops' },
  { id: 'Bottoms & Denim', label: 'Denim & Bottoms' },
  { id: 'Dresses & Jumpsuits', label: 'Dresses' },
  { id: 'Footwear & Shoes', label: 'Footwear' },
  { id: 'Accessories & Bags', label: 'Accessories' }
];

export default function Storefront({
  catalog,
  selectedCategory,
  onSelectCategory,
  isLoading,
  onAddToCart,
  onInspect
}) {
  return (
    <section className="tab-pane active" id="pane-storefront">
      {/* Controls & Filters Bar (Category pills only) */}
      <div className="catalog-controls">
        {/* Category Filters */}
        <div className="filter-pills" id="categoryPills">
          {CATEGORIES.map(cat => (
            <button
              key={cat.id}
              className={`pill ${selectedCategory === cat.id ? 'active' : ''}`}
              data-cat={cat.id}
              onClick={() => onSelectCategory(cat.id)}
            >
              {cat.label}
            </button>
          ))}
        </div>
      </div>

      {/* Product Catalog Grid */}
      <div className="products-grid" id="productsGrid">
        {isLoading ? (
          <div className="loading-state">
            <div className="spinner"></div>
            <p>Loading catalogue...</p>
          </div>
        ) : catalog.length === 0 ? (
          <div style={{ gridColumn: '1 / -1', textAlign: 'center', padding: '60px 20px', color: 'var(--text-dim)' }}>
            <p>No products match your current filters.</p>
          </div>
        ) : (
          catalog.map(item => (
            <ProductCard
              key={item.id}
              item={item}
              isRecommendation={false}
              onAddToCart={onAddToCart}
              onInspect={onInspect}
            />
          ))
        )}
      </div>
    </section>
  );
}
