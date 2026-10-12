import React from 'react';

export default function CartDrawer({
  isOpen,
  onClose,
  cart,
  sessionLength,
  onRemoveItem,
  onClearCart,
  onCheckout
}) {
  const total = cart.reduce((acc, item) => acc + (item.price || 0), 0);

  return (
    <>
      <div
        className={`drawer-backdrop ${isOpen ? 'open' : ''}`}
        id="cartDrawerBackdrop"
        onClick={onClose}
      />
      <aside className={`cart-drawer ${isOpen ? 'open' : ''}`} id="cartDrawer">
        <div className="drawer-header">
          <h3>Your Bag & Session Sequence</h3>
          <button className="drawer-close" onClick={onClose}>✕</button>
        </div>

        {/* Active Session Tracker Inside Drawer */}
        <div className="drawer-session-meta">
          <div className="session-stat-row">
            <span>Session Sequence Items:</span>
            <strong id="drawerSeqCount">{sessionLength} items</strong>
          </div>
          <p className="drawer-session-desc">
            Every added or viewed product immediately informs the SASRec sequential attention mechanism.
          </p>
        </div>

        <div className="drawer-items" id="cartItemsList">
          {cart.length === 0 ? (
            <div className="empty-cart-state">
              <span className="empty-icon">🛒</span>
              <p>Your bag is empty.</p>
              <button className="btn btn-sm btn-outline" onClick={onClose}>
                Start Exploring
              </button>
            </div>
          ) : (
            cart.map((item, idx) => (
              <div className="cart-item-row" key={`${item.id}-${idx}`}>
                <img src={item.image} alt={item.title} />
                <div className="cart-item-info">
                  <span className="cart-item-title">{item.title}</span>
                  <span className="cart-item-price">${item.price.toFixed(2)}</span>
                  <button
                    className="cart-item-remove"
                    onClick={() => onRemoveItem(idx)}
                  >
                    Remove
                  </button>
                </div>
              </div>
            ))
          )}
        </div>

        <div className="drawer-footer">
          <div className="drawer-total-row">
            <span>Estimated Total</span>
            <strong id="cartTotalPrice">${total.toFixed(2)}</strong>
          </div>
          <button
            className="btn btn-primary btn-block"
            onClick={onCheckout}
            disabled={cart.length === 0}
          >
            Checkout Demo
          </button>
          <button
            className="btn btn-outline btn-block"
            onClick={onClearCart}
            disabled={cart.length === 0}
          >
            Clear Bag
          </button>
        </div>
      </aside>
    </>
  );
}
