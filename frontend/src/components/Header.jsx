import React, { useState, useEffect, useRef } from 'react';

export default function Header({
  activeTab,
  onTabChange,
  personas,
  currentPersona,
  onSelectPersona,
  cartCount,
  onOpenCart
}) {
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const dropdownRef = useRef(null);

  useEffect(() => {
    function handleClickOutside(event) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setDropdownOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <header className="site-header" id="mainHeader">
      <div className="header-inner">
        {/* Logo - Name changed from AURA HYBRID to only HYBRID */}
        <div className="brand-logo" onClick={() => onTabChange('storefront')} style={{ cursor: 'pointer' }}>
          <div className="logo-gem">
            <span className="gem-core"></span>
          </div>
          <div className="brand-text">
            <span className="brand-name">HYBRID</span>
            <span className="brand-sub">Recommendation Platform</span>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="nav-tabs" id="navTabs">
          <button
            className={`nav-tab ${activeTab === 'storefront' ? 'active' : ''}`}
            data-tab="storefront"
            onClick={() => onTabChange('storefront')}
          >
            <span className="tab-icon">🛍️</span>
            <span className="tab-label">Storefront</span>
          </button>
          <button
            className={`nav-tab ${activeTab === 'recommendations' ? 'active' : ''}`}
            data-tab="recommendations"
            onClick={() => onTabChange('recommendations')}
          >
            <span className="tab-icon">✨</span>
            <span className="tab-label">Recommended For You</span>
          </button>
          <button
            className={`nav-tab ${activeTab === 'simulator' ? 'active' : ''}`}
            data-tab="simulator"
            onClick={() => onTabChange('simulator')}
          >
            <span className="tab-icon">🎛️</span>
            <span className="tab-label">Model Tuning Lab</span>
          </button>
        </nav>

        {/* User Persona & Cart Action Bar */}
        <div className="header-actions">
          {/* Persona Switcher */}
          <div className="persona-selector-wrap" ref={dropdownRef}>
            <div
              className="persona-pill"
              onClick={() => setDropdownOpen(prev => !prev)}
              id="currentPersonaBtn"
              title="Change active customer persona"
            >
              <img
                src={currentPersona?.avatar || 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=100&q=80'}
                alt="Avatar"
                className="persona-avatar"
                id="headerAvatar"
              />
              <div className="persona-info">
                <span className="persona-label">Persona</span>
                <span className="persona-name" id="headerPersonaName">
                  {currentPersona ? currentPersona.name.split(' ')[0] + '.' : 'Guest'}
                </span>
              </div>
              <span className="chevron-down" style={{ transform: dropdownOpen ? 'rotate(180deg)' : 'none', transition: '0.2s ease' }}>▼</span>
            </div>

            {/* Persona Dropdown Menu */}
            <div className={`persona-dropdown ${dropdownOpen ? 'open' : ''}`} id="personaDropdown">
              <div className="dropdown-header">
                <span>Select Active Test Persona</span>
                <span className="sub">Simulate different customer profiles</span>
              </div>
              <div className="dropdown-list" id="personaList">
                {personas.map(p => (
                  <div
                    key={p.user_id}
                    className={`dropdown-item ${currentPersona?.user_id === p.user_id ? 'active' : ''}`}
                    onClick={() => {
                      onSelectPersona(p);
                      setDropdownOpen(false);
                    }}
                  >
                    <img src={p.avatar} alt={p.name} />
                    <div className="dropdown-item-meta">
                      <div className="dropdown-item-name">{p.name}</div>
                      <div className="dropdown-item-tag">{p.tagline}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Cart Button */}
          <button className="cart-btn" onClick={onOpenCart} id="cartBtn" title="View Bag & Session Sequence">
            <span className="cart-icon">🛒</span>
            <span className="cart-badge" id="cartCount">{cartCount}</span>
          </button>
        </div>
      </div>
    </header>
  );
}
