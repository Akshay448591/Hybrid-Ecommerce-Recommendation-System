/**
 * AURA HYBRID - Frontend Application Engine
 * Connects with FastAPI Hybrid V2 Engine for Live E-Commerce & Major Project Defense.
 */

class AuraHybridApp {
  constructor() {
    // Determine API Base URL: auto-detects local vs production
    this.apiBase = this.detectApiBase();

    // State
    this.activeTab = 'storefront';
    this.personas = [];
    this.currentPersona = null;
    this.catalog = [];
    this.filteredCatalog = [];
    this.recommendations = [];
    this.simRecommendations = [];
    this.sessionSequence = [];
    this.cart = [];
    
    // Filters & Simulation Params
    this.currentCategory = 'all';
    this.searchQuery = '';
    this.currentSort = 'popular';
    
    this.simGraphWeight = 1.0;
    this.simSeqWeight = 1.0;
    this.simMode = 'hybrid_v2';

    // Initialize
    this.init();
  }

  detectApiBase() {
    // If hosted on Render as combined or separate, support auto detection or query param ?api=...
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('api')) {
      return urlParams.get('api');
    }
    if (window.location.port === '8000') {
      return '';
    }
    return 'http://localhost:8000';
  }

  async init() {
    try {
      await this.checkApiHealth();
      await this.loadPersonas();
      await this.loadCatalog();
      await this.refreshRecommendations();
      this.populateSimulatorControls();
    } catch (err) {
      console.warn("Initial load notice:", err);
      this.showToast("Connected in offline preview mode (waiting for backend)", "warning");
    }
  }

  // ============================================================
  // API HEALTH & PERSONAS
  // ============================================================

  async checkApiHealth() {
    try {
      const res = await fetch(`${this.apiBase}/api/health`);
      if (res.ok) {
        const data = await res.json();
        const el = document.getElementById('apiStatus');
        if (el) {
          el.innerHTML = `
            <span class="status-dot"></span>
            <span class="status-text">Engine Ready</span>
          `;
        }
      }
    } catch (e) {
      const el = document.getElementById('apiStatus');
      if (el) {
        el.innerHTML = `
          <span class="status-dot" style="background:#f43f5e;box-shadow:0 0 8px #f43f5e"></span>
          <span class="status-text" style="color:#f43f5e">Connecting...</span>
        `;
      }
    }
  }

  async loadPersonas() {
    try {
      const res = await fetch(`${this.apiBase}/api/personas`);
      if (!res.ok) throw new Error("Could not fetch personas");
      this.personas = await res.json();

      // Set default persona (Sophia Martinez, id: 0)
      this.currentPersona = this.personas.find(p => p.user_id === 0) || this.personas[0];
      this.updatePersonaUI();
      this.renderPersonaDropdown();
    } catch (e) {
      console.error(e);
    }
  }

  renderPersonaDropdown() {
    const list = document.getElementById('personaList');
    if (!list) return;

    list.innerHTML = this.personas.map(p => `
      <div class="dropdown-item ${this.currentPersona && this.currentPersona.user_id === p.user_id ? 'active' : ''}" 
           onclick="app.selectPersona(${p.user_id})">
        <img src="${p.avatar}" alt="${p.name}">
        <div class="dropdown-item-meta">
          <div class="dropdown-item-name">${p.name}</div>
          <div class="dropdown-item-tag">${p.tagline}</div>
        </div>
      </div>
    `).join('');
  }

  selectPersona(userId) {
    const found = this.personas.find(p => p.user_id === userId);
    if (!found) return;

    this.currentPersona = found;
    // Set seed session from persona history if user desires
    this.sessionSequence = [...(found.history_item_ids || [])];

    this.updatePersonaUI();
    this.renderPersonaDropdown();
    this.updateSessionTrackerUI();
    this.togglePersonaDropdown(false);
    this.refreshRecommendations();
    this.showToast(`Switched active profile to ${found.name}`);
  }

  updatePersonaUI() {
    if (!this.currentPersona) return;
    const p = this.currentPersona;

    // Header Pill
    const avatar = document.getElementById('headerAvatar');
    const name = document.getElementById('headerPersonaName');
    if (avatar) avatar.src = p.avatar;
    if (name) name.textContent = p.name.split(' ')[0] + '.';

    // "For You" hero profile
    const recAvatar = document.getElementById('recProfileAvatar');
    const recName = document.getElementById('recProfileName');
    const recTag = document.getElementById('recProfileTag');
    const recDesc = document.getElementById('recProfileDesc');
    const recSessionLen = document.getElementById('recSessionLen');

    if (recAvatar) recAvatar.src = p.avatar;
    if (recName) recName.textContent = p.name;
    if (recTag) recTag.textContent = p.tagline.split('&')[0];
    if (recDesc) recDesc.textContent = p.description;
    if (recSessionLen) recSessionLen.textContent = `${this.sessionSequence.length} items`;
  }

  togglePersonaDropdown(forceState) {
    const dd = document.getElementById('personaDropdown');
    if (!dd) return;
    if (typeof forceState === 'boolean') {
      dd.classList.toggle('open', forceState);
    } else {
      dd.classList.toggle('open');
    }
  }

  // ============================================================
  // CATALOG & STOREFRONT
  // ============================================================

  async loadCatalog() {
    try {
      const res = await fetch(`${this.apiBase}/api/products?limit=60`);
      if (!res.ok) throw new Error("Could not fetch products");
      const data = await res.json();
      this.catalog = data.items;
      this.filteredCatalog = [...this.catalog];
      this.renderCatalog();
    } catch (e) {
      console.error(e);
      document.getElementById('productsGrid').innerHTML = `
        <div style="grid-column: 1/-1; text-align:center; padding: 60px 20px;">
          <p style="color: var(--accent-rose)">Unable to connect to recommendation backend service.</p>
          <p style="color: var(--text-dim); font-size: 13px; margin-top: 8px;">Ensure backend is running at ${this.apiBase}</p>
        </div>
      `;
    }
  }

  renderCatalog() {
    const grid = document.getElementById('productsGrid');
    if (!grid) return;

    if (this.filteredCatalog.length === 0) {
      grid.innerHTML = `
        <div style="grid-column: 1/-1; text-align: center; padding: 60px 20px; color: var(--text-dim)">
          <p>No products match your current filters.</p>
        </div>
      `;
      return;
    }

    grid.innerHTML = this.filteredCatalog.map(item => this.createProductCardHtml(item, false)).join('');
  }

  createProductCardHtml(item, isRecommendation = false) {
    const meta = item.recommendation_metadata;
    const badgeText = isRecommendation ? `Rank #${meta?.rank || 1}` : 'Hybrid Candidate';

    return `
      <div class="product-card" id="product-${item.id}">
        <div class="product-media">
          <img src="${item.image}" alt="${item.title}" class="product-img" loading="lazy">
          
          <div class="product-badge-stack">
            <span class="${isRecommendation ? 'badge-hybrid' : 'badge-tag'}">${badgeText}</span>
            ${isRecommendation && meta ? `<span class="badge-graph">${meta.contributions.graph_percent}% Graph</span>` : ''}
          </div>

          <div class="product-overlay-actions">
            <button class="btn btn-primary btn-sm btn-block" onclick="app.addToCart(${item.id})">
              Add to Bag & Track Session
            </button>
            <button class="btn btn-outline btn-sm btn-block" onclick="app.inspectProduct(${item.id})">
              ${isRecommendation ? 'Inspect XAI Reasoning' : 'Interact with Item'}
            </button>
          </div>
        </div>

        <div class="product-info">
          <div class="product-brand-row">
            <span class="product-brand">${item.brand}</span>
            <span class="product-rating">★ ${item.rating}</span>
          </div>

          <h3 class="product-title">${item.title}</h3>
          <p class="product-desc">${item.description}</p>

          ${isRecommendation && meta ? `
            <div class="rec-reasoning-pill" onclick="app.openXaiModal(${item.id})" style="cursor: pointer" title="Click to view deep XAI breakdown">
              <span class="rec-score-chip">Score: ${meta.hybrid_score}</span>
              <span>${meta.explanation}</span>
            </div>
          ` : ''}

          <div class="product-footer">
            <span class="product-price">$${item.price.toFixed(2)}</span>
            <button class="btn btn-outline btn-sm" onclick="app.addToCart(${item.id})">
              + Bag
            </button>
          </div>
        </div>
      </div>
    `;
  }

  filterCategory(cat) {
    this.currentCategory = cat;
    document.querySelectorAll('#categoryPills .pill').forEach(p => {
      p.classList.toggle('active', p.getAttribute('data-cat') === cat);
    });
    this.applyFilters();
  }

  handleSearch(query) {
    this.searchQuery = query.toLowerCase().trim();
    this.applyFilters();
  }

  handleSort(sortVal) {
    this.currentSort = sortVal;
    this.applyFilters();
  }

  applyFilters() {
    let items = [...this.catalog];

    if (this.currentCategory && this.currentCategory !== 'all') {
      items = items.filter(x => x.category.toLowerCase() === this.currentCategory.toLowerCase());
    }

    if (this.searchQuery) {
      items = items.filter(x => 
        x.title.toLowerCase().includes(this.searchQuery) ||
        x.brand.toLowerCase().includes(this.searchQuery) ||
        x.description.toLowerCase().includes(this.searchQuery)
      );
    }

    if (this.currentSort === 'price_asc') {
      items.sort((a, b) => a.price - b.price);
    } else if (this.currentSort === 'price_desc') {
      items.sort((a, b) => b.price - a.price);
    } else if (this.currentSort === 'rating') {
      items.sort((a, b) => b.rating - a.rating);
    } else {
      items.sort((a, b) => b.reviews_count - a.reviews_count);
    }

    this.filteredCatalog = items;
    this.renderCatalog();
  }

  // ============================================================
  // SESSION TRACKER & SASREC INTEGRATION
  // ============================================================

  recordInteraction(itemId) {
    // Append to sequence if not immediately duplicate
    if (this.sessionSequence.length === 0 || this.sessionSequence[this.sessionSequence.length - 1] !== itemId) {
      this.sessionSequence.push(itemId);
      if (this.sessionSequence.length > 20) {
        this.sessionSequence.shift(); // Keep last 20
      }
      this.updateSessionTrackerUI();
      // Auto refresh recommendation feed
      this.refreshRecommendations();
    }
  }

  updateSessionTrackerUI() {
    const summary = document.getElementById('sessionItemsSummary');
    const chips = document.getElementById('sessionChips');
    const drawerCount = document.getElementById('drawerSeqCount');
    const recSessionLen = document.getElementById('recSessionLen');

    const count = this.sessionSequence.length;
    if (summary) summary.textContent = `${count} items in SASRec attention window`;
    if (drawerCount) drawerCount.textContent = `${count} items`;
    if (recSessionLen) recSessionLen.textContent = `${count} items`;

    if (!chips) return;

    if (count === 0) {
      chips.innerHTML = `<span class="chip-empty">Browse items to feed the sequential transformer in real-time...</span>`;
      return;
    }

    chips.innerHTML = this.sessionSequence.map((id, idx) => {
      const prod = this.catalog.find(p => p.id === id);
      const name = prod ? prod.title.split(' ')[0] : `Item #${id}`;
      return `
        <span class="session-chip">
          <span>#${idx + 1}: ${name}</span>
          <span style="cursor:pointer; opacity:0.7" onclick="app.removeFromSession(${idx})">×</span>
        </span>
      `;
    }).join('');
  }

  removeFromSession(idx) {
    this.sessionSequence.splice(idx, 1);
    this.updateSessionTrackerUI();
    this.refreshRecommendations();
  }

  clearSession() {
    this.sessionSequence = [];
    this.updateSessionTrackerUI();
    this.refreshRecommendations();
    this.showToast("Sequential session reset to neutral state");
  }

  // ============================================================
  // "FOR YOU" RECOMMENDATIONS
  // ============================================================

  async refreshRecommendations() {
    const userId = this.currentPersona ? this.currentPersona.user_id : 0;
    const grid = document.getElementById('recommendationsGrid');
    if (grid) {
      grid.innerHTML = `
        <div style="grid-column: 1/-1; text-align: center; padding: 40px; color: var(--text-dim)">
          <div class="spinner"></div>
          <p style="margin-top: 12px">Computing personalized recommendations for ${this.currentPersona ? this.currentPersona.name : 'User'}...</p>
        </div>
      `;
    }

    try {
      const res = await fetch(`${this.apiBase}/api/recommendations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          session_sequence: this.sessionSequence,
          top_k: 12,
          mode: 'hybrid_v2'
        })
      });

      if (!res.ok) throw new Error("Recommendation endpoint failed");
      const data = await res.json();
      this.recommendations = data.recommendations;
      this.renderRecommendations();
    } catch (e) {
      console.error(e);
      if (grid) {
        grid.innerHTML = `
          <div style="grid-column: 1/-1; text-align: center; padding: 40px; color: var(--accent-rose)">
            <p>Could not compute recommendations. Ensure the FastAPI backend is running.</p>
          </div>
        `;
      }
    }
  }

  renderRecommendations() {
    const grid = document.getElementById('recommendationsGrid');
    if (!grid) return;

    if (this.recommendations.length === 0) {
      grid.innerHTML = `<p style="grid-column: 1/-1; text-align: center;">No recommendations generated.</p>`;
      return;
    }

    grid.innerHTML = this.recommendations.map(item => this.createProductCardHtml(item, true)).join('');
  }

  // ============================================================
  // EXPLAINABLE AI (XAI) MODAL
  // ============================================================

  openXaiModal(productId) {
    const item = this.recommendations.find(p => p.id === productId) || 
                 this.catalog.find(p => p.id === productId);
    if (!item) return;

    const meta = item.recommendation_metadata || {
      hybrid_score: 8.42,
      graph_score: 0.12,
      sequential_score: 0.18,
      cross_domain_score: 0.05,
      contributions: { graph_percent: 42.0, sequential_percent: 38.0, cross_percent: 20.0 },
      explanation: "Calculated based on collaborative LightGCN graph affinity and SASRec sequential attention."
    };

    // Populate modal
    document.getElementById('xaiProductTitle').textContent = item.title;
    document.getElementById('xaiProductBrand').textContent = item.brand;
    document.getElementById('xaiProductPrice').textContent = `$${item.price.toFixed(2)}`;
    document.getElementById('xaiProductCategory').textContent = item.category;
    document.getElementById('xaiProductImg').src = item.image;

    document.getElementById('xaiTotalScore').textContent = meta.hybrid_score.toFixed(4);

    // Bars
    document.getElementById('xaiGraphVal').textContent = `${meta.contributions.graph_percent}% (Raw: ${meta.graph_score})`;
    document.getElementById('xaiGraphFill').style.width = `${meta.contributions.graph_percent}%`;

    document.getElementById('xaiSeqVal').textContent = `${meta.contributions.sequential_percent}% (Raw: ${meta.sequential_score})`;
    document.getElementById('xaiSeqFill').style.width = `${meta.contributions.sequential_percent}%`;

    document.getElementById('xaiCrossVal').textContent = `${meta.contributions.cross_percent}% (Raw: ${meta.cross_domain_score})`;
    document.getElementById('xaiCrossFill').style.width = `${meta.contributions.cross_percent}%`;

    document.getElementById('xaiExplanationText').textContent = meta.explanation;

    document.getElementById('xaiModal').classList.add('open');
  }

  inspectProduct(productId) {
    this.recordInteraction(productId);
    this.openXaiModal(productId);
  }

  closeModal(modalId) {
    const m = document.getElementById(modalId);
    if (m) m.classList.remove('open');
  }

  // ============================================================
  // SHOPPING CART & SESSIONS
  // ============================================================

  addToCart(productId) {
    const product = this.catalog.find(p => p.id === productId);
    if (!product) return;

    this.cart.push(product);
    this.recordInteraction(productId); // Feed into sequential transformer!

    this.updateCartUI();
    this.showToast(`Added ${product.title} to your bag`);
  }

  openCart() {
    document.getElementById('cartDrawer').classList.add('open');
    document.getElementById('cartDrawerBackdrop').classList.add('open');
    this.updateCartUI();
  }

  closeCart() {
    document.getElementById('cartDrawer').classList.remove('open');
    document.getElementById('cartDrawerBackdrop').classList.remove('open');
  }

  updateCartUI() {
    const countBadge = document.getElementById('cartCount');
    if (countBadge) countBadge.textContent = this.cart.length;

    const list = document.getElementById('cartItemsList');
    const totalEl = document.getElementById('cartTotalPrice');
    if (!list) return;

    if (this.cart.length === 0) {
      list.innerHTML = `
        <div class="empty-cart-state">
          <span class="empty-icon">🛒</span>
          <p>Your bag is empty.</p>
          <button class="btn btn-sm btn-outline" onclick="app.closeCart()">Start Exploring</button>
        </div>
      `;
      if (totalEl) totalEl.textContent = '$0.00';
      return;
    }

    let total = 0;
    list.innerHTML = this.cart.map((item, idx) => {
      total += item.price;
      return `
        <div class="cart-item-row">
          <img src="${item.image}" alt="${item.title}">
          <div class="cart-item-info">
            <span class="cart-item-title">${item.title}</span>
            <span class="cart-item-price">$${item.price.toFixed(2)}</span>
            <button class="cart-item-remove" onclick="app.removeFromCart(${idx})">Remove</button>
          </div>
        </div>
      `;
    }).join('');

    if (totalEl) totalEl.textContent = `$${total.toFixed(2)}`;
  }

  removeFromCart(idx) {
    this.cart.splice(idx, 1);
    this.updateCartUI();
  }

  clearCart() {
    this.cart = [];
    this.updateCartUI();
    this.showToast("Bag emptied");
  }

  checkoutDemo() {
    if (this.cart.length === 0) return;
    this.showToast("Checkout simulated! Active session captured by SASRec.");
    this.clearCart();
    this.closeCart();
  }

  // ============================================================
  // INTERACTIVE MODEL LAB / SIMULATOR
  // ============================================================

  populateSimulatorControls() {
    const select = document.getElementById('simPersonaSelect');
    if (!select || !this.personas.length) return;

    select.innerHTML = this.personas.map(p => `
      <option value="${p.user_id}" ${this.currentPersona && this.currentPersona.user_id === p.user_id ? 'selected' : ''}>
        User #${p.user_id}: ${p.name} (${p.tagline})
      </option>
    `).join('');

    this.runSimulation();
  }

  updateSimWeight(type, val) {
    const num = parseFloat(val);
    if (type === 'graph') {
      this.simGraphWeight = num;
      document.getElementById('valGraphWeight').textContent = num.toFixed(2);
    } else {
      this.simSeqWeight = num;
      document.getElementById('valSeqWeight').textContent = num.toFixed(2);
    }
  }

  setAblationMode(mode) {
    this.simMode = mode;
    document.querySelectorAll('.btn-ablation').forEach(b => {
      b.classList.toggle('active', b.getAttribute('data-mode') === mode);
    });

    const badge = document.getElementById('simModeBadge');
    if (badge) {
      badge.textContent = `Mode: ${mode.replace('_', ' ').toUpperCase()}`;
    }
    this.runSimulation();
  }

  changePersona(userId) {
    this.selectPersona(userId);
    this.runSimulation();
  }

  async runSimulation() {
    const grid = document.getElementById('simProductsGrid');
    if (grid) {
      grid.innerHTML = `<div style="grid-column: 1/-1; text-align: center; padding: 40px; color: var(--text-dim)"><div class="spinner"></div><p>Simulating inference...</p></div>`;
    }

    try {
      const userId = this.currentPersona ? this.currentPersona.user_id : 0;
      const res = await fetch(`${this.apiBase}/api/demonstration/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          session_sequence: this.sessionSequence,
          top_k: 8,
          graph_weight: this.simGraphWeight,
          seq_weight: this.simSeqWeight,
          mode: this.simMode
        })
      });

      if (!res.ok) throw new Error("Simulation request failed");
      const data = await res.json();
      this.simRecommendations = data.recommendations;

      if (grid) {
        grid.innerHTML = this.simRecommendations.map(item => this.createProductCardHtml(item, true)).join('');
      }

      const summary = document.getElementById('simResultSummary');
      if (summary) {
        summary.textContent = `α=${this.simGraphWeight.toFixed(2)} (Graph), β=${this.simSeqWeight.toFixed(2)} (Seq) • ${this.simMode}`;
      }
    } catch (e) {
      console.error(e);
      if (grid) grid.innerHTML = `<p style="grid-column: 1/-1; text-align: center; color: var(--accent-rose)">Simulation failed.</p>`;
    }
  }

  // ============================================================
  // TAB NAVIGATION
  // ============================================================

  switchTab(tabId) {
    this.activeTab = tabId;

    // Nav button state
    document.querySelectorAll('#navTabs .nav-tab').forEach(t => {
      t.classList.toggle('active', t.getAttribute('data-tab') === tabId);
    });

    // Content pane state
    document.querySelectorAll('.main-content .tab-pane').forEach(p => {
      p.classList.remove('active');
    });

    const target = document.getElementById(`pane-${tabId}`);
    if (target) {
      target.classList.add('active');
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    if (tabId === 'recommendations') {
      this.refreshRecommendations();
    } else if (tabId === 'simulator') {
      this.runSimulation();
    }
  }

  // ============================================================
  // TOAST NOTIFICATIONS
  // ============================================================

  showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.innerHTML = `
      <span>${type === 'warning' ? '⚠️' : '✨'}</span>
      <span>${message}</span>
    `;

    container.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      toast.style.transition = '0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, 3200);
  }
}

// Global instance
const app = new AuraHybridApp();
window.app = app;
