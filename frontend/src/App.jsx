import React, { useState, useEffect, useCallback, useMemo } from 'react';
import Header from './components/Header';
import SessionTracker from './components/SessionTracker';
import Storefront from './components/Storefront';
import Recommendations from './components/Recommendations';
import ModelTuningLab from './components/ModelTuningLab';
import XaiModal from './components/XaiModal';
import CartDrawer from './components/CartDrawer';
import Toast from './components/Toast';
import {
  checkApiHealth,
  fetchPersonas,
  fetchProducts,
  fetchRecommendations,
  simulateRecommendations
} from './services/api';

export default function App() {
  // Navigation
  const [activeTab, setActiveTab] = useState('storefront');

  // Personas & Customer State
  const [personas, setPersonas] = useState([]);
  const [currentPersona, setCurrentPersona] = useState(null);

  // Catalog State
  const [catalog, setCatalog] = useState([]);
  const [loadingCatalog, setLoadingCatalog] = useState(true);
  const [selectedCategory, setSelectedCategory] = useState('all');
  // Recommendations State
  const [recommendations, setRecommendations] = useState([]);
  const [loadingRecs, setLoadingRecs] = useState(false);

  // Session & SASRec State
  const [sessionSequence, setSessionSequence] = useState([]);

  // Cart State
  const [cart, setCart] = useState([]);
  const [isCartOpen, setIsCartOpen] = useState(false);

  // Simulator / Model Tuning State
  const [simGraphWeight, setSimGraphWeight] = useState(1.0);
  const [simSeqWeight, setSimSeqWeight] = useState(1.0);
  const [simMode, setSimMode] = useState('hybrid_v2');
  const [simUserId, setSimUserId] = useState(0);
  const [simRecommendations, setSimRecommendations] = useState([]);
  const [loadingSim, setLoadingSim] = useState(false);

  // Modal & Notifications
  const [activeXaiProduct, setActiveXaiProduct] = useState(null);
  const [toasts, setToasts] = useState([]);

  // Toast Helper
  const showToast = useCallback((message, type = 'info') => {
    const id = Date.now() + Math.random();
    setToasts(prev => [...prev, { id, message, type }]);
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id));
    }, 3200);
  }, []);

  // Compute recommendations
  const loadRecommendations = useCallback(async (userId, sequence) => {
    setLoadingRecs(true);
    try {
      const data = await fetchRecommendations({
        userId: userId !== undefined ? userId : (currentPersona?.user_id ?? 0),
        sessionSequence: sequence !== undefined ? sequence : sessionSequence,
        topK: 12,
        mode: 'hybrid_v2'
      });
      setRecommendations(data.recommendations || []);
    } catch (err) {
      console.warn('Recommendation computation notice:', err);
    } finally {
      setLoadingRecs(false);
    }
  }, [currentPersona, sessionSequence]);

  // Run simulation
  const runSimulation = useCallback(async (opts = {}) => {
    setLoadingSim(true);
    const uId = opts.userId !== undefined ? opts.userId : simUserId;
    const gWeight = opts.graphWeight !== undefined ? opts.graphWeight : simGraphWeight;
    const sWeight = opts.seqWeight !== undefined ? opts.seqWeight : simSeqWeight;
    const sMode = opts.mode !== undefined ? opts.mode : simMode;

    try {
      const data = await simulateRecommendations({
        userId: uId,
        sessionSequence: sessionSequence,
        topK: 12,
        graphWeight: gWeight,
        seqWeight: sWeight,
        mode: sMode
      });
      setSimRecommendations(data.recommendations || []);
    } catch (err) {
      console.warn('Simulation notice:', err);
    } finally {
      setLoadingSim(false);
    }
  }, [simUserId, simGraphWeight, simSeqWeight, simMode, sessionSequence]);

  // Initial Data Fetch
  useEffect(() => {
    async function init() {
      try {
        await checkApiHealth();
      } catch {
        showToast('Running in local mode - connect backend at http://localhost:8000', 'warning');
      }

      try {
        const personasData = await fetchPersonas();
        setPersonas(personasData);
        const defaultPersona = personasData.find(p => p.user_id === 0) || personasData[0];
        if (defaultPersona) {
          setCurrentPersona(defaultPersona);
          setSimUserId(defaultPersona.user_id);
          const initialSequence = [...(defaultPersona.history_item_ids || [])];
          setSessionSequence(initialSequence);
          loadRecommendations(defaultPersona.user_id, initialSequence);
        }
      } catch (err) {
        console.error('Failed to load personas:', err);
      }

      try {
        setLoadingCatalog(true);
        const catalogData = await fetchProducts({ limit: 60 });
        setCatalog(catalogData.items || []);
      } catch (err) {
        console.error('Failed to load catalog:', err);
      } finally {
        setLoadingCatalog(false);
      }
    }

    init();
  }, []);

  // Filter catalog items by category
  const filteredCatalog = useMemo(() => {
    if (selectedCategory && selectedCategory !== 'all') {
      return catalog.filter(
        x => x.category.toLowerCase() === selectedCategory.toLowerCase()
      );
    }
    return catalog;
  }, [catalog, selectedCategory]);

  // Session interaction recorder (feeds SASRec)
  const recordInteraction = useCallback((productId) => {
    setSessionSequence(prev => {
      if (prev.length > 0 && prev[prev.length - 1] === productId) {
        return prev;
      }
      const updated = [...prev, productId];
      const trimmed = updated.length > 20 ? updated.slice(updated.length - 20) : updated;
      // Trigger live recomputation
      loadRecommendations(currentPersona ? currentPersona.user_id : 0, trimmed);
      return trimmed;
    });
  }, [currentPersona, loadRecommendations]);

  // Persona Selection
  const handleSelectPersona = useCallback((persona) => {
    setCurrentPersona(persona);
    setSimUserId(persona.user_id);
    const newSeq = [...(persona.history_item_ids || [])];
    setSessionSequence(newSeq);
    loadRecommendations(persona.user_id, newSeq);
    showToast(`Switched active profile to ${persona.name}`);
  }, [loadRecommendations, showToast]);

  // Cart actions
  const handleAddToCart = useCallback((product) => {
    setCart(prev => [...prev, product]);
    recordInteraction(product.id);
    showToast(`Added ${product.title} to your bag`);
  }, [recordInteraction, showToast]);

  const handleRemoveFromCart = useCallback((index) => {
    setCart(prev => prev.filter((_, idx) => idx !== index));
  }, []);

  const handleClearCart = useCallback(() => {
    setCart([]);
    showToast('Bag emptied');
  }, [showToast]);

  const handleCheckoutDemo = useCallback(() => {
    if (cart.length === 0) return;
    showToast('Checkout simulated! Active session captured by SASRec.');
    setCart([]);
    setIsCartOpen(false);
  }, [cart, showToast]);

  // Session actions
  const handleRemoveFromSession = useCallback((index) => {
    setSessionSequence(prev => {
      const updated = prev.filter((_, idx) => idx !== index);
      loadRecommendations(currentPersona?.user_id ?? 0, updated);
      return updated;
    });
  }, [currentPersona, loadRecommendations]);

  const handleClearSession = useCallback(() => {
    setSessionSequence([]);
    loadRecommendations(currentPersona?.user_id ?? 0, []);
    showToast('Sequential session reset to neutral state');
  }, [currentPersona, loadRecommendations, showToast]);

  // Inspection & XAI
  const handleInspect = useCallback((product) => {
    recordInteraction(product.id);
    setActiveXaiProduct(product);
  }, [recordInteraction]);

  // Tab switching with side-effects
  const handleTabChange = useCallback((tabId) => {
    setActiveTab(tabId);
    window.scrollTo({ top: 0, behavior: 'smooth' });
    if (tabId === 'recommendations') {
      loadRecommendations();
    } else if (tabId === 'simulator') {
      runSimulation();
    }
  }, [loadRecommendations, runSimulation]);

  return (
    <div className="theme-dark app-root">
      {/* Top Navigation Header */}
      <Header
        activeTab={activeTab}
        onTabChange={handleTabChange}
        personas={personas}
        currentPersona={currentPersona}
        onSelectPersona={handleSelectPersona}
        cartCount={cart.length}
        onOpenCart={() => setIsCartOpen(true)}
      />

      {/* Live Session Banner / Tracker */}
      <SessionTracker
        sessionSequence={sessionSequence}
        catalog={catalog}
        onRemoveItem={handleRemoveFromSession}
        onClearSession={handleClearSession}
      />

      {/* Main Application Body */}
      <main className="main-content" id="mainContent">
        {activeTab === 'storefront' && (
          <Storefront
            catalog={filteredCatalog}
            selectedCategory={selectedCategory}
            onSelectCategory={setSelectedCategory}
            isLoading={loadingCatalog}
            onAddToCart={handleAddToCart}
            onInspect={handleInspect}
          />
        )}

        {activeTab === 'recommendations' && (
          <Recommendations
            currentPersona={currentPersona}
            sessionLength={sessionSequence.length}
            recommendations={recommendations}
            isLoading={loadingRecs}
            onRefresh={() => loadRecommendations()}
            onAddToCart={handleAddToCart}
            onInspect={handleInspect}
          />
        )}

        {activeTab === 'simulator' && (
          <ModelTuningLab
            graphWeight={simGraphWeight}
            onChangeGraphWeight={setSimGraphWeight}
            seqWeight={simSeqWeight}
            onChangeSeqWeight={setSimSeqWeight}
            mode={simMode}
            onChangeMode={(newMode) => {
              setSimMode(newMode);
              runSimulation({ mode: newMode });
            }}
            personas={personas}
            selectedUserId={simUserId}
            onChangePersona={(userId) => {
              setSimUserId(userId);
              const found = personas.find(p => p.user_id === userId);
              if (found) setCurrentPersona(found);
              runSimulation({ userId });
            }}
            onRunSimulation={() => runSimulation()}
            simRecommendations={simRecommendations}
            isLoading={loadingSim}
            onAddToCart={handleAddToCart}
            onInspect={handleInspect}
          />
        )}
      </main>

      {/* Explainability (XAI) Modal */}
      {activeXaiProduct && (
        <XaiModal
          product={activeXaiProduct}
          onClose={() => setActiveXaiProduct(null)}
        />
      )}

      {/* Shopping Cart Drawer */}
      <CartDrawer
        isOpen={isCartOpen}
        onClose={() => setIsCartOpen(false)}
        cart={cart}
        sessionLength={sessionSequence.length}
        onRemoveItem={handleRemoveFromCart}
        onClearCart={handleClearCart}
        onCheckout={handleCheckoutDemo}
      />

      {/* Toast Notifications */}
      <Toast toasts={toasts} />
    </div>
  );
}
