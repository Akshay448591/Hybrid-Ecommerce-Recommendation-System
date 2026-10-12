import React from 'react';

export default function SessionTracker({
  sessionSequence,
  catalog,
  onRemoveItem,
  onClearSession
}) {
  const count = sessionSequence.length;

  return (
    <div className="session-tracker-bar" id="sessionTrackerBar">
      <div className="tracker-container">
        <div className="tracker-meta">
          <span className="tracker-badge">⚡ Active Browsing Session:</span>
          <span className="tracker-desc" id="sessionItemsSummary">
            {count === 0 ? '0 items recorded' : `${count} items in SASRec attention window`}
          </span>
        </div>

        <div className="tracker-items" id="sessionChips">
          {count === 0 ? (
            <span className="chip-empty">Browse or add items to feed the real-time sequential model...</span>
          ) : (
            sessionSequence.map((id, idx) => {
              const prod = catalog.find(p => p.id === id);
              const name = prod ? prod.title.split(' ')[0] : `Item #${id}`;
              return (
                <span className="session-chip" key={`${id}-${idx}`}>
                  <span>#{idx + 1}: {name}</span>
                  <span
                    style={{ cursor: 'pointer', opacity: 0.7, marginLeft: '6px' }}
                    onClick={() => onRemoveItem(idx)}
                    title="Remove from session window"
                  >
                    ×
                  </span>
                </span>
              );
            })
          )}
        </div>

        <button
          className="btn-text-sm"
          onClick={onClearSession}
          title="Clear active session sequence"
        >
          Reset Session
        </button>
      </div>
    </div>
  );
}
