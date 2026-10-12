import React from 'react';

export default function Toast({ toasts }) {
  if (!toasts || toasts.length === 0) return null;

  return (
    <div className="toast-container" id="toastContainer">
      {toasts.map(t => (
        <div className="toast" key={t.id} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span>{t.type === 'warning' ? '⚠️' : '✨'}</span>
          <span>{t.message}</span>
        </div>
      ))}
    </div>
  );
}
