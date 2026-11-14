import React from 'react';

export default function Header({ experimentCount }) {
  return (
    <header className="header">
      <div className="header-brand">
        <div>
          <h1>Sentiment-Driven Stock Trend Prediction</h1>
          <span className="subtitle">Hybrid Dual-Branch LSTM · Walk-Forward Dashboard</span>
        </div>
      </div>
      <div className="header-badge">
        <span className="dot" />
        {experimentCount} Experiments Loaded
      </div>
    </header>
  );
}
