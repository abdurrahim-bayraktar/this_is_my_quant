import React, { useState, useMemo } from 'react';
import { shortenExpName, formatNum } from '../utils/formatters';

const FILTERS = ['all', 'classification', 'regression', 'sentiment', 'no-sentiment', 'backtest'];

export default function Sidebar({ experiments, selected, compared, onSelect, onClearCompare, loading }) {
  const [filter, setFilter] = useState('all');

  const filtered = useMemo(() => {
    if (filter === 'all') return experiments;
    if (filter === 'classification') return experiments.filter(e => e.task === 'classification');
    if (filter === 'regression') return experiments.filter(e => e.task === 'regression');
    if (filter === 'sentiment') return experiments.filter(e => e.use_sentiment);
    if (filter === 'no-sentiment') return experiments.filter(e => !e.use_sentiment);
    if (filter === 'backtest') return experiments.filter(e => e.has_backtest);
    return experiments;
  }, [experiments, filter]);

  const handleClick = (e, name) => {
    onSelect(name, e.shiftKey);
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h2>Experiments</h2>
      </div>

      <div className="sidebar-filters">
        {FILTERS.map(f => (
          <button
            key={f}
            className={`filter-chip ${filter === f ? 'active' : ''}`}
            onClick={() => setFilter(f)}
          >
            {f}
          </button>
        ))}
      </div>

      {compared && (
        <div className="compare-bar">
          <span className="compare-label">COMPARING</span>
          <span className="compare-name">{shortenExpName(compared)}</span>
          <button className="compare-clear" onClick={onClearCompare}>✕</button>
        </div>
      )}

      <div className="sidebar-list">
        {loading ? (
          <div className="empty-state">
            <div className="loading-shimmer" style={{ width: '80%', height: 24, marginBottom: 8 }} />
            <div className="loading-shimmer" style={{ width: '60%', height: 24, marginBottom: 8 }} />
            <div className="loading-shimmer" style={{ width: '70%', height: 24 }} />
          </div>
        ) : filtered.length === 0 ? (
          <div className="empty-state">
            <span className="empty-icon">📂</span>
            <span className="empty-title">No experiments found</span>
          </div>
        ) : (
          filtered.map(exp => (
            <div
              key={exp.name}
              className={`experiment-item ${selected === exp.name ? 'selected' : ''} ${compared === exp.name ? 'compared' : ''}`}
              onClick={(e) => handleClick(e, exp.name)}
              title={`Shift+click to compare\n${exp.name}`}
            >
              <span className="exp-name">{shortenExpName(exp.name)}</span>
              <div className="exp-meta">
                <span className={`tag ${exp.task}`}>{exp.task}</span>
                <span className={`tag ${exp.use_sentiment ? 'sentiment' : 'no-sentiment'}`}>
                  {exp.use_sentiment ? '🧠 SENT' : 'NO SENT'}
                </span>
                {exp.has_backtest && <span className="tag classification" style={{ fontSize: '0.58rem' }}>BT</span>}
                {exp.has_folds && <span style={{ color: 'var(--text-muted)', fontSize: '0.62rem' }}>F{exp.n_folds}</span>}
                <span className="exp-metric">
                  {formatNum(exp.headline_metric_value)}
                </span>
              </div>
            </div>
          ))
        )}
      </div>

      <div className="compare-hint">
        Shift+click to compare two experiments
      </div>
    </aside>
  );
}
