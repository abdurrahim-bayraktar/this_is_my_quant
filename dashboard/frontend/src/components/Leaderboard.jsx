import React, { useState, useMemo } from 'react';
import { shortenExpName, formatNum, formatPct, formatInt } from '../utils/formatters';

export default function Leaderboard({ experiments, onSelect, compact }) {
  const [sortKey, setSortKey] = useState('headline_metric_value');
  const [sortAsc, setSortAsc] = useState(false);

  const sorted = useMemo(() => {
    return [...experiments].sort((a, b) => {
      const va = a[sortKey] ?? 0;
      const vb = b[sortKey] ?? 0;
      if (typeof va === 'string') return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
      return sortAsc ? va - vb : vb - va;
    });
  }, [experiments, sortKey, sortAsc]);

  const handleSort = (key) => {
    if (sortKey === key) setSortAsc(!sortAsc);
    else { setSortKey(key); setSortAsc(false); }
  };

  const cols = [
    { key: 'rank', label: '#', width: '40px' },
    { key: 'name', label: 'Experiment' },
    { key: 'task', label: 'Type' },
    { key: 'use_sentiment', label: 'Sent' },
    { key: 'architecture', label: 'Arch' },
    { key: 'headline_metric_value', label: 'Metric' },
    { key: 'n_params', label: 'Params' },
    { key: 'n_folds', label: 'Folds' },
  ];

  const displayData = compact ? sorted.slice(0, 10) : sorted;

  return (
    <div className="section-card">
      <div className="section-card-header">
        <h3>Experiment Leaderboard</h3>
        <span className="section-badge">{experiments.length} experiments · sorted by {sortKey.replace(/_/g, ' ')}</span>
      </div>
      <div className="section-card-body" style={{ overflowX: 'auto' }}>
        <table className="leaderboard-table">
          <thead>
            <tr>
              {cols.map(col => (
                <th
                  key={col.key}
                  className={sortKey === col.key ? 'sorted' : ''}
                  onClick={() => col.key !== 'rank' && handleSort(col.key)}
                  style={col.width ? { width: col.width } : {}}
                >
                  {col.label} {sortKey === col.key ? (sortAsc ? '↑' : '↓') : ''}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {displayData.map((exp, i) => (
              <tr key={exp.name} onClick={() => onSelect(exp.name)} style={{ cursor: 'pointer' }}>
                <td className="rank-cell">{i + 1}</td>
                <td className="name-cell" title={exp.name}>{shortenExpName(exp.name)}</td>
                <td><span className={`tag ${exp.task}`}>{exp.task}</span></td>
                <td>
                  <span className={`tag ${exp.use_sentiment ? 'sentiment' : 'no-sentiment'}`}>
                    {exp.use_sentiment ? '🧠' : '—'}
                  </span>
                </td>
                <td style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>{exp.architecture || '—'}</td>
                <td style={{ color: 'var(--accent-secondary)', fontWeight: 700 }}>
                  {formatNum(exp.headline_metric_value)}
                </td>
                <td>{formatInt(exp.n_params)}</td>
                <td>{exp.has_folds ? exp.n_folds : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
