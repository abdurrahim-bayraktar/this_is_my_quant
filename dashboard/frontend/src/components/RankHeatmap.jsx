import React, { useMemo } from 'react';
import { formatNum, shortenExpName } from '../utils/formatters';

function getColor(value, min, max) {
  if (value == null) return 'var(--bg-card)';
  const ratio = max === min ? 0.5 : (value - min) / (max - min);
  // Green (good) to red (bad)
  const r = Math.round(255 * (1 - ratio));
  const g = Math.round(200 * ratio);
  return `rgba(${r}, ${g}, 80, 0.25)`;
}

function getDeltaColor(delta) {
  if (delta == null) return 'var(--bg-card)';
  if (delta > 0) return `rgba(0, 211, 149, ${Math.min(0.4, Math.abs(delta) * 5)})`;
  if (delta < 0) return `rgba(255, 71, 87, ${Math.min(0.4, Math.abs(delta) * 5)})`;
  return 'var(--bg-card)';
}

export default function RankHeatmap({ stocks, comparedStocks, taskType, name, comparedName }) {
  const isClassification = taskType === 'classification';
  const metricKeys = isClassification
    ? ['accuracy', 'lift', 'mcc', 'f1_macro']
    : ['ic', 'directional_accuracy', 'mse', 'r2'];

  const isComparing = comparedStocks && comparedStocks.length > 0;

  const data = useMemo(() => {
    if (!isComparing) {
      // Single experiment: rank by each metric
      return stocks
        .map(s => ({ ticker: s.ticker, ...s }))
        .sort((a, b) => {
          const key = metricKeys[0];
          return (b[key] ?? 0) - (a[key] ?? 0);
        });
    }

    // Comparison: show delta
    const comparedMap = {};
    comparedStocks.forEach(s => { comparedMap[s.ticker] = s; });

    return stocks
      .filter(s => comparedMap[s.ticker])
      .map(s => {
        const c = comparedMap[s.ticker];
        const deltas = {};
        metricKeys.forEach(k => {
          deltas[`${k}_delta`] = (s[k] ?? 0) - (c[k] ?? 0);
          deltas[`${k}_base`] = s[k];
          deltas[`${k}_comp`] = c[k];
        });
        return { ticker: s.ticker, ...deltas };
      })
      .sort((a, b) => {
        const key = `${metricKeys[0]}_delta`;
        return Math.abs(b[key] ?? 0) - Math.abs(a[key] ?? 0); // Sort by biggest mover
      });
  }, [stocks, comparedStocks, isComparing, metricKeys]);

  if (data.length === 0) return null;

  // Get min/max for color scaling
  const ranges = {};
  metricKeys.forEach(k => {
    const vals = stocks.map(s => s[k]).filter(v => v != null);
    ranges[k] = { min: Math.min(...vals), max: Math.max(...vals) };
  });

  return (
    <div className="section-card">
      <div className="section-card-header">
        <h3>{isComparing ? 'Rank-Difference Heatmap' : 'Per-Stock Metrics'}</h3>
        <span className="section-badge">
          {data.length} stocks · sorted by {isComparing ? '|Δ|' : metricKeys[0]}
        </span>
      </div>
      <div className="section-card-body" style={{ overflowX: 'auto' }}>
        <table className="leaderboard-table">
          <thead>
            <tr>
              <th>Ticker</th>
              {metricKeys.map(k => (
                <th key={k}>{isComparing ? `Δ ${k}` : k}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((row, i) => (
              <tr key={row.ticker}>
                <td className="name-cell">{row.ticker}</td>
                {metricKeys.map(k => {
                  if (isComparing) {
                    const delta = row[`${k}_delta`];
                    return (
                      <td key={k} style={{
                        background: getDeltaColor(delta),
                        color: delta > 0 ? 'var(--color-up)' : delta < 0 ? 'var(--color-down)' : 'var(--text-secondary)',
                        fontWeight: 600,
                      }}>
                        {delta > 0 ? '+' : ''}{formatNum(delta)}
                      </td>
                    );
                  } else {
                    const val = row[k];
                    return (
                      <td key={k} style={{ background: getColor(val, ranges[k].min, ranges[k].max) }}>
                        {formatNum(val)}
                      </td>
                    );
                  }
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
