import React, { useMemo } from 'react';
import { formatPct, formatNum } from '../utils/formatters';

export default function StrategyTable({ strategies, name }) {
  const sorted = useMemo(() => {
    return [...strategies].sort((a, b) => (b['Sharpe Ratio'] ?? 0) - (a['Sharpe Ratio'] ?? 0));
  }, [strategies]);

  const bestSharpe = sorted.length > 0 ? sorted[0]['Sharpe Ratio'] : 0;

  const cols = [
    { key: 'Strategy', label: 'Strategy', align: 'left' },
    { key: 'Total Return', label: 'Total Return', fmt: 'pct' },
    { key: 'Annualized Return', label: 'Ann. Return', fmt: 'pct' },
    { key: 'Sharpe Ratio', label: 'Sharpe', fmt: 'num2' },
    { key: 'Sortino Ratio', label: 'Sortino', fmt: 'num2' },
    { key: 'Max Drawdown', label: 'Max DD', fmt: 'pct' },
    { key: 'Daily Win Rate', label: 'Win Rate', fmt: 'pct' },
    { key: 'Total Turnover', label: 'Turnover', fmt: 'num1' },
  ];

  const format = (val, fmt) => {
    if (val == null) return '—';
    if (fmt === 'pct') return formatPct(val);
    if (fmt === 'num2') return formatNum(val, 2);
    if (fmt === 'num1') return formatNum(val, 1);
    return String(val);
  };

  return (
    <div className="section-card">
      <div className="section-card-header">
        <h3>Backtesting Results</h3>
        <span className="section-badge">{strategies.length} strategies · sorted by Sharpe</span>
      </div>
      <div className="section-card-body" style={{ overflowX: 'auto' }}>
        <table className="strategy-table">
          <thead>
            <tr>
              {cols.map(c => (
                <th key={c.key} style={c.align === 'left' ? { textAlign: 'left' } : {}}>{c.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((s, i) => {
              const isBest = s['Sharpe Ratio'] === bestSharpe && s.Strategy?.includes('Regression');
              const isBaseline = !s.Strategy?.includes('Regression');
              return (
                <tr key={i} className={isBest ? 'best-strategy' : ''}>
                  {cols.map(c => {
                    const val = s[c.key];
                    let style = {};
                    if (c.key === 'Total Return' && val != null) {
                      style.color = val >= 0 ? 'var(--color-up)' : 'var(--color-down)';
                    }
                    if (c.key === 'Max Drawdown' && val != null) {
                      style.color = 'var(--color-down)';
                    }
                    if (isBaseline && c.key !== 'Strategy') {
                      style.opacity = 0.6;
                    }
                    return (
                      <td key={c.key} style={style}>
                        {c.key === 'Strategy' ? val : format(val, c.fmt)}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
