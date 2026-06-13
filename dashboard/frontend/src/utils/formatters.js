export function formatPct(val, decimals = 2) {
  if (val == null || isNaN(val)) return '—';
  return `${(val * 100).toFixed(decimals)}%`;
}

export function formatNum(val, decimals = 4) {
  if (val == null || isNaN(val)) return '—';
  return val.toFixed(decimals);
}

export function formatInt(val) {
  if (val == null || isNaN(val)) return '—';
  return Number(val).toLocaleString();
}

export function formatMetric(val, name) {
  if (val == null || isNaN(val)) return '—';
  const pctMetrics = [
    'accuracy', 'lift', 'directional_accuracy', 'hit_rate', 'ic_hit_rate',
    'wf_agg_accuracy', 'wf_agg_lift', 'wf_agg_directional_accuracy',
    'per_stock_avg_accuracy', 'Total Return', 'Annualized Return',
    'Annualized Volatility', 'Max Drawdown', 'Daily Win Rate',
  ];
  const lowerName = (name || '').toLowerCase().replace(/\s+/g, '_');
  if (pctMetrics.some(m => lowerName.includes(m.toLowerCase().replace(/\s+/g, '_')))) {
    return formatPct(val);
  }
  return formatNum(val);
}

export function shortenExpName(name) {
  // Remove date suffix for cleaner display
  return name.replace(/_\d{8}_\d{6}$/, '');
}
