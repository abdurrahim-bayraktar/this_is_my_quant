import React from 'react';
import { formatPct, formatNum, formatInt } from '../utils/formatters';

function KpiCard({ label, value, format = 'num', sub, positive }) {
  const formatted = format === 'pct' ? formatPct(value) :
                    format === 'int' ? formatInt(value) :
                    formatNum(value);
  const colorClass = positive === true ? 'positive' : positive === false ? 'negative' : '';

  return (
    <div className="kpi-card">
      <div className="kpi-label">{label}</div>
      <div className={`kpi-value ${colorClass}`}>{formatted}</div>
      {sub && <div className="kpi-sub">{sub}</div>}
    </div>
  );
}

export default function MetricsCards({ detail, comparedDetail, taskType }) {
  const s = detail?.summary || {};
  const isClassification = taskType === 'classification';

  const cards = isClassification ? [
    { label: 'WF Accuracy', value: s.wf_agg_accuracy, format: 'pct', sub: `vs ${formatPct(s.wf_agg_zero_rule)} zero-rule`, positive: s.wf_agg_lift > 0 },
    { label: 'Lift', value: s.wf_agg_lift, format: 'pct', sub: 'Over majority-class baseline', positive: s.wf_agg_lift > 0 },
    { label: 'MCC', value: s.wf_agg_mcc, format: 'num', sub: 'Matthews correlation', positive: s.wf_agg_mcc > 0 },
    { label: 'F1 Macro', value: s.wf_agg_f1_macro, format: 'num', sub: 'Per-class F1 average' },
    { label: 'Valid Folds', value: s.n_valid_folds, format: 'int', sub: `of ${s.n_folds} total` },
    { label: 'Parameters', value: s.n_params, format: 'int', sub: `${s.n_features || '?'} features` },
  ] : [
    { label: 'WF IC', value: s.wf_agg_ic, format: 'num', sub: 'Aggregate Spearman rank IC', positive: s.wf_agg_ic > 0 },
    { label: 'Directional Acc', value: s.wf_agg_directional_accuracy, format: 'pct', sub: 'Correct return sign', positive: s.wf_agg_directional_accuracy > 0.5 },
    { label: 'CS-IC', value: s.cross_sectional_ic, format: 'num', sub: 'Cross-sectional ranking', positive: s.cross_sectional_ic > 0 },
    { label: 'IC-IR', value: s.ic_ir, format: 'num', sub: 'Information ratio (mean/std)', positive: s.ic_ir > 0 },
    { label: 'Q-Spread', value: s.quantile_spread, format: 'num', sub: 'Top-Q minus Bottom-Q', positive: s.quantile_spread > 0 },
    { label: 'Parameters', value: s.n_params, format: 'int', sub: `${s.n_features || '?'} features` },
  ];

  return (
    <div className="section-card">
      <div className="section-card-header">
        <h3>Key Metrics</h3>
        <span className="section-badge">{isClassification ? 'Classification' : 'Regression'}</span>
      </div>
      <div className="section-card-body">
        <div className="kpi-grid">
          {cards.map((c, i) => (
            <KpiCard key={i} {...c} />
          ))}
        </div>
      </div>
    </div>
  );
}
