import React from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine, Area, ComposedChart, Bar
} from 'recharts';
import { shortenExpName, formatNum } from '../utils/formatters';

export default function FoldTimeline({ folds, comparedFolds, taskType, name, comparedName }) {
  const isClassification = taskType === 'classification';
  const metricKey = isClassification ? 'accuracy' : 'ic';
  const metricLabel = isClassification ? 'Accuracy' : 'IC';

  // Compute mean
  const values = folds.map(f => f[metricKey]).filter(v => v != null);
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const std = Math.sqrt(values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length);

  // Find worst fold
  const worstIdx = values.indexOf(Math.min(...values));

  const data = folds.map((f, i) => {
    const item = {
      name: `F${f.fold || i + 1}`,
      period: f.val_start ? `${f.val_start.slice(0, 7)} → ${f.val_end?.slice(0, 7) || ''}` : `Fold ${i + 1}`,
      [metricKey]: f[metricKey],
      samples: f.val_samples || f.train_samples,
      trainSamples: f.train_samples,
      isWorst: i === worstIdx,
    };
    if (comparedFolds && comparedFolds[i]) {
      item[`${metricKey}_compared`] = comparedFolds[i][metricKey];
    }
    return item;
  });

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null;
    const d = payload[0]?.payload;
    return (
      <div style={{
        background: 'var(--bg-secondary)',
        border: '1px solid var(--border-medium)',
        borderRadius: 8,
        padding: '10px 14px',
        fontSize: '0.75rem',
      }}>
        <div style={{ fontWeight: 700, marginBottom: 4 }}>{d.period}</div>
        {payload.map((p, i) => (
          <div key={i} style={{ color: p.color, fontFamily: 'var(--font-mono)' }}>
            {p.name}: {formatNum(p.value)}
          </div>
        ))}
        {d.trainSamples && (
          <div style={{ color: 'var(--text-muted)', marginTop: 4, fontSize: '0.68rem' }}>
            Train: {d.trainSamples?.toLocaleString()} samples
          </div>
        )}
        {d.isWorst && <div style={{ color: 'var(--color-down)', fontWeight: 600, marginTop: 4 }}>⚠ Worst fold</div>}
      </div>
    );
  };

  return (
    <div className="section-card">
      <div className="section-card-header">
        <h3>Walk-Forward Fold Performance</h3>
        <span className="section-badge">
          {folds.length} folds · μ={formatNum(mean)} · σ={formatNum(std)}
        </span>
      </div>
      <div className="section-card-body">
        <div className="fold-timeline-container">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(100,120,220,0.08)" />
              <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#5c6394' }} />
              <YAxis tick={{ fontSize: 11, fill: '#5c6394' }} domain={['auto', 'auto']} />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              <ReferenceLine y={mean} stroke="#6c7bff" strokeDasharray="6 4" strokeWidth={1.5}
                label={{ value: `μ = ${formatNum(mean)}`, position: 'right', fill: '#6c7bff', fontSize: 11 }} />
              <Line
                type="monotone"
                dataKey={metricKey}
                name={shortenExpName(name)}
                stroke="#6c7bff"
                strokeWidth={2.5}
                dot={{ r: 5, fill: '#6c7bff', strokeWidth: 2, stroke: '#07080f' }}
                activeDot={{ r: 7 }}
              />
              {comparedFolds && (
                <Line
                  type="monotone"
                  dataKey={`${metricKey}_compared`}
                  name={shortenExpName(comparedName)}
                  stroke="#4ecdc4"
                  strokeWidth={2.5}
                  dot={{ r: 5, fill: '#4ecdc4', strokeWidth: 2, stroke: '#07080f' }}
                  strokeDasharray="5 5"
                />
              )}
              <Bar dataKey="samples" fill="rgba(108,123,255,0.08)" radius={[4, 4, 0, 0]} yAxisId="right" />
              <YAxis yAxisId="right" orientation="right" tick={false} axisLine={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
