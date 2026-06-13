import React, { useMemo } from 'react';
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, Legend, Tooltip
} from 'recharts';
import { shortenExpName, formatNum } from '../utils/formatters';

function normalize(val, min, max) {
  if (max === min) return 0.5;
  return Math.max(0, Math.min(1, (val - min) / (max - min)));
}

export default function RadarComparison({ detail1, detail2, name1, name2, taskType }) {
  const s1 = detail1?.summary || {};
  const s2 = detail2?.summary || {};
  const isClassification = taskType === 'classification';

  const data = useMemo(() => {
    if (isClassification) {
      const axes = [
        { key: 'wf_agg_accuracy', label: 'Accuracy', min: 0.3, max: 0.55 },
        { key: 'wf_agg_lift', label: 'Lift', min: -0.05, max: 0.15 },
        { key: 'wf_agg_mcc', label: 'MCC', min: -0.1, max: 0.3 },
        { key: 'wf_agg_f1_macro', label: 'F1 Macro', min: 0.2, max: 0.5 },
        { key: 'stability', label: 'Stability', min: 0, max: 1 },
      ];

      // Compute stability (inverse of std)
      const std1 = s1.wf_std_accuracy || 0.05;
      const std2 = s2.wf_std_accuracy || 0.05;
      s1.stability = 1 / (1 + std1 * 10);
      s2.stability = 1 / (1 + std2 * 10);

      return axes.map(a => ({
        axis: a.label,
        exp1: normalize(s1[a.key] ?? 0, a.min, a.max),
        exp2: normalize(s2[a.key] ?? 0, a.min, a.max),
        raw1: s1[a.key],
        raw2: s2[a.key],
      }));
    } else {
      const axes = [
        { key: 'wf_agg_ic', label: 'WF IC', min: -0.05, max: 0.1 },
        { key: 'wf_agg_directional_accuracy', label: 'Dir. Accuracy', min: 0.45, max: 0.6 },
        { key: 'cross_sectional_ic', label: 'CS-IC', min: -0.02, max: 0.06 },
        { key: 'ic_ir', label: 'IC-IR', min: -0.1, max: 0.3 },
        { key: 'quantile_spread', label: 'Q-Spread', min: -0.02, max: 0.06 },
      ];

      return axes.map(a => ({
        axis: a.label,
        exp1: normalize(s1[a.key] ?? 0, a.min, a.max),
        exp2: normalize(s2[a.key] ?? 0, a.min, a.max),
        raw1: s1[a.key],
        raw2: s2[a.key],
      }));
    }
  }, [s1, s2, isClassification]);

  const CustomTooltip = ({ active, payload }) => {
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
        <div style={{ fontWeight: 700, marginBottom: 4 }}>{d.axis}</div>
        <div style={{ color: '#6c7bff' }}>{shortenExpName(name1)}: {formatNum(d.raw1)}</div>
        <div style={{ color: '#4ecdc4' }}>{shortenExpName(name2)}: {formatNum(d.raw2)}</div>
      </div>
    );
  };

  return (
    <div className="section-card">
      <div className="section-card-header">
        <h3>A/B Comparison</h3>
        <span className="section-badge">Radar overlay</span>
      </div>
      <div className="section-card-body">
        <div className="radar-container">
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart data={data}>
              <PolarGrid stroke="rgba(100,120,220,0.12)" />
              <PolarAngleAxis
                dataKey="axis"
                tick={{ fill: '#9ca3c4', fontSize: 12, fontWeight: 600 }}
              />
              <PolarRadiusAxis domain={[0, 1]} tick={false} axisLine={false} />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              <Radar
                name={shortenExpName(name1)}
                dataKey="exp1"
                stroke="#6c7bff"
                fill="#6c7bff"
                fillOpacity={0.2}
                strokeWidth={2}
              />
              <Radar
                name={shortenExpName(name2)}
                dataKey="exp2"
                stroke="#4ecdc4"
                fill="#4ecdc4"
                fillOpacity={0.2}
                strokeWidth={2}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
