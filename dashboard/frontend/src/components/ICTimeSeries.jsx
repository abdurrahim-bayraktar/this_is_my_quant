import React, { useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine, AreaChart, Area, BarChart, Bar
} from 'recharts';
import { formatNum } from '../utils/formatters';

export default function ICTimeSeries({ icData, name }) {
  // Compute summary stats
  const stats = useMemo(() => {
    const ics = icData.map(d => d.IC).filter(v => v != null);
    const mean = ics.reduce((a, b) => a + b, 0) / ics.length;
    const std = Math.sqrt(ics.reduce((s, v) => s + (v - mean) ** 2, 0) / ics.length);
    const ir = std > 0 ? mean / std : 0;
    const hitRate = ics.filter(v => v > 0).length / ics.length;
    return { mean, std, ir, hitRate, count: ics.length };
  }, [icData]);

  // Downsample if too many points
  const data = useMemo(() => {
    const maxPoints = 200;
    if (icData.length <= maxPoints) return icData;
    const step = Math.ceil(icData.length / maxPoints);
    return icData.filter((_, i) => i % step === 0);
  }, [icData]);

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
        <div style={{ fontWeight: 700, marginBottom: 4 }}>{d.Date}</div>
        <div style={{ color: '#6c7bff' }}>IC: {formatNum(d.IC)}</div>
        {d.IC_Rolling_20 != null && <div style={{ color: '#4ecdc4' }}>Rolling 20d: {formatNum(d.IC_Rolling_20)}</div>}
        <div style={{ color: 'var(--text-muted)' }}>Stocks: {d.N_Stocks}</div>
        {d.Cumulative_Spread != null && (
          <div style={{ color: d.Cumulative_Spread >= 0 ? 'var(--color-up)' : 'var(--color-down)' }}>
            Cum. Spread: {(d.Cumulative_Spread * 100).toFixed(2)}%
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="section-card">
      <div className="section-card-header">
        <h3>Cross-Sectional IC Time Series</h3>
        <span className="section-badge">
          μ={formatNum(stats.mean)} · IR={formatNum(stats.ir)} · Hit={(stats.hitRate * 100).toFixed(0)}% · {stats.count} dates
        </span>
      </div>
      <div className="section-card-body">
        {/* IC Line Chart */}
        <div style={{ width: '100%', height: 280 }}>
          <ResponsiveContainer>
            <LineChart data={data} margin={{ top: 5, right: 30, left: 10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(100,120,220,0.08)" />
              <XAxis dataKey="Date" tick={{ fontSize: 10, fill: '#5c6394' }} interval="preserveStartEnd" />
              <YAxis tick={{ fontSize: 10, fill: '#5c6394' }} domain={['auto', 'auto']} />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              <ReferenceLine y={0} stroke="rgba(255,255,255,0.15)" />
              <ReferenceLine y={stats.mean} stroke="#6c7bff" strokeDasharray="6 4" strokeWidth={1}
                label={{ value: `μ=${formatNum(stats.mean)}`, position: 'right', fill: '#6c7bff', fontSize: 10 }} />
              <Line type="monotone" dataKey="IC" stroke="rgba(108,123,255,0.4)" strokeWidth={1} dot={false} name="Daily IC" />
              <Line type="monotone" dataKey="IC_Rolling_20" stroke="#4ecdc4" strokeWidth={2.5} dot={false} name="Rolling 20d IC" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Cumulative Spread */}
        <div style={{ width: '100%', height: 180, marginTop: 16 }}>
          <ResponsiveContainer>
            <AreaChart data={data} margin={{ top: 5, right: 30, left: 10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(100,120,220,0.08)" />
              <XAxis dataKey="Date" tick={{ fontSize: 10, fill: '#5c6394' }} interval="preserveStartEnd" />
              <YAxis tick={{ fontSize: 10, fill: '#5c6394' }} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine y={0} stroke="rgba(255,255,255,0.15)" />
              <Area
                type="monotone"
                dataKey="Cumulative_Spread"
                stroke="#00d395"
                fill="rgba(0,211,149,0.15)"
                strokeWidth={2}
                name="Cumulative Q-Spread"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
