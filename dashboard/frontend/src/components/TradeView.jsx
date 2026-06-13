import React, { useState, useEffect, useMemo, useRef } from 'react';
import { fetchApi } from '../hooks/useApi';
import { createChart, CandlestickSeries, HistogramSeries } from 'lightweight-charts';

function TickerRankingList({ rankings, activeTicker, onSelect }) {
  if (!rankings || rankings.length === 0) {
    return (
      <div className="empty-state" style={{ padding: 'var(--space-lg)' }}>
        <span className="empty-icon">📊</span>
        <span className="empty-title">No trade data</span>
      </div>
    );
  }

  return (
    <div className="ticker-ranking-list">
      {rankings.map((r, i) => (
        <div
          key={r.ticker}
          className={`ticker-rank-item ${activeTicker === r.ticker ? 'active' : ''}`}
          onClick={() => onSelect(r.ticker)}
        >
          <span className="ticker-symbol">{r.ticker}</span>
          <div className="trade-bar">
            <div className="buy-segment" style={{ width: `${r.buy_pct}%` }} />
            <div className="sell-segment" style={{ width: `${r.sell_pct}%` }} />
          </div>
          <span className="trade-count">{r.total_trades} trades</span>
          <span className="trade-count">{r.buys}B / {r.sells}S</span>
          <span className={`bias-label ${r.net_bias}`}>{r.net_bias}</span>
        </div>
      ))}
    </div>
  );
}

function CandlestickPanel({ ticker, trades }) {
  const containerRef = useRef(null);
  const chartRef = useRef(null);
  const [priceData, setPriceData] = useState(null);
  const [loading, setLoading] = useState(false);

  // Fetch price data
  useEffect(() => {
    if (!ticker) return;
    setLoading(true);
    fetchApi(`/price/${ticker}`)
      .then(d => setPriceData(d.data))
      .catch(() => setPriceData(null))
      .finally(() => setLoading(false));
  }, [ticker]);

  // Create chart
  useEffect(() => {
    if (!containerRef.current || !priceData || priceData.length === 0) return;

    // Clean up previous chart
    if (chartRef.current) {
      chartRef.current.remove();
      chartRef.current = null;
    }

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: 400,
      layout: {
        background: { color: '#0d0f1a' },
        textColor: '#5c6394',
        fontFamily: "'Inter', sans-serif",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: 'rgba(100,120,220,0.06)' },
        horzLines: { color: 'rgba(100,120,220,0.06)' },
      },
      crosshair: {
        mode: 0,
        vertLine: { color: 'rgba(108,123,255,0.3)', labelBackgroundColor: '#131629' },
        horzLine: { color: 'rgba(108,123,255,0.3)', labelBackgroundColor: '#131629' },
      },
      timeScale: {
        borderColor: 'rgba(100,120,220,0.1)',
        timeVisible: false,
      },
      rightPriceScale: {
        borderColor: 'rgba(100,120,220,0.1)',
      },
    });

    chartRef.current = chart;

    // Candlestick series
    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#00d395',
      downColor: '#ff4757',
      borderUpColor: '#00d395',
      borderDownColor: '#ff4757',
      wickUpColor: '#00d395',
      wickDownColor: '#ff4757',
    });

    candleSeries.setData(priceData);

    // Volume series
    const volumeSeries = chart.addSeries(HistogramSeries, {
      color: 'rgba(108,123,255,0.15)',
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
    });

    chart.priceScale('volume').applyOptions({
      scaleMargins: { top: 0.85, bottom: 0 },
    });

    const volumeData = priceData.map(d => ({
      time: d.time,
      value: d.volume,
      color: d.close >= d.open ? 'rgba(0,211,149,0.2)' : 'rgba(255,71,87,0.2)',
    }));
    volumeSeries.setData(volumeData);

    // Trade markers
    if (trades && trades.length > 0) {
      const tradesByDate = {};
      trades.forEach(t => {
        const date = t.Date || t.date;
        if (!tradesByDate[date]) tradesByDate[date] = [];
        tradesByDate[date].push(t);
      });

      const markers = Object.entries(tradesByDate)
        .map(([date, dayTrades]) => {
          const direction = String(dayTrades[0].Direction || dayTrades[0].Action || '').toLowerCase();
          const isBuy = direction.includes('buy');
          return {
            time: date,
            position: isBuy ? 'belowBar' : 'aboveBar',
            color: isBuy ? '#00d395' : '#ff4757',
            shape: isBuy ? 'arrowUp' : 'arrowDown',
            text: isBuy ? 'B' : 'S',
          };
        })
        .sort((a, b) => a.time.localeCompare(b.time));

      if (markers.length > 0) {
        candleSeries.setMarkers(markers);
      }
    }

    chart.timeScale().fitContent();

    // Resize handler
    const handleResize = () => {
      if (containerRef.current && chartRef.current) {
        chartRef.current.applyOptions({ width: containerRef.current.clientWidth });
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }
    };
  }, [priceData, trades]);

  if (!ticker) {
    return (
      <div className="empty-state" style={{ height: 400 }}>
        <span className="empty-icon">📈</span>
        <span className="empty-title">Select a ticker to view chart</span>
        <span className="empty-hint">Click a ticker from the ranking panel on the left</span>
      </div>
    );
  }

  if (loading) {
    return <div className="loading-shimmer" style={{ height: 400, borderRadius: 10 }} />;
  }

  if (!priceData) {
    return (
      <div className="empty-state" style={{ height: 400 }}>
        <span className="empty-icon">⚠️</span>
        <span className="empty-title">No price data for {ticker}</span>
        <span className="empty-hint">Price data could not be loaded from cache or yfinance</span>
      </div>
    );
  }

  return <div ref={containerRef} className="candlestick-container" />;
}

export default function TradeView({ experimentName, strategies, backtest }) {
  const [activeStrategy, setActiveStrategy] = useState(null);
  const [rankings, setRankings] = useState([]);
  const [activeTicker, setActiveTicker] = useState(null);
  const [tickerTrades, setTickerTrades] = useState([]);

  // Default to best strategy by Sharpe
  useEffect(() => {
    if (strategies.length > 0 && !activeStrategy) {
      if (backtest && backtest.length > 0) {
        const regressionStrats = backtest.filter(s => s.Strategy?.includes('Regression'));
        const best = regressionStrats.length > 0
          ? regressionStrats.reduce((a, b) => (a['Sharpe Ratio'] ?? 0) > (b['Sharpe Ratio'] ?? 0) ? a : b)
          : backtest[0];
        // Find matching strategy name in files
        const stratName = strategies.find(s => best.Strategy?.includes(s.replace(/_/g, ' ').replace(/  +/g, ' ')));
        setActiveStrategy(stratName || strategies[0]);
      } else {
        setActiveStrategy(strategies[0]);
      }
    }
  }, [strategies, backtest, activeStrategy]);

  // Load ticker rankings when strategy changes
  useEffect(() => {
    if (!activeStrategy || !experimentName) return;
    fetchApi(`/experiment/${experimentName}/ticker-rankings/${activeStrategy}`)
      .then(d => {
        setRankings(d.rankings || []);
        // Auto-select most traded ticker
        if (d.rankings?.length > 0) {
          setActiveTicker(d.rankings[0].ticker);
        }
      })
      .catch(() => setRankings([]));
  }, [activeStrategy, experimentName]);

  // Load trades for selected ticker
  useEffect(() => {
    if (!activeStrategy || !experimentName || !activeTicker) {
      setTickerTrades([]);
      return;
    }
    fetchApi(`/experiment/${experimentName}/trades/${activeStrategy}`)
      .then(d => {
        const trades = (d.trades || []).filter(t => t.Ticker === activeTicker);
        setTickerTrades(trades);
      })
      .catch(() => setTickerTrades([]));
  }, [activeStrategy, experimentName, activeTicker]);

  return (
    <div className="section-card">
      <div className="section-card-header">
        <h3>Per-Stock Trade Analysis</h3>
        <span className="section-badge">
          {activeTicker ? `${activeTicker} · ${tickerTrades.length} trades` : 'Select a strategy'}
        </span>
      </div>
      <div className="section-card-body">
        {/* Strategy Tabs */}
        <div className="section-tabs">
          {strategies.map(s => (
            <button
              key={s}
              className={`section-tab ${activeStrategy === s ? 'active' : ''}`}
              onClick={() => { setActiveStrategy(s); setActiveTicker(null); }}
            >
              {s.replace(/_/g, ' ').slice(0, 30)}
            </button>
          ))}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr', gap: 'var(--space-lg)' }}>
          {/* Ticker Ranking */}
          <div style={{ borderRight: '1px solid var(--border-subtle)', paddingRight: 'var(--space-md)' }}>
            <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 'var(--space-sm)' }}>
              Tickers by Trade Frequency
            </div>
            <TickerRankingList
              rankings={rankings}
              activeTicker={activeTicker}
              onSelect={setActiveTicker}
            />
          </div>

          {/* Candlestick Chart */}
          <div>
            <CandlestickPanel
              ticker={activeTicker}
              trades={tickerTrades}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
