import React, { useState, useEffect, useCallback } from 'react';
import { fetchApi } from './hooks/useApi';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import MetricsCards from './components/MetricsCards';
import Leaderboard from './components/Leaderboard';
import FoldTimeline from './components/FoldTimeline';
import RadarComparison from './components/RadarComparison';
import RankHeatmap from './components/RankHeatmap';
import StrategyTable from './components/StrategyTable';
import ArchitectureDiagram from './components/ArchitectureDiagram';
import ICTimeSeries from './components/ICTimeSeries';
import TradeView from './components/TradeView';

export default function App() {
  const [experiments, setExperiments] = useState([]);
  const [selected, setSelected] = useState(null);
  const [compared, setCompared] = useState(null);
  const [selectedDetail, setSelectedDetail] = useState(null);
  const [comparedDetail, setComparedDetail] = useState(null);
  const [loading, setLoading] = useState(true);

  // Load experiment list
  useEffect(() => {
    fetchApi('/experiments')
      .then(d => {
        setExperiments(d.experiments || []);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  // Load selected experiment detail
  useEffect(() => {
    if (!selected) { setSelectedDetail(null); return; }
    fetchApi(`/experiment/${selected}`)
      .then(setSelectedDetail)
      .catch(() => setSelectedDetail(null));
  }, [selected]);

  // Load compared experiment detail
  useEffect(() => {
    if (!compared) { setComparedDetail(null); return; }
    fetchApi(`/experiment/${compared}`)
      .then(setComparedDetail)
      .catch(() => setComparedDetail(null));
  }, [compared]);

  const handleSelect = useCallback((name, isShift) => {
    if (isShift && selected && name !== selected) {
      setCompared(name);
    } else {
      setSelected(name);
      if (compared === name) setCompared(null);
    }
  }, [selected, compared]);

  const clearCompare = useCallback(() => setCompared(null), []);

  const selectedExp = experiments.find(e => e.name === selected);
  const comparedExp = experiments.find(e => e.name === compared);
  const taskType = selectedDetail?.summary?.task || selectedExp?.task || 'unknown';
  const isRegression = taskType === 'regression';

  return (
    <div className="app-layout">
      <Header experimentCount={experiments.length} />
      <div className="app-body">
        <Sidebar
          experiments={experiments}
          selected={selected}
          compared={compared}
          onSelect={handleSelect}
          onClearCompare={clearCompare}
          loading={loading}
        />
        <main className="main-content">
          {!selected ? (
            <div className="fade-in">
              <Leaderboard experiments={experiments} onSelect={(name) => handleSelect(name, false)} />
            </div>
          ) : (
            <div className="fade-in" key={selected}>
              {/* KPI Cards */}
              {selectedDetail && (
                <MetricsCards
                  detail={selectedDetail}
                  comparedDetail={comparedDetail}
                  taskType={taskType}
                />
              )}

              {/* Architecture Diagram */}
              {selectedDetail && (
                <ArchitectureDiagram
                  config={selectedDetail.config}
                  summary={selectedDetail.summary}
                />
              )}

              {/* Walk-Forward Fold Timeline */}
              {selectedDetail?.folds?.length > 0 && (
                <FoldTimeline
                  folds={selectedDetail.folds}
                  comparedFolds={comparedDetail?.folds}
                  taskType={taskType}
                  name={selected}
                  comparedName={compared}
                />
              )}

              {/* Radar Comparison */}
              {selectedDetail && comparedDetail && (
                <RadarComparison
                  detail1={selectedDetail}
                  detail2={comparedDetail}
                  name1={selected}
                  name2={compared}
                  taskType={taskType}
                />
              )}

              {/* Rank Heatmap */}
              {selectedDetail?.stocks?.length > 0 && (
                <RankHeatmap
                  stocks={selectedDetail.stocks}
                  comparedStocks={comparedDetail?.stocks}
                  taskType={taskType}
                  name={selected}
                  comparedName={compared}
                />
              )}

              {/* IC Time Series (Regression only) */}
              {isRegression && selectedDetail?.ic?.length > 0 && (
                <ICTimeSeries
                  icData={selectedDetail.ic}
                  name={selected}
                />
              )}

              {/* Backtesting Strategy Table */}
              {selectedDetail?.backtest?.length > 0 && (
                <StrategyTable
                  strategies={selectedDetail.backtest}
                  name={selected}
                />
              )}

              {/* Trade View */}
              {selectedDetail?.strategies?.length > 0 && (
                <TradeView
                  experimentName={selected}
                  strategies={selectedDetail.strategies}
                  backtest={selectedDetail.backtest}
                />
              )}

              {/* Leaderboard at bottom */}
              <Leaderboard
                experiments={experiments}
                onSelect={(name) => handleSelect(name, false)}
                compact
              />
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
