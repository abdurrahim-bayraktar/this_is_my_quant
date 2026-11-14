import React from 'react';
import { formatInt } from '../utils/formatters';

export default function ArchitectureDiagram({ config, summary }) {
  const useSentiment = config?.use_sentiment ?? summary?.use_sentiment ?? false;
  const nTech = config?.n_technical_features ?? summary?.n_technical_features ?? 20;
  const nSent = config?.n_sentiment_features ?? summary?.n_sentiment_features ?? 2;
  const arch = config?.architecture ?? summary?.architecture ?? 'unknown';
  const nParams = summary?.n_params ?? config?.n_params ?? 0;
  const modelType = config?.model_type ?? summary?.model_type ?? 'unknown';
  const task = summary?.task ?? config?.task ?? 'unknown';

  const isHybrid = arch === 'dual_branch_hybrid' || modelType === 'hybrid';
  const isDarnn = arch?.includes('darnn') || modelType?.includes('darnn') || modelType === 'DARNN';

  if (isDarnn) {
    return (
      <div className="section-card">
        <div className="section-card-header">
          <h3>Model Architecture</h3>
          <span className="section-badge">DA-RNN · {formatInt(nParams)} params</span>
        </div>
        <div className="section-card-body">
          <div className="arch-diagram">
            <div className="arch-block tech">
              <div className="block-label">Input</div>
              <div className="block-detail">{config?.n_features || '110'} features</div>
              <div className="block-detail">seq_len={config?.sequence_length || 20}</div>
            </div>
            <span className="arch-arrow">→</span>
            <div className="arch-block tech">
              <div className="block-label">Encoder LSTM</div>
              <div className="block-detail">Input Attention</div>
              <div className="block-detail">hidden={config?.encoder_hidden || 64}</div>
            </div>
            <span className="arch-arrow">→</span>
            <div className="arch-block fusion">
              <div className="block-label">Decoder LSTM</div>
              <div className="block-detail">Temporal Attention</div>
              <div className="block-detail">hidden={config?.decoder_hidden || 64}</div>
            </div>
            <span className="arch-arrow">→</span>
            <div className="arch-block fusion">
              <div className="block-label">Output</div>
              <div className="block-detail">{task === 'classification' ? '3-class trend' : 'vol-adj return'}</div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="section-card">
      <div className="section-card-header">
        <h3>Model Architecture</h3>
        <span className="section-badge">
          {isHybrid ? 'Dual-Branch Hybrid' : 'Attention LSTM'} · {formatInt(nParams)} params
        </span>
      </div>
      <div className="section-card-body">
        <div className="arch-diagram">
          {/* Input */}
          <div className="arch-block">
            <div className="block-label">Input</div>
            <div className="block-detail">[B, {config?.sequence_length || 20}, {nTech + nSent}]</div>
          </div>
          <span className="arch-arrow">→</span>

          {/* Branches */}
          {isHybrid ? (
            <div className="arch-branches">
              {/* Technical Branch */}
              <div className="arch-block tech">
                <div className="block-label">Technical Branch</div>
                <div className="block-detail">LSTM(128×2) + Attn(4h)</div>
                <div className="block-detail">{nTech} features</div>
                <div className="block-params">~260K params</div>
              </div>

              {/* Sentiment Branch */}
              <div className={`arch-block sent ${!useSentiment ? 'dimmed' : ''}`}>
                <div className="block-label">
                  Sentiment Branch {!useSentiment && '(disabled)'}
                </div>
                <div className="block-detail">GRU(32×1)</div>
                <div className="block-detail">{nSent} features</div>
                <div className="block-params">~3K params</div>
              </div>
            </div>
          ) : (
            <div className="arch-block tech">
              <div className="block-label">LSTM + Attention</div>
              <div className="block-detail">hidden=128, layers=2</div>
              <div className="block-detail">{nTech + nSent} features</div>
            </div>
          )}

          <span className="arch-arrow">→</span>

          {/* Fusion */}
          <div className="arch-block fusion">
            <div className="block-label">Fusion</div>
            <div className="block-detail">
              Concat → FC({useSentiment ? 160 : 128}→64)
            </div>
          </div>
          <span className="arch-arrow">→</span>

          {/* Head */}
          <div className="arch-block">
            <div className="block-label">
              {task === 'classification' ? 'Trend Head' : 'Regression Head'}
            </div>
            <div className="block-detail">
              {task === 'classification' ? '64→32→3 (softmax)' : '64→32→1'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
