# Data Model & Dynamic Model — Thesis Sections 3.1 & 3.3

> [!NOTE]
> All schemas, types, and relationships below were derived from live inspection of the codebase, including `pd.read_parquet()` on actual cached files. LaTeX/TikZ equivalents are provided at the end.

---

## 3.1 Data Model

### Cached File Schemas

The system uses a **three-layer caching architecture**. Each layer has a distinct storage format and schema.

#### Layer 1 — Price Cache (`data/price_cache/`)

| Aspect | Detail |
|---|---|
| **Format** | Python Pickle (`.pkl`) |
| **Structure** | `Dict[str, pd.DataFrame]` — keys are ticker symbols, values are OHLCV DataFrames |
| **Key** | `prices_{start_date}_{end_date}{suffix}.pkl` |
| **Example** | `prices_2012-01-01_2019-12-31.pkl` (14 MB, ~150 stocks) |

**Per-ticker DataFrame schema (DatetimeIndex):**

| Column | Dtype | Description |
|---|---|---|
| *(index)* | `DatetimeIndex` | Trading day |
| `Open` | `float64` | Opening price |
| `High` | `float64` | Intraday high |
| `Low` | `float64` | Intraday low |
| `Close` | `float64` | Closing price |
| `Volume` | `int64` | Trading volume |
| `Ticker` | `str` | Stock symbol |

---

#### Layer 2 — Sentiment Cache (`data/cache/sentiment/`)

Three Parquet files per ticker: `{TICKER}_news.parquet`, `{TICKER}_sentiment.parquet`, `{TICKER}_prices.parquet`.

**`{TICKER}_news.parquet` — Raw headlines (pre-FinBERT):**

| Column | Dtype | Description |
|---|---|---|
| `date` | `datetime64[ns, UTC]` | Publication timestamp (UTC) |
| `headline` | `str` | Raw headline text from FNSPID |
| `ticker` | `str` | Stock symbol |

**`{TICKER}_sentiment.parquet` — FinBERT-enriched headlines:**

| Column | Dtype | Description |
|---|---|---|
| `date` | `datetime64[ns, UTC]` | Publication timestamp (UTC) |
| `headline` | `str` | Raw headline text |
| `ticker` | `str` | Stock symbol |
| `sentiment_label` | `str` | Predicted class: `"positive"`, `"negative"`, `"neutral"` |
| `sentiment_score` | `float64` | Probability of the predicted class ∈ [0, 1] |
| `sentiment_positive` | `float64` | P(positive) ∈ [0, 1] |
| `sentiment_negative` | `float64` | P(negative) ∈ [0, 1] |
| `sentiment_neutral` | `float64` | P(neutral) ∈ [0, 1] |
| `sentiment_value` | `float64` | Signed polarity: `P(pos) − P(neg)` ∈ [−1, +1] |

**`{TICKER}_prices.parquet` — OHLCV from yfinance (per-ticker copy):**

| Column | Dtype | Description |
|---|---|---|
| `date` | `datetime64[ns, America/New_York]` | Trading day (US/Eastern) |
| `open` | `float64` | Opening price |
| `high` | `float64` | Intraday high |
| `low` | `float64` | Intraday low |
| `close` | `float64` | Closing price |
| `volume` | `int64` | Trading volume |
| `dividends` | `float64` | Dividends paid |
| `stock_splits` | `float64` | Split factor |
| `ticker` | `str` | Stock symbol |

---

#### Layer 3 — Feature Cache (`data/feature_cache/`)

| Aspect | Detail |
|---|---|
| **Format** | NumPy compressed archive (`.npz`) |
| **Key** | `features_v7_{N}stocks_{start}_{end}.npz` |
| **Contents** | `X` (float32), `y` (float32/int64), `feature_cols` (str[]), `tickers` (str[]) |

---

### Entity-Relationship Diagram

```mermaid
erDiagram
    TICKER {
        string symbol PK "e.g. MSFT, AAPL"
        float  sentiment_density "fraction of biz days with news"
        bool   excluded "in EXCLUDE_TICKERS set"
    }

    PRICE_DATA_OHLCV {
        date   trading_date PK "DatetimeIndex"
        string ticker FK "→ TICKER.symbol"
        float  open
        float  high
        float  low
        float  close
        int    volume
    }

    NEWS_HEADLINE {
        datetime date PK "UTC timestamp"
        string   ticker FK "→ TICKER.symbol"
        string   headline "raw text from FNSPID"
    }

    SENTIMENT_SCORE {
        datetime date PK "UTC timestamp"
        string   ticker FK "→ TICKER.symbol"
        string   headline
        string   sentiment_label "positive|negative|neutral"
        float    sentiment_score "P(predicted class)"
        float    sentiment_positive "P(pos) from FinBERT"
        float    sentiment_negative "P(neg) from FinBERT"
        float    sentiment_neutral "P(neu) from FinBERT"
        float    sentiment_value "P(pos) - P(neg)"
    }

    DAILY_AGGREGATED_SENTIMENT {
        date   trading_date PK "business day"
        string ticker FK "→ TICKER.symbol"
        float  sent_mean "daily mean of sentiment_value"
        int    sent_count "number of headlines"
        float  sent_count_log "log1p(sent_count)"
        float  sent_strength "sent_mean x sent_count_log"
    }

    TECHNICAL_FEATURES {
        date   trading_date PK "DatetimeIndex"
        string ticker FK "→ TICKER.symbol"
        float  atr_pct "SHAP rank 1"
        float  intraday_range "SHAP rank 2"
        float  bb_BBB "SHAP rank 3"
        float  gap "SHAP rank 4"
        float  rsi_14 "SHAP rank 18"
        float  return_5d "SHAP rank 20"
    }

    FEATURE_VECTOR {
        date   target_date PK "prediction target date"
        string ticker FK "→ TICKER.symbol"
        float  features_22d "20 tech + 2 sent features"
        float  vol_adj_return "regression target"
    }

    TICKER ||--o{ PRICE_DATA_OHLCV : "has daily"
    TICKER ||--o{ NEWS_HEADLINE : "mentioned in"
    NEWS_HEADLINE ||--|| SENTIMENT_SCORE : "FinBERT extracts"
    TICKER ||--o{ SENTIMENT_SCORE : "has scored"
    SENTIMENT_SCORE }o--|| DAILY_AGGREGATED_SENTIMENT : "aggregated to"
    TICKER ||--o{ DAILY_AGGREGATED_SENTIMENT : "has daily"
    PRICE_DATA_OHLCV ||--|| TECHNICAL_FEATURES : "indicators computed"
    TECHNICAL_FEATURES ||--|| FEATURE_VECTOR : "combined with"
    DAILY_AGGREGATED_SENTIMENT ||--|| FEATURE_VECTOR : "combined with"
```

### FinBERT Score Storage Detail

The FinBERT scores are stored **alongside their source timestamp** in the per-ticker sentiment Parquet files. The key data structure is:

```
{TICKER}_sentiment.parquet
├── date: datetime64[ns, UTC]      ← Original publication timestamp
├── headline: str                  ← Source text (retained for debugging)
├── sentiment_label: str           ← Argmax class label
├── sentiment_score: float64       ← Confidence of predicted class
├── sentiment_positive: float64    ← P(positive)  ┐
├── sentiment_negative: float64    ← P(negative)  ├─ Full softmax triple
├── sentiment_neutral: float64     ← P(neutral)   ┘
└── sentiment_value: float64       ← P(pos) - P(neg), the signed polarity
```

Each row is a **single headline** scored by FinBERT. Multiple rows may share the same timestamp (multiple headlines published simultaneously). The `date` column preserves the original UTC publication time to sub-minute precision. During aggregation, `date` is normalised to the corresponding business day (`trading_date`) for alignment with price data.

---

## 3.3 Dynamic Model

### UML Sequence Diagram — Full Pipeline

```mermaid
sequenceDiagram
    actor User as User / Script
    participant Exp as Experiment Runner
    participant PC as PriceCache
    participant YF as yfinance API
    participant SC as Sentiment Cache
    participant FB as FinBERT (GPU)
    participant TI as Indicator Engine V7
    participant WF as Walk-Forward Loop
    participant Model as HybridLSTM (GPU)

    User->>Exp: python regression_v3_hybrid_branch.py

    Note over Exp: Phase 1 — Load Sentiment Data
    Exp->>SC: glob("*_sentiment.parquet")
    SC-->>Exp: {ticker: DataFrame} for ~200 tickers
    Exp->>Exp: Exclude noisy tickers (90 excluded)
    Exp->>Exp: Rank by density, take top 150

    Note over Exp: Phase 2 — Load Price Data
    Exp->>PC: load(start, end, tickers)
    alt Cache Hit
        PC-->>Exp: Dict[str, DataFrame] from pickle
    else Cache Miss
        PC->>YF: yf.download(tickers, start, end)
        YF-->>PC: Raw OHLCV DataFrames
        PC->>PC: save() → prices_{start}_{end}.pkl
        PC-->>Exp: Dict[str, DataFrame]
    end

    Note over Exp: Phase 3 — Pre-compute Features (once, reused across folds)
    loop For each ticker
        Exp->>TI: compute_shap_top20(ohlcv_df)
        TI-->>Exp: DataFrame with 20 technical columns

        Exp->>SC: lookup sentiment for ticker
        alt Sentiment Available
            SC-->>Exp: sent_count_log, sent_strength (with decay fill)
        else No Sentiment
            Exp->>Exp: Fill sentiment columns with 0.0
        end

        Exp->>Exp: Compute vol_adj_return target
        Exp->>Exp: Store (features, targets, dates) per ticker
    end

    Note over Exp: Phase 4 — Walk-Forward Training Loop
    loop For each quarterly fold (12 folds)
        Exp->>WF: Define train=[2012, fold_start), val=[fold_start, fold_end)
        WF->>WF: Build sliding-window sequences (L=20)
        WF->>WF: Pool all tickers' sequences
        WF->>WF: Fit StandardScaler on train only
        WF->>WF: Transform train + val

        WF->>Model: Create FRESH HybridLSTM
        Note over Model: Tech branch: LSTM(128×2) + Attn(4h)<br/>Sent branch: GRU(32×1)<br/>Fusion: 160→64→32→1

        loop For each epoch (up to 100)
            Model->>Model: train_epoch() with FP16 + Huber loss
            Model->>Model: validate() → val_loss
            alt val_loss improved
                Model->>Model: Save best checkpoint
            else No improvement for 15 epochs
                Model->>Model: Early stop, restore best
            end
        end

        WF->>WF: Evaluate: MSE, IC, Dir. Accuracy
        WF->>WF: Store per-stock predictions for CS-IC
    end

    Note over Exp: Phase 5 — Aggregate & Save
    Exp->>Exp: Concatenate all fold predictions
    Exp->>Exp: Compute aggregate metrics + cross-sectional IC
    Exp->>Exp: Save summary.csv, per_fold_results.csv, model.pt
    Exp-->>User: Results in reports/{experiment_name}_{timestamp}/
```

### Statechart — Lifecycle of a Single Data Point

This diagram traces a single news headline from raw text to its final role as a component in a model feature vector.

```mermaid
stateDiagram-v2
    [*] --> RawHeadline: FNSPID CSV ingested

    state "Raw Headline" as RawHeadline {
        note right of RawHeadline
            Fields: date (UTC), headline (str), ticker (str)
            Storage: {TICKER}_news.parquet
        end note
    }

    RawHeadline --> Tokenised: WordPiece tokeniser (max 512 tokens)

    state "Tokenised Sequence" as Tokenised {
        note right of Tokenised
            input_ids, attention_mask
            On GPU (CUDA)
        end note
    }

    Tokenised --> ScoredHeadline: FinBERT forward pass (GPU)

    state "Scored Headline" as ScoredHeadline {
        note right of ScoredHeadline
            + sentiment_label, sentiment_score
            + sentiment_positive/negative/neutral
            + sentiment_value = P(pos) - P(neg)
            Storage: {TICKER}_sentiment.parquet
        end note
    }

    ScoredHeadline --> DailyAggregated: Group by trading_date, compute mean/count

    state "Daily Aggregated" as DailyAggregated {
        note right of DailyAggregated
            sent_mean, sent_count
            sent_count_log = log1p(count)
            sent_strength = mean × count_log
        end note
    }

    DailyAggregated --> DecayFilled: Forward-fill gaps with 0.95^d decay

    state "Decay-Filled Signal" as DecayFilled {
        note right of DecayFilled
            Complete business-day index
            No NaN gaps remaining
            Sentiment fades gradually
        end note
    }

    DecayFilled --> FeatureSelected: RF permutation importance selects top 2

    state "Selected Features" as FeatureSelected {
        note right of FeatureSelected
            sent_count_log (perm rank 1)
            sent_strength  (perm rank 2)
        end note
    }

    FeatureSelected --> MergedVector: Concatenate with 20 SHAP tech features

    state "22-D Feature Vector" as MergedVector {
        note right of MergedVector
            [20 technical | 2 sentiment]
            Per trading day, per ticker
        end note
    }

    MergedVector --> Scaled: StandardScaler (z-score, per-fold)

    state "Normalised Vector" as Scaled {
        note right of Scaled
            Zero mean, unit variance
            Fitted on train data only
        end note
    }

    Scaled --> Sequenced: Sliding window (L=20 days)

    state "Input Sequence" as Sequenced {
        note right of Sequenced
            Shape: (20, 22) = (seq_len, features)
            Target: vol_adj_return at day t+L
        end note
    }

    Sequenced --> ModelInput: Batched (batch_size=1024)

    state "Model Input Tensor" as ModelInput {
        note right of ModelInput
            Shape: (B, 20, 22) float32
            Split: x[:,:,:20] → Tech branch
                   x[:,:,20:] → Sent branch
        end note
    }

    ModelInput --> [*]: Fed to HybridLSTM for training/inference
```

---

