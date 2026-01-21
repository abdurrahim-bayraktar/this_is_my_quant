# Data Pipeline Documentation

> Complete documentation for data loading, preprocessing, and feature engineering.

---

## Table of Contents

1. [Overview](#overview)
2. [Data Sources](#data-sources)
3. [Dataset Loader](#dataset-loader)
4. [Preprocessing](#preprocessing)
5. [Feature Engineering](#feature-engineering)
6. [Sentiment Extraction](#sentiment-extraction)
7. [Sequence Creation](#sequence-creation)

---

## Overview

The data pipeline transforms raw financial data into model-ready sequences:

```
Raw Data                     Processed                    Model Input
─────────────────────────────────────────────────────────────────────
News Headlines  ─────→  Sentiment Scores  ─────┐
                                               ├───→  Feature Matrix  ───→  LSTM
Price Data     ─────→  Technical Indicators ──┘      [batch, 20, 25]
```

---

## Data Sources

### 1. FNSPID (Primary)

**Financial News and Stock Price Integration Dataset**

| Property | Value |
|----------|-------|
| Source | [Zihan1004/FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID) |
| Size | 5.73 GB |
| News Records | 15.7 million |
| Price Records | 29.7 million |
| Companies | 4,775 S&P 500 stocks |
| Time Range | 1999-2023 |

**Columns**:
```
news: date, ticker, headline, content, source
prices: date, ticker, open, high, low, close, volume, adj_close
```

### 2. Twitter Financial Sentiment

**Pre-labeled financial tweets for sentiment validation**

| Property | Value |
|----------|-------|
| Source | [zeroshot/twitter-financial-news-sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) |
| Size | ~50 MB |
| Samples | 9,543 |
| Labels | Bearish (0), Bullish (1), Neutral (2) |

**Example**:
```json
{
  "text": "$BYND - JPMorgan reels in expectations on Beyond Meat",
  "label": 0
}
```

### 3. Yahoo Finance (Fallback)

**Real-time price data via yfinance API**

```python
import yfinance as yf
stock = yf.Ticker("AAPL")
hist = stock.history(start="2016-01-01", end="2024-12-31")
```

---

## Dataset Loader

### Class: `DatasetLoader`

```python
from src.data import DatasetLoader

loader = DatasetLoader(
    tickers=["AAPL", "MSFT", "GOOGL"],  # Stock symbols
    date_range=("2016-01-01", "2024-12-31"),
    use_cache=True,  # Cache to parquet files
)
```

### Methods

#### `load_fnspid()`

```python
news_df, prices_df = loader.load_fnspid()
```

**Returns**:
- `news_df`: DataFrame with headlines, tickers, dates
- `prices_df`: DataFrame with OHLCV data

**Behavior**:
1. Check cache (`data/cache/*.parquet`)
2. If not cached, load from HuggingFace
3. Filter by tickers and date range
4. Cache for future use

#### `load_twitter_sentiment()`

```python
twitter_data = loader.load_twitter_sentiment()
```

**Returns**: HuggingFace Dataset with `text` and `label` columns

#### `_load_prices_yfinance()`

```python
prices_df = loader._load_prices_yfinance()
```

**Fallback method** when FNSPID prices unavailable.

---

## Preprocessing

### Class: `DataPreprocessor`

```python
from src.data import DataPreprocessor

preprocessor = DataPreprocessor(
    trend_threshold=0.005,  # ±0.5%
    sequence_length=20,
    normalization="zscore",
)
```

### Trend Labeling

**Threshold**: ±0.5% daily return

```python
def label_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Labels:
      0 = Down  (return < -0.5%)
      1 = Neutral (-0.5% ≤ return ≤ +0.5%)
      2 = Up (return > +0.5%)
    """
    df['return'] = df['close'].pct_change()
    
    conditions = [
        df['return'] < -0.005,  # Down
        df['return'] > 0.005,   # Up
    ]
    choices = [0, 2]
    df['trend'] = np.select(conditions, choices, default=1)
    
    return df
```

**Class Distribution** (typical):
- Down: ~28%
- Neutral: ~44%
- Up: ~28%

### Return Calculation

```python
def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    # Simple return
    df['return'] = df['close'].pct_change()
    
    # Log return (more stable for compounding)
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    return df
```

### Normalization

**Z-Score Normalization** (per stock, rolling window):

```python
def normalize_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for col in columns:
        rolling_mean = df.groupby('ticker')[col].transform(
            lambda x: x.rolling(60, min_periods=1).mean()
        )
        rolling_std = df.groupby('ticker')[col].transform(
            lambda x: x.rolling(60, min_periods=1).std()
        )
        df[col] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
    
    return df
```

**Why rolling window?**
- Financial data is non-stationary
- Rolling normalization adapts to regime changes
- 60-day window captures ~3 months of behavior

### News-to-Trading-Day Alignment

```python
def align_news_to_trading_day(news_df: pd.DataFrame, prices_df: pd.DataFrame):
    """
    Aligns news to the trading day they can affect.
    
    Rules:
    - Pre-market news (before 9:30 AM) → Same day
    - Market hours news → Same day
    - After-hours news (after 4:00 PM) → Next trading day
    - Weekend/holiday news → Next trading day
    """
```

---

## Feature Engineering

### Class: `FeatureEngineer`

```python
from src.data import FeatureEngineer

engineer = FeatureEngineer()
```

### Technical Indicators

All indicators computed using the `ta` library:

```python
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    import ta
    
    # Momentum
    df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['mfi_14'] = ta.volume.MFIIndicator(
        df['high'], df['low'], df['close'], df['volume'], window=14
    ).money_flow_index()
    
    # Trend
    macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    
    # Volatility
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    df['atr_14'] = ta.volatility.AverageTrueRange(
        df['high'], df['low'], df['close'], window=14
    ).average_true_range()
    
    # Moving Averages
    df['sma_5'] = ta.trend.SMAIndicator(df['close'], window=5).sma_indicator()
    df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
    df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
    df['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
    
    return df
```

### Feature Column List

```python
def get_feature_columns() -> List[str]:
    return [
        # Price
        'open', 'high', 'low', 'close', 'volume',
        # Returns
        'return', 'log_return',
        # Momentum
        'rsi_14', 'mfi_14',
        # Trend
        'macd', 'macd_signal', 'macd_hist',
        # Volatility
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'atr_14',
        # Moving Averages
        'sma_5', 'sma_20', 'ema_12', 'ema_26',
        # Sentiment
        'sentiment_mean', 'sentiment_std', 'news_count', 'sentiment_momentum',
    ]
```

---

## Sentiment Extraction

### Class: `SentimentExtractor`

```python
from src.nlp import SentimentExtractor

extractor = SentimentExtractor(
    model_name="ProsusAI/finbert",
    device="cuda",
    max_length=512,
    batch_size=16,
)
```

### FinBERT Model

**ProsusAI/finbert** is a BERT model fine-tuned on financial text:

| Property | Value |
|----------|-------|
| Base Model | BERT-base |
| Parameters | 110M |
| Classes | Positive, Negative, Neutral |
| Training Data | Financial PhraseBank + Reuters |

### Single Text Extraction

```python
result = extractor.extract("Apple reports record quarterly revenue")

# SentimentResult(
#     label='positive',
#     score=0.944,
#     positive=0.944,
#     negative=0.032,
#     neutral=0.024,
# )
```

### Batch Extraction

```python
results = extractor.extract_batch([
    "Apple reports record revenue",
    "Tesla faces investigation",
    "Fed holds rates steady",
])
# Returns list of SentimentResult objects
```

### DataFrame Extraction

```python
news_df = extractor.extract_dataframe(
    df=news_df,
    text_col='headline',
    cache_key='aapl_news',  # Cache results to disk
)
# Adds columns: sentiment_positive, sentiment_negative, sentiment_neutral, sentiment_label
```

### Embedding Extraction

For advanced models requiring embeddings:

```python
embeddings = extractor.extract_embeddings(
    texts=["Apple reports record revenue"],
    pooling="mean",  # or "cls"
)
# Shape: [n_texts, 768]
```

---

## Sentiment Aggregation

### Class: `SentimentAggregator`

```python
from src.nlp import SentimentAggregator

aggregator = SentimentAggregator(strategy='time_weighted')
```

### Aggregation Strategies

#### 1. Simple Average

```python
daily_sentiment = news_df.groupby(['date', 'ticker']).agg({
    'sentiment_positive': 'mean',
    'sentiment_negative': 'mean',
    'sentiment_neutral': 'mean',
    'headline': 'count',  # news_count
})
```

#### 2. Time-Weighted Average

More recent news has higher weight:

```python
def time_weighted_aggregate(group):
    hours = group['hours_to_close']  # Hours until market close
    weights = np.exp(-hours / 12)  # Decay factor
    weights /= weights.sum()  # Normalize
    
    return (group['sentiment_value'] * weights).sum()
```

#### 3. Volume-Weighted Average

Weight by article importance (if available):

```python
def volume_weighted_aggregate(group):
    weights = group['article_importance']  # e.g., word count, source rank
    return (group['sentiment_value'] * weights).sum() / weights.sum()
```

### Sentiment Momentum

3-day change in sentiment:

```python
df['sentiment_momentum'] = df.groupby('ticker')['sentiment_mean'].diff(3)
```

---

## Sequence Creation

### Creating LSTM Input Sequences

```python
def create_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'trend',
    sequence_length: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates sequences for LSTM input.
    
    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names
        target_col: Target column name
        sequence_length: Number of timesteps per sequence
    
    Returns:
        X: [n_samples, sequence_length, n_features]
        y: [n_samples]
        tickers: [n_samples] - ticker for each sequence
    """
    X, y, tickers = [], [], []
    
    for ticker, group in df.groupby('ticker'):
        group = group.sort_values('date')
        features = group[feature_cols].values
        targets = group[target_col].values
        
        for i in range(len(group) - sequence_length):
            X.append(features[i:i+sequence_length])
            y.append(targets[i+sequence_length])  # Predict next day
            tickers.append(ticker)
    
    return np.array(X), np.array(y), np.array(tickers)
```

### Sequence Visualization

```
Day:    1   2   3   4   5  ...  18  19  20  │  21
        ├───────────────────────────────────┤
        │     Input Sequence (X)            │  Target (y)
        │     [20 days × 25 features]       │  [trend]
```

### Data Split (Time-Series Aware)

```python
# Maintain temporal order - NO shuffling across time
train_end = int(0.7 * n_samples)    # 70% training
val_end = int(0.85 * n_samples)     # 15% validation
# Remaining 15% for testing

X_train = X[:train_end]
X_val = X[train_end:val_end]
X_test = X[val_end:]
```

---

## Caching Strategy

### Parquet Files

All intermediate data is cached as parquet files:

```
data/cache/
├── prices_50stocks_2016-01-01_2024-12-31.parquet
├── fnspid_50stocks_2016-01-01_2024-12-31.news.parquet
├── fnspid_50stocks_2016-01-01_2024-12-31.prices.parquet
├── sentiment_aapl_news.parquet
└── sentiment_msft_news.parquet
```

### Benefits

1. **Speed**: Avoids re-downloading/re-processing
2. **Compression**: Parquet is columnar and compressed
3. **Compatibility**: Works with pandas, polars, spark

### Cache Invalidation

```python
# Force reload by deleting cache
import shutil
shutil.rmtree('data/cache')

# Or disable caching
loader = DatasetLoader(use_cache=False)
```

---

## Configuration

All data pipeline settings are in `config/settings.py`:

```python
@dataclass
class DatasetConfig:
    # Dataset sources
    fnspid_repo: str = "Zihan1004/FNSPID"
    twitter_sentiment_repo: str = "zeroshot/twitter-financial-news-sentiment"
    
    # Time splits
    train_end_date: str = "2022-12-31"
    val_end_date: str = "2023-06-30"
    # Test: 2023-07-01 to 2024-12-31
    
    # Caching
    use_cache: bool = True
    cache_sentiment: bool = True

@dataclass
class FeatureConfig:
    # Price features
    price_features: List[str] = ["open", "high", "low", "close", "volume"]
    
    # Technical indicators
    technical_indicators: List[str] = [
        "rsi_14", "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_middle", "bb_lower", "bb_width",
        "sma_5", "sma_20", "ema_12", "ema_26", "atr_14",
    ]
    
    # Sentiment aggregation
    sentiment_aggregation: str = "time_weighted"
    
    # Normalization
    normalization: str = "zscore"

@dataclass
class TrendConfig:
    up_threshold: float = 0.005    # > 0.5% = Up
    down_threshold: float = -0.005  # < -0.5% = Down
```

---

## Troubleshooting

### 1. FNSPID Download Fails

```
Error: Dataset 'Zdong104/FNSPID_Financial_News_Dataset' doesn't exist
```

**Solution**: Use correct repo `Zihan1004/FNSPID`

### 2. HuggingFace Rate Limiting

```
Error: Too many requests
```

**Solution**: Use caching, set `use_cache=True`

### 3. Missing Technical Indicators

```
KeyError: 'rsi_14'
```

**Solution**: Ensure `ta` library is installed: `pip install ta`

### 4. NaN in Features

**Cause**: Technical indicators need warmup period
**Solution**: Drop first N rows where N = max lookback period (e.g., 50)

```python
df = df.dropna(subset=feature_cols)
```
