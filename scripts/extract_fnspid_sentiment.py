"""
Bulk FinBERT Sentiment Extraction for FNSPID Dataset.

Processes the raw FNSPID CSV (~23GB) through FinBERT to produce per-ticker
sentiment parquet files in data/cache/sentiment/.

Features:
- Resume support: skips tickers that already have sentiment files
- Memory-efficient: processes one ticker at a time
- Progress logging with ETA
- Optimized for 6GB VRAM (batch_size=32)

Usage:
    # Process top 100 tickers by news count
    python scripts/extract_fnspid_sentiment.py --top 100

    # Process specific tickers
    python scripts/extract_fnspid_sentiment.py --tickers AAPL TSLA NVDA

    # Resume (re-run same command, already-done tickers are skipped)
    python scripts/extract_fnspid_sentiment.py --top 200
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import time
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent.parent / "logs" / "extract_sentiment.log"),
    ]
)
logger = logging.getLogger(__name__)

# === Paths ===
PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
SENTIMENT_DIR = CACHE_DIR / "sentiment"
COVERAGE_FILE = CACHE_DIR / "fnspid_ticker_coverage.csv"

# Known FNSPID CSV location
FNSPID_CSV = Path(
    r"C:\Users\abdurrahim\.cache\huggingface\hub"
    r"\datasets--Zihan1004--FNSPID\snapshots"
    r"\bf9189c41527198897d1af3e17b1a0095279fc45"
    r"\Stock_news\All_external.csv"
)


def get_target_tickers(args) -> list:
    """Determine which tickers to process."""
    if args.tickers:
        return args.tickers

    # Use coverage file to get top N by news count
    if not COVERAGE_FILE.exists():
        logger.error(f"Coverage file not found: {COVERAGE_FILE}")
        logger.info("Run with --tickers to specify manually, or create coverage file first.")
        sys.exit(1)

    coverage = pd.read_csv(COVERAGE_FILE)
    all_tickers = coverage['ticker'].tolist()

    top_n = args.top or 100
    return all_tickers[:top_n]


def get_already_processed() -> set:
    """Get set of tickers that already have sentiment parquets."""
    if not SENTIMENT_DIR.exists():
        return set()
    return {
        f.stem.replace("_sentiment", "")
        for f in SENTIMENT_DIR.glob("*_sentiment.parquet")
    }


def load_ticker_news(csv_path: Path, ticker: str, chunk_size: int = 100_000) -> pd.DataFrame:
    """Load all news for a single ticker from the FNSPID CSV."""
    chunks = []

    for chunk in pd.read_csv(
        csv_path,
        chunksize=chunk_size,
        on_bad_lines='skip',
        engine='c',
        low_memory=False,
    ):
        # Normalize column names
        chunk.columns = [c.lower() for c in chunk.columns]

        ticker_col = "stock_symbol" if "stock_symbol" in chunk.columns else "ticker"
        if ticker_col not in chunk.columns:
            continue

        filtered = chunk[chunk[ticker_col] == ticker].copy()
        if len(filtered) > 0:
            # Rename to standard columns
            title_col = "article_title" if "article_title" in chunk.columns else "headline"
            rename_map = {ticker_col: "ticker"}
            if title_col in filtered.columns:
                rename_map[title_col] = "headline"

            filtered = filtered.rename(columns=rename_map)

            # Keep minimum columns
            keep = ["date", "ticker", "headline"]
            filtered = filtered[[c for c in keep if c in filtered.columns]]
            chunks.append(filtered)

    if not chunks:
        return pd.DataFrame()

    news_df = pd.concat(chunks, ignore_index=True)
    news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce')
    news_df = news_df.dropna(subset=['date', 'headline'])

    # Remove duplicate headlines on same date
    news_df = news_df.drop_duplicates(subset=['date', 'headline'])

    return news_df.sort_values('date').reset_index(drop=True)


def extract_sentiment_for_ticker(
    news_df: pd.DataFrame,
    extractor,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Run FinBERT on all headlines for a ticker."""
    headlines = news_df['headline'].fillna('').tolist()

    # Process in batches
    results = extractor.batch_extract(headlines, show_progress=True)

    # Add results to DataFrame
    news_df = news_df.copy()
    news_df['sentiment_label'] = [r.label for r in results]
    news_df['sentiment_score'] = [r.score for r in results]
    news_df['sentiment_positive'] = [r.positive for r in results]
    news_df['sentiment_negative'] = [r.negative for r in results]
    news_df['sentiment_neutral'] = [r.neutral for r in results]
    news_df['sentiment_value'] = [r.positive - r.negative for r in results]

    return news_df


def process_single_ticker(
    ticker: str,
    csv_path: Path,
    extractor,
    batch_size: int = 32,
) -> bool:
    """Process a single ticker end-to-end. Returns True on success."""
    try:
        logger.info(f"Loading news for {ticker}...")
        news_df = load_ticker_news(csv_path, ticker)

        if len(news_df) == 0:
            logger.warning(f"No news found for {ticker}")
            return False

        logger.info(f"  {ticker}: {len(news_df):,} headlines to process")

        # Extract sentiment
        sentiment_df = extract_sentiment_for_ticker(news_df, extractor, batch_size)

        # Save results
        SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)

        # Save news (raw headlines + dates)
        news_path = SENTIMENT_DIR / f"{ticker}_news.parquet"
        news_df[['date', 'headline', 'ticker']].to_parquet(news_path, index=False)

        # Save sentiment (full results)
        sent_path = SENTIMENT_DIR / f"{ticker}_sentiment.parquet"
        sentiment_df.to_parquet(sent_path, index=False)

        logger.info(f"  {ticker}: Saved to {sent_path} ({len(sentiment_df):,} rows)")
        return True

    except Exception as e:
        logger.error(f"  {ticker}: FAILED — {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(description="Bulk FinBERT Sentiment Extraction")
    parser.add_argument("--top", type=int, default=100,
                        help="Process top N tickers by news count (default: 100)")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Process specific tickers")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="FinBERT batch size (default: 32 for 6GB VRAM)")
    parser.add_argument("--force", action="store_true",
                        help="Re-process even if sentiment file exists")
    parser.add_argument("--csv-path", type=str, default=None,
                        help="Path to FNSPID CSV (auto-detected if not provided)")
    args = parser.parse_args()

    # Resolve CSV path
    csv_path = Path(args.csv_path) if args.csv_path else FNSPID_CSV
    if not csv_path.exists():
        logger.error(f"FNSPID CSV not found: {csv_path}")
        logger.info("Provide path with --csv-path or ensure FNSPID is downloaded.")
        sys.exit(1)

    logger.info(f"FNSPID CSV: {csv_path}")

    # Determine targets
    target_tickers = get_target_tickers(args)
    already_done = get_already_processed() if not args.force else set()
    pending = [t for t in target_tickers if t not in already_done]

    logger.info(f"Target tickers: {len(target_tickers)}")
    logger.info(f"Already processed: {len(already_done)}")
    logger.info(f"Pending: {len(pending)}")

    if not pending:
        logger.info("Nothing to do — all tickers already processed.")
        return

    # Initialize FinBERT (load once, reuse)
    logger.info("Loading FinBERT model...")
    from src.nlp import SentimentExtractor
    extractor = SentimentExtractor(batch_size=args.batch_size)

    # Process tickers
    successes = 0
    failures = 0
    start_time = time.time()

    for i, ticker in enumerate(pending):
        elapsed = time.time() - start_time
        if successes > 0:
            avg_time = elapsed / successes
            remaining = avg_time * (len(pending) - i)
            eta = timedelta(seconds=int(remaining))
            logger.info(f"\n{'='*60}")
            logger.info(f"[{i+1}/{len(pending)}] {ticker} — ETA: {eta}")
            logger.info(f"{'='*60}")
        else:
            logger.info(f"\n{'='*60}")
            logger.info(f"[{i+1}/{len(pending)}] {ticker}")
            logger.info(f"{'='*60}")

        ticker_start = time.time()
        success = process_single_ticker(ticker, csv_path, extractor, args.batch_size)
        ticker_time = time.time() - ticker_start

        if success:
            successes += 1
            logger.info(f"  {ticker}: Done in {ticker_time:.1f}s")
        else:
            failures += 1

    # Summary
    total_time = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info("EXTRACTION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Processed: {successes + failures}")
    logger.info(f"Successes: {successes}")
    logger.info(f"Failures:  {failures}")
    logger.info(f"Total time: {timedelta(seconds=int(total_time))}")
    if successes > 0:
        logger.info(f"Avg per ticker: {total_time / successes:.1f}s")


if __name__ == "__main__":
    main()
