"""
Comprehensive Technical Indicators Module.

This module provides a centralized, reusable class for computing 70+ technical
indicators using pandas_ta. All experiment scripts should import from here
instead of redefining indicators.

Usage:
    from src.features.indicators import ComprehensiveIndicators
    
    indicators = ComprehensiveIndicators()
    df = indicators.compute_all(df)  # Returns df with all indicators added
    feature_cols = indicators.get_indicator_columns(df)  # Get list of indicator column names
"""

import numpy as np
import pandas as pd
from typing import List
import logging

logger = logging.getLogger(__name__)


class ComprehensiveIndicators:
    """
    Compute 70+ technical indicators using pandas_ta.
    
    Indicator Categories:
    - Moving Averages (SMA, EMA, VWMA, KAMA, Ichimoku)
    - Momentum Oscillators (RSI, Stochastic, MACD, Williams %R, CCI, ROC, PPO, TRIX, Aroon, Coppock, KST)
    - Volume Indicators (MFI, BOP, PVO, OBV)
    - Volatility Indicators (Bollinger Bands, ATR, True Range, Choppiness)
    - Trend Indicators (ADX, Bull/Bear Power)
    - Price Features (Returns, Gap, Intraday Range)
    
    Attributes:
        ta_available (bool): Whether pandas_ta is installed
        
    Example:
        >>> from src.features.indicators import ComprehensiveIndicators
        >>> import yfinance as yf
        >>> df = yf.download("AAPL", start="2020-01-01", end="2023-12-31")
        >>> indicators = ComprehensiveIndicators()
        >>> df = indicators.compute_all(df)
        >>> print(f"Added {len(indicators.get_indicator_columns(df))} indicators")
    """
    
    def __init__(self):
        """Initialize the indicator computer."""
        try:
            import pandas_ta as ta
            self.ta = ta
            self.ta_available = True
        except ImportError:
            logger.warning("pandas_ta not installed. Run: pip install pandas_ta")
            self.ta_available = False
    
    def compute_all(self, df: pd.DataFrame, exclude: List[str] = None) -> pd.DataFrame:
        """
        Compute all technical indicators.
        
        Args:
            df: DataFrame with OHLCV columns (Open, High, Low, Close, Volume).
                Accepts both capitalized and lowercase column names.
        
        Returns:
            DataFrame with all indicators added as new columns.
            Original OHLCV columns are preserved.
        """
        exclude = exclude or []
        df = df.copy()
        
        # Normalize column names
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required:
            if col not in df.columns:
                if col.lower() in df.columns:
                    df[col] = df[col.lower()]
                elif col.capitalize() in df.columns:
                    df[col] = df[col.capitalize()]
        
        if not self.ta_available:
            return self._compute_basic_indicators(df)
        
        # ===== MOVING AVERAGES & TREND =====
        df['sma_5'] = self.ta.sma(df['Close'], length=5)
        df['sma_10'] = self.ta.sma(df['Close'], length=10)
        df['sma_20'] = self.ta.sma(df['Close'], length=20)
        df['sma_50'] = self.ta.sma(df['Close'], length=50)
        
        df['ema_5'] = self.ta.ema(df['Close'], length=5)
        df['ema_12'] = self.ta.ema(df['Close'], length=12)
        df['ema_26'] = self.ta.ema(df['Close'], length=26)
        
        df['vwma_20'] = self.ta.vwma(df['Close'], df['Volume'], length=20)
        df['kama'] = self.ta.kama(df['Close'], length=10)
        
        # Ichimoku Cloud
        ichimoku = self.ta.ichimoku(df['High'], df['Low'], df['Close'])
        if ichimoku is not None and len(ichimoku) > 0 and ichimoku[0] is not None:
            for col in ichimoku[0].columns:
                col_name = f'ichimoku_{col}'
                if col_name not in exclude:
                    df[col_name] = ichimoku[0][col]
        
        # ===== MOMENTUM OSCILLATORS =====
        df['rsi_14'] = self.ta.rsi(df['Close'], length=14)
        df['rsi_7'] = self.ta.rsi(df['Close'], length=7)
        
        stoch = self.ta.stoch(df['High'], df['Low'], df['Close'])
        if stoch is not None:
            for col in stoch.columns:
                df[f'stoch_{col}'] = stoch[col]
        
        stochrsi = self.ta.stochrsi(df['Close'])
        if stochrsi is not None:
            for col in stochrsi.columns:
                df[f'stochrsi_{col}'] = stochrsi[col]
        
        macd = self.ta.macd(df['Close'])
        if macd is not None:
            for col in macd.columns:
                df[f'macd_{col}'] = macd[col]
        
        df['willr'] = self.ta.willr(df['High'], df['Low'], df['Close'])
        df['cci'] = self.ta.cci(df['High'], df['Low'], df['Close'])
        
        df['roc_10'] = self.ta.roc(df['Close'], length=10)
        df['roc_20'] = self.ta.roc(df['Close'], length=20)
        
        ppo = self.ta.ppo(df['Close'])
        if ppo is not None:
            if isinstance(ppo, pd.DataFrame):
                for col in ppo.columns:
                    df[f'ppo_{col}'] = ppo[col]
            else:
                df['ppo'] = ppo
        
        trix = self.ta.trix(df['Close'])
        if trix is not None:
            if isinstance(trix, pd.DataFrame):
                for col in trix.columns:
                    df[f'trix_{col}'] = trix[col]
            else:
                df['trix'] = trix
        
        df['ao'] = self.ta.ao(df['High'], df['Low'])
        
        aroon = self.ta.aroon(df['High'], df['Low'])
        if aroon is not None:
            for col in aroon.columns:
                df[f'aroon_{col}'] = aroon[col]
        
        df['coppock'] = self.ta.coppock(df['Close'])
        
        kst = self.ta.kst(df['Close'])
        if kst is not None:
            for col in kst.columns:
                df[f'kst_{col}'] = kst[col]
        
        df['psl'] = (df['Close'] > df['Close'].shift(1)).rolling(12).mean() * 100
        
        # ===== VOLUME INDICATORS =====
        df['mfi'] = self.ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'])
        df['bop'] = self.ta.bop(df['Open'], df['High'], df['Low'], df['Close'])
        
        pvo = self.ta.pvo(df['Volume'])
        if pvo is not None:
            if isinstance(pvo, pd.DataFrame):
                for col in pvo.columns:
                    df[f'pvo_{col}'] = pvo[col]
            else:
                df['pvo'] = pvo
        
        df['obv'] = self.ta.obv(df['Close'], df['Volume'])
        
        df['volume_sma_20'] = self.ta.sma(df['Volume'], length=20)
        df['volume_ratio'] = df['Volume'] / df['volume_sma_20'].replace(0, np.nan)
        
        df['rvgi'] = (df['Close'] - df['Open']) / (df['High'] - df['Low']).replace(0, np.nan)
        
        # ===== VOLATILITY INDICATORS =====
        bbands = self.ta.bbands(df['Close'])
        if bbands is not None:
            for col in bbands.columns:
                df[f'bb_{col}'] = bbands[col]
        
        df['atr'] = self.ta.atr(df['High'], df['Low'], df['Close'])
        df['atr_pct'] = df['atr'] / df['Close'] * 100
        
        df['true_range'] = self.ta.true_range(df['High'], df['Low'], df['Close'])
        df['mstd_20'] = self.ta.stdev(df['Close'], length=20)
        df['chop'] = self.ta.chop(df['High'], df['Low'], df['Close'])
        
        # ===== TREND STRENGTH =====
        adx = self.ta.adx(df['High'], df['Low'], df['Close'])
        if adx is not None:
            for col in adx.columns:
                df[f'adx_{col}'] = adx[col]
        
        df['bull_power'] = df['High'] - self.ta.ema(df['Close'], length=13)
        df['bear_power'] = df['Low'] - self.ta.ema(df['Close'], length=13)
        
        # ===== PRICE FEATURES =====
        df['return_1d'] = df['Close'].pct_change()
        df['return_5d'] = df['Close'].pct_change(5)
        df['return_10d'] = df['Close'].pct_change(10)
        df['return_20d'] = df['Close'].pct_change(20)
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['high_low_pct'] = (df['Close'] - df['Low']) / (df['High'] - df['Low']).replace(0, np.nan)
        df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['intraday_range'] = (df['High'] - df['Low']) / df['Open']
        
        # Drop intermediate columns
        if 'volume_sma_20' in df.columns:
            df = df.drop(columns=['volume_sma_20'])
        
        return df
    
    def _compute_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fallback when pandas_ta is not available.
        Computes a minimal set of indicators using only pandas.
        """
        logger.warning("Using basic indicators only (pandas_ta not installed)")
        
        df['sma_5'] = df['Close'].rolling(5).mean()
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['sma_50'] = df['Close'].rolling(50).mean()
        
        df['ema_12'] = df['Close'].ewm(span=12).mean()
        df['ema_26'] = df['Close'].ewm(span=26).mean()
        
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        df['bb_mid'] = df['Close'].rolling(20).mean()
        df['bb_std'] = df['Close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        df['bb_pctb'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['return_1d'] = df['Close'].pct_change()
        df['return_5d'] = df['Close'].pct_change(5)
        
        return df
    
    def get_indicator_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of indicator column names (excludes OHLCV and metadata).
        
        Args:
            df: DataFrame with indicators computed
            
        Returns:
            List of column names that are indicators
        """
        exclude = ['Open', 'High', 'Low', 'Close', 'Volume', 'Date', 'date', 
                   'Adj Close', 'Dividends', 'Stock Splits', 'ticker', 'Ticker',
                   'trend', 'return_next', 'actual_return', 'actual_trend']
        return [col for col in df.columns if col not in exclude]
    
    def get_indicator_count(self, df: pd.DataFrame) -> int:
        """Get the number of indicator columns."""
        return len(self.get_indicator_columns(df))
