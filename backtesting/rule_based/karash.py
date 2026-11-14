# indicator_engine.py
import numpy as np
import pandas as pd
import pandas_ta as ta
import torch

def knn_score(close, rsi_vals, macd_hist_vals, bb_pct_b_vals, atr_vals, k=5, lookback=100, device='cuda'):
    n = len(close)
    scores = np.zeros(n)
    
    if n < lookback + 10:
        return scores

    # 1. Convert arrays to PyTorch tensors and push to GPU
    c = torch.tensor(close, dtype=torch.float32, device=device)
    f1 = torch.tensor(rsi_vals, dtype=torch.float32, device=device) / 100.0
    f2 = torch.tensor(macd_hist_vals, dtype=torch.float32, device=device) / (torch.tensor(atr_vals, dtype=torch.float32, device=device) + 1e-10)
    f3 = torch.tensor(bb_pct_b_vals, dtype=torch.float32, device=device)

    # Stack features into a single matrix of shape (N, 3)
    features = torch.stack([f1, f2, f3], dim=1)

    # 2. Precompute all future votes at once (Vectorized)
    # This replaces the slow `if future_ret > 0 else -1` loop
    votes = torch.zeros(n, dtype=torch.float32, device=device)
    # Compare current close to close 5 bars ahead
    votes[:-5] = torch.where(c[5:] > c[:-5], 1.0, -1.0)

    # 3. Calculate distances efficiently on the GPU
    for i in range(lookback + 10, n):
        # Target feature: shape (1, 3)
        target = features[i].unsqueeze(0)
        
        # History window: shape (lookback - 10, 3)
        window_start = i - lookback
        window_end = i - 10
        window = features[window_start:window_end]

        # PyTorch's optimized pairwise distance function
        dists = torch.cdist(target, window).squeeze(0)

        # Get indices of the top K smallest distances
        # largest=False acts exactly like np.argsort()[:k] but much faster
        _, topk_idx = torch.topk(dists, k, largest=False)

        # Map the local window indices back to the global array to fetch the correct votes
        global_idx = topk_idx + window_start

        # Calculate the final kNN score and pull it back to the CPU
        knn_vote = (torch.sum(votes[global_idx]) / k) * 100
        scores[i] = knn_vote.item()

    return scores


def compute_score(df: pd.DataFrame, params: dict = None) -> dict:
    """
    df: OHLCV DataFrame (columns: open, high, low, close, volume)
    params: indikatör parametreleri (None ise default)
    Döner: {"score": float, "signal": str, "components": dict}
    """
    if params is None:
        params = {}

    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    volume = df["Volume"].values

    # Parametreler
    ema_fast_p = params.get("ema_fast", 9)
    ema_mid_p = params.get("ema_mid", 21)
    ema_slow_p = params.get("ema_slow", 50)
    ema_trend_p = params.get("ema_trend", 200)
    rsi_len_p = params.get("rsi_len", 14)
    rsi_ob = params.get("rsi_ob", 70)
    rsi_os = params.get("rsi_os", 30)
    macd_fast_p = params.get("macd_fast", 12)
    macd_slow_p = params.get("macd_slow", 26)
    macd_sig_p = params.get("macd_signal", 9)
    stoch_k_p = params.get("stoch_k", 14)
    stoch_d_p = params.get("stoch_d", 3)
    stoch_sm_p = params.get("stoch_smooth", 3)
    bb_len_p = params.get("bb_len", 20)
    bb_mult_p = params.get("bb_mult", 2.0)
    atr_len_p = params.get("atr_len", 14)
    vol_ma_len_p = params.get("vol_ma_len", 20)
    w_trend = params.get("w_trend", 25)
    w_momentum = params.get("w_momentum", 20)
    w_volatility = params.get("w_volatility", 15)
    w_volume = params.get("w_volume", 15)
    w_sr = params.get("w_sr", 10)
    w_ml = params.get("w_ml", 10)
    w_mtf = params.get("w_mtf", 5)
    strong_thresh = params.get("strong_thresh", 50)

    n = len(close)
    if n < 210:
        return {"score": 0, "signal": "BEKLE", "components": {}}

    # ── 1. TREND ──────────────────────────────────────────────────────────────
    e9 = ta.ema(df["Close"], length=ema_fast_p).fillna(0).values
    e21 = ta.ema(df["Close"], length=ema_mid_p).fillna(0).values
    e50 = ta.ema(df["Close"], length=ema_slow_p).fillna(0).values
    e200 = ta.ema(df["Close"], length=ema_trend_p).fillna(0).values
    
    st_df = ta.supertrend(df["High"], df["Low"], df["Close"], length=10, multiplier=3.0)
    st_dir = st_df[st_df.columns[1]].fillna(1).values  # default to uptrend if NaN

    ema_bull = 1.0 if (e9[-1] > e21[-1] > e50[-1] > e200[-1]) else 0.0
    ema_bear = -1.0 if (e9[-1] < e21[-1] < e50[-1] < e200[-1]) else 0.0
    ema_align = ema_bull + ema_bear
    pve200 = 1.0 if close[-1] > e200[-1] else -1.0
    pve50 = 1.0 if close[-1] > e50[-1] else -1.0
    golden = 1.0 if (e50[-1] > e200[-1] and e50[-2] <= e200[-2]) else 0.0
    death = -1.0 if (e50[-1] < e200[-1] and e50[-2] >= e200[-2]) else 0.0
    st_sig = 1.0 if st_dir[-1] == 1 else -1.0
    trend_score = ema_align * 30 + pve200 * 20 + pve50 * 15 + (golden + death) * 15 + st_sig * 20

    # ── 2. MOMENTUM ───────────────────────────────────────────────────────────
    rsi_vals = ta.rsi(df["Close"], length=rsi_len_p).fillna(50).values
    rv = rsi_vals[-1]
    if rv > rsi_ob:
        rsi_score = -((rv - rsi_ob) / (100 - rsi_ob)) * 100
    elif rv < rsi_os:
        rsi_score = ((rsi_os - rv) / rsi_os) * 100
    else:
        rsi_score = ((rv - 50) / 50) * 50

    macd_df = ta.macd(df["Close"], fast=macd_fast_p, slow=macd_slow_p, signal=macd_sig_p).fillna(0)
    macd_l = macd_df[macd_df.columns[0]].values
    macd_h = macd_df[macd_df.columns[1]].values
    macd_s = macd_df[macd_df.columns[2]].values

    bull_cross = macd_l[-1] > macd_s[-1] and macd_l[-2] <= macd_s[-2]
    bear_cross = macd_l[-1] < macd_s[-1] and macd_l[-2] >= macd_s[-2]
    if bull_cross:
        macd_score = 100.0
    elif bear_cross:
        macd_score = -100.0
    else:
        macd_score = 50.0 if macd_l[-1] > macd_s[-1] else -50.0
        macd_score += 25.0 if macd_h[-1] > macd_h[-2] else -25.0

    stoch_df = ta.stoch(df["High"], df["Low"], df["Close"], k=stoch_k_p, d=stoch_d_p, smooth_k=stoch_sm_p).fillna(50)
    stk = stoch_df[stoch_df.columns[0]].values
    std = stoch_df[stoch_df.columns[1]].values
    
    sv = stk[-1]
    stoch_score = -(sv - 80) / 20 * 100 if sv > 80 else (20 - sv) / 20 * 100 if sv < 20 else (sv - 50) / 50 * 60
    if stk[-1] < 30 and stk[-1] > stk[-2] and stk[-2] <= std[-2]:
        stoch_score = 100.0
    elif stk[-1] > 70 and stk[-1] < stk[-2] and stk[-2] >= std[-2]:
        stoch_score = -100.0

    momentum_score = rsi_score * 0.35 + macd_score * 0.40 + stoch_score * 0.25

    # ── 3. VOLATİLİTE ─────────────────────────────────────────────────────────
    bb_df = ta.bbands(df["Close"], length=bb_len_p, std=bb_mult_p).fillna(0)
    bb_lo = bb_df[bb_df.columns[0]].values
    bb_up = bb_df[bb_df.columns[2]].values
    bb_pct_b_arr = bb_df[bb_df.columns[4]].values
    
    bb_pct_b = bb_pct_b_arr[-1]
    bb_score = -80.0 if bb_pct_b > 1.0 else 80.0 if bb_pct_b < 0.0 else (bb_pct_b - 0.5) * 160
    volatility_score = bb_score

    atr_vals = ta.atr(df["High"], df["Low"], df["Close"], length=atr_len_p).fillna(0).values

    # ── 4. HACİM ──────────────────────────────────────────────────────────────
    vol_ma_v = ta.sma(df["Volume"], length=vol_ma_len_p).fillna(1).values
    vol_ratio = volume[-1] / (vol_ma_v[-1] + 1e-10)
    
    obv_vals = ta.obv(df["Close"], df["Volume"]).fillna(0).values
    obv_ma_v = ta.sma(pd.Series(obv_vals), length=vol_ma_len_p).fillna(0).values
    
    vwma_vals = ta.vwma(df["Close"], df["Volume"], length=vol_ma_len_p).fillna(0).values
    vwap = vwma_vals[-1]

    vol_confirm_bull = 40.0 if (close[-1] > close[-2] and vol_ratio > 1.2) else 0.0
    vol_confirm_bear = -40.0 if (close[-1] < close[-2] and vol_ratio > 1.2) else 0.0
    obv_trend = 30.0 if obv_vals[-1] > obv_ma_v[-1] else -30.0
    vwap_sig = 30.0 if close[-1] > vwap else -30.0
    volume_score = vol_confirm_bull + vol_confirm_bear + obv_trend + vwap_sig

    # ── 5. DESTEK/DİRENÇ ──────────────────────────────────────────────────────
    window = 10
    pivot_highs = [i for i in range(window, n - window) if high[i] == max(high[i - window:i + window + 1])]
    pivot_lows = [i for i in range(window, n - window) if low[i] == min(low[i - window:i + window + 1])]
    last_res = high[pivot_highs[-1]] if pivot_highs else None
    last_sup = low[pivot_lows[-1]] if pivot_lows else None

    if last_res and last_sup:
        d_res = (last_res - close[-1]) / close[-1] * 100
        d_sup = (close[-1] - last_sup) / close[-1] * 100
        if d_res < 1.0:
            sr_score = -60.0
        elif d_sup < 1.0:
            sr_score = 60.0
        else:
            sr_score = (d_res - d_sup) / (d_res + d_sup + 0.001) * 100
    else:
        sr_score = 0.0

    # ── 6. kNN ML ─────────────────────────────────────────────────────────────
    knn_device = params.get("device", "cuda")
    knn_scores = knn_score(close, rsi_vals, macd_h, bb_pct_b_arr, atr_vals, device=knn_device)
    ml_score = float(knn_scores[-1])

    # ── 7. MTF (günlük üzerine haftalık EMA bakışı) ────────────────────────────
    mtf_score = 0.0  # yfinance ile aynı sembolden haftalık veri çekilecek (bot.py'de)

    # ── BİRLEŞİK SKOR ─────────────────────────────────────────────────────────
    total_w = w_trend + w_momentum + w_volatility + w_volume + w_sr + w_ml + w_mtf
    total_w = total_w if total_w > 0 else 100
    mc_raw = (
        trend_score * w_trend
        + momentum_score * w_momentum
        + volatility_score * w_volatility
        + volume_score * w_volume
        + sr_score * w_sr
        + ml_score * w_ml
        + mtf_score * w_mtf
    ) / total_w

    # DEMA MACD
    d_fast = ta.dema(df["Close"], length=12).fillna(0)
    d_slow = ta.dema(df["Close"], length=26).fillna(0)
    dema_line = d_fast - d_slow
    dema_sig = ta.dema(dema_line, length=9).fillna(0)
    dema_hist = dema_line - dema_sig
    
    dema_l = dema_line.values
    dema_s = dema_sig.values
    dema_h = dema_hist.values

    dema_bull_x = dema_l[-1] > dema_s[-1] and dema_l[-2] <= dema_s[-2]
    dema_bear_x = dema_l[-1] < dema_s[-1] and dema_l[-2] >= dema_s[-2]
    if dema_bull_x:
        dema_score = 100.0
    elif dema_bear_x:
        dema_score = -100.0
    else:
        dema_score = 50.0 if dema_l[-1] > dema_s[-1] else -50.0
        dema_score += 25.0 if dema_h[-1] > dema_h[-2] else -25.0

    htf_dema_bull = dema_l[-1] > dema_s[-1]  # simplification; bot.py injects weekly

    prediction_score = mc_raw * 0.55 + dema_score * 0.25 + 0.0 * 0.10 + (40.0 if htf_dema_bull else -40.0) * 0.10
    prediction_score = max(-100.0, min(100.0, prediction_score))

    # Smooth (3-bar EMA emülasyonu — tek bar; yeterli)
    smooth = prediction_score

    if smooth >= strong_thresh:
        signal = "AL"
    elif smooth <= -strong_thresh:
        signal = "SAT"
    else:
        signal = "BEKLE"

    return {
        "score": round(smooth, 2),
        "signal": signal,
        "components": {
            "trend": round(trend_score, 1),
            "momentum": round(momentum_score, 1),
            "volatility": round(volatility_score, 1),
            "volume": round(volume_score, 1),
            "sr": round(sr_score, 1),
            "ml": round(ml_score, 1),
            "dema": round(dema_score, 1),
            "rsi": round(rv, 1),
            "atr": round(float(atr_vals[-1]), 4),
        },
    }