"""
Diagnose whether a Long_Top_N result comes from genuine selection/timing skill
or from simply holding the most volatile names in the universe.

Works on any predictions.csv with columns:
    Date, Ticker, Prob_Up, Prob_Down, Return_Next

Tests:
  1. Control strategies (same simulator, same costs):
       - Model Top-N (by Prob_Up)
       - Vol Top-N         : top N by trailing realized vol (no model)
       - Random-from-VolTopK: N random names from the K most volatile, many seeds
       - Random-from-ModelTopK: N random names from model's top K by Prob_Up, many seeds
     -> If Vol Top-N ~ Model Top-N, the edge is volatility exposure, not skill.
     -> If Model beats Random-from-VolTopK, the model times *within* the high-vol pool.
  2. Rank correlations (daily Spearman, averaged):
       Prob_Up vs vol, Prob_Down vs vol, signal IC overall and within high-vol pool.
  3. Timing test: for frequently picked tickers, next-day return on picked vs not-picked days.
  4. Robustness: drop top contributors / top days.

All features use only data known at the close of day t (trailing vol uses returns up to t).

Usage:
    python experiments/diagnose_topn_skill.py --eval-dir reports/bt_... [--n 3] [--k 20] [--seeds 500]
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

COST_BPS = 5.0  # matches Backtester default


# ----------------------------------------------------------------------------
# Data prep
# ----------------------------------------------------------------------------

def load_wide(eval_dir: Path, vol_lookback: int):
    df = pd.read_csv(eval_dir / "predictions.csv", parse_dates=["Date"])
    df = df.sort_values(["Ticker", "Date"])

    # Return realized at close of t (known at t) = previous row's Return_Next
    df["Ret_Today"] = df.groupby("Ticker")["Return_Next"].shift(1)
    df["Trail_Vol"] = (df.groupby("Ticker")["Ret_Today"]
                         .transform(lambda s: s.rolling(vol_lookback, min_periods=vol_lookback // 2).std()))

    wide = {c: df.pivot(index="Date", columns="Ticker", values=c)
            for c in ["Return_Next", "Prob_Up", "Prob_Down", "Trail_Vol"]}
    # Align
    idx, cols = wide["Return_Next"].index, wide["Return_Next"].columns
    for k in wide:
        wide[k] = wide[k].reindex(index=idx, columns=cols)
    return df, wide


# ----------------------------------------------------------------------------
# Fast equal-weight simulator (daily rebalance, turnover costs, no drift)
# ----------------------------------------------------------------------------

def select_top(score: np.ndarray, n: int, valid: np.ndarray) -> np.ndarray:
    """Boolean mask (D x T) of top-n by score per row, among valid entries."""
    s = np.where(valid, score, -np.inf)
    idx = np.argsort(-s, axis=1)[:, :n]
    mask = np.zeros_like(valid, dtype=bool)
    rows = np.arange(s.shape[0])[:, None]
    mask[rows, idx] = True
    return mask & valid


def select_random_from_topk(score: np.ndarray, n: int, k: int, valid: np.ndarray, rng) -> np.ndarray:
    s = np.where(valid, score, -np.inf)
    topk = np.argsort(-s, axis=1)[:, :k]                       # D x k
    pick = np.argsort(rng.random(topk.shape), axis=1)[:, :n]   # D x n positions inside top-k
    chosen = np.take_along_axis(topk, pick, axis=1)
    mask = np.zeros_like(valid, dtype=bool)
    mask[np.arange(s.shape[0])[:, None], chosen] = True
    return mask & valid


def simulate(mask: np.ndarray, R: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    cnt = mask.sum(axis=1, keepdims=True)
    W = np.divide(mask, cnt, out=np.zeros(mask.shape), where=cnt > 0)
    gross = np.nansum(W * np.nan_to_num(R), axis=1)
    turnover = 0.5 * np.abs(np.diff(np.vstack([np.zeros(W.shape[1]), W]), axis=0)).sum(axis=1)
    ret = gross - turnover * COST_BPS / 1e4
    return metrics(pd.Series(ret, index=dates))


def metrics(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    total = eq.iloc[-1] - 1
    years = (r.index[-1] - r.index[0]).days / 365.25
    ann = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else np.nan
    vol = r.std() * np.sqrt(252)
    return {
        "Total": total,
        "Ann": ann,
        "Vol": vol,
        "Sharpe": ann / vol if vol > 0 else np.nan,   # same definition as Backtester
        "MaxDD": (eq / eq.cummax() - 1).min(),
        "_series": r,
    }


# ----------------------------------------------------------------------------
# Rank statistics
# ----------------------------------------------------------------------------

def daily_spearman(a: pd.DataFrame, b: pd.DataFrame, pool: pd.DataFrame = None) -> pd.Series:
    if pool is not None:
        a, b = a.where(pool), b.where(pool)
    both = a.notna() & b.notna()
    ra = a.where(both).rank(axis=1)
    rb = b.where(both).rank(axis=1)
    ra = ra.sub(ra.mean(axis=1), axis=0)
    rb = rb.sub(rb.mean(axis=1), axis=0)
    num = (ra * rb).sum(axis=1)
    den = np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1))
    return (num / den).where(both.sum(axis=1) >= 5)


def tstat(s: pd.Series) -> float:
    s = s.dropna()
    return s.mean() / (s.std() / np.sqrt(len(s))) if len(s) > 1 and s.std() > 0 else np.nan


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--n", type=int, default=3, help="Top-N size (match the strategy)")
    ap.add_argument("--k", type=int, default=20, help="Pool size for random-from-top-K controls")
    ap.add_argument("--seeds", type=int, default=500)
    ap.add_argument("--vol-lookback", type=int, default=20)
    ap.add_argument("--drop-top", type=int, default=3, help="Exclude this many top-contributing tickers")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    df, w = load_wide(eval_dir, args.vol_lookback)
    dates = w["Return_Next"].index
    R = w["Return_Next"].values
    PU = w["Prob_Up"].values
    VOL = w["Trail_Vol"].values
    valid = ~np.isnan(R) & ~np.isnan(PU)
    valid_vol = valid & ~np.isnan(VOL)
    rng = np.random.default_rng(0)
    N, K = args.n, args.k
    logger.info(f"{len(dates)} dates x {w['Return_Next'].shape[1]} tickers, N={N}, K={K}")

    # ---------------- 1. Control strategies ----------------
    rows = []
    model_mask = select_top(PU, N, valid)
    m_model = simulate(model_mask, R, dates)
    rows.append(("Model Top-N (Prob_Up)", m_model))

    # Model restricted to days/tickers where vol is defined (fair comparison to vol strategies)
    rows.append(("Model Top-N (vol-defined universe)", simulate(select_top(PU, N, valid_vol), R, dates)))
    rows.append(("Vol Top-N (no model)", simulate(select_top(VOL, N, valid_vol), R, dates)))
    rows.append(("Equal-weight universe", simulate(valid.copy(), R, dates)))

    def dist(score, v, label):
        sh = np.array([simulate(select_random_from_topk(score, N, K, v, rng), R, dates)["Sharpe"]
                       for _ in range(args.seeds)])
        tot = np.array([np.nan])  # Sharpe distribution is what we compare
        return sh

    logger.info(f"Running {args.seeds} random seeds for each pool control...")
    sh_vol = dist(VOL, valid_vol, "vol")
    sh_mod = dist(PU, valid, "model")

    res = pd.DataFrame([{**{k: v for k, v in m.items() if k != "_series"}, "Strategy": name}
                        for name, m in rows]).set_index("Strategy")

    print("\n" + "=" * 78)
    print(f"1. CONTROL STRATEGIES  (equal-weight, daily rebalance, {COST_BPS:.0f} bps cost)")
    print("=" * 78)
    print(res.to_string(float_format=lambda x: f"{x:,.3f}"))
    pct_vs_vol = (sh_vol < m_model["Sharpe"]).mean()
    pct_vs_mod = (sh_mod < m_model["Sharpe"]).mean()
    print(f"\nRandom {N} from Vol top-{K}:    Sharpe median={np.nanmedian(sh_vol):.2f} "
          f"[p5={np.nanpercentile(sh_vol, 5):.2f}, p95={np.nanpercentile(sh_vol, 95):.2f}]  "
          f"-> model beats {pct_vs_vol:.0%} of seeds")
    print(f"Random {N} from Model top-{K}:  Sharpe median={np.nanmedian(sh_mod):.2f} "
          f"[p5={np.nanpercentile(sh_mod, 5):.2f}, p95={np.nanpercentile(sh_mod, 95):.2f}]  "
          f"-> model beats {pct_vs_mod:.0%} of seeds")
    print("  (Model beating ~95%+ of seeds = ranking inside the pool carries information.)")

    # ---------------- 2. Rank correlations ----------------
    PUdf, PDdf, VOLdf, Rdf = w["Prob_Up"], w["Prob_Down"], w["Trail_Vol"], w["Return_Next"]
    spread = PUdf - PDdf
    vol_rank = VOLdf.rank(axis=1, ascending=False)
    hv_pool = vol_rank <= max(K, 50)

    c_up_vol = daily_spearman(PUdf, VOLdf)
    c_dn_vol = daily_spearman(PDdf, VOLdf)
    ic_up = daily_spearman(PUdf, Rdf)
    ic_spread = daily_spearman(spread, Rdf)
    ic_spread_hv = daily_spearman(spread, Rdf, hv_pool)
    ic_up_hv = daily_spearman(PUdf, Rdf, hv_pool)

    print("\n" + "=" * 78)
    print("2. DAILY CROSS-SECTIONAL RANK CORRELATIONS (mean, t-stat)")
    print("=" * 78)
    for name, s in [("Prob_Up   vs trailing vol", c_up_vol),
                    ("Prob_Down vs trailing vol", c_dn_vol),
                    ("IC: Prob_Up vs next return (all)", ic_up),
                    ("IC: Up-Down vs next return (all)", ic_spread),
                    (f"IC: Prob_Up vs next return (top-{max(K, 50)} vol)", ic_up_hv),
                    (f"IC: Up-Down vs next return (top-{max(K, 50)} vol)", ic_spread_hv)]:
        print(f"  {name:<45s} {s.mean():+.4f}   t={tstat(s):+.2f}")
    print("  (Both Prob_Up AND Prob_Down strongly +corr with vol = the model mostly ranks volatility.)")
    print("  (The high-vol-pool IC is the direct test of 'timing among volatile stocks'.)")

    # Overlap: how volatile are the top-Up vs top-Down names?
    dn_mask = select_top(PDdf.values, N, valid & ~np.isnan(PDdf.values))
    vr = vol_rank.values
    print(f"\n  Median vol rank of model's top-{N} Up picks:   {np.nanmedian(np.where(model_mask, vr, np.nan)):.0f}")
    print(f"  Median vol rank of model's top-{N} Down picks: {np.nanmedian(np.where(dn_mask, vr, np.nan)):.0f}")
    print(f"  (out of ~{int(np.nanmedian(np.isfinite(VOL).sum(axis=1)))} names; 1 = most volatile)")

    # ---------------- 3. Timing test ----------------
    picked = pd.DataFrame(model_mask, index=dates, columns=Rdf.columns)
    timing = []
    for t in Rdf.columns:
        p = picked[t] & Rdf[t].notna()
        q = ~picked[t] & Rdf[t].notna()
        if p.sum() >= 20 and q.sum() >= 20:
            timing.append({"Ticker": t, "Picks": int(p.sum()),
                           "Ret_Picked": Rdf[t][p].mean(), "Ret_NotPicked": Rdf[t][q].mean()})
    print("\n" + "=" * 78)
    print("3. TIMING TEST (tickers picked >=20 days): next-day return picked vs not picked")
    print("=" * 78)
    if timing:
        tdf = pd.DataFrame(timing)
        tdf["Diff"] = tdf["Ret_Picked"] - tdf["Ret_NotPicked"]
        print(tdf.sort_values("Picks", ascending=False).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
        print(f"  Pick-weighted mean diff: {np.average(tdf['Diff'], weights=tdf['Picks']):+.4%} per day")
        print("  (Positive & meaningful = the model times entries/exits in names it likes.)")
    else:
        print("  No ticker picked >=20 days.")

    # ---------------- 4. Robustness ----------------
    contrib = pd.DataFrame(np.where(model_mask, np.nan_to_num(R) / N, 0.0),
                           index=dates, columns=Rdf.columns).sum().sort_values(ascending=False)
    drop = list(contrib.index[:args.drop_top])
    keep = ~Rdf.columns.isin(drop)
    m_drop = simulate(select_top(PU, N, valid & keep[None, :]), R, dates)

    s = m_model["_series"]
    top_days = s.nlargest(5).index
    m_nodays = metrics(s.drop(top_days))

    print("\n" + "=" * 78)
    print("4. ROBUSTNESS")
    print("=" * 78)
    print(f"  Top contributors: " + ", ".join(f"{t} ({contrib[t]:+.2f})" for t in contrib.index[:5]))
    print(f"  Model Top-N:                         Total={m_model['Total']:+,.1%}  Sharpe={m_model['Sharpe']:.2f}")
    print(f"  Excluding {drop} from universe: Total={m_drop['Total']:+,.1%}  Sharpe={m_drop['Sharpe']:.2f}")
    print(f"  Removing 5 best days:                Total={m_nodays['Total']:+,.1%}  Sharpe={m_nodays['Sharpe']:.2f}")
    print(f"  5 best days: " + ", ".join(f"{d.date()} ({s[d]:+.1%})" for d in top_days))

    out = eval_dir / "topn_skill_diagnostics.csv"
    res.assign(**{}).to_csv(out)
    logger.info(f"Saved control-strategy table to {out}")


if __name__ == "__main__":
    main()
