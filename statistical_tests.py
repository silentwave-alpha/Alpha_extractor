import numpy as np
import pandas as pd
from scipy.stats import norm


# ─────────────────────────────────────────────────────────────────────────────
# BASIC
# ─────────────────────────────────────────────────────────────────────────────

def sharpe_ratio(returns):
    """Raw Sharpe per bar (tidak annualized)."""
    returns = np.asarray(returns)
    returns = returns[~np.isnan(returns)]
    if len(returns) == 0:
        return np.nan
    return returns.mean() / (returns.std() + 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# PROBABILISTIC SHARPE RATIO (Bailey & Lopez de Prado)
#
# Mengukur probabilitas bahwa Sharpe yang diobservasi benar-benar > 0
# setelah memperhitungkan ukuran sampel.
# PSR < 0.95 → sampel terlalu kecil untuk dipercaya
# ─────────────────────────────────────────────────────────────────────────────

def probabilistic_sharpe(returns, benchmark_sr=0.0):
    """
    Prob(SR_real > benchmark_sr) berdasarkan observasi.
    Returns float 0-1. < 0.95 = tidak reliable.
    """
    returns = np.asarray(returns)
    returns = returns[~np.isnan(returns)]
    n = len(returns)
    if n < 2:
        return 0.0
    sr = sharpe_ratio(returns)
    return float(norm.cdf((sr - benchmark_sr) * np.sqrt(n - 1)))


# ─────────────────────────────────────────────────────────────────────────────
# DEFLATED SHARPE RATIO (Lopez de Prado)
#
# Koreksi multiple testing: makin banyak kombinasi ditest,
# makin tinggi Sharpe yang diperlukan agar tidak dianggap keberuntungan.
# DSR < 0.95 → hasil bisa jadi false positive dari banyak trials
# ─────────────────────────────────────────────────────────────────────────────

def deflated_sharpe_ratio(returns, n_trials):
    """
    Deflated Sharpe Ratio — koreksi multiple testing.

    Args:
        returns  : array return dari strategy
        n_trials : total kombinasi yang ditest (termasuk yang reject)

    Returns:
        float: probabilitas 0-1. < 0.95 = kemungkinan false positive.
    """
    returns = np.asarray(returns)
    returns = returns[~np.isnan(returns)]
    n = len(returns)
    if n < 2 or n_trials < 1:
        return 0.0

    sr     = sharpe_ratio(returns)
    sr_std = np.sqrt((1 + 0.5 * sr**2) / (n - 1))

    # Expected max Sharpe dari n_trials percobaan acak
    sr_max_expected = sr_std * norm.ppf(1 - 1.0 / max(n_trials, 1))
    dsr = (sr - sr_max_expected) / (sr_std + 1e-9)
    return float(norm.cdf(dsr))


# ─────────────────────────────────────────────────────────────────────────────
# RSS — REGIME STABILITY SCORE (diperbaiki dari v3)
#
# v3: max(0, 1 - std/mean) → tidak cek magnitude, bisa menipu
# v4: geometric mean dari KONSISTENSI × MAGNITUDE
#     → kalau salah satu buruk, RSS = 0
# ─────────────────────────────────────────────────────────────────────────────

def regime_stability_score(regime_sharpes, min_magnitude=0.05):
    """
    RSS yang memperhitungkan konsistensi DAN magnitude.

    Args:
        regime_sharpes : array/Series Sharpe per regime
        min_magnitude  : minimum mean Sharpe agar dianggap berguna (default 0.05)

    Returns:
        float: RSS 0-1
    """
    arr = np.asarray(regime_sharpes, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return 0.0

    mean = arr.mean()
    std  = arr.std()

    # Konsistensi: 1 - CV (dibatasi 0-1)
    consistency = max(0.0, min(1.0, 1.0 - std / (abs(mean) + 1e-6)))

    # Magnitude: seberapa berguna edge-nya secara absolut
    # Naik linear dari 0 ke 1 saat mean naik dari 0 ke min_magnitude*3
    if mean <= 0:
        magnitude = 0.0
    else:
        magnitude = min(1.0, mean / (min_magnitude * 3))

    # Geometric mean: keduanya harus baik
    rss = (consistency * magnitude) ** 0.5
    return float(rss)


# ─────────────────────────────────────────────────────────────────────────────
# MONTE CARLO EQUITY TEST
#
# Verifikasi urutan return bukan kebetulan.
# Kalau equity_original >> distribusi random → ada sequential edge.
# ─────────────────────────────────────────────────────────────────────────────

def monte_carlo_equity_test(returns, simulations=500):
    """
    Monte Carlo shuffle test.

    Returns:
        percentiles : (p5, p50, p95) equity akhir distribusi random
        eq_original : equity original untuk perbandingan
        p_value     : fraksi simulasi >= original (kecil = bagus, < 0.05 = signifikan)
    """
    returns = np.asarray(returns)
    returns = returns[~np.isnan(returns)]
    if len(returns) == 0:
        return (np.nan, np.nan, np.nan), np.nan, np.nan

    eq_original = float(np.sum(returns))

    final_equities = np.array([
        np.sum(np.random.permutation(returns))
        for _ in range(simulations)
    ])

    percentiles = tuple(np.percentile(final_equities, [5, 50, 95]))
    p_value     = float(np.mean(final_equities >= eq_original))

    return percentiles, eq_original, p_value


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_feature_importance(importances_list):
    df = pd.DataFrame(importances_list)
    return df.mean().sort_values(ascending=False)


def regime_walkforward_stats(df, config):
    """Stats per regime — hanya dari OOS bars (is_unseen=True)."""
    summary    = {}
    regime_col = config["data"]["regime_col"]
    for r in df[regime_col].unique():
        sub = df[df[regime_col] == r]
        if "is_unseen" in sub.columns:
            sub = sub[sub["is_unseen"] == True]
        sr = sharpe_ratio(sub["strategy_return_test"].dropna().values)
        summary[r] = sr
    return summary
