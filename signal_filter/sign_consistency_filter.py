"""
signal_filter/sign_consistency_filter.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Layer C — Sign Consistency & Monotonicity

Tujuan:
  Verifikasi bahwa hubungan feature → return bersifat MONOTONIC dan KONSISTEN.
  IC yang tinggi bisa saja datang dari outlier atau non-linear relationship
  yang tidak exploitable. Layer ini mengkonfirmasi struktur sinyal.

Metrics:
  1. Quantile Spread     — spread return antara Q5 (top) dan Q1 (bottom)
  2. Monotonicity Score  — seberapa monotonic Q1 < Q2 < Q3 < Q4 < Q5
  3. Sign Consistency    — % rolling windows dimana spread(Q5-Q1) searah IC
  4. Long Signal Quality — apakah Q5 return > 0 secara konsisten
  5. Short Signal Quality— apakah Q1 return < 0 secara konsisten
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional
from scipy.stats import spearmanr


@dataclass
class SignConsistencyConfig:
    n_quantiles: int = 5              # Bagi fitur ke N quantile (default quintile)
    window: int = 500                 # Rolling window untuk cek konsistensi
    step: int = 100                   # Step antar windows

    # Thresholds
    min_monotonicity: float = 0.60    # Min monotonicity score (0=chaotic, 1=perfect)
    min_spread_pct: float = 0.55      # Min % windows dengan Q5>Q1 return spread positif
    min_long_quality: float = 0.50    # Min % windows dimana Q5 mean return > 0
    min_short_quality: float = 0.40   # Min % windows dimana Q1 mean return < 0


@dataclass
class SignConsistencyResult:
    feature: str
    passed: bool

    # Quantile returns (global, bukan rolling)
    quantile_returns: List[float] = field(default_factory=list)

    # Consistency metrics
    monotonicity_score: float = 0.0   # Fraction of adjacent pairs Q(i) < Q(i+1)
    spread_consistency: float = 0.0   # % windows spread > 0
    long_quality: float = 0.0         # % windows Q5 return > 0
    short_quality: float = 0.0        # % windows Q1 return < 0
    n_windows: int = 0

    reject_reason: Optional[str] = None


class SignConsistencyFilter:
    """
    Layer C: Sign Consistency & Monotonicity Check.

    Complement dari IC Filter — IC bisa tinggi karena outlier,
    tapi kalau quantile spread tidak monotonic, sinyal tidak exploitable
    dalam praktik (karena kita tidak tahu kapan outlier terjadi).
    """

    def __init__(self, config: Optional[SignConsistencyConfig] = None):
        self.config = config or SignConsistencyConfig()

    def _quantile_returns(
        self,
        feature: np.ndarray,
        forward_ret: np.ndarray,
        n_quantiles: int
    ) -> List[float]:
        """
        Hitung mean return per quantile bucket.
        Q1 = fitur rendah, Q5 = fitur tinggi (untuk n_quantiles=5).
        """
        mask = ~(np.isnan(feature) | np.isnan(forward_ret))
        f = feature[mask]
        r = forward_ret[mask]

        if len(f) < n_quantiles * 10:
            return []

        boundaries = np.percentile(f, np.linspace(0, 100, n_quantiles + 1))
        q_returns = []

        for i in range(n_quantiles):
            lo = boundaries[i]
            hi = boundaries[i + 1]
            if i == n_quantiles - 1:
                in_bucket = (f >= lo) & (f <= hi)
            else:
                in_bucket = (f >= lo) & (f < hi)

            if in_bucket.sum() < 5:
                q_returns.append(np.nan)
            else:
                q_returns.append(float(r[in_bucket].mean()))

        return q_returns

    def _monotonicity_score(self, q_returns: List[float]) -> float:
        """
        Fraction of adjacent pairs dimana Q(i+1) > Q(i).
        Score = 1.0 → perfectly monotonic (increasing).
        Score = 0.0 → perfectly decreasing.
        Score = 0.5 → random.

        Note: kita juga cek arah terbalik (feature inversi),
        yang juga valid sebagai sinyal short.
        """
        valid = [q for q in q_returns if not np.isnan(q)]
        if len(valid) < 2:
            return 0.0

        n_pairs = len(valid) - 1
        n_increasing = sum(1 for i in range(n_pairs) if valid[i + 1] > valid[i])

        # Score bisa > 0.5 (increasing) atau < 0.5 (decreasing = short signal)
        # Return max dari kedua arah (0.5 = no info, 1.0 = perfect)
        frac = n_increasing / n_pairs
        return max(frac, 1 - frac)  # symmetrize: 0.5-1.0 range

    def check(
        self,
        series:    pd.Series,
        close:     pd.Series,
        horizon:   int = 1,
        direction: str = "long",
    ) -> SignConsistencyResult:
        """
        Evaluasi sign consistency satu fitur.

        Args:
            series    : feature series
            close     : close price
            horizon   : forward return horizon
            direction : "long" / "short" / "both" dari ICResult
        """
        cfg = self.config
        name = series.name or "unknown"
        n = len(series)

        # Untuk short signals, flip feature sign agar Q5 = high feature = short edge
        # Ini membuat semua downstream logic (spread, monotonicity, quality) bekerja
        # dengan asumsi yang sama: "high feature value → positive edge"
        feature_raw = series.values.astype(float)
        feature     = -feature_raw if direction == "short" else feature_raw
        close_arr   = close.values.astype(float)

        # Simple forward return
        fwd_ret = np.full(n, np.nan)
        fwd_ret[:n - horizon] = (
            (close_arr[horizon:] - close_arr[:n - horizon])
            / (close_arr[:n - horizon] + 1e-10)
        )

        # ── Global quantile returns ───────────────────────────────
        q_returns = self._quantile_returns(feature, fwd_ret, cfg.n_quantiles)
        if not q_returns or any(np.isnan(q) for q in q_returns):
            return SignConsistencyResult(
                feature=name, passed=False,
                reject_reason="insufficient_data_for_quantiles"
            )

        mono_score = self._monotonicity_score(q_returns)

        if mono_score < cfg.min_monotonicity:
            return SignConsistencyResult(
                feature=name, passed=False,
                quantile_returns=q_returns,
                monotonicity_score=mono_score,
                reject_reason=f"monotonicity={mono_score:.3f} < {cfg.min_monotonicity}"
            )

        # ── Rolling consistency ───────────────────────────────────
        spread_consistent_count = 0
        long_quality_count = 0
        short_quality_count = 0
        n_windows = 0

        for start in range(0, n - cfg.window, cfg.step):
            end = start + cfg.window
            f_win = feature[start:end]
            r_win = fwd_ret[start:end]

            q_win = self._quantile_returns(f_win, r_win, cfg.n_quantiles)
            if not q_win or any(np.isnan(q) for q in q_win):
                continue

            n_windows += 1

            # Spread: Q_top - Q_bottom > 0
            q_top = q_win[-1]   # Q5
            q_bot = q_win[0]    # Q1
            spread = q_top - q_bot
            if spread > 0:
                spread_consistent_count += 1

            # Long quality: Q5 return > 0
            if q_top > 0:
                long_quality_count += 1

            # Short quality: Q1 return < 0
            if q_bot < 0:
                short_quality_count += 1

        if n_windows == 0:
            return SignConsistencyResult(
                feature=name, passed=False,
                quantile_returns=q_returns,
                monotonicity_score=mono_score,
                reject_reason="no_valid_rolling_windows"
            )

        spread_consistency = spread_consistent_count / n_windows
        long_quality       = long_quality_count / n_windows
        short_quality      = short_quality_count / n_windows

        # ── Rejection checks ─────────────────────────────────────
        if spread_consistency < cfg.min_spread_pct:
            return SignConsistencyResult(
                feature=name, passed=False,
                quantile_returns=q_returns,
                monotonicity_score=mono_score,
                spread_consistency=spread_consistency,
                long_quality=long_quality,
                short_quality=short_quality,
                n_windows=n_windows,
                reject_reason=f"spread_consistency={spread_consistency:.2%} < {cfg.min_spread_pct:.2%}"
            )

        # Direction-aware quality check
        if direction == "short":
            quality_ok     = short_quality >= cfg.min_short_quality
            quality_reason = f"short_quality={short_quality:.2%} < {cfg.min_short_quality:.2%}"
        elif direction == "both":
            quality_ok     = (long_quality  >= cfg.min_long_quality or
                              short_quality >= cfg.min_short_quality)
            quality_reason = (f"long_quality={long_quality:.2%} & "
                              f"short_quality={short_quality:.2%} both below threshold")
        else:
            quality_ok     = long_quality >= cfg.min_long_quality
            quality_reason = f"long_quality={long_quality:.2%} < {cfg.min_long_quality:.2%}"

        if not quality_ok:
            return SignConsistencyResult(
                feature=name, passed=False,
                quantile_returns=q_returns,
                monotonicity_score=mono_score,
                spread_consistency=spread_consistency,
                long_quality=long_quality,
                short_quality=short_quality,
                n_windows=n_windows,
                reject_reason=quality_reason,
            )

        return SignConsistencyResult(
            feature=name, passed=True,
            quantile_returns=q_returns,
            monotonicity_score=mono_score,
            spread_consistency=spread_consistency,
            long_quality=long_quality,
            short_quality=short_quality,
            n_windows=n_windows
        )

    def run(
        self,
        df:            pd.DataFrame,
        feature_list:  List[str],
        close_col:     str = "close",
        horizon:       int = 1,
        direction_map: dict = None,   # {feature: "long"/"short"/"both"}
        horizon_map:   dict = None,   # {feature: int} — per-feature horizon override
    ) -> tuple[List[str], pd.DataFrame]:
        """
        Run sign consistency check pada semua fitur.

        Args:
            direction_map : dari ICFilter, per-feature direction
            horizon_map   : dari PassportBuilder, per-feature optimal horizon.
                           Jika None, pakai default horizon untuk semua feature.
                           Ini penting: feature dengan best_horizon=48 tidak boleh
                           di-check sign consistency di h=1.
        """
        if close_col not in df.columns:
            raise ValueError(f"close_col '{close_col}' not found in df")

        close   = df[close_col]
        dmap    = direction_map or {}
        hmap    = horizon_map   or {}
        results = []

        for feat in feature_list:
            if feat not in df.columns:
                results.append(SignConsistencyResult(
                    feature=feat, passed=False,
                    reject_reason="column_not_found"
                ))
                continue
            direction   = dmap.get(feat, "long")
            feat_horizon = hmap.get(feat, horizon)  # per-feature horizon, fallback ke default
            result      = self.check(df[feat], close, feat_horizon, direction=direction)
            results.append(result)

        rows = []
        for r in results:
            row = {
                "feature":            r.feature,
                "passed":             r.passed,
                "monotonicity_score": r.monotonicity_score,
                "spread_consistency": r.spread_consistency,
                "long_quality":       r.long_quality,
                "short_quality":      r.short_quality,
                "n_windows":          r.n_windows,
                "reject_reason":      r.reject_reason,
            }
            for i, q in enumerate(r.quantile_returns, 1):
                row[f"q{i}_return"] = q
            rows.append(row)

        results_df = pd.DataFrame(rows)
        passed = [r.feature for r in results if r.passed]
        return passed, results_df