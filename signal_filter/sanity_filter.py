"""
signal_filter/sanity_filter.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Layer A — Statistical Sanity

Tujuan:
  Buang fitur yang secara statistik tidak layak SEBELUM hitung IC.
  Ini yang paling murah secara komputasi — microseconds per fitur.

Checks:
  1. NaN fraction    — terlalu banyak missing value
  2. Near-constant   — variance terlalu kecil (sinyal tidak bergerak)
  3. Extreme kurtosis— distribusi terlalu fat-tailed / spikey (likely data error)
  4. Zero-variance window — terlalu banyak periode dimana nilai tidak berubah sama sekali
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SanityConfig:
    max_nan_fraction: float = 0.05        # Max 5% NaN
    min_unique_ratio: float = 0.01        # Min 1% unique values (anti-constant)
    max_kurtosis: float = 100.0           # Kurtosis > 100 = likely outlier/error
    max_zero_variance_window_pct: float = 0.30  # Max 30% windows dimana std=0
    zero_variance_window: int = 20        # Rolling window untuk cek zero-variance


@dataclass
class SanityResult:
    feature: str
    passed: bool
    nan_fraction: float
    unique_ratio: float
    kurtosis: float
    zero_var_pct: float
    reject_reason: Optional[str] = None


class SanityFilter:
    """
    Layer A: Statistical Sanity Check.
    Run ini PERTAMA sebelum semua filter lain.
    """

    def __init__(self, config: Optional[SanityConfig] = None):
        self.config = config or SanityConfig()

    def check(self, series: pd.Series) -> SanityResult:
        cfg = self.config
        name = series.name or "unknown"
        arr = series.values.astype(float)

        # ── 1. NaN fraction ──────────────────────────────────────
        nan_fraction = float(np.isnan(arr).mean())
        if nan_fraction > cfg.max_nan_fraction:
            return SanityResult(
                feature=name, passed=False,
                nan_fraction=nan_fraction, unique_ratio=0.0,
                kurtosis=0.0, zero_var_pct=0.0,
                reject_reason=f"nan_fraction={nan_fraction:.3f} > {cfg.max_nan_fraction}"
            )

        # Drop NaN untuk kalkulasi selanjutnya
        clean = arr[~np.isnan(arr)]
        n = len(clean)

        if n < 30:
            return SanityResult(
                feature=name, passed=False,
                nan_fraction=nan_fraction, unique_ratio=0.0,
                kurtosis=0.0, zero_var_pct=0.0,
                reject_reason=f"insufficient_data n={n} < 30"
            )

        # ── 2. Unique ratio (near-constant check) ────────────────
        unique_ratio = float(pd.Series(clean).nunique() / n)
        if unique_ratio < cfg.min_unique_ratio:
            return SanityResult(
                feature=name, passed=False,
                nan_fraction=nan_fraction, unique_ratio=unique_ratio,
                kurtosis=0.0, zero_var_pct=0.0,
                reject_reason=f"unique_ratio={unique_ratio:.4f} < {cfg.min_unique_ratio}"
            )

        # ── 3. Kurtosis check ────────────────────────────────────
        mean = clean.mean()
        std = clean.std()
        if std < 1e-10:
            return SanityResult(
                feature=name, passed=False,
                nan_fraction=nan_fraction, unique_ratio=unique_ratio,
                kurtosis=0.0, zero_var_pct=0.0,
                reject_reason="std=0 (constant feature)"
            )

        kurt = float(((clean - mean) ** 4).mean() / (std ** 4 + 1e-10))
        if kurt > cfg.max_kurtosis:
            return SanityResult(
                feature=name, passed=False,
                nan_fraction=nan_fraction, unique_ratio=unique_ratio,
                kurtosis=kurt, zero_var_pct=0.0,
                reject_reason=f"kurtosis={kurt:.1f} > {cfg.max_kurtosis}"
            )

        # ── 4. Zero-variance window percentage ───────────────────
        w = cfg.zero_variance_window
        s = pd.Series(clean)
        rolling_std = s.rolling(w).std().dropna()
        zero_var_pct = float((rolling_std < 1e-10).mean())
        if zero_var_pct > cfg.max_zero_variance_window_pct:
            return SanityResult(
                feature=name, passed=False,
                nan_fraction=nan_fraction, unique_ratio=unique_ratio,
                kurtosis=kurt, zero_var_pct=zero_var_pct,
                reject_reason=f"zero_var_pct={zero_var_pct:.2%} > {cfg.max_zero_variance_window_pct:.2%}"
            )

        return SanityResult(
            feature=name, passed=True,
            nan_fraction=nan_fraction, unique_ratio=unique_ratio,
            kurtosis=kurt, zero_var_pct=zero_var_pct
        )

    def run(self, df: pd.DataFrame, feature_list: List[str]) -> tuple[List[str], pd.DataFrame]:
        """
        Run sanity check pada semua fitur.

        Returns:
            passed_features : list fitur yang lolos
            results_df      : scorecard lengkap semua fitur
        """
        results = []
        for feat in feature_list:
            if feat not in df.columns:
                results.append(SanityResult(
                    feature=feat, passed=False,
                    nan_fraction=1.0, unique_ratio=0.0,
                    kurtosis=0.0, zero_var_pct=0.0,
                    reject_reason="column_not_found"
                ))
                continue
            results.append(self.check(df[feat]))

        results_df = pd.DataFrame([vars(r) for r in results])
        passed = [r.feature for r in results if r.passed]
        return passed, results_df
