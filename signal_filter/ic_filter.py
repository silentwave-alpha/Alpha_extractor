"""
signal_filter/ic_filter.py  v3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Layer B — Information Coefficient Filter + Decay Analysis

FIXES dari v2:
  - Bearish signal fix: ic_pos_pct diganti dengan ic_consistency check
    IC negatif yang konsisten (short signal) sekarang lolos filter
  - Direction field: "long" / "short" / "both" per feature
  - Decay curve dihitung dengan signed IC (bukan abs) agar direction terjaga
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, linregress
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

SIGNAL_MEAN_REVERSION = "mean_reversion"
SIGNAL_TREND          = "trend"
SIGNAL_DELAYED_PEAK   = "delayed_peak"
SIGNAL_FLAT           = "flat"
SIGNAL_UNKNOWN        = "unknown"

DIRECTION_LONG  = "long"
DIRECTION_SHORT = "short"
DIRECTION_BOTH  = "both"


@dataclass
class ICConfig:
    ic_window: int = 500
    ic_step:   int = 100

    decay_horizons: List[int] = field(
        default_factory=lambda: [1, 5, 10, 24, 48, 72, 100, 150, 200]
    )

    min_ic_mean:   float = 0.02   # |IC mean| minimum
    min_ic_ir:     float = 0.30   # |IC IR| minimum
    min_t_stat:    float = 2.0    # |t-stat| minimum

    # Consistency threshold — menggantikan min_ic_positive_pct
    # Feature lolos jika IC konsisten ke SALAH SATU arah:
    #   long  : ic_pos_pct  >= min_ic_consistency  (IC sering positif)
    #   short : ic_neg_pct  >= min_ic_consistency  (IC sering negatif)
    # Jika keduanya < threshold → IC random → reject
    min_ic_consistency: float = 0.55

    # Legacy alias — masih dibaca tapi tidak dipakai untuk reject
    min_ic_positive_pct: float = 0.50

    # Signal type thresholds
    trend_ratio_threshold:    float = 2.0
    delayed_peak_min_horizon: int   = 5
    flat_cv_threshold:        float = 0.30

    halflife_interpolate: bool = True
    halflife_extrapolate: bool = True


@dataclass
class ICResult:
    feature: str
    passed:  bool

    ic_mean:           float = 0.0
    ic_std:            float = 0.0
    ic_ir:             float = 0.0
    t_stat:            float = 0.0   # naive (overlapping windows)
    t_stat_corrected:  float = 0.0   # Newey-West effective N correction
    ic_positive_pct:   float = 0.0
    ic_negative_pct:   float = 0.0
    n_windows:         int   = 0
    n_windows_eff:     float = 0.0   # effective N setelah autocorr correction

    direction: str = DIRECTION_LONG  # NEW: long / short / both

    decay_curve:        Dict[int, float] = field(default_factory=dict)
    signal_type:        str              = SIGNAL_UNKNOWN
    peak_ic:            float            = 0.0
    peak_horizon:       Optional[int]    = None
    half_life:          Optional[float]  = None
    half_life_reliable: bool             = True
    best_horizon:       Optional[int]    = None

    reject_reason: Optional[str] = None


class ICFilter:

    def __init__(self, config: Optional[ICConfig] = None):
        self.config = config or ICConfig()

    # ── Effective N (Newey-West AC correction) ───────────────────

    def _effective_n(self, ic_arr: np.ndarray) -> float:
        n = len(ic_arr)
        if n < 4:
            return float(n)
        from scipy.stats import pearsonr
        rho, _ = pearsonr(ic_arr[:-1], ic_arr[1:])
        rho    = max(0.0, float(rho))
        eff_n  = n / (1.0 + 2.0 * rho)
        return float(max(2.0, eff_n))

    # ── Rolling Spearman IC ───────────────────────────────────────

    def _rolling_ic(self, feature: np.ndarray, forward_ret: np.ndarray) -> np.ndarray:
        cfg = self.config
        n   = len(feature)
        ics = []
        for start in range(0, n - cfg.ic_window, cfg.ic_step):
            end   = start + cfg.ic_window
            f_win = feature[start:end]
            r_win = forward_ret[start:end]
            mask  = ~(np.isnan(f_win) | np.isnan(r_win))
            if mask.sum() < 30:
                continue
            # Skip window jika feature atau return konstan (spearmanr undefined)
            if np.std(f_win[mask]) == 0 or np.std(r_win[mask]) == 0:
                continue
            ic, _ = spearmanr(f_win[mask], r_win[mask])
            if not np.isnan(ic):
                ics.append(float(ic))
        return np.array(ics)

    # ── Direction Assignment ──────────────────────────────────────

    def _assign_direction(
        self,
        ic_arr:     np.ndarray,
        ic_pos_pct: float,
        ic_neg_pct: float,
    ) -> str:
        """
        Tentukan direction trading berdasarkan konsistensi IC.

        long  : IC lebih sering positif (ic_pos_pct dominan)
        short : IC lebih sering negatif (ic_neg_pct dominan)
        both  : IC relatif seimbang (tidak ada dominasi jelas)
        """
        cfg = self.config
        consistent_long  = ic_pos_pct >= cfg.min_ic_consistency
        consistent_short = ic_neg_pct >= cfg.min_ic_consistency

        if consistent_long and consistent_short:
            return DIRECTION_BOTH
        elif consistent_long:
            return DIRECTION_LONG
        elif consistent_short:
            return DIRECTION_SHORT
        else:
            # Tidak konsisten ke arah manapun — masih bisa lolos
            # jika |IC mean| kuat, tapi direction ambiguous
            return DIRECTION_BOTH

    # ── Decay Curve ───────────────────────────────────────────────

    def _compute_decay_curve(
        self, feature: np.ndarray, close: np.ndarray, horizons: List[int]
    ) -> Dict[int, float]:
        """
        Hitung IC signed (bukan abs) per horizon.
        Signed penting untuk menjaga direction information.
        """
        n     = len(feature)
        decay = {}
        for h in sorted(horizons):
            if h >= n:
                continue
            fwd_ret         = np.full(n, np.nan)
            fwd_ret[:n - h] = (close[h:] - close[:n-h]) / (close[:n-h] + 1e-10)
            ics             = self._rolling_ic(feature, fwd_ret)
            decay[h]        = float(np.mean(ics)) if len(ics) > 0 else np.nan
        return decay

    # ── Signal Type Classification ────────────────────────────────

    def _classify_signal(
        self, decay_curve: Dict[int, float]
    ) -> Tuple[str, int, float]:
        """
        Klasifikasi shape decay curve menggunakan |IC| (magnitude).
        Direction sudah di-handle terpisah di _assign_direction.
        """
        cfg   = self.config
        valid = {h: ic for h, ic in decay_curve.items() if not np.isnan(ic)}
        if len(valid) < 2:
            return SIGNAL_UNKNOWN, 1, 0.0

        horizons_sorted = sorted(valid.keys())
        ic_abs  = {h: abs(v) for h, v in valid.items()}
        peak_h  = max(ic_abs, key=ic_abs.get)
        peak_ic = ic_abs[peak_h]

        h1       = horizons_sorted[0]
        ic_at_h1 = ic_abs.get(h1, 0.0)
        max_h    = horizons_sorted[-1]
        ic_vals  = [ic_abs[h] for h in horizons_sorted]
        cv       = np.std(ic_vals) / (np.mean(ic_vals) + 1e-10)

        if cv < cfg.flat_cv_threshold:
            return SIGNAL_FLAT, peak_h, peak_ic

        if peak_h == h1:
            return SIGNAL_MEAN_REVERSION, peak_h, peak_ic

        ratio = peak_ic / (ic_at_h1 + 1e-10)
        if ratio > cfg.trend_ratio_threshold and peak_h >= cfg.delayed_peak_min_horizon:
            if peak_h == max_h or peak_h >= horizons_sorted[int(len(horizons_sorted)*0.7)]:
                return SIGNAL_TREND, peak_h, peak_ic

        if peak_h > h1 and peak_h < max_h:
            return SIGNAL_DELAYED_PEAK, peak_h, peak_ic

        if peak_h == max_h:
            return SIGNAL_TREND, peak_h, peak_ic

        return SIGNAL_UNKNOWN, peak_h, peak_ic

    # ── Half-Life Estimation (dari PEAK) ─────────────────────────

    def _estimate_half_life(
        self,
        decay_curve:  Dict[int, float],
        peak_horizon: int,
        peak_ic:      float,
    ) -> Tuple[Optional[float], bool]:
        cfg    = self.config
        target = 0.5 * peak_ic
        if peak_ic < 1e-6:
            return None, False

        post_peak = sorted([
            h for h, ic in decay_curve.items()
            if h >= peak_horizon and not np.isnan(ic)
        ])
        if len(post_peak) < 2:
            return None, False

        ic_post = {h: abs(decay_curve[h]) for h in post_peak}
        prev_h  = post_peak[0]
        prev_ic = ic_post[prev_h]

        for h in post_peak[1:]:
            curr_ic = ic_post[h]
            if curr_ic <= target:
                if cfg.halflife_interpolate and prev_ic > target:
                    frac      = (prev_ic - target) / (prev_ic - curr_ic + 1e-10)
                    half_life = prev_h + frac * (h - prev_h)
                    return float(half_life), True
                return float(h), True
            prev_h, prev_ic = h, curr_ic

        if cfg.halflife_extrapolate and len(post_peak) >= 3:
            h_arr  = np.array(post_peak, dtype=float)
            ic_arr = np.array([ic_post[h] for h in post_peak])
            diffs  = np.diff(ic_arr)
            if (diffs < 0).sum() >= len(diffs) // 2:
                slope, intercept, r, _, _ = linregress(h_arr, ic_arr)
                if slope < 0 and abs(r) > 0.5:
                    h_extrap = (target - intercept) / slope
                    if h_extrap > post_peak[-1] and h_extrap < post_peak[-1] * 10:
                        return float(h_extrap), False

        return None, False

    # ── Best Horizon Assignment ───────────────────────────────────

    def _assign_best_horizon(
        self,
        decay_curve:        Dict[int, float],
        signal_type:        str,
        peak_horizon:       int,
        half_life:          Optional[float],
        available_horizons: List[int],
    ) -> int:
        sorted_h = sorted(available_horizons)

        def snap(target: float) -> int:
            return min(sorted_h, key=lambda h: abs(h - target))

        if signal_type == SIGNAL_FLAT:
            if half_life is not None:
                return snap(half_life)
            return sorted_h[len(sorted_h) // 2]

        best = snap(peak_horizon)

        if signal_type == SIGNAL_DELAYED_PEAK and half_life is not None:
            midpoint = (peak_horizon + half_life) / 2.0
            alt      = snap(midpoint)
            ic_best  = abs(decay_curve.get(best, 0.0))
            ic_alt   = abs(decay_curve.get(alt,  0.0))
            if ic_alt > ic_best:
                best = alt

        return best

    # ── Check single feature ──────────────────────────────────────

    def check(
        self,
        series:       pd.Series,
        close:        pd.Series,
        base_horizon: int = 1,
    ) -> ICResult:
        cfg  = self.config
        name = series.name or "unknown"
        n    = len(series)

        feature   = series.values.astype(float)
        close_arr = close.values.astype(float)

        # Step 1: Rolling IC di h=1
        fwd_ret         = np.full(n, np.nan)
        fwd_ret[:n - base_horizon] = (
            (close_arr[base_horizon:] - close_arr[:n - base_horizon])
            / (close_arr[:n - base_horizon] + 1e-10)
        )
        ic_arr    = self._rolling_ic(feature, fwd_ret)
        n_windows = len(ic_arr)

        if n_windows < 3:
            return ICResult(
                feature=name, passed=False,
                reject_reason=f"insufficient_windows={n_windows} < 3"
            )

        ic_mean          = float(np.mean(ic_arr))
        ic_std           = float(np.std(ic_arr) + 1e-10)
        ic_ir            = ic_mean / ic_std
        t_stat           = ic_mean / (ic_std / np.sqrt(n_windows))
        n_eff            = self._effective_n(ic_arr)
        t_stat_corrected = ic_mean / (ic_std / np.sqrt(n_eff))
        ic_pos_pct       = float(np.mean(ic_arr > 0))
        ic_neg_pct       = float(np.mean(ic_arr < 0))

        # Step 2: Hard rejection — semua pakai abs() untuk direction-agnostic
        if abs(ic_mean) < cfg.min_ic_mean:
            return ICResult(
                feature=name, passed=False,
                ic_mean=ic_mean, ic_std=ic_std, ic_ir=ic_ir,
                t_stat=t_stat, t_stat_corrected=t_stat_corrected,
                ic_positive_pct=ic_pos_pct,
                ic_negative_pct=ic_neg_pct, n_windows=n_windows, n_windows_eff=n_eff,
                reject_reason=f"|ic_mean|={abs(ic_mean):.4f} < {cfg.min_ic_mean}"
            )
        if abs(ic_ir) < cfg.min_ic_ir:
            return ICResult(
                feature=name, passed=False,
                ic_mean=ic_mean, ic_std=ic_std, ic_ir=ic_ir,
                t_stat=t_stat, t_stat_corrected=t_stat_corrected,
                ic_positive_pct=ic_pos_pct,
                ic_negative_pct=ic_neg_pct, n_windows=n_windows, n_windows_eff=n_eff,
                reject_reason=f"|ic_ir|={abs(ic_ir):.3f} < {cfg.min_ic_ir}"
            )
        if abs(t_stat_corrected) < cfg.min_t_stat:
            return ICResult(
                feature=name, passed=False,
                ic_mean=ic_mean, ic_std=ic_std, ic_ir=ic_ir,
                t_stat=t_stat, ic_positive_pct=ic_pos_pct,
                ic_negative_pct=ic_neg_pct, n_windows=n_windows,
                reject_reason=f"|t_stat|={abs(t_stat):.3f} < {cfg.min_t_stat}"
            )

        # Step 2b: Consistency check — gantikan ic_pos_pct rejection
        # Lolos jika konsisten ke SALAH SATU arah (long ATAU short)
        consistent_long  = ic_pos_pct >= cfg.min_ic_consistency
        consistent_short = ic_neg_pct >= cfg.min_ic_consistency

        if not consistent_long and not consistent_short:
            return ICResult(
                feature=name, passed=False,
                ic_mean=ic_mean, ic_std=ic_std, ic_ir=ic_ir,
                t_stat=t_stat, t_stat_corrected=t_stat_corrected,
                ic_positive_pct=ic_pos_pct,
                ic_negative_pct=ic_neg_pct, n_windows=n_windows, n_windows_eff=n_eff,
                reject_reason=(
                    f"inconsistent: ic_pos%={ic_pos_pct:.2%}, "
                    f"ic_neg%={ic_neg_pct:.2%} — "
                    f"keduanya < {cfg.min_ic_consistency:.2%}"
                )
            )

        # Step 3: Assign direction
        direction = self._assign_direction(ic_arr, ic_pos_pct, ic_neg_pct)

        # Step 4: Decay curve
        # Untuk short signals, flip sign feature sebelum hitung decay
        # agar decay curve shape-nya konsisten (IC positif = signal kuat)
        feature_for_decay = feature if direction != DIRECTION_SHORT else -feature
        decay_curve = self._compute_decay_curve(feature_for_decay, close_arr, cfg.decay_horizons)

        # Step 5: Signal type classification
        signal_type, peak_horizon, peak_ic = self._classify_signal(decay_curve)

        # Step 6: Half-life dari PEAK
        half_life, hl_reliable = self._estimate_half_life(
            decay_curve, peak_horizon, peak_ic
        )

        # Step 7: Best horizon
        best_horizon = self._assign_best_horizon(
            decay_curve, signal_type, peak_horizon,
            half_life if hl_reliable else None,
            cfg.decay_horizons
        )

        return ICResult(
            feature=name, passed=True,
            ic_mean=ic_mean, ic_std=ic_std, ic_ir=ic_ir,
            t_stat=t_stat, t_stat_corrected=t_stat_corrected,
            ic_positive_pct=ic_pos_pct,
            ic_negative_pct=ic_neg_pct, n_windows=n_windows, n_windows_eff=n_eff,
            direction=direction,
            decay_curve=decay_curve,
            signal_type=signal_type,
            peak_ic=peak_ic,
            peak_horizon=peak_horizon,
            half_life=half_life,
            half_life_reliable=hl_reliable,
            best_horizon=best_horizon,
        )

    # ── Run on all features ───────────────────────────────────────

    def run(
        self,
        df:           pd.DataFrame,
        feature_list: List[str],
        close_col:    str = "close",
        base_horizon: int = 1,
    ) -> tuple[List[str], pd.DataFrame]:

        if close_col not in df.columns:
            raise ValueError(f"close_col '{close_col}' not found in df")

        close   = df[close_col]
        results = []

        for feat in feature_list:
            if feat not in df.columns:
                results.append(ICResult(
                    feature=feat, passed=False,
                    reject_reason="column_not_found"
                ))
                continue
            results.append(self.check(df[feat], close, base_horizon))

        rows = []
        for r in results:
            row = {
                "feature":            r.feature,
                "passed":             r.passed,
                "ic_mean":            r.ic_mean,
                "ic_std":             r.ic_std,
                "ic_ir":              r.ic_ir,
                "t_stat":             r.t_stat,
                "t_stat_corrected":   r.t_stat_corrected,
                "n_windows_eff":      r.n_windows_eff,
                "ic_positive_pct":    r.ic_positive_pct,
                "ic_negative_pct":    r.ic_negative_pct,
                "n_windows":          r.n_windows,
                "direction":          r.direction,
                "signal_type":        r.signal_type,
                "peak_ic":            r.peak_ic,
                "peak_horizon":       r.peak_horizon,
                "half_life":          r.half_life,
                "half_life_reliable": r.half_life_reliable,
                "best_horizon":       r.best_horizon,
                "reject_reason":      r.reject_reason,
            }
            for h, ic_val in r.decay_curve.items():
                row[f"ic_h{h}"] = ic_val
            rows.append(row)

        results_df = pd.DataFrame(rows)
        passed     = [r.feature for r in results if r.passed]
        return passed, results_df