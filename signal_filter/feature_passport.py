"""
signal_filter/feature_passport.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Feature Passport — Characterization & Structure-Based Routing

Posisi dalam pipeline:
  SanityFilter → ICFilter → [PassportBuilder] → EdgeMiner

Filosofi:
  ICFilter menjawab: "Apakah feature ini punya predictive power?"
  PassportBuilder menjawab: "Apa KARAKTER signal ini dan bagaimana cara exploit-nya?"

  Routing bukan lagi berdasarkan IC metrics semata, tapi berdasarkan
  STRUKTUR statistik feature → treatment yang berbeda per karakter.

Route yang dihasilkan:
  graveyard         → IC ada tapi struktur terlalu rusak untuk di-exploit
  retransform       → Ada signal tapi butuh feature engineering ulang
  regime_conditional→ IC flip di bull vs bear — hanya valid per regime
  rank_based        → Tail-driven: signal hanya di extreme quantile
  linear_stable     → Monotonic + linear + temporal stabil → model sederhana
  nonlinear         → Monotonic tapi non-linear → butuh tree/boosting
  structural        → Persistent + horizon panjang → bukan entry signal
  horizon_specific  → Non-monotonic decay tapi tidak flip sign → pakai di horizon tertentu
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from scipy.stats import spearmanr, pearsonr, kendalltau


# ─────────────────────────────────────────────────────────────────
# ROUTE LABELS
# ─────────────────────────────────────────────────────────────────

ROUTE_GRAVEYARD          = "graveyard"
ROUTE_RETRANSFORM        = "retransform"
ROUTE_REGIME_CONDITIONAL = "regime_conditional"
ROUTE_RANK_BASED         = "rank_based"
ROUTE_LINEAR_STABLE      = "linear_stable"
ROUTE_NONLINEAR          = "nonlinear"
ROUTE_STRUCTURAL         = "structural"
ROUTE_HORIZON_SPECIFIC   = "horizon_specific"

# Decay shapes
DECAY_IMMEDIATE     = "immediate"
DECAY_LAGGED        = "lagged"
DECAY_PERSISTENT    = "persistent"
DECAY_NON_MONOTONIC = "non_monotonic"
DECAY_UNKNOWN       = "unknown"

# Confidence
CONF_HIGH   = "HIGH"
CONF_MEDIUM = "MEDIUM"
CONF_LOW    = "LOW"


# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────

@dataclass
class PassportConfig:
    # Decay horizons untuk mapping
    decay_horizons: List[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 12, 24, 48, 72, 100, 150, 200]
    )

    # Decay shape thresholds
    immediate_max_horizon: int   = 4     # peak_h <= 4 → IMMEDIATE
    lagged_max_horizon:    int   = 24    # peak_h <= 24 → LAGGED
    persistent_cv:         float = 0.30  # CV IC lintas horizon < ini → PERSISTENT
    structural_min_horizon: int  = 72    # peak_h >= ini → kandidat STRUCTURAL

    # Stability — sub-period split
    n_sub_periods:     int   = 6
    min_stability:     float = 0.55   # min fraksi sub-period IC sign benar
    degrade_tau:       float = -0.4   # Kendall tau threshold
    degrade_p:         float = 0.10

    # Regime proxy
    regime_window:        int   = 168   # rolling bars untuk bull/bear proxy
    min_regime_samples:   int   = 50

    # Quantile structure
    n_quantiles:          int   = 5
    tail_ratio_threshold: float = 0.20  # middle/total spread < ini → tail_driven
    linearity_threshold:  float = 0.80  # |pearson|/|spearman| > ini → linear

    # Confidence weights (sum = 100)
    w_stability:    float = 35.0
    w_no_degrade:   float = 20.0
    w_no_flip:      float = 20.0
    w_monotonicity: float = 25.0

    conf_high:   float = 70.0
    conf_medium: float = 45.0


# ─────────────────────────────────────────────────────────────────
# PASSPORT DATACLASS
# ─────────────────────────────────────────────────────────────────

@dataclass
class FeaturePassport:
    name: str

    # Decay
    peak_horizon:  int
    halflife:      Optional[float]
    decay_shape:   str

    # Stability
    temporal_stability: float
    ic_degrading:       bool
    regime_flip:        bool
    ic_bull:            float
    ic_bear:            float
    sub_period_ics:     List[float]

    # Quantile structure
    monotonicity_score: float
    is_linear:          bool
    tail_driven:        bool
    quantile_returns:   List[float]

    # Confidence
    confidence:       str
    confidence_score: float

    # Routing output
    route:            str
    route_reason:     str
    transform:        str
    model_type:       str
    optimal_horizon:  int


# ─────────────────────────────────────────────────────────────────
# PASSPORT BUILDER
# ─────────────────────────────────────────────────────────────────

class PassportBuilder:
    """
    Membangun FeaturePassport untuk setiap feature yang lolos ICFilter.

    Cara pakai:
        builder = PassportBuilder(config)
        passport = builder.build(feature_series, close_series, ic_direction)

    ic_direction: "long" / "short" / "both" dari ICResult
    """

    def __init__(self, config: Optional[PassportConfig] = None):
        self.cfg = config or PassportConfig()

    # ─────────────────────────────────────────────
    # PUBLIC: BUILD
    # ─────────────────────────────────────────────

    def build(
        self,
        feature:      pd.Series,
        close:        pd.Series,
        ic_direction: str = "long",
        ic_mean:      float = 0.0,
    ) -> FeaturePassport:
        """
        Bangun passport lengkap untuk satu feature.

        Args:
            feature      : feature series (sudah stationary)
            close        : close price series
            ic_direction : dari ICResult — "long" / "short" / "both"
            ic_mean      : IC mean dari ICResult, dipakai untuk sign reference
        """
        cfg  = self.cfg
        name = feature.name or "unknown"

        # Untuk short signal, flip feature agar analisis downstream
        # konsisten (high feature value = positive expected return)
        feat = -feature if ic_direction == "short" else feature
        sign_ref = np.sign(ic_mean) if ic_mean != 0 else 1.0

        # ── 1. Decay Mapping ──────────────────────────────────────
        decay_curve, peak_horizon, halflife, decay_shape = \
            self._compute_decay(feat, close)

        optimal_horizon = peak_horizon

        # ── 2. Temporal Stability ─────────────────────────────────
        fwd = self._make_fwd_return(close, optimal_horizon)
        aligned = pd.concat([feat, fwd], axis=1).dropna()
        aligned.columns = ["f", "r"]

        sub_ics, temporal_stability, ic_degrading = \
            self._stability_analysis(aligned, sign_ref)

        # ── 3. Regime Flip ────────────────────────────────────────
        regime_flip, ic_bull, ic_bear = \
            self._regime_analysis(aligned, close)

        # ── 4. Quantile Structure ─────────────────────────────────
        monotonicity_score, is_linear, tail_driven, q_returns = \
            self._quantile_analysis(aligned)

        # ── 5. Confidence Score ───────────────────────────────────
        confidence, confidence_score = self._score_confidence(
            temporal_stability, ic_degrading, regime_flip, monotonicity_score
        )

        # ── 6. Route berdasarkan karakter ─────────────────────────
        route, route_reason, transform, model_type = self._route(
            decay_shape      = decay_shape,
            peak_horizon     = peak_horizon,
            temporal_stability = temporal_stability,
            ic_degrading     = ic_degrading,
            regime_flip      = regime_flip,
            is_linear        = is_linear,
            tail_driven      = tail_driven,
            monotonicity_score = monotonicity_score,
            confidence       = confidence,
        )

        return FeaturePassport(
            name              = name,
            peak_horizon      = peak_horizon,
            halflife          = halflife,
            decay_shape       = decay_shape,
            temporal_stability = temporal_stability,
            ic_degrading      = ic_degrading,
            regime_flip       = regime_flip,
            ic_bull           = ic_bull,
            ic_bear           = ic_bear,
            sub_period_ics    = sub_ics,
            monotonicity_score = monotonicity_score,
            is_linear         = is_linear,
            tail_driven       = tail_driven,
            quantile_returns  = q_returns,
            confidence        = confidence,
            confidence_score  = confidence_score,
            route             = route,
            route_reason      = route_reason,
            transform         = transform,
            model_type        = model_type,
            optimal_horizon   = optimal_horizon,
        )

    # ─────────────────────────────────────────────
    # 1. DECAY MAPPING
    # ─────────────────────────────────────────────

    def _compute_decay(
        self,
        feat:  pd.Series,
        close: pd.Series,
    ) -> Tuple[dict, int, Optional[float], str]:
        """
        Hitung IC per horizon → peak → halflife → decay shape.
        Menggunakan global (bukan rolling) Spearman IC per horizon
        untuk speed, karena ini characterization bukan validation.
        """
        cfg = self.cfg
        feat_arr  = feat.values.astype(float)
        close_arr = close.values.astype(float)
        n = len(feat_arr)

        decay_curve = {}
        for h in cfg.decay_horizons:
            if h >= n - 10:
                continue
            fwd = np.full(n, np.nan)
            fwd[:n - h] = (close_arr[h:] - close_arr[:n - h]) / (close_arr[:n - h] + 1e-10)

            mask = ~(np.isnan(feat_arr) | np.isnan(fwd))
            if mask.sum() < 50:
                continue
            ic, _ = spearmanr(feat_arr[mask], fwd[mask])
            if not np.isnan(ic):
                decay_curve[h] = float(ic)

        if not decay_curve:
            return {}, 1, None, DECAY_UNKNOWN

        # Peak
        ic_abs = {h: abs(v) for h, v in decay_curve.items()}
        peak_h = max(ic_abs, key=ic_abs.get)
        peak_ic = ic_abs[peak_h]

        # Halflife dari peak
        halflife = self._estimate_halflife(decay_curve, peak_h, peak_ic)

        # Decay shape
        decay_shape = self._classify_decay(decay_curve, peak_h, peak_ic)

        return decay_curve, int(peak_h), halflife, decay_shape

    def _estimate_halflife(
        self,
        decay_curve: dict,
        peak_h:      int,
        peak_ic:     float,
    ) -> Optional[float]:
        target = 0.5 * peak_ic
        if peak_ic < 1e-6:
            return None

        post = sorted([h for h in decay_curve if h >= peak_h])
        if len(post) < 2:
            return None

        prev_h, prev_ic = post[0], abs(decay_curve[post[0]])
        for h in post[1:]:
            curr_ic = abs(decay_curve[h])
            if curr_ic <= target:
                if prev_ic > target:
                    frac = (prev_ic - target) / (prev_ic - curr_ic + 1e-10)
                    return float(prev_h + frac * (h - prev_h))
                return float(h)
            prev_h, prev_ic = h, curr_ic
        return None

    def _classify_decay(
        self,
        decay_curve: dict,
        peak_h:      int,
        peak_ic:     float,
    ) -> str:
        cfg = self.cfg
        if not decay_curve or len(decay_curve) < 2:
            return DECAY_UNKNOWN

        horizons = sorted(decay_curve.keys())
        ic_vals  = [decay_curve[h] for h in horizons]
        ic_abs   = [abs(v) for v in ic_vals]

        # Non-monotonic: ada sign flip di decay curve
        # Threshold minimum IC untuk dihitung sebagai signal (bukan noise)
        # IC < 0.01 dianggap noise, tidak dihitung dalam sign check
        # Non-monotonic hanya jika IC yang cukup signifikan flip sign.
        # Threshold 0.02 — di bawah ini dianggap noise, bukan genuine flip.
        # Tambahan: butuh setidaknya 30% dari horizons punya IC signifikan
        # agar classification reliable.
        min_ic_for_sign   = 0.02
        significant_ic    = [v for v in ic_vals if abs(v) > min_ic_for_sign]
        min_coverage      = max(2, int(len(ic_vals) * 0.30))

        if len(significant_ic) >= min_coverage:
            signs = [np.sign(v) for v in significant_ic]
            if len(set(signs)) > 1:
                return DECAY_NON_MONOTONIC

        # Persistent: CV rendah lintas horizon
        mean_abs = np.mean(ic_abs)
        cv = np.std(ic_abs) / (mean_abs + 1e-10)
        if cv < cfg.persistent_cv:
            return DECAY_PERSISTENT

        # Immediate: peak di horizon sangat awal
        if peak_h <= cfg.immediate_max_horizon:
            return DECAY_IMMEDIATE

        # Lagged: peak di horizon menengah
        if peak_h <= cfg.lagged_max_horizon:
            return DECAY_LAGGED

        # Default: jika peak di horizon panjang → trend/structural candidate
        return DECAY_LAGGED

    # ─────────────────────────────────────────────
    # 2. TEMPORAL STABILITY
    # ─────────────────────────────────────────────

    def _stability_analysis(
        self,
        aligned:   pd.DataFrame,  # columns: f, r
        sign_ref:  float,
    ) -> Tuple[List[float], float, bool]:
        cfg = self.cfg
        n   = len(aligned)
        chunk = n // cfg.n_sub_periods

        sub_ics = []
        for i in range(cfg.n_sub_periods):
            s = aligned.iloc[i * chunk : (i + 1) * chunk]
            if len(s) < 20:
                sub_ics.append(0.0)
                continue
            ic, _ = spearmanr(s["f"], s["r"])
            sub_ics.append(float(ic) if not np.isnan(ic) else 0.0)

        # Temporal stability: fraksi sub-period dengan IC sign benar
        correct = sum(1 for x in sub_ics if np.sign(x) == np.sign(sign_ref))
        temporal_stability = correct / cfg.n_sub_periods

        # Degradation: apakah IC memburuk over time?
        tau, p_tau = kendalltau(range(cfg.n_sub_periods), sub_ics)
        ic_degrading = float(tau) < cfg.degrade_tau and float(p_tau) < cfg.degrade_p

        return sub_ics, float(temporal_stability), bool(ic_degrading)

    # ─────────────────────────────────────────────
    # 3. REGIME FLIP
    # ─────────────────────────────────────────────

    def _regime_analysis(
        self,
        aligned: pd.DataFrame,  # columns: f, r
        close:   pd.Series,
    ) -> Tuple[bool, float, float]:
        cfg = self.cfg

        # Bull/bear proxy: rolling return MA
        close_aligned = close.reindex(aligned.index)
        ma = close_aligned.pct_change().rolling(cfg.regime_window).mean()
        bull_mask = (ma > 0).reindex(aligned.index).fillna(False)
        bear_mask = ~bull_mask

        ic_bull, ic_bear = 0.0, 0.0

        bull_data = aligned[bull_mask]
        if len(bull_data) >= cfg.min_regime_samples:
            ic_b, _ = spearmanr(bull_data["f"], bull_data["r"])
            ic_bull  = float(ic_b) if not np.isnan(ic_b) else 0.0

        bear_data = aligned[bear_mask]
        if len(bear_data) >= cfg.min_regime_samples:
            ic_e, _ = spearmanr(bear_data["f"], bear_data["r"])
            ic_bear  = float(ic_e) if not np.isnan(ic_e) else 0.0

        regime_flip = (
            len(bull_data) >= cfg.min_regime_samples
            and len(bear_data) >= cfg.min_regime_samples
            and np.sign(ic_bull) != np.sign(ic_bear)
        )

        return bool(regime_flip), ic_bull, ic_bear

    # ─────────────────────────────────────────────
    # 4. QUANTILE STRUCTURE
    # ─────────────────────────────────────────────

    def _quantile_analysis(
        self,
        aligned: pd.DataFrame,  # columns: f, r
    ) -> Tuple[float, bool, bool, List[float]]:
        cfg = self.cfg
        n_q = cfg.n_quantiles

        f = aligned["f"].values
        r = aligned["r"].values

        # Quantile bucket returns
        boundaries = np.percentile(f, np.linspace(0, 100, n_q + 1))
        q_returns  = []
        for i in range(n_q):
            lo, hi = boundaries[i], boundaries[i + 1]
            mask = (f >= lo) & (f <= hi) if i == n_q - 1 else (f >= lo) & (f < hi)
            q_returns.append(float(r[mask].mean()) if mask.sum() >= 5 else 0.0)

        # Monotonicity: Spearman corr antara quantile rank dan quantile return
        ranks = np.arange(1, n_q + 1)
        mono_corr, _ = spearmanr(ranks, q_returns)
        monotonicity_score = float(mono_corr) if not np.isnan(mono_corr) else 0.0

        # Linearity: Pearson vs Spearman IC pada raw data
        pearson_ic,  _ = pearsonr(f, r)
        spearman_ic, _ = spearmanr(f, r)
        is_linear = (
            abs(spearman_ic) > 1e-6
            and abs(pearson_ic) / (abs(spearman_ic) + 1e-8) > cfg.linearity_threshold
        )

        # Tail-driven: apakah spread didominasi extreme quantile?
        if len(q_returns) >= 5:
            total_spread  = q_returns[-1] - q_returns[0]
            middle_spread = q_returns[n_q // 2] - q_returns[0]
            tail_driven = (
                abs(total_spread) > 1e-8
                and abs(middle_spread / (total_spread + 1e-10)) < cfg.tail_ratio_threshold
            )
        else:
            tail_driven = False

        return monotonicity_score, bool(is_linear), bool(tail_driven), q_returns

    # ─────────────────────────────────────────────
    # 5. CONFIDENCE SCORING
    # ─────────────────────────────────────────────

    def _score_confidence(
        self,
        temporal_stability: float,
        ic_degrading:       bool,
        regime_flip:        bool,
        monotonicity_score: float,
    ) -> Tuple[str, float]:
        cfg = self.cfg

        score  = temporal_stability * cfg.w_stability
        score += (1 - int(ic_degrading)) * cfg.w_no_degrade
        score += (1 - int(regime_flip))  * cfg.w_no_flip
        score += max(0.0, monotonicity_score) * cfg.w_monotonicity

        if score >= cfg.conf_high:
            label = CONF_HIGH
        elif score >= cfg.conf_medium:
            label = CONF_MEDIUM
        else:
            label = CONF_LOW

        return label, float(score)

    # ─────────────────────────────────────────────
    # 6. ROUTING BERDASARKAN KARAKTER
    # ─────────────────────────────────────────────

    def _route(
        self,
        decay_shape:        str,
        peak_horizon:       int,
        temporal_stability: float,
        ic_degrading:       bool,
        regime_flip:        bool,
        is_linear:          bool,
        tail_driven:        bool,
        monotonicity_score: float,
        confidence:         str,
    ) -> Tuple[str, str, str, str]:
        """
        Routing berdasarkan karakter struktural feature, bukan IC metrics.

        Urutan prioritas:
          1. Graveyard   — struktur terlalu rusak
          2. Retransform — ada signal tapi perlu re-engineering
          3. Structural  — persistent + long horizon
          4. Regime Conditional — IC flip di bull/bear
          5. Rank Based  — tail-driven, hanya extreme quantile
          6. Horizon Specific — non-monotonic decay tapi tidak flip sign
          7. Linear Stable — monotonic + linear + stabil
          8. NonLinear   — monotonic + non-linear

        Returns: (route, reason, transform, model_type)
        """
        cfg = self.cfg

        # ── 1. GRAVEYARD ─────────────────────────────────────────
        # Struktur yang tidak bisa di-exploit dalam kondisi apapun
        if ic_degrading and temporal_stability < 0.4:
            return (
                ROUTE_GRAVEYARD,
                f"IC degrading (tau<{cfg.degrade_tau}) + temporal_stability={temporal_stability:.2f} < 0.4 — signal sedang mati",
                "none",
                "none",
            )

        if decay_shape == DECAY_NON_MONOTONIC and temporal_stability < cfg.min_stability:
            return (
                ROUTE_GRAVEYARD,
                f"Non-monotonic decay + temporal_stability={temporal_stability:.2f} < {cfg.min_stability} — tidak exploitable",
                "none",
                "none",
            )

        if monotonicity_score < -0.3 and confidence == CONF_LOW:
            return (
                ROUTE_GRAVEYARD,
                f"Reverse monotonicity ({monotonicity_score:.2f}) + LOW confidence — inverse relationship tidak konsisten",
                "none",
                "none",
            )

        # ── 2. RETRANSFORM ────────────────────────────────────────
        # Ada signal tapi butuh feature engineering berbeda
        if decay_shape == DECAY_NON_MONOTONIC and temporal_stability >= cfg.min_stability:
            return (
                ROUTE_RETRANSFORM,
                f"Non-monotonic decay tapi temporal_stability={temporal_stability:.2f} OK — coba difference/ratio transform",
                "diff_or_ratio",
                "none",
            )

        if ic_degrading and temporal_stability >= 0.4:
            return (
                ROUTE_RETRANSFORM,
                f"IC degrading tapi masih punya {temporal_stability:.2f} stability — coba rolling window lebih pendek",
                "shorter_window",
                "none",
            )

        # ── 3. STRUCTURAL ─────────────────────────────────────────
        # Persistent + horizon panjang → bukan entry signal, tapi regime descriptor
        if (
            decay_shape == DECAY_PERSISTENT
            and peak_horizon >= cfg.structural_min_horizon
            and not regime_flip
            and temporal_stability >= cfg.min_stability
        ):
            return (
                ROUTE_STRUCTURAL,
                f"Persistent decay + peak_h={peak_horizon} >= {cfg.structural_min_horizon} + stable — gunakan sebagai regime conditioning",
                "rolling_zscore",
                "regime_descriptor",
            )

        # ── 4. REGIME CONDITIONAL ─────────────────────────────────
        # IC flip di bull vs bear — harus di-train terpisah per regime
        if regime_flip:
            return (
                ROUTE_REGIME_CONDITIONAL,
                f"IC flip: bull={self._fmt(0.0)}, bear={self._fmt(0.0)} — train model terpisah per regime",
                "rolling_zscore",
                "regime_conditional",
            )

        # ── 5. RANK BASED ─────────────────────────────────────────
        # Signal hanya di extreme quantile — jangan pakai sebagai continuous signal
        if tail_driven:
            return (
                ROUTE_RANK_BASED,
                f"Tail-driven: IC concentrated di extreme quantile — gunakan sebagai binary signal top/bottom {100//cfg.n_quantiles}%",
                "signed_log",
                "rank_based",
            )

        # ── 6. HORIZON SPECIFIC ───────────────────────────────────
        # Non-monotonic decay tapi sudah di-handle di retransform di atas
        # Kalau sampai sini, ini case yang lebih mild
        if decay_shape == DECAY_NON_MONOTONIC:
            return (
                ROUTE_HORIZON_SPECIFIC,
                f"Non-monotonic decay tapi stabil — gunakan HANYA di horizon {peak_horizon}",
                "horizon_specific_zscore",
                "horizon_specific",
            )

        # ── 7. LINEAR STABLE ─────────────────────────────────────
        # Best case: linear, monotonic, stabil → model sederhana sudah cukup
        if is_linear and temporal_stability >= cfg.min_stability:
            return (
                ROUTE_LINEAR_STABLE,
                f"Linear ({is_linear}) + temporal_stability={temporal_stability:.2f} >= {cfg.min_stability} — zscore + linear model",
                "zscore",
                "linear",
            )

        # ── 8. NONLINEAR ──────────────────────────────────────────
        # Monotonic tapi non-linear → perlu tree/boosting
        return (
            ROUTE_NONLINEAR,
            f"Non-linear relationship + temporal_stability={temporal_stability:.2f} — rank_transform + gradient boosting",
            "rank_transform",
            "gradient_boosting",
        )

    @staticmethod
    def _fmt(v: float) -> str:
        return f"{v:.3f}"

    # ─────────────────────────────────────────────
    # HELPER: forward return
    # ─────────────────────────────────────────────

    def _make_fwd_return(self, close: pd.Series, horizon: int) -> pd.Series:
        close_arr = close.values.astype(float)
        n = len(close_arr)
        fwd = np.full(n, np.nan)
        if horizon < n:
            fwd[:n - horizon] = (
                (close_arr[horizon:] - close_arr[:n - horizon])
                / (close_arr[:n - horizon] + 1e-10)
            )
        return pd.Series(fwd, index=close.index)

    # ─────────────────────────────────────────────
    # PUBLIC: RUN BATCH
    # ─────────────────────────────────────────────

    def run(
        self,
        df:            pd.DataFrame,
        feature_list:  List[str],
        ic_results:    dict,        # {feature_name: ICResult}
        close_col:     str = "close",
    ) -> Tuple[dict, pd.DataFrame]:
        """
        Build passport untuk semua feature yang lolos ICFilter.

        Args:
            df           : DataFrame dengan semua kolom
            feature_list : feature yang sudah lolos ICFilter
            ic_results   : dict {name: ICResult} dari ICFilter
            close_col    : nama kolom close price

        Returns:
            passports    : dict {name: FeaturePassport}
            scorecard_df : DataFrame ringkasan semua passport
        """
        close    = df[close_col]
        passports = {}

        for feat_name in feature_list:
            if feat_name not in df.columns:
                continue

            ic_res      = ic_results.get(feat_name)
            ic_direction = getattr(ic_res, "direction", "long") if ic_res else "long"
            ic_mean      = getattr(ic_res, "ic_mean",  0.0)     if ic_res else 0.0

            passport = self.build(
                feature      = df[feat_name].rename(feat_name),
                close        = close,
                ic_direction = ic_direction,
                ic_mean      = ic_mean,
            )
            passports[feat_name] = passport

        scorecard_df = self._to_dataframe(passports)
        return passports, scorecard_df

    def _to_dataframe(self, passports: dict) -> pd.DataFrame:
        rows = []
        for name, p in passports.items():
            rows.append({
                "feature":            name,
                "route":              p.route,
                "confidence":         p.confidence,
                "confidence_score":   p.confidence_score,
                "decay_shape":        p.decay_shape,
                "peak_horizon":       p.peak_horizon,
                "halflife":           p.halflife,
                "optimal_horizon":    p.optimal_horizon,
                "temporal_stability": p.temporal_stability,
                "ic_degrading":       p.ic_degrading,
                "regime_flip":        p.regime_flip,
                "ic_bull":            p.ic_bull,
                "ic_bear":            p.ic_bear,
                "monotonicity_score": p.monotonicity_score,
                "is_linear":          p.is_linear,
                "tail_driven":        p.tail_driven,
                "transform":          p.transform,
                "model_type":         p.model_type,
                "route_reason":       p.route_reason,
                "sub_period_ics":     str([f"{x:.3f}" for x in p.sub_period_ics]),
                "quantile_returns":   str([f"{x:.4f}" for x in p.quantile_returns]),
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        # Sort: linear_stable dulu (paling exploitable), graveyard terakhir
        route_order = {
            ROUTE_LINEAR_STABLE:      0,
            ROUTE_NONLINEAR:          1,
            ROUTE_RANK_BASED:         2,
            ROUTE_REGIME_CONDITIONAL: 3,
            ROUTE_HORIZON_SPECIFIC:   4,
            ROUTE_STRUCTURAL:         5,
            ROUTE_RETRANSFORM:        6,
            ROUTE_GRAVEYARD:          7,
        }
        df["_sort"] = df["route"].map(route_order).fillna(99)
        df = df.sort_values(["_sort", "confidence_score"], ascending=[True, False])
        df = df.drop(columns=["_sort"]).reset_index(drop=True)
        return df
