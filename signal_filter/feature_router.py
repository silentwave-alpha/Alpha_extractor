"""
signal_filter/feature_router.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage 1 Router — Klasifikasi & Routing per Feature

Mengambil output IC Filter (scorecard dengan semua metrics)
dan mengklasifikasi setiap feature ke salah satu dari 4 jalur:

  ROUTE_GRAVEYARD       → dead: sinyal tidak ada, buang
  ROUTE_RETRANSFORM     → noisy: ada sinyal tapi tidak exploitable as-is
  ROUTE_MICRO_ALPHA     → exploitable signal, IC stabil, horizon pendek/menengah
  ROUTE_STRUCTURAL      → market condition descriptor, IC naik di horizon panjang
                          → bukan entry signal, tapi conditioning variable

Kriteria routing ditentukan secara empiris dari data, bukan asumsi domain.
Hasilnya bisa dipakai untuk:
  1. Update category tags di FeatureRegistry (base_builder)
  2. Guard di InteractionBuilder (skip structural × signal)
  3. Populate feature_pool dan regime_features di Atomic Mining
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────────────
# ROUTE LABELS
# ─────────────────────────────────────────────────────────────────

ROUTE_GRAVEYARD   = "graveyard"        # dead feature
ROUTE_RETRANSFORM = "retransform"      # noisy, butuh re-engineering
ROUTE_MICRO_ALPHA = "micro_alpha"      # → Atomic Mining feature_pool
ROUTE_STRUCTURAL  = "structural"       # → Atomic Mining regime conditioning


@dataclass
class RouterConfig:
    # ── GRAVEYARD thresholds ──────────────────────────────────────
    # Feature masuk graveyard jika SALAH SATU terpenuhi
    dead_max_ic_mean:    float = 0.02   # |IC mean| terlalu kecil
    dead_max_t_stat:     float = 1.5    # t-stat tidak signifikan
    dead_max_ic_pos_pct: float = 0.45   # DEPRECATED: tidak dipakai untuk reject short signals

    # ── NOISY thresholds ─────────────────────────────────────────
    # Feature masuk retransform jika KOMBINASI terpenuhi
    noisy_min_ic_ir:            float = 0.25   # IC IR terlalu rendah (tidak stabil)
    noisy_max_half_life:        float = 3.0    # half-life sangat pendek (< 3 bars)
    noisy_min_significant_horizons: int = 2    # IC signifikan di < 2 horizon

    # IC dianggap "signifikan" di suatu horizon jika |IC| > threshold ini
    horizon_significance_threshold: float = 0.015

    # ── STRUCTURAL thresholds ─────────────────────────────────────
    # Feature masuk structural jika SEMUA terpenuhi
    structural_min_peak_horizon: int   = 72    # peak IC di horizon panjang
    structural_min_trend_ratio:  float = 1.5   # peak_ic / ic_h1 > threshold
                                               # (IC naik signifikan ke horizon panjang)

    # ── MICRO_ALPHA thresholds ────────────────────────────────────
    # Default: semua yang lolos graveyard + noisy + bukan structural
    micro_alpha_max_peak_horizon: int = 100    # peak horizon tidak terlalu panjang


@dataclass
class RouteResult:
    """Hasil routing untuk satu feature."""
    feature:    str
    route:      str             # salah satu dari ROUTE_* constants
    signal_type: str            # dari ICResult: mean_reversion/trend/delayed_peak/flat
    direction:  str             # long / short / both
    peak_horizon: Optional[int]
    half_life:    Optional[float]
    half_life_reliable: bool
    best_horizon: Optional[int]
    ic_mean:    float
    ic_ir:      float
    t_stat:     float
    peak_ic:    float
    n_significant_horizons: int
    route_reason: str           # penjelasan kenapa masuk route ini


@dataclass
class RouterResult:
    """Output lengkap Stage 1 Router."""

    # Feature lists per route
    micro_alpha:  List[str] = field(default_factory=list)
    structural:   List[str] = field(default_factory=list)
    retransform:  List[str] = field(default_factory=list)
    graveyard:    List[str] = field(default_factory=list)

    # Horizon map — hanya untuk micro_alpha
    # structural tidak punya horizon_map karena tidak di-mine sebagai signal
    horizon_map:   Dict[str, int] = field(default_factory=dict)

    # Direction map — long / short / both per micro_alpha feature
    direction_map: Dict[str, str] = field(default_factory=dict)

    # Detail per feature
    route_details: List[RouteResult] = field(default_factory=list)

    # Scorecard DataFrame (gabungan IC scorecard + routing)
    scorecard: Optional[pd.DataFrame] = None

    @property
    def summary(self) -> Dict[str, int]:
        return {
            "micro_alpha":  len(self.micro_alpha),
            "structural":   len(self.structural),
            "retransform":  len(self.retransform),
            "graveyard":    len(self.graveyard),
            "total":        len(self.route_details),
        }


class FeatureRouter:
    """
    Stage 1 Router.

    Input : IC scorecard DataFrame (output dari ICFilter.run())
    Output: RouterResult dengan 4 jalur routing + horizon_map

    Urutan evaluasi per feature:
      1. Graveyard check  — IC terlalu lemah atau tidak signifikan
      2. Noisy check      — IC ada tapi tidak stabil/exploitable
      3. Structural check — IC naik di horizon panjang (regime descriptor)
      4. Default          — Micro Alpha (exploitable signal)
    """

    def __init__(self, config: Optional[RouterConfig] = None):
        self.config = config or RouterConfig()

    # ─────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────

    def _count_significant_horizons(self, row: pd.Series) -> int:
        """
        Hitung berapa horizon yang punya |IC| > threshold.
        Kolom IC per horizon ada di scorecard sebagai ic_h1, ic_h5, dst.
        """
        threshold = self.config.horizon_significance_threshold
        ic_cols   = [c for c in row.index if c.startswith("ic_h")]
        count     = 0
        for col in ic_cols:
            val = row.get(col, np.nan)
            if pd.notna(val) and abs(val) > threshold:
                count += 1
        return count

    def _get_trend_ratio(self, row: pd.Series) -> float:
        """
        Hitung rasio peak_ic / ic_h1.
        Ratio tinggi = IC naik signifikan dari h=1 ke horizon panjang.
        """
        ic_h1    = abs(row.get("ic_h1", 0.0) or 0.0)
        peak_ic  = abs(row.get("peak_ic", 0.0) or 0.0)
        return peak_ic / (ic_h1 + 1e-10)

    # ─────────────────────────────────────────────────────────────
    # ROUTING LOGIC PER FEATURE
    # ─────────────────────────────────────────────────────────────

    def _route_one(self, row: pd.Series) -> RouteResult:
        cfg     = self.config
        feature = row["feature"]

        # Extract metrics
        ic_mean    = float(row.get("ic_mean",   0.0) or 0.0)
        ic_ir      = float(row.get("ic_ir",     0.0) or 0.0)
        t_stat     = float(row.get("t_stat",    0.0) or 0.0)
        ic_pos_pct = float(row.get("ic_positive_pct", 0.0) or 0.0)
        peak_ic    = float(row.get("peak_ic",   0.0) or 0.0)
        signal_type    = str(row.get("signal_type",  "unknown") or "unknown")
        peak_horizon   = row.get("peak_horizon")
        half_life      = row.get("half_life")
        hl_reliable    = bool(row.get("half_life_reliable", False))
        best_horizon   = row.get("best_horizon")

        peak_horizon = int(peak_horizon) if pd.notna(peak_horizon) else None
        best_horizon = int(best_horizon) if pd.notna(best_horizon) else None
        half_life    = float(half_life)  if pd.notna(half_life)    else None

        n_sig_horizons = self._count_significant_horizons(row)
        trend_ratio    = self._get_trend_ratio(row)

        direction = str(row.get("direction", "long") or "long")

        def make(route, reason):
            return RouteResult(
                feature=feature,
                route=route,
                signal_type=signal_type,
                direction=direction,
                peak_horizon=peak_horizon,
                half_life=half_life,
                half_life_reliable=hl_reliable,
                best_horizon=best_horizon,
                ic_mean=ic_mean,
                ic_ir=ic_ir,
                t_stat=t_stat,
                peak_ic=peak_ic,
                n_significant_horizons=n_sig_horizons,
                route_reason=reason,
            )

        # ── 1. GRAVEYARD ─────────────────────────────────────────
        # Cek kondisi dead — salah satu saja sudah cukup
        if abs(ic_mean) < cfg.dead_max_ic_mean:
            return make(ROUTE_GRAVEYARD,
                f"|ic_mean|={abs(ic_mean):.4f} < {cfg.dead_max_ic_mean}")

        if abs(t_stat) < cfg.dead_max_t_stat:
            return make(ROUTE_GRAVEYARD,
                f"|t_stat|={abs(t_stat):.2f} < {cfg.dead_max_t_stat}")

        # Direction-aware consistency check (gantikan ic_pos_pct dead check)
        # long/both : ic_pos_pct harus cukup tinggi
        # short     : ic_neg_pct harus cukup tinggi
        ic_neg_pct = float(row.get("ic_negative_pct", 1 - ic_pos_pct) or 0.0)
        if direction == "short":
            consistency_ok = ic_neg_pct >= cfg.dead_max_ic_pos_pct
        else:
            consistency_ok = ic_pos_pct >= cfg.dead_max_ic_pos_pct
        if not consistency_ok:
            return make(ROUTE_GRAVEYARD,
                f"ic_consistency: direction={direction}, "
                f"pos%={ic_pos_pct:.2%}, neg%={ic_neg_pct:.2%} — "
                f"tidak konsisten ke arah {direction}")

        # ── 2. NOISY / RETRANSFORM ────────────────────────────────
        # Kombinasi beberapa kondisi noisy
        noisy_flags = []

        if abs(ic_ir) < cfg.noisy_min_ic_ir:
            noisy_flags.append(f"ic_ir={ic_ir:.3f} < {cfg.noisy_min_ic_ir}")

        if (half_life is not None and hl_reliable
                and half_life < cfg.noisy_max_half_life):
            noisy_flags.append(
                f"half_life={half_life:.1f} < {cfg.noisy_max_half_life} (spike only)"
            )

        if n_sig_horizons < cfg.noisy_min_significant_horizons:
            noisy_flags.append(
                f"significant_horizons={n_sig_horizons} < {cfg.noisy_min_significant_horizons}"
            )

        # Butuh >= 2 noisy flags untuk di-route ke retransform
        # (1 flag saja bisa false positive)
        if len(noisy_flags) >= 2:
            return make(ROUTE_RETRANSFORM, " | ".join(noisy_flags))

        # ── 3. STRUCTURAL ─────────────────────────────────────────
        # IC naik signifikan ke horizon panjang → regime descriptor
        is_structural = (
            peak_horizon is not None
            and peak_horizon >= cfg.structural_min_peak_horizon
            and trend_ratio >= cfg.structural_min_trend_ratio
        )

        # Signal type dari ICFilter juga bisa konfirmasi
        if signal_type == "trend" and peak_horizon is not None:
            is_structural = is_structural or (
                peak_horizon >= cfg.structural_min_peak_horizon
            )

        if is_structural:
            return make(ROUTE_STRUCTURAL,
                f"peak_h={peak_horizon} >= {cfg.structural_min_peak_horizon} "
                f"& trend_ratio={trend_ratio:.2f} >= {cfg.structural_min_trend_ratio} "
                f"(signal_type={signal_type})"
            )

        # ── 4. MICRO ALPHA (default) ──────────────────────────────
        return make(ROUTE_MICRO_ALPHA,
            f"IC stabil: ic_ir={ic_ir:.3f}, t={t_stat:.2f}, "
            f"n_sig_h={n_sig_horizons}, peak_h={peak_horizon}, "
            f"signal_type={signal_type}"
        )

    # ─────────────────────────────────────────────────────────────
    # PUBLIC: Run router
    # ─────────────────────────────────────────────────────────────

    def run(
        self,
        ic_scorecard: pd.DataFrame,
        only_passed: bool = True,
    ) -> RouterResult:
        """
        Route semua features berdasarkan IC scorecard.

        Args:
            ic_scorecard : DataFrame output dari ICFilter.run()
                           Harus punya kolom: feature, passed, ic_mean,
                           ic_ir, t_stat, ic_positive_pct, signal_type,
                           peak_horizon, half_life, best_horizon, peak_ic
                           + ic_h{N} columns dari decay curve

            only_passed  : jika True, hanya route features yang passed=True
                           di IC filter. Features yang failed di IC filter
                           langsung masuk graveyard tanpa re-evaluate.

        Returns:
            RouterResult dengan 4 jalur + horizon_map + scorecard
        """
        result = RouterResult()
        details = []

        for _, row in ic_scorecard.iterrows():
            feature = row["feature"]
            passed  = bool(row.get("passed", False))

            # Features yang gagal IC filter → langsung graveyard
            if only_passed and not passed:
                reject = str(row.get("reject_reason", "") or
                             row.get("reject_ic", "") or
                             row.get("reject_sanity", "") or
                             "failed_ic_filter")
                details.append(RouteResult(
                    feature=feature,
                    route=ROUTE_GRAVEYARD,
                    signal_type="unknown",
                    direction="unknown",
                    peak_horizon=None,
                    half_life=None,
                    half_life_reliable=False,
                    best_horizon=None,
                    ic_mean=float(row.get("ic_mean", 0.0) or 0.0),
                    ic_ir=float(row.get("ic_ir", 0.0) or 0.0),
                    t_stat=float(row.get("t_stat", 0.0) or 0.0),
                    peak_ic=float(row.get("peak_ic", 0.0) or 0.0),
                    n_significant_horizons=0,
                    route_reason=f"failed_ic_filter: {reject}",
                ))
                result.graveyard.append(feature)
                continue

            # Guard: jika IC metrics NaN meski passed=True → graveyard
            if pd.isna(row.get("ic_mean", np.nan)):
                details.append(RouteResult(
                    feature=feature, route=ROUTE_GRAVEYARD,
                    signal_type="unknown", direction="unknown", peak_horizon=None,
                    half_life=None, half_life_reliable=False, best_horizon=None,
                    ic_mean=0.0, ic_ir=0.0, t_stat=0.0, peak_ic=0.0,
                    n_significant_horizons=0,
                    route_reason="ic_metrics_nan",
                ))
                result.graveyard.append(feature)
                continue

            # Route berdasarkan IC metrics
            route_result = self._route_one(row)
            details.append(route_result)

            if route_result.route == ROUTE_MICRO_ALPHA:
                result.micro_alpha.append(feature)
                bh = route_result.best_horizon
                result.horizon_map[feature]   = bh if bh is not None else 24
                result.direction_map[feature] = route_result.direction

            elif route_result.route == ROUTE_STRUCTURAL:
                result.structural.append(feature)
                # Structural tidak punya horizon_map (bukan entry signal)

            elif route_result.route == ROUTE_RETRANSFORM:
                result.retransform.append(feature)

            elif route_result.route == ROUTE_GRAVEYARD:
                result.graveyard.append(feature)

        result.route_details = details

        # Build scorecard: merge IC scorecard + routing info
        result.scorecard = self._build_scorecard(ic_scorecard, details)

        return result

    # ─────────────────────────────────────────────────────────────
    # SCORECARD
    # ─────────────────────────────────────────────────────────────

    def _build_scorecard(
        self,
        ic_scorecard: pd.DataFrame,
        details:      List[RouteResult],
    ) -> pd.DataFrame:
        """
        Gabungkan IC scorecard dengan routing result.
        Tambah kolom: route, route_reason, n_significant_horizons, trend_ratio.
        """
        routing_rows = []
        for d in details:
            routing_rows.append({
                "feature":               d.feature,
                "route":                 d.route,
                "direction":             d.direction,
                "route_reason":          d.route_reason,
                "n_significant_horizons": d.n_significant_horizons,
            })

        routing_df = pd.DataFrame(routing_rows)

        # Merge dengan IC scorecard
        merged = ic_scorecard.merge(routing_df, on="feature", how="left")

        # Tambah trend_ratio kolom
        if "ic_h1" in merged.columns and "peak_ic" in merged.columns:
            merged["trend_ratio"] = (
                merged["peak_ic"].abs()
                / (merged["ic_h1"].abs() + 1e-10)
            )

        # Sort: micro_alpha dulu, lalu structural, retransform, graveyard
        route_order = {
            ROUTE_MICRO_ALPHA: 0,
            ROUTE_STRUCTURAL:  1,
            ROUTE_RETRANSFORM: 2,
            ROUTE_GRAVEYARD:   3,
        }
        merged["_sort"] = merged["route"].map(route_order).fillna(99)
        merged = merged.sort_values(["_sort", "ic_ir"], ascending=[True, False])
        merged = merged.drop(columns=["_sort"]).reset_index(drop=True)

        return merged

    # ─────────────────────────────────────────────────────────────
    # DISPLAY HELPERS
    # ─────────────────────────────────────────────────────────────

    def print_summary(self, result: RouterResult) -> None:
        """Print ringkasan routing ke console."""
        s = result.summary
        print("═" * 60)
        print("STAGE 1 ROUTER RESULTS")
        print(f"  Total features evaluated : {s['total']}")
        print(f"  ⚛️  Micro Alpha            : {s['micro_alpha']}")
        print(f"  🏛️  Structural (regime)    : {s['structural']}")
        print(f"  🔧 Retransform (noisy)    : {s['retransform']}")
        print(f"  💀 Graveyard (dead)       : {s['graveyard']}")
        print("─" * 60)

        if result.micro_alpha:
            print("\n⚛️  MICRO ALPHA → Atomic Mining:")
            for feat in result.micro_alpha:
                h  = result.horizon_map.get(feat, "?")
                dr = next((d for d in result.route_details if d.feature == feat), None)
                sig = dr.signal_type if dr else "?"
                hl  = f"{dr.half_life:.1f}" if dr and dr.half_life else "?"
                print(f"  {feat:<35} h={h:<5} type={sig:<18} hl={hl}")

        if result.structural:
            print("\n🏛️  STRUCTURAL → Regime Conditioning:")
            for feat in result.structural:
                dr = next((d for d in result.route_details if d.feature == feat), None)
                ph = dr.peak_horizon if dr else "?"
                tr = f"{dr.peak_ic / (abs(dr.ic_mean) + 1e-10):.1f}x" if dr else "?"
                print(f"  {feat:<35} peak_h={ph:<5} trend_ratio={tr}")

        if result.retransform:
            print("\n🔧 RETRANSFORM → Butuh Feature Engineering:")
            for feat in result.retransform:
                dr = next((d for d in result.route_details if d.feature == feat), None)
                reason = dr.route_reason if dr else "?"
                print(f"  {feat:<35} {reason}")

        print("═" * 60)
