"""
signal_filter/filter_pipeline.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Signal Filter Pipeline — Orchestrator

Urutan eksekusi:
  Layer A → Sanity Filter      : buang fitur statistik rusak
  Layer B → IC Filter + Decay  : IC mean, IR, t-stat, decay curve, signal type
  Stage 1 → Feature Router     : routing ke 4 jalur (micro_alpha / structural /
                                  retransform / graveyard)
  Layer C → Sign Consistency   : hanya untuk micro_alpha + structural
  Layer D → Decorrelation      : hanya untuk micro_alpha

Output utama:
  router_result  : RouterResult dengan 4 jalur + horizon_map
  scorecard_df   : tabel lengkap semua metrics + routing label
"""

import time
import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from .sanity_filter import SanityFilter, SanityConfig
from .ic_filter import ICFilter, ICConfig
from .sign_consistency_filter import SignConsistencyFilter, SignConsistencyConfig
from .decorrelation_filter import DecorrelationFilter, DecorrelationConfig
from .feature_router import (
    FeatureRouter, RouterConfig, RouterResult,
    ROUTE_MICRO_ALPHA, ROUTE_STRUCTURAL, ROUTE_RETRANSFORM, ROUTE_GRAVEYARD
)
from .feature_passport import (
    PassportBuilder, PassportConfig, FeaturePassport,
    ROUTE_LINEAR_STABLE, ROUTE_NONLINEAR, ROUTE_RANK_BASED,
    ROUTE_REGIME_CONDITIONAL, ROUTE_HORIZON_SPECIFIC,
    ROUTE_STRUCTURAL as PASSPORT_STRUCTURAL,
    ROUTE_RETRANSFORM as PASSPORT_RETRANSFORM,
    ROUTE_GRAVEYARD as PASSPORT_GRAVEYARD,
)


@dataclass
class FilterPipelineConfig:
    sanity:           SanityConfig          = field(default_factory=SanityConfig)
    ic:               ICConfig              = field(default_factory=ICConfig)
    sign_consistency: SignConsistencyConfig = field(default_factory=SignConsistencyConfig)
    decorrelation:    DecorrelationConfig   = field(default_factory=DecorrelationConfig)
    router:           RouterConfig          = field(default_factory=RouterConfig)
    passport:         PassportConfig        = field(default_factory=PassportConfig)

    # Layer control
    run_sanity:           bool = True
    run_ic:               bool = True
    run_passport:         bool = True
    run_router:           bool = True
    run_sign_consistency: bool = True
    run_decorrelation:    bool = True

    close_col:       str = "close"
    default_horizon: int = 24


class FilterPipeline:
    """
    Orchestrator Pipeline Stage 1.

    Usage:
        pipeline = FilterPipeline(config)
        router_result, scorecard = pipeline.run(df, feature_list)

        # Micro alpha → Atomic Mining
        feature_pool = router_result.micro_alpha
        horizon_map  = router_result.horizon_map

        # Structural → Regime Conditioning (bukan entry signal)
        regime_features = router_result.structural
    """

    def __init__(
        self,
        config: Optional[FilterPipelineConfig] = None,
        logger: Optional[logging.Logger]        = None,
    ):
        self.config = config or FilterPipelineConfig()
        self.logger = logger or self._default_logger()

        cfg = self.config
        self.sanity_filter   = SanityFilter(cfg.sanity)
        self.ic_filter       = ICFilter(cfg.ic)
        self.passport_builder = PassportBuilder(cfg.passport)
        self.sign_filter     = SignConsistencyFilter(cfg.sign_consistency)
        self.decorr_filter   = DecorrelationFilter(cfg.decorrelation)
        self.router          = FeatureRouter(cfg.router)

    def _default_logger(self) -> logging.Logger:
        logger = logging.getLogger("filter_pipeline_" + str(id(self)))
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(ch)
        return logger

    def _log(self, msg: str):
        self.logger.info(msg)

    # ─────────────────────────────────────────────────────────────
    # MAIN RUN
    # ─────────────────────────────────────────────────────────────

    def run(
        self,
        df:           pd.DataFrame,
        feature_list: List[str],
    ) -> Tuple[RouterResult, pd.DataFrame]:
        """
        Jalankan full pipeline.

        Returns:
            router_result : RouterResult dengan routing berdasarkan passport character
            scorecard_df  : DataFrame lengkap semua metrics + route label + passport info
        """
        cfg     = self.config
        t0      = time.time()
        n_input = len(feature_list)

        self._log("═" * 60)
        self._log("SIGNAL FILTER PIPELINE — STAGE 1")
        self._log(f"  Input  : {n_input} features")
        self._log("═" * 60)

        current_pool  = list(feature_list)
        sanity_df     = pd.DataFrame()
        ic_df         = pd.DataFrame()
        sign_df       = pd.DataFrame()
        passport_df   = pd.DataFrame()
        passports     = {}          # {feature_name: FeaturePassport}
        ic_result_map = {}          # {feature_name: ICResult} — dipakai PassportBuilder

        # ── Layer A: Sanity ───────────────────────────────────────
        if cfg.run_sanity:
            t1 = time.time()
            current_pool, sanity_df = self.sanity_filter.run(df, current_pool)
            n_drop = n_input - len(current_pool)
            self._log(f"[A] Sanity    : {n_input} → {len(current_pool)} "
                      f"(-{n_drop}) [{time.time()-t1:.1f}s]")
        else:
            self._log("[A] Sanity    : SKIPPED")

        # ── Layer B: IC Filter + Decay ────────────────────────────
        if cfg.run_ic and current_pool:
            n_before = len(current_pool)
            t1 = time.time()
            passed_ic, ic_df = self.ic_filter.run(
                df, current_pool, close_col=cfg.close_col, base_horizon=1
            )
            self._log(f"[B] IC Filter : {n_before} → {len(passed_ic)} passed "
                      f"[{time.time()-t1:.1f}s]")

            # Log signal type breakdown dari semua yang dievaluasi
            if not ic_df.empty and "signal_type" in ic_df.columns:
                passed_sc = ic_df[ic_df["passed"] == True]
                type_counts = passed_sc["signal_type"].value_counts()
                for stype, cnt in type_counts.items():
                    self._log(f"    {stype:<20} : {cnt}")
        else:
            self._log("[B] IC Filter : SKIPPED")

        # ── Layer B.5: Passport Characterization ──────────────────
        # Build passport untuk setiap feature yang lolos ICFilter.
        # Passport berisi: decay shape, temporal stability, regime flip,
        # quantile structure, confidence → basis routing decision.
        if cfg.run_passport and not ic_df.empty:
            t1 = time.time()
            passed_ic = ic_df[ic_df["passed"] == True]["feature"].tolist()

            # Build ic_result_map dari ICResult objects
            # ICFilter.run() returns list + df — kita pakai df untuk rebuild map
            if passed_ic:
                ic_result_map = self._build_ic_result_map(ic_df)
                passports, passport_df = self.passport_builder.run(
                    df           = df,
                    feature_list = passed_ic,
                    ic_results   = ic_result_map,
                    close_col    = cfg.close_col,
                )

                # Log route distribution dari passport
                if not passport_df.empty:
                    route_counts = passport_df["route"].value_counts()
                    self._log(f"[P] Passport  : {len(passed_ic)} features characterized [{time.time()-t1:.1f}s]")
                    for route, cnt in route_counts.items():
                        self._log(f"    {route:<25} : {cnt}")

                # Features yang passport-nya graveyard/retransform
                # langsung di-exclude dari pool sebelum masuk router
                passport_graveyard   = passport_df[passport_df["route"] == PASSPORT_GRAVEYARD]["feature"].tolist()
                passport_retransform = passport_df[passport_df["route"] == PASSPORT_RETRANSFORM]["feature"].tolist()
                passport_structural  = passport_df[passport_df["route"] == PASSPORT_STRUCTURAL]["feature"].tolist()

                if passport_graveyard:
                    self._log(f"    → {len(passport_graveyard)} ke graveyard (struktur rusak)")
                if passport_retransform:
                    self._log(f"    → {len(passport_retransform)} ke retransform (butuh re-engineering)")
                if passport_structural:
                    self._log(f"    → {len(passport_structural)} ke structural (persistent/long-horizon)")
        else:
            self._log("[P] Passport  : SKIPPED")

        # ── Stage 1: Router (diperkaya dengan Passport) ───────────
        router_result = RouterResult()
        if cfg.run_router and not ic_df.empty:
            t1 = time.time()

            # Merge sanity + IC + passport untuk input router
            full_scorecard = self._merge_sanity_ic(sanity_df, ic_df, feature_list)

            # Inject passport route sebagai override hint ke router
            if not passport_df.empty:
                full_scorecard = self._inject_passport(full_scorecard, passport_df)

            router_result = self.router.run(full_scorecard, only_passed=True)

            # Override router result dengan passport decisions
            if passports:
                router_result = self._apply_passport_routing(
                    router_result, passports, passport_df
                )

            s = router_result.summary
            self._log(f"[R] Router    : [{time.time()-t1:.1f}s]")
            self._log(f"    ⚛️  micro_alpha : {s['micro_alpha']}")
            self._log(f"    🏛️  structural  : {s['structural']}")
            self._log(f"    🔧 retransform : {s['retransform']}")
            self._log(f"    💀 graveyard   : {s['graveyard']}")

            if router_result.micro_alpha:
                self._log("    Micro Alpha pool:")
                for feat in router_result.micro_alpha:
                    h   = router_result.horizon_map.get(feat, "?")
                    p   = passports.get(feat)
                    sig = p.decay_shape if p else "?"
                    mod = p.model_type  if p else "?"
                    conf= p.confidence  if p else "?"
                    self._log(f"      {feat:<30} h={h:<5} shape={sig:<15} model={mod:<20} conf={conf}")

            if router_result.structural:
                self._log("    Structural (→ regime conditioning):")
                for feat in router_result.structural:
                    p = passports.get(feat)
                    ph = p.peak_horizon if p else "?"
                    self._log(f"      {feat:<30} peak_h={ph}")
        else:
            self._log("[R] Router    : SKIPPED")

        # ── Layer C: Sign Consistency (hanya micro_alpha) ─────────
        # Structural tidak perlu sign consistency — dia bukan entry signal
        micro_after_sign = router_result.micro_alpha.copy()
        sign_dropped     = []

        if cfg.run_sign_consistency and router_result.micro_alpha:
            n_before = len(router_result.micro_alpha)
            t1 = time.time()

            # Build direction_map dari IC results untuk di-pass ke sign filter
            ic_direction_map = {}
            if not ic_df.empty and "direction" in ic_df.columns:
                ic_direction_map = dict(zip(ic_df["feature"], ic_df["direction"].fillna("long")))

            # Gunakan passport optimal_horizon per feature (bukan hardcoded h=1)
            # Karena setiap feature punya best horizon berbeda
            passport_horizon_map = {}
            if passports:
                passport_horizon_map = {
                    f: p.optimal_horizon
                    for f, p in passports.items()
                    if f in router_result.micro_alpha
                }

            passed_sign, sign_df = self.sign_filter.run(
                df, router_result.micro_alpha,
                close_col=cfg.close_col,
                horizon=1,  # fallback default
                direction_map=ic_direction_map,
                horizon_map=passport_horizon_map,  # per-feature override
            )
            micro_after_sign = passed_sign
            sign_dropped     = [f for f in router_result.micro_alpha if f not in passed_sign]
            self._log(f"[C] Sign      : {n_before} → {len(passed_sign)} "
                      f"(-{len(sign_dropped)}) [{time.time()-t1:.1f}s]")

            # Features yang drop di sign consistency → pindah ke retransform
            if sign_dropped:
                router_result.micro_alpha = passed_sign
                router_result.retransform.extend(sign_dropped)
                self._log(f"    → {len(sign_dropped)} dipindah ke retransform (sign consistency)")
        else:
            self._log("[C] Sign      : SKIPPED")

        # ── Layer D: Decorrelation (hanya micro_alpha) ────────────
        if cfg.run_decorrelation and len(micro_after_sign) > 1:
            n_before = len(micro_after_sign)
            t1 = time.time()

            ic_ir_scores = {}
            if not ic_df.empty and "ic_ir" in ic_df.columns:
                ic_ir_scores = dict(zip(ic_df["feature"], ic_df["ic_ir"].fillna(0)))

            decorr_features, decorr_result = self.decorr_filter.run(
                df, micro_after_sign, ic_ir_scores
            )
            decorr_dropped = [f for f in micro_after_sign if f not in decorr_features]

            router_result.micro_alpha = decorr_features
            # Decorr-dropped masuk graveyard (redundant, bukan butuh retransform)
            router_result.graveyard.extend(decorr_dropped)

            # Update horizon_map — hapus yang decorr-dropped
            for f in decorr_dropped:
                router_result.horizon_map.pop(f, None)

            self._log(f"[D] Decorr    : {n_before} → {len(decorr_features)} "
                      f"(-{len(decorr_dropped)} redundant) [{time.time()-t1:.1f}s]")
        else:
            self._log("[D] Decorr    : SKIPPED")

        # ── Build final scorecard ─────────────────────────────────
        final_scorecard = self._build_final_scorecard(
            feature_list, sanity_df, ic_df, sign_df, router_result, passport_df
        )
        router_result.scorecard = final_scorecard

        # ── Summary ───────────────────────────────────────────────
        elapsed = time.time() - t0
        s = router_result.summary
        self._log("─" * 60)
        self._log(f"DONE [{elapsed:.1f}s] | "
                  f"⚛️ {s['micro_alpha']} micro_alpha | "
                  f"🏛️ {s['structural']} structural | "
                  f"🔧 {s['retransform']} retransform | "
                  f"💀 {s['graveyard']} graveyard")
        self._log("═" * 60)

        return router_result, final_scorecard

    # ─────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────

    def _merge_sanity_ic(
        self,
        sanity_df:    pd.DataFrame,
        ic_df:        pd.DataFrame,
        feature_list: List[str],
    ) -> pd.DataFrame:
        """Merge sanity dan IC scorecard untuk input ke router."""
        base = pd.DataFrame({"feature": feature_list})

        if not sanity_df.empty:
            s_cols = ["feature", "passed", "reject_reason"]
            s_merge = sanity_df[[c for c in s_cols if c in sanity_df.columns]].copy()
            s_merge = s_merge.rename(columns={
                "passed": "passed_sanity",
                "reject_reason": "reject_sanity"
            })
            base = base.merge(s_merge, on="feature", how="left")

        if not ic_df.empty:
            base = base.merge(ic_df, on="feature", how="left")

        return base

    def _build_ic_result_map(self, ic_df: pd.DataFrame) -> dict:
        """
        Buat lightweight proxy dari IC scorecard DataFrame.
        PassportBuilder butuh .direction dan .ic_mean per feature.
        """
        class _ICProxy:
            def __init__(self, direction, ic_mean):
                self.direction = direction
                self.ic_mean   = ic_mean

        result = {}
        for _, row in ic_df.iterrows():
            if row.get("passed", False):
                result[row["feature"]] = _ICProxy(
                    direction = str(row.get("direction", "long") or "long"),
                    ic_mean   = float(row.get("ic_mean",  0.0)  or 0.0),
                )
        return result

    def _inject_passport(
        self,
        full_scorecard: pd.DataFrame,
        passport_df:    pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Tambahkan kolom passport_route ke full_scorecard.
        Router bisa membaca ini sebagai hint.
        """
        p_cols = ["feature", "route", "confidence", "confidence_score",
                  "decay_shape", "temporal_stability", "ic_degrading",
                  "regime_flip", "monotonicity_score", "is_linear",
                  "tail_driven", "transform", "model_type", "optimal_horizon"]
        p_sub = passport_df[[c for c in p_cols if c in passport_df.columns]].copy()
        p_sub = p_sub.rename(columns={
            "route":            "passport_route",
            "transform":        "passport_transform",
            "model_type":       "passport_model",
            "optimal_horizon":  "passport_horizon",
        })
        return full_scorecard.merge(p_sub, on="feature", how="left")

    def _apply_passport_routing(
        self,
        router_result: RouterResult,
        passports:     dict,
        passport_df:   pd.DataFrame,
    ) -> RouterResult:
        """
        Override router_result berdasarkan passport decisions.

        Passport routes yang di-override:
          PASSPORT_GRAVEYARD   → pindah dari micro_alpha ke graveyard
          PASSPORT_RETRANSFORM → pindah dari micro_alpha ke retransform
          PASSPORT_STRUCTURAL  → pindah dari micro_alpha ke structural
          Lainnya (linear_stable, nonlinear, rank_based, regime_conditional,
                   horizon_specific) → TETAP di micro_alpha, tapi update
                   horizon_map dengan passport optimal_horizon

        Kenapa tidak semua di-override?
          Router lama masih valid sebagai second opinion untuk IC metrics.
          Passport menambahkan structural knowledge, bukan menggantikan IC check.
        """
        from .feature_passport import (
            ROUTE_GRAVEYARD as PG, ROUTE_RETRANSFORM as PR,
            ROUTE_STRUCTURAL as PS
        )

        # Build set untuk quick lookup
        currently_micro = set(router_result.micro_alpha)

        for feat_name, passport in passports.items():
            p_route = passport.route

            if feat_name not in currently_micro:
                continue  # Sudah di-handle oleh IC router, skip

            if p_route == PG:
                # Passport: graveyard → pindahkan dari micro_alpha
                router_result.micro_alpha = [
                    f for f in router_result.micro_alpha if f != feat_name
                ]
                router_result.graveyard.append(feat_name)
                router_result.horizon_map.pop(feat_name, None)
                router_result.direction_map.pop(feat_name, None)

            elif p_route == PR:
                # Passport: retransform → pindahkan dari micro_alpha
                router_result.micro_alpha = [
                    f for f in router_result.micro_alpha if f != feat_name
                ]
                router_result.retransform.append(feat_name)
                router_result.horizon_map.pop(feat_name, None)
                router_result.direction_map.pop(feat_name, None)

            elif p_route == PS:
                # Passport: structural → pindahkan dari micro_alpha
                router_result.micro_alpha = [
                    f for f in router_result.micro_alpha if f != feat_name
                ]
                router_result.structural.append(feat_name)
                router_result.horizon_map.pop(feat_name, None)
                router_result.direction_map.pop(feat_name, None)

            else:
                # Feature tetap di micro_alpha, update horizon_map
                # dengan passport optimal_horizon (lebih akurat dari IC h=1)
                if passport.optimal_horizon:
                    router_result.horizon_map[feat_name] = passport.optimal_horizon

        return router_result

    def _build_final_scorecard(
        self,
        feature_list:  List[str],
        sanity_df:     pd.DataFrame,
        ic_df:         pd.DataFrame,
        sign_df:       pd.DataFrame,
        router_result: RouterResult,
        passport_df:   pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Build scorecard akhir dengan semua metrics + route label + passport info."""

        if router_result.scorecard is not None and not router_result.scorecard.empty:
            base = router_result.scorecard.copy()
        else:
            base = pd.DataFrame({"feature": feature_list})
            if not ic_df.empty:
                base = base.merge(ic_df, on="feature", how="left")

        # Merge sign consistency
        if not sign_df.empty:
            s_cols = ["feature", "passed", "monotonicity_score",
                      "spread_consistency", "long_quality", "reject_reason"]
            s_merge = sign_df[[c for c in s_cols if c in sign_df.columns]].copy()
            s_merge = s_merge.rename(columns={
                "passed": "passed_sign",
                "reject_reason": "reject_sign"
            })
            base = base.merge(s_merge, on="feature", how="left")

        # Merge passport info
        if passport_df is not None and not passport_df.empty:
            p_cols = ["feature", "route", "confidence", "confidence_score",
                      "decay_shape", "temporal_stability", "ic_degrading",
                      "regime_flip", "monotonicity_score", "is_linear",
                      "tail_driven", "transform", "model_type",
                      "route_reason", "optimal_horizon", "halflife"]
            p_sub = passport_df[[c for c in p_cols if c in passport_df.columns]].copy()
            p_sub = p_sub.rename(columns={
                "route":         "passport_route",
                "route_reason":  "passport_reason",
            })
            base = base.merge(p_sub, on="feature", how="left")

        base["assigned_horizon"] = base["feature"].map(router_result.horizon_map)

        route_order = {
            ROUTE_MICRO_ALPHA: 0,
            ROUTE_STRUCTURAL:  1,
            ROUTE_RETRANSFORM: 2,
            ROUTE_GRAVEYARD:   3,
        }
        if "route" in base.columns:
            base["_sort"] = base["route"].map(route_order).fillna(99)
            ic_ir_col = "ic_ir" if "ic_ir" in base.columns else base.columns[0]
            base = base.sort_values(["_sort", ic_ir_col], ascending=[True, False])
            base = base.drop(columns=["_sort"])

        return base.reset_index(drop=True)
