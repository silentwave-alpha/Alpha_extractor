"""
signal_filter/
━━━━━━━━━━━━━━
Stage 1: Statistical Filter + Router

Pipeline:
  Layer A → Sanity Filter
  Layer B → IC Filter + Decay Analysis
  Stage 1 → Feature Router (4 jalur)
  Layer C → Sign Consistency  (micro_alpha only)
  Layer D → Decorrelation     (micro_alpha only)

Output:
  micro_alpha  → Atomic Mining feature_pool
  structural   → Regime Conditioning (bukan entry signal)
  retransform  → User review, butuh feature engineering ulang
  graveyard    → Discard

Usage:
    from signal_filter import FilterPipeline, FilterPipelineConfig
    from signal_filter.feature_router import ROUTE_MICRO_ALPHA, ROUTE_STRUCTURAL

    pipeline = FilterPipeline()
    router_result, scorecard = pipeline.run(df, feature_list)

    # Ke Atomic Mining
    feature_pool    = router_result.micro_alpha
    horizon_map     = router_result.horizon_map

    # Ke Regime Conditioning
    regime_features = router_result.structural
"""

from .sanity_filter          import SanityFilter, SanityConfig, SanityResult
from .ic_filter              import ICFilter, ICConfig, ICResult
from .sign_consistency_filter import SignConsistencyFilter, SignConsistencyConfig, SignConsistencyResult
from .decorrelation_filter   import DecorrelationFilter, DecorrelationConfig, DecorrelationResult
from .feature_router         import (
    FeatureRouter, RouterConfig, RouterResult, RouteResult,
    ROUTE_MICRO_ALPHA, ROUTE_STRUCTURAL, ROUTE_RETRANSFORM, ROUTE_GRAVEYARD,
)
from .feature_passport       import (
    PassportBuilder, PassportConfig, FeaturePassport,
    ROUTE_LINEAR_STABLE, ROUTE_NONLINEAR, ROUTE_RANK_BASED,
    ROUTE_REGIME_CONDITIONAL, ROUTE_HORIZON_SPECIFIC,
    ROUTE_STRUCTURAL as PASSPORT_ROUTE_STRUCTURAL,
    ROUTE_RETRANSFORM as PASSPORT_ROUTE_RETRANSFORM,
    ROUTE_GRAVEYARD as PASSPORT_ROUTE_GRAVEYARD,
    DECAY_IMMEDIATE, DECAY_LAGGED, DECAY_PERSISTENT, DECAY_NON_MONOTONIC,
    CONF_HIGH, CONF_MEDIUM, CONF_LOW,
)
from .filter_pipeline        import FilterPipeline, FilterPipelineConfig

__all__ = [
    # Pipeline
    "FilterPipeline", "FilterPipelineConfig",
    # IC Router (lama)
    "FeatureRouter",  "RouterConfig",  "RouterResult", "RouteResult",
    "ROUTE_MICRO_ALPHA", "ROUTE_STRUCTURAL", "ROUTE_RETRANSFORM", "ROUTE_GRAVEYARD",
    # Passport (baru)
    "PassportBuilder", "PassportConfig", "FeaturePassport",
    "ROUTE_LINEAR_STABLE", "ROUTE_NONLINEAR", "ROUTE_RANK_BASED",
    "ROUTE_REGIME_CONDITIONAL", "ROUTE_HORIZON_SPECIFIC",
    "PASSPORT_ROUTE_STRUCTURAL", "PASSPORT_ROUTE_RETRANSFORM", "PASSPORT_ROUTE_GRAVEYARD",
    "DECAY_IMMEDIATE", "DECAY_LAGGED", "DECAY_PERSISTENT", "DECAY_NON_MONOTONIC",
    "CONF_HIGH", "CONF_MEDIUM", "CONF_LOW",
    # Filters
    "SanityFilter",   "SanityConfig",
    "ICFilter",       "ICConfig",
    "SignConsistencyFilter", "SignConsistencyConfig",
    "DecorrelationFilter",   "DecorrelationConfig",
]
