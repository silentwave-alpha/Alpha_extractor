import streamlit as st
import pandas as pd
import yaml
import os
import sys
import copy
import numpy as _np
from itertools import combinations as _comb
from collections import Counter
from governance.manager import OutputManager
from engine import EdgeEngine
from next.edge_miner import EdgeMiner

# Pre-import signal_filter at module level to avoid repeated import overhead
try:
    from signal_filter import FilterPipeline, FilterPipelineConfig
    from signal_filter.ic_filter import ICConfig
    from signal_filter.sanity_filter import SanityConfig
    from signal_filter.sign_consistency_filter import SignConsistencyConfig
    from signal_filter.decorrelation_filter import DecorrelationConfig
    from signal_filter.feature_router import RouterConfig
    from signal_filter.feature_passport import PassportConfig
    _SIGNAL_FILTER_AVAILABLE = True
except ImportError:
    _SIGNAL_FILTER_AVAILABLE = False

st.set_page_config(layout="wide")
st.title("Silentwave - Ensemble Research Platform")

# =====================================================
# SESSION STATE
# =====================================================

_defaults = {
    "df_built":               None,
    "registry_built":         None,
    # Filter results
    "filter_feature_pool":    None,   # micro_alpha → masuk Atomic
    "filter_structural":      None,   # structural → regime conditioning
    "filter_horizon_map":     None,   # {feature: optimal_horizon}
    "filter_scorecard":       None,   # IC + passport scorecard lengkap
    "filter_router_result":   None,   # RouterResult object
    # Passport results (NEW)
    "filter_passports":       None,   # {feature: FeaturePassport}
    "filter_passport_df":     None,   # DataFrame ringkasan passport
    "filter_model_map":       None,   # {feature: model_type} → ke Atomic
    "filter_transform_map":   None,   # {feature: transform}  → ke Atomic
    # Mining results
    "mining_results_df":       None,
    "mining_artifacts_path":   None,
    "mining_feature_pool":     None,
    "mining_config_snapshot":  None,
    "mining_trial_scorecard":  None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

HOURS_PER_DAY = 24

# =====================================================
# 1. LOAD CONFIG
# =====================================================

yaml_path = "config.yaml"
if os.path.exists(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
else:
    config = {}

config.setdefault("modeling",        {})
config.setdefault("mining",          {})
config.setdefault("feature_builder", {})
config.setdefault("walkforward",     {})
config.setdefault("feature_pruning", {})
config.setdefault("logging",         {})

# =====================================================
# 2. SIDEBAR - FEATURE BUILDER
# =====================================================

st.sidebar.header("1. Feature Builder")

col1, col2 = st.sidebar.columns(2)
with col1:
    config["feature_builder"]["technical"] = st.checkbox(
        "Technical", value=config.get("feature_builder", {}).get("technical", True)
    )
with col2:
    config["feature_builder"]["onchain"] = st.checkbox(
        "Onchain", value=config.get("feature_builder", {}).get("onchain", True)
    )

col1, col2, col3 = st.sidebar.columns(3)
with col1:
    config["feature_builder"]["enable_transform"] = st.checkbox(
        "Transform",
        value=config.get("feature_builder", {}).get("enable_transform", False),
        help="Z-score, Percentile, Vol-Adjust, etc."
    )
with col2:
    config["feature_builder"]["enable_interaction"] = st.checkbox(
        "Interaction",
        value=config.get("feature_builder", {}).get("enable_interaction", False),
        help="Fixed + Dynamic feature combinations"
    )
with col3:
    config["feature_builder"]["dynamic_interactions"] = st.checkbox(
        "Dynamic",
        value=config.get("feature_builder", {}).get("dynamic_interactions", False),
        help="All pairwise interactions (slow!)"
    )

feature_windows_str = st.sidebar.text_input(
    "Feature window",
    value=",".join([str(x) for x in config.get("feature_builder", {}).get("feature_windows", [5, 10, 20, 50])])
)
config["feature_builder"]["feature_windows"] = [
    int(x.strip()) for x in feature_windows_str.split(",") if x.strip().isdigit()
]

build_clicked = st.sidebar.button("🔨 Build Features", width='stretch')

st.sidebar.divider()

# =====================================================
# 3. SIDEBAR - ENGINE CONFIGURATOR
# =====================================================

st.sidebar.header("2. Engine Configurator")

execution_type = st.sidebar.radio(
    "Select Execution Type",
    ["Alpha Statistical Filter", "Atomic Edge Discovery", "Ensemble Engine"],
    help=(
        "Filter: pre-screen fitur via IC/t-stat/decay  |  "
        "Atomic: ML walkforward grid search  |  "
        "Ensemble: engine dengan feature pool pilihan"
    )
)

is_filtering = execution_type == "Alpha Statistical Filter"
is_mining    = execution_type == "Atomic Edge Discovery"
is_ensemble  = execution_type == "Ensemble Engine"

# ── Mode & Walkforward (Atomic + Ensemble saja) ──────────────────
if is_mining or is_ensemble:
    mode = st.sidebar.selectbox(
        "Mode", ["simple", "walkforward"],
        index=0 if config.get("mode") == "simple" else 1
    )
    config["mode"] = mode

    if mode == "simple":
        config["split_ratio"] = st.sidebar.number_input(
            "Train/Test split ratio",
            value=float(config.get("split_ratio", 0.7)),
            min_value=0.1, max_value=0.9
        )
    elif mode == "walkforward":
        col1, col2, col3 = st.sidebar.columns(3)
        with col1:
            train_days = st.number_input(
                "Train (hari)",
                value=max(1, int(config.get("walkforward", {}).get("train_size", 8640) // HOURS_PER_DAY)),
                min_value=1
            )
        with col2:
            test_days = st.number_input(
                "Test (hari)",
                value=max(1, int(config.get("walkforward", {}).get("test_size", 720) // HOURS_PER_DAY)),
                min_value=1
            )
        with col3:
            step_days = st.number_input(
                "Step (hari)",
                value=max(1, int(config.get("walkforward", {}).get("step_size", 720) // HOURS_PER_DAY)),
                min_value=1
            )
        config["walkforward"]["train_size"] = train_days * HOURS_PER_DAY
        config["walkforward"]["test_size"]  = test_days  * HOURS_PER_DAY
        config["walkforward"]["step_size"]  = step_days  * HOURS_PER_DAY
        st.sidebar.caption(
            f"→ bars: train={config['walkforward']['train_size']} | "
            f"test={config['walkforward']['test_size']} | "
            f"step={config['walkforward']['step_size']}"
        )

# ── Triple Barrier ───────────────────────────────────────────────
if is_mining or is_ensemble:
    st.sidebar.subheader("Triple Barrier Labeling")

if is_mining:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        tp_grid_str = st.text_input(
            "TP % (grid)",
            value=",".join([str(round(v*100,2)) for v in config["mining"].get("tp_pcts",[0.05])]),
            help="Contoh: 1,2,3 → TP 1%, 2%, 3%"
        )
        tp_pcts_pct = [float(x.strip()) for x in tp_grid_str.split(",") if x.strip().replace(".","").isdigit()]
        config["mining"]["tp_pcts"] = [v/100.0 for v in tp_pcts_pct]
    with col2:
        horizons_str = st.text_input(
            "Horizon (grid, bars)",
            value=",".join([str(x) for x in config["mining"].get("horizons",[12,48,72])]),
            help="Akan di-override oleh horizon_map dari Filter jika tersedia"
        )
        config["mining"]["horizons"] = [
            int(x.strip()) for x in horizons_str.split(",") if x.strip().isdigit()
        ]

    if st.session_state.filter_horizon_map:
        unique_h = sorted(set(st.session_state.filter_horizon_map.values()))
        st.sidebar.success(f"✅ Filter aktif — horizon akan di-override ke {unique_h}")
        # Tampilkan model distribution dari passport jika tersedia
        _mmap = st.session_state.get("filter_model_map") or {}
        if _mmap:
            from collections import Counter as _Ctr
            _mc = Counter(_mmap.values())
            st.sidebar.caption("Model types: " + " · ".join(f"{m}: {c}" for m, c in _mc.items()))
    else:
        st.sidebar.caption("ℹ️ Jalankan Alpha Filter untuk guided horizon.")

    total_label_combos = len(config["mining"]["tp_pcts"]) * len(config["mining"]["horizons"])
    st.sidebar.caption(
        f"→ {len(config['mining']['tp_pcts'])} TP × {len(config['mining']['horizons'])} horizon "
        f"= {total_label_combos} kombinasi labeling"
    )

elif is_ensemble:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        tp_pct_display = st.number_input(
            "TP %", value=round(float(config.get("tp_pct",0.02))*100,2),
            min_value=0.1, max_value=50.0, step=0.5, format="%.1f"
        )
        config["tp_pct"] = tp_pct_display / 100.0
    with col2:
        config["max_hold"] = st.number_input(
            "Max Hold (bars)", value=int(config.get("max_hold",100)), min_value=1
        )
    config["horizon"] = config["max_hold"]

if is_mining or is_ensemble:
    # ── Threshold config ─────────────────────────────────────────
    st.sidebar.markdown("**Threshold**")

    _opt = st.sidebar.toggle(
        "Auto-optimize threshold (IS → OOS)",
        value=bool(config.get("optimize_threshold", True)),
        help="ON = cari threshold optimal dari IS data (precision-based), apply ke OOS.\n"
             "OFF = pakai nilai fixed di bawah.",
    )
    config["optimize_threshold"] = _opt

    if _opt:
        # Grid editor — compact 3 kolom: min / max / step
        st.sidebar.caption("Grid search range")
        _gcol1, _gcol2, _gcol3 = st.sidebar.columns(3)
        _g_min  = _gcol1.number_input("Min",  value=0.30, min_value=0.30, max_value=0.95, step=0.05, format="%.2f")
        _g_max  = _gcol2.number_input("Max",  value=0.55, min_value=0.31, max_value=0.99, step=0.05, format="%.2f")
        _g_step = _gcol3.number_input("Step", value=0.05, min_value=0.01, max_value=0.10, step=0.05, format="%.2f")

        # Build grid dari min/max/step
        _grid = [round(float(v), 4) for v in _np.arange(_g_min, _g_max + _g_step * 0.5, _g_step)]
        config["threshold_grid"] = _grid
        st.sidebar.caption(f"Grid: {[f'{v:.2f}' for v in _grid]}")

        # Precision target
        config["threshold_precision_target"] = st.sidebar.number_input(
            "Precision target", value=float(config.get("threshold_precision_target", 0.55)),
            min_value=0.30, max_value=0.95, step=0.01, format="%.2f",
            help="Pilih threshold dengan weighted precision tertinggi yang ≥ nilai ini.",
        )
        config["threshold_min_trades"] = st.sidebar.number_input(
            "Min trades (per thr)", value=int(config.get("threshold_min_trades", 10)),
            min_value=1, step=1,
            help="Minimum jumlah trade di IS agar threshold dianggap valid.",
        )
        # Fallback fixed threshold (dipakai kalau tidak ada IS data atau optimize gagal)
        config["prob_threshold"] = float(config.get("prob_threshold", 0.55))
    else:
        # Fixed threshold
        config["prob_threshold"] = st.sidebar.number_input(
            "Prob threshold (fixed)", value=float(config.get("prob_threshold", 0.55)),
            min_value=0.30, max_value=0.99, step=0.01, format="%.2f",
            help="Threshold fixed — tidak ada pencarian otomatis.",
        )
        config["threshold_grid"] = [config["prob_threshold"]]

    config["direction"] = "both"

# =====================================================
# 4. SIDEBAR - STAGE-SPECIFIC CONFIG
# =====================================================

# ─── ALPHA FILTER config ─────────────────────────────────────────
if is_filtering:
    st.sidebar.subheader("Filter Config")

    st.sidebar.markdown("**IC Filter**")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        ic_window     = st.number_input("IC window (bars)", value=500, min_value=100, step=50)
        ic_step       = st.number_input("IC step (bars)",   value=100, min_value=20,  step=20)
    with col2:
        min_ic_mean   = st.number_input("Min |IC mean|", value=0.02, min_value=0.001, max_value=0.5, format="%.3f")
        min_ic_ir     = st.number_input("Min IC IR",     value=0.30, min_value=0.0,   max_value=5.0, format="%.2f")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        min_t_stat    = st.number_input("Min |t-stat|",  value=2.0,  min_value=0.0,  max_value=10.0, format="%.1f")
    with col2:
        min_ic_pos_pct= st.number_input("Min IC pos%",   value=0.50, min_value=0.0,  max_value=1.0,  format="%.2f")

    decay_horizons_str = st.sidebar.text_input(
        "Decay horizons (bars)", value="1,5,10,24,48,72,100",
        help="Horizon untuk hitung decay curve & half-life"
    )
    decay_horizons = [int(x.strip()) for x in decay_horizons_str.split(",") if x.strip().isdigit()]

    st.sidebar.markdown("**Sign Consistency**")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        min_monotonicity = st.number_input("Min monotonicity", value=0.60, min_value=0.0, max_value=1.0, format="%.2f")
    with col2:
        min_spread_pct   = st.number_input("Min spread%",      value=0.55, min_value=0.0, max_value=1.0, format="%.2f")

    st.sidebar.markdown("**Decorrelation**")
    max_corr = st.sidebar.slider(
        "Max correlation", min_value=0.50, max_value=1.00, value=0.80, step=0.05,
        help="Fitur |corr| > threshold dianggap redundant"
    )

    st.sidebar.markdown("**Router — Structural Threshold**")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        structural_horizon_threshold = st.number_input(
            "Min peak horizon", value=72, min_value=10, max_value=500, step=1,
            help="peak_horizon >= nilai ini → kandidat structural"
        )
    with col2:
        structural_trend_ratio = st.number_input(
            "Min trend ratio", value=1.5, min_value=1.0, max_value=20.0,
            format="%.1f",
            help="peak_IC / IC(h=1) >= nilai ini → structural (IC naik signifikan)"
        )

    st.sidebar.markdown("**Layer control**")
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        run_sign    = st.checkbox("Sign Consistency", value=True)
    with col2:
        run_decorr  = st.checkbox("Decorrelation",    value=True)
    with col3:
        run_passport = st.checkbox("Passport",         value=True,
                                   help="Characterize setiap feature: decay shape, stability, regime flip, quantile structure")

    if run_passport:
        st.sidebar.markdown("**Passport Config**")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            passport_n_periods = st.number_input(
                "Sub-periods", value=6, min_value=3, max_value=12,
                help="Jumlah sub-period untuk temporal stability check"
            )
            passport_min_stability = st.number_input(
                "Min stability", value=0.55, min_value=0.0, max_value=1.0, format="%.2f",
                help="Min fraksi sub-period dengan IC sign benar"
            )
        with col2:
            passport_regime_window = st.number_input(
                "Regime window (bars)", value=168, min_value=20, max_value=1000,
                help="Rolling window untuk bull/bear proxy"
            )
            passport_tail_ratio = st.number_input(
                "Tail ratio threshold", value=0.20, min_value=0.01, max_value=0.5, format="%.2f",
                help="middle/total spread < ini → tail_driven signal"
            )
    else:
        passport_n_periods     = 6
        passport_min_stability = 0.55
        passport_regime_window = 168
        passport_tail_ratio    = 0.20

    filter_feature_pool = []
    if st.session_state.registry_built is not None:
        available_features = st.session_state.registry_built.get_feature_list()
        filter_feature_pool = st.sidebar.multiselect(
            "Feature pool to filter",
            options=available_features, default=available_features
        )
        if not filter_feature_pool:
            filter_feature_pool = available_features
    else:
        st.sidebar.info("Build features first.")

# ─── ATOMIC config ───────────────────────────────────────────────
if is_mining:
    st.sidebar.subheader("Mining Config")

    _mode_opts   = ["A","B","C","D"]
    _mode_labels = {"A":"A – Single, tanpa regime","B":"B – Multi, tanpa regime",
                    "C":"C – Single, dengan regime","D":"D – Multi, dengan regime"}
    _cfg_modes_str = config["mining"].get("modes","ABCD")
    _default_modes = [m for m in _mode_opts if m in str(_cfg_modes_str)] or _mode_opts

    selected_modes = st.sidebar.multiselect(
        "Mining Modes", options=_mode_opts, default=_default_modes,
        format_func=lambda m: _mode_labels[m]
    )
    config["mining"]["modes"] = "".join(selected_modes)

    max_kombinasi = st.sidebar.number_input(
        "Max kombinasi fitur", value=int(config["mining"].get("max_kombinasi",2)),
        min_value=1, max_value=3
    )
    config["mining"]["max_kombinasi"] = max_kombinasi

    col1, col2 = st.sidebar.columns(2)
    with col1:
        mining_n_jobs = st.number_input("n_jobs", value=int(config["mining"].get("n_jobs",-1)), min_value=-1, max_value=64)
    with col2:
        batch_size    = st.number_input("Batch size", value=int(config["mining"].get("batch_size",500)), min_value=50, max_value=5000, step=50)
    config["mining"]["n_jobs"]     = mining_n_jobs
    config["mining"]["batch_size"] = batch_size

    # Feature pool
    feature_pool = []
    if st.session_state.filter_feature_pool is not None:
        feature_pool = st.session_state.filter_feature_pool
        st.sidebar.success(f"✅ {len(feature_pool)} fitur dari Alpha Filter")
        st.sidebar.caption(", ".join(feature_pool))
    elif st.session_state.registry_built is not None:
        available_features = st.session_state.registry_built.get_feature_list()
        default_features   = [f for f in config["mining"].get("feature_pool",[]) if f in available_features]
        feature_pool = st.sidebar.multiselect("Feature pool (manual)", options=available_features, default=default_features)
        if not feature_pool:
            feature_pool = available_features
    else:
        st.sidebar.info("Build features first.")
        fp_str = st.sidebar.text_input("Feature pool (comma separated)", value=",".join(config["mining"].get("feature_pool",[])))
        feature_pool = [f.strip() for f in fp_str.split(",") if f.strip()]

    config["mining"]["feature_pool"] = feature_pool

    # Pruning (hanya jika tidak ada Filter)
    if st.session_state.filter_feature_pool is None:
        col1, col2 = st.sidebar.columns([1,2])
        with col1:
            pruning_enabled = st.checkbox("Pruning", value=config["feature_pruning"].get("enabled",False))
        with col2:
            corr_threshold = st.slider("Corr threshold", 0.50, 1.00,
                float(config["feature_pruning"].get("correlation_threshold",0.95)), 0.01)
        config["feature_pruning"] = {"enabled":pruning_enabled,"correlation_threshold":corr_threshold}

    # Task estimator
    if feature_pool:
        n_f     = len(feature_pool)
        n_pairs = len(list(_comb(range(n_f),2))) if max_kombinasi >= 2 else 0
        n_tp    = len(config["mining"].get("tp_pcts",[]))
        if st.session_state.filter_horizon_map:
            n_h = len(set(st.session_state.filter_horizon_map.values()))
        else:
            n_h = len(config["mining"].get("horizons",[]))
        n_single_modes = sum(1 for m in ["A","C"] if m in selected_modes)
        n_multi_modes  = sum(1 for m in ["B","D"] if m in selected_modes)
        est_tasks = (n_f*n_single_modes + n_pairs*n_multi_modes) * n_h * n_tp
        n_batches = max(1,(est_tasks+batch_size-1)//batch_size)
        color = "red" if est_tasks > 5000 else "orange" if est_tasks > 1000 else "green"
        st.sidebar.markdown(
            f"<span style='color:{color}'>⚡ Est. tasks: **{est_tasks:,}**</span> "
            f"· {n_batches} batch · {n_f} feat · {n_pairs:,} pairs",
            unsafe_allow_html=True
        )

# ─── ENSEMBLE config ─────────────────────────────────────────────
if is_ensemble:
    st.sidebar.subheader("Regime Settings")
    use_regime_conditioned = st.sidebar.checkbox(
        "Use regime conditioned", value=config["modeling"].get("use_regime_conditioned",False)
    )
    config["modeling"]["use_regime_conditioned"] = use_regime_conditioned
    if use_regime_conditioned:
        use_allowed = st.sidebar.checkbox("Use allowed regime", value=config["modeling"].get("use_allowed_regimes",False))
        config["modeling"]["use_allowed_regimes"] = use_allowed
        if use_allowed:
            allowed_str = st.sidebar.text_input(
                "Allowed regime column",
                value=",".join([str(x) for x in config["modeling"].get("allowed_regimes",[2,8])])
            )
            config["modeling"]["allowed_regimes"] = [int(x.strip()) for x in allowed_str.split(",") if x.strip().isdigit()]

    feature_pool_ensemble = []
    if st.session_state.filter_feature_pool is not None:
        feature_pool_ensemble = st.session_state.filter_feature_pool
        st.sidebar.success(f"✅ {len(feature_pool_ensemble)} fitur dari Alpha Filter")
    elif st.session_state.registry_built is not None:
        available_features = st.session_state.registry_built.get_feature_list()
        default_features   = [f for f in config["modeling"].get("feature_pool",[]) if f in available_features]
        feature_pool_ensemble = st.sidebar.multiselect("Feature pool (ensemble)", options=available_features, default=default_features)
    else:
        st.sidebar.info("Build features first.")
        fp_str = st.sidebar.text_input("Feature pool (ensemble, comma separated)", value=",".join(config["modeling"].get("feature_pool",[])))
        feature_pool_ensemble = [f.strip() for f in fp_str.split(",") if f.strip()]

    config["modeling"]["feature_pool"] = feature_pool_ensemble
    config["mining"] = {"enabled": False}

st.sidebar.divider()

_run_labels = {
    "Alpha Statistical Filter": "🔬 Run Filter",
    "Atomic Edge Discovery":    "⚛️  Run Atomic",
    "Ensemble Engine":          "▶  Run Ensemble",
}
run_clicked = st.sidebar.button(_run_labels[execution_type], width='stretch')

if st.session_state.filter_feature_pool is not None:
    if st.sidebar.button("🗑 Reset Filter Results", width='stretch'):
        st.session_state.filter_feature_pool  = None
        st.session_state.filter_structural    = None
        st.session_state.filter_horizon_map   = None
        st.session_state.filter_scorecard     = None
        st.session_state.filter_router_result = None
        st.session_state.filter_passports     = None
        st.session_state.filter_passport_df   = None
        st.session_state.filter_model_map     = None
        st.session_state.filter_transform_map = None
        st.rerun()

# =====================================================
# 5. BUILD FEATURE LOGIC
# =====================================================

@st.cache_data(show_spinner=False)
def _cached_build_features(config_hash: str, _config: dict):
    """Cache feature building — only re-runs when config changes."""
    silent_manager = OutputManager(save_artifacts=False)
    engine         = EdgeEngine(_config, silent_manager)
    engine.load_data()
    df_built       = engine.build_features(log_registry=False)
    return df_built, engine.registry

if build_clicked:
    try:
        import hashlib, json
        config_hash = hashlib.md5(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()
        with st.spinner("Building features..."):
            df_built, registry_built = _cached_build_features(config_hash, config)
        st.session_state.df_built       = df_built
        st.session_state.registry_built = registry_built
        st.success("Features built successfully!")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Technical",   "✓" if config["feature_builder"]["technical"]           else "✗")
        with col2: st.metric("Onchain",     "✓" if config["feature_builder"]["onchain"]             else "✗")
        with col3: st.metric("Transform",   "✓" if config["feature_builder"]["enable_transform"]    else "✗")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Interaction", "✓" if config["feature_builder"]["enable_interaction"]  else "✗")
        with col2: st.metric("Dynamic",     "✓" if config["feature_builder"]["dynamic_interactions"] else "✗")
        with col3: st.metric("Total Features", len(registry_built.get_feature_list()))
    except Exception as e:
        st.error(f"Feature build error: {e}")
        raise

@st.fragment
def render_feature_overview():
    if st.session_state.df_built is None:
        return
    with st.expander("📊 Dataframe Overview", expanded=False):
        st.dataframe(st.session_state.df_built.head(10))
    if (st.session_state.registry_built is not None
        and hasattr(st.session_state.registry_built,"feature_stats_df")
        and st.session_state.registry_built.feature_stats_df is not None):
        with st.expander("🔨 Feature Build", expanded=False):
            st.dataframe(st.session_state.registry_built.feature_stats_df)
    with st.expander("📋 Feature List", expanded=False):
        feature_list = st.session_state.registry_built.get_feature_list()
        st.write(f"Total Features: **{len(feature_list)}**")
        cols = st.columns(3)
        for i, feat in enumerate(feature_list):
            cols[i%3].write(f"• {feat}")

render_feature_overview()

# =====================================================
# 6. RUN LOGIC
# =====================================================

if run_clicked:
    if st.session_state.df_built is None:
        st.warning("Please build features first.")
        st.stop()

    try:
        if is_filtering:
            _exp_subdir = os.path.join("experiments", "alpha_statistical_filter")
        elif is_mining:
            _exp_subdir = os.path.join("experiments", "atomic")
        else:  # is_ensemble
            _exp_subdir = os.path.join("experiments", "ensemble")
        active_manager = OutputManager(save_artifacts=True, base_dir=_exp_subdir)

        # ─── STAGE 1: ALPHA STATISTICAL FILTER ──────────────────────────
        if is_filtering:
            if not filter_feature_pool:
                st.warning("Feature pool kosong untuk filter.")
                st.stop()

            _sf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
            if _sf_path not in sys.path:
                sys.path.insert(0, _sf_path)

            if not _SIGNAL_FILTER_AVAILABLE:
                st.error("signal_filter module tidak ditemukan. Pastikan folder signal_filter/ ada di parent directory engine.")
                st.stop()

            # Build PassportConfig hanya jika run_passport diaktifkan
            _passport_cfg = PassportConfig(
                n_sub_periods    = passport_n_periods    if run_passport else 6,
                min_stability    = passport_min_stability if run_passport else 0.55,
                regime_window    = passport_regime_window if run_passport else 168,
                tail_ratio_threshold = passport_tail_ratio if run_passport else 0.20,
            ) if run_passport else PassportConfig()

            pipeline_config = FilterPipelineConfig(
                ic=ICConfig(
                    ic_window=ic_window, ic_step=ic_step,
                    decay_horizons=decay_horizons,
                    min_ic_mean=min_ic_mean, min_ic_ir=min_ic_ir,
                    min_t_stat=min_t_stat, min_ic_positive_pct=min_ic_pos_pct,
                ),
                sign_consistency=SignConsistencyConfig(
                    window=ic_window, step=ic_step,
                    min_monotonicity=min_monotonicity, min_spread_pct=min_spread_pct,
                ),
                decorrelation=DecorrelationConfig(max_correlation=max_corr),
                router=RouterConfig(
                    structural_min_peak_horizon=structural_horizon_threshold,
                    structural_min_trend_ratio=structural_trend_ratio,
                ),
                passport=_passport_cfg,
                run_sign_consistency=run_sign,
                run_decorrelation=run_decorr,
                run_passport=run_passport,
                close_col="close",
            )

            with st.spinner("🔬 Running Alpha Statistical Filter..."):
                pipeline    = FilterPipeline(
                    config=pipeline_config,
                    logger=active_manager.logger,
                )
                router_result, scorecard_df = pipeline.run(
                    st.session_state.df_built, filter_feature_pool
                )

            st.session_state.filter_feature_pool  = router_result.micro_alpha + router_result.structural
            st.session_state.filter_structural     = router_result.structural
            st.session_state.filter_horizon_map    = router_result.horizon_map
            st.session_state.filter_scorecard      = scorecard_df
            st.session_state.filter_router_result  = router_result

            # Simpan passport results jika tersedia
            if run_passport and hasattr(pipeline, "passport_builder"):
                # Ekstrak passport data dari scorecard_df
                _passport_df = scorecard_df[
                    scorecard_df["feature"].isin(router_result.micro_alpha)
                ] if scorecard_df is not None else pd.DataFrame()

                # Build model_map dan transform_map dari passport columns
                _model_map     = {}
                _transform_map = {}
                if not _passport_df.empty and "model_type" in _passport_df.columns:
                    _model_map = dict(zip(
                        _passport_df["feature"],
                        _passport_df["model_type"].fillna("gradient_boosting")
                    ))
                if not _passport_df.empty and "transform" in _passport_df.columns:
                    _transform_map = dict(zip(
                        _passport_df["feature"],
                        _passport_df["transform"].fillna("zscore")
                    ))

                st.session_state.filter_passport_df   = _passport_df
                st.session_state.filter_model_map     = _model_map
                st.session_state.filter_transform_map = _transform_map

            # Simpan scorecard + routing summary ke artifacts
            if active_manager.save_artifacts and scorecard_df is not None:
                import os as _os
                sc_path = _os.path.join(active_manager.path, "filter_scorecard.csv")
                scorecard_df.to_csv(sc_path, index=False)
                active_manager.logger.info(f"Filter scorecard saved: {sc_path}")

                s = router_result.summary
                summary_lines = [
                    "=== STAGE 1 ROUTER SUMMARY ===",
                    f"micro_alpha  : {s['micro_alpha']} | {router_result.micro_alpha}",
                    f"structural   : {s['structural']} | {router_result.structural}",
                    f"retransform  : {s['retransform']} | {router_result.retransform}",
                    f"graveyard    : {s['graveyard']}",
                    f"horizon_map  : {router_result.horizon_map}",
                ]
                rsum_path = _os.path.join(active_manager.path, "filter_routing_summary.txt")
                with open(rsum_path, "w", encoding="utf-8") as _f:
                    _f.write("\n".join(summary_lines))
                active_manager.logger.info(f"Routing summary saved: {rsum_path}")

            active_manager.save_config_yaml(config)
            st.info(f"Artifacts saved to: {active_manager.path}")

        # ─── STAGE 2: ATOMIC EDGE DISCOVERY ─────────────────────────────
        elif is_mining:
            if not config["mining"].get("feature_pool"):
                st.warning("Feature pool kosong.")
                st.stop()
            if not selected_modes:
                st.warning("Pilih minimal satu mining mode.")
                st.stop()

            # Override horizons dari Filter jika tersedia
            if st.session_state.filter_horizon_map:
                all_horizons = sorted(set(st.session_state.filter_horizon_map.values()))
                config["mining"]["horizons"] = all_horizons

            with st.spinner("⚛️ Running atomic engine (parallel, batched)..."):
                config["logging"]["run_type"] = "atomic"
                active_manager.log_feature_built(st.session_state.registry_built)
                miner                = EdgeMiner(config, active_manager)
                miner.df_built       = st.session_state.df_built
                miner.registry_built = copy.deepcopy(st.session_state.registry_built)

                # Inject passport maps ke miner jika tersedia dari Filter
                _model_map_for_mining     = st.session_state.get("filter_model_map")     or {}
                _transform_map_for_mining = st.session_state.get("filter_transform_map") or {}
                if _model_map_for_mining:
                    miner.model_map     = _model_map_for_mining
                    miner.transform_map = _transform_map_for_mining
                    active_manager.logger.info(
                        f"Passport model_map injected: {len(_model_map_for_mining)} features"
                    )

                results_df = miner.mine(
                    feature_pool  = config["mining"]["feature_pool"],
                    modes         = tuple(selected_modes),
                    horizons      = config["mining"].get("horizons",[48]),
                    tp_pcts       = config["mining"].get("tp_pcts",[0.02]),
                    max_kombinasi = config["mining"].get("max_kombinasi",2),
                )

            # Simpan trial scorecard terlepas dari apakah ada alpha yang lolos
            if miner.trial_scorecard is not None:
                st.session_state.mining_trial_scorecard = miner.trial_scorecard

            if results_df is not None and not results_df.empty:
                st.session_state.mining_results_df      = results_df.sort_values("composite_score",ascending=False).reset_index(drop=True)
                st.session_state.mining_artifacts_path  = active_manager.path
                st.session_state.mining_feature_pool    = config["mining"]["feature_pool"]
                st.session_state.mining_config_snapshot = {
                    "horizons":    config["mining"].get("horizons",[48]),
                    "tp_pcts":     config["mining"].get("tp_pcts",[0.02]),
                    "modes":       selected_modes,
                    "from_filter": st.session_state.filter_horizon_map is not None,
                }
                active_manager.save_atomic_summary(st.session_state.mining_results_df)
            else:
                st.session_state.mining_results_df = None
                st.warning("Tidak ada alpha valid yang ditemukan.")

        # ─── STAGE 3: ENSEMBLE ENGINE ────────────────────────────────────
        elif is_ensemble:
            with st.spinner("▶ Running ensemble engine..."):
                config["logging"]["run_type"] = "ensemble"
                feature_pool_ensemble          = config["modeling"].get("feature_pool",[])
                active_manager.log_feature_built(
                    st.session_state.registry_built,
                    feature_pool=feature_pool_ensemble if feature_pool_ensemble else None
                )
                engine          = EdgeEngine(config, active_manager)
                engine.df       = st.session_state.df_built
                engine.registry = copy.deepcopy(st.session_state.registry_built)
                if feature_pool_ensemble:
                    engine.override_feature_list(feature_pool_ensemble)
                engine.run(use_prebuilt_data=True)

            with st.expander("📋 Feature Pool Used", expanded=False):
                pool = config["modeling"].get("feature_pool",[])
                if pool:
                    st.write(f"Total: **{len(pool)}** features (subset)")
                    cols = st.columns(3)
                    for i,f in enumerate(pool): cols[i%3].write(f"• {f}")
                else:
                    feat_list = st.session_state.registry_built.get_feature_list()
                    st.write(f"Total: **{len(feat_list)}** features (semua)")
                    cols = st.columns(3)
                    for i,f in enumerate(feat_list): cols[i%3].write(f"• {f}")

            st.subheader("Global Metrics")
            c1,c2,c3 = st.columns(3)
            c1.metric("Sharpe OOS",   f"{engine.sharpe_oos:.4f}"   if hasattr(engine,"sharpe_oos")   else "—")
            c2.metric("Sharpe IS",    f"{engine.sharpe_is:.4f}"    if hasattr(engine,"sharpe_is")    else "—")
            c3.metric("Sharpe Total", f"{engine.sharpe_total:.4f}" if hasattr(engine,"sharpe_total") else "—")

            st.subheader("Breakdown")
            tab_dir, tab_reg = st.tabs(["↕️ Per Direction","📊 Per Regime"])
            with tab_reg:
                if hasattr(engine,"stats_df") and not engine.stats_df.empty:
                    st.dataframe(engine.stats_df, width='stretch', hide_index=True)
                else:
                    st.info("No regime breakdown available.")
            with tab_dir:
                if hasattr(engine,"direction_df") and not engine.direction_df.empty:
                    st.dataframe(engine.direction_df, width='stretch', hide_index=True)
                else:
                    st.info("No direction breakdown available.")

        active_manager.save_config_yaml(config)

    except Exception as e:
        st.error(f"Error: {e}")
        raise

# =====================================================
# 7. RENDER FILTER RESULTS (persistent)
# =====================================================

@st.fragment
def render_filter_results():
    if st.session_state.filter_scorecard is None or not is_filtering:
        return
    scorecard       = st.session_state.filter_scorecard
    router_result   = st.session_state.get("filter_router_result")
    micro_alpha     = router_result.micro_alpha if router_result else []
    structural      = router_result.structural  if router_result else []
    horizon_map     = st.session_state.filter_horizon_map or {}
    n_input         = len(scorecard)

    st.subheader("🔬 Alpha Statistical Filter Results")

    # ── Summary metrics ──────────────────────────────────────────
    passport_df = st.session_state.get("filter_passport_df")
    model_map   = st.session_state.get("filter_model_map") or {}

    # Confidence counts dari passport
    _n_high   = int((passport_df["confidence"] == "HIGH").sum())   if passport_df is not None and "confidence" in passport_df.columns else 0
    _n_medium = int((passport_df["confidence"] == "MEDIUM").sum()) if passport_df is not None and "confidence" in passport_df.columns else 0

    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
    c1.metric("Input Features", n_input)
    c2.metric("⚛️ Micro Alpha",  len(micro_alpha))
    c3.metric("🏛️ Structural",   len(structural))
    c4.metric("Unique Horizons", len(set(horizon_map.values())) if horizon_map else 0)
    c5.metric("🟢 HIGH conf",     _n_high)
    c6.metric("🟡 MEDIUM conf",   _n_medium)
    c7.metric("Ready for Atomic","✅" if micro_alpha else "⚠️")

    st.divider()

    # ── Route breakdown tabs ──────────────────────────────────────
    tab_ma, tab_st, tab_rt, tab_gv = st.tabs([
        f"⚛️ Micro Alpha ({len(micro_alpha)})",
        f"🏛️ Structural ({len(structural)})",
        f"🔧 Retransform ({len(st.session_state.get('filter_router_result').retransform) if router_result else 0})",
        f"💀 Graveyard ({len(router_result.graveyard) if router_result else 0})",
    ])

    with tab_ma:
        if micro_alpha:
            st.caption("Fitur ini masuk Atomic Mining. Horizon, model type, dan transform sudah di-assign berdasarkan karakter passport.")
            hmap_rows = []
            for feat in micro_alpha:
                h     = horizon_map.get(feat, "?")
                row   = scorecard[scorecard["feature"] == feat]

                # IC metrics
                ic_ir_val = row["ic_ir"].values[0]    if not row.empty and "ic_ir"    in row.columns else float("nan")
                ic_mean_v = row["ic_mean"].values[0]  if not row.empty and "ic_mean"  in row.columns else float("nan")
                hl        = f"{row['half_life'].values[0]:.1f}" if not row.empty and "half_life" in row.columns and pd.notna(row["half_life"].values[0]) else "?"

                # Passport metrics (jika ada)
                has_passport = passport_df is not None and not passport_df.empty and feat in passport_df["feature"].values
                p_row        = passport_df[passport_df["feature"] == feat] if has_passport else None

                decay_shape  = p_row["decay_shape"].values[0]        if p_row is not None and "decay_shape"         in p_row.columns else row["signal_type"].values[0] if not row.empty and "signal_type" in row.columns else "?"
                conf         = p_row["confidence"].values[0]          if p_row is not None and "confidence"          in p_row.columns else "—"
                stab         = f"{p_row['temporal_stability'].values[0]:.2f}" if p_row is not None and "temporal_stability" in p_row.columns else "—"
                model        = p_row["model_type"].values[0]          if p_row is not None and "model_type"          in p_row.columns else "—"
                transform    = p_row["transform"].values[0]           if p_row is not None and "transform"           in p_row.columns else "—"
                reg_flip     = "⚠️ YES" if (p_row is not None and "regime_flip" in p_row.columns and p_row["regime_flip"].values[0]) else "no"
                tail         = "⚡ YES" if (p_row is not None and "tail_driven" in p_row.columns and p_row["tail_driven"].values[0]) else "no"

                hmap_rows.append({
                    "Feature":          feat,
                    "Horizon":          h,
                    "Decay Shape":      decay_shape,
                    "Half-life":        hl,
                    "IC IR":            round(ic_ir_val, 4) if pd.notna(ic_ir_val) else "—",
                    "IC Mean":          round(ic_mean_v, 4) if pd.notna(ic_mean_v) else "—",
                    "Stability":        stab,
                    "Confidence":       conf,
                    "Model Type":       model,
                    "Transform":        transform,
                    "Regime Flip":      reg_flip,
                    "Tail Driven":      tail,
                })

            st.dataframe(
                pd.DataFrame(hmap_rows), width='stretch', hide_index=True,
                column_config={
                    "Confidence": st.column_config.TextColumn("Conf",   width="small"),
                    "Horizon":    st.column_config.NumberColumn("H",     width="small"),
                    "IC IR":      st.column_config.NumberColumn("IC IR", format="%.4f", width="small"),
                    "IC Mean":    st.column_config.NumberColumn("IC Mean", format="%.4f", width="small"),
                }
            )

            # Breakdown model type distribution
            if model_map:
                from collections import Counter
                model_counts = Counter(model_map.values())
                st.caption("Model distribution: " + " | ".join(f"**{m}**: {c}" for m,c in model_counts.items()))
        else:
            st.info("Tidak ada micro alpha features.")

    with tab_st:
        if structural:
            st.caption("Fitur ini adalah market condition descriptor — bukan entry signal langsung. Akan dipakai sebagai regime conditioning di Atomic modes C/D.")
            st_rows = []
            for feat in structural:
                row = scorecard[scorecard["feature"] == feat]
                ph  = row["peak_horizon"].values[0] if not row.empty and "peak_horizon" in row.columns else "?"
                tr  = row["trend_ratio"].values[0]  if not row.empty and "trend_ratio"  in row.columns else "?"
                ic_ir_val = row["ic_ir"].values[0] if not row.empty and "ic_ir" in row.columns else float("nan")
                reason = row["route_reason"].values[0] if not row.empty and "route_reason" in row.columns else "?"
                st_rows.append({
                    "Feature": feat,
                    "Peak Horizon": ph,
                    "Trend Ratio": round(float(tr),2) if tr != "?" else "?",
                    "IC IR": round(ic_ir_val,4) if pd.notna(ic_ir_val) else "—",
                    "Routing Reason": reason,
                })
            st.dataframe(pd.DataFrame(st_rows), width='stretch', hide_index=True)
        else:
            st.info("Tidak ada structural features.")

    with tab_rt:
        rt_features = router_result.retransform if router_result else []
        if rt_features:
            st.caption("Fitur ini punya sinyal tapi tidak stabil atau terlalu noisy. Pertimbangkan transformasi ulang di Feature Builder.")
            rt_rows = []
            for feat in rt_features:
                row = scorecard[scorecard["feature"] == feat]
                reason = row["route_reason"].values[0] if not row.empty and "route_reason" in row.columns else "?"
                rt_rows.append({"Feature": feat, "Reason": reason})
            st.dataframe(pd.DataFrame(rt_rows), width='stretch', hide_index=True)
        else:
            st.info("Tidak ada features yang butuh retransform.")

    with tab_gv:
        gv_features = router_result.graveyard if router_result else []
        if gv_features:
            st.caption("Fitur ini tidak punya edge yang detectable. Bisa di-ignore.")
            gv_rows = []
            for feat in gv_features:
                row = scorecard[scorecard["feature"] == feat]
                reason = row["route_reason"].values[0] if not row.empty and "route_reason" in row.columns else                          row.get("reject_reason", "?") if isinstance(row, dict) else "?"
                if not reason or reason == "?" and not row.empty:
                    reason = row["reject_reason"].values[0] if "reject_reason" in row.columns else "?"
                gv_rows.append({"Feature": feat, "Reason": reason})
            st.dataframe(pd.DataFrame(gv_rows), width='stretch', hide_index=True)
        else:
            st.info("Tidak ada features di graveyard.")

    # Full scorecard
    st.divider()
    st.markdown("#### 📋 Trial Scorecard — ALL ")
    # Tab: IC scorecard vs Passport scorecard
    sc_tab_ic, sc_tab_passport = st.tabs(["📊 IC Scorecard", "🪪 Passport Scorecard"])

    with sc_tab_ic:
        display_cols = [c for c in [
            "feature", "route",
            "ic_mean","ic_ir","t_stat","ic_positive_pct",
            "signal_type","peak_horizon","half_life","half_life_reliable",
            "assigned_horizon","trend_ratio",
            "monotonicity_score","spread_consistency",
            "route_reason",
        ] if c in scorecard.columns]
        _sc_display = scorecard[display_cols]
        st.caption(f"Menampilkan {min(500, len(_sc_display)):,} dari {len(_sc_display):,} fitur")
        st.dataframe(
            _sc_display.head(500),
            width='stretch', hide_index=True,
            column_config={
                "route":              st.column_config.TextColumn("Route",      width="small"),
                "ic_mean":            st.column_config.NumberColumn("IC Mean",   format="%.4f"),
                "ic_ir":              st.column_config.NumberColumn("IC IR",     format="%.3f"),
                "t_stat":             st.column_config.NumberColumn("t-stat",    format="%.2f"),
                "ic_positive_pct":    st.column_config.NumberColumn("IC pos%",   format="%.2f"),
                "half_life":          st.column_config.NumberColumn("Half-life", format="%.1f"),
                "assigned_horizon":   st.column_config.NumberColumn("Horizon",   format="%d"),
                "monotonicity_score": st.column_config.NumberColumn("Mono",      format="%.3f"),
                "spread_consistency": st.column_config.NumberColumn("Spread%",   format="%.2f"),
            }
        )

    with sc_tab_passport:
        if passport_df is not None and not passport_df.empty:
            passport_display_cols = [c for c in [
                "feature", "route", "confidence", "confidence_score",
                "decay_shape", "peak_horizon", "halflife", "optimal_horizon",
                "temporal_stability", "ic_degrading", "regime_flip",
                "monotonicity_score", "is_linear", "tail_driven",
                "transform", "model_type",
                "ic_bull", "ic_bear",
                "passport_reason",
            ] if c in passport_df.columns]
            st.dataframe(
                passport_df[passport_display_cols] if passport_display_cols else passport_df,
                width='stretch', hide_index=True,
                column_config={
                    "confidence":          st.column_config.TextColumn("Conf",     width="small"),
                    "confidence_score":    st.column_config.NumberColumn("Score",   format="%.1f",  width="small"),
                    "decay_shape":         st.column_config.TextColumn("Decay",     width="small"),
                    "peak_horizon":        st.column_config.NumberColumn("Peak H",  format="%d",    width="small"),
                    "halflife":            st.column_config.NumberColumn("Halflife", format="%.1f", width="small"),
                    "optimal_horizon":     st.column_config.NumberColumn("Opt H",   format="%d",    width="small"),
                    "temporal_stability":  st.column_config.NumberColumn("Stability", format="%.2f", width="small"),
                    "ic_degrading":        st.column_config.CheckboxColumn("Degrade", width="small"),
                    "regime_flip":         st.column_config.CheckboxColumn("Reg Flip", width="small"),
                    "monotonicity_score":  st.column_config.NumberColumn("Mono",    format="%.3f",  width="small"),
                    "is_linear":           st.column_config.CheckboxColumn("Linear", width="small"),
                    "tail_driven":         st.column_config.CheckboxColumn("Tail",   width="small"),
                    "ic_bull":             st.column_config.NumberColumn("IC Bull",  format="%.4f", width="small"),
                    "ic_bear":             st.column_config.NumberColumn("IC Bear",  format="%.4f", width="small"),
                }
            )
        else:
            st.info("Passport data tidak tersedia. Aktifkan Passport di Layer control dan jalankan filter ulang.")

    # Decay curves
    decay_cols = [c for c in scorecard.columns if c.startswith("ic_h")]
    if decay_cols:
        with st.expander("📉 Decay Profile (IC vs Horizon)", expanded=False):
            _dc1, _dc2, _dc3 = st.columns(3)
            with _dc1:
                _show_ma  = st.checkbox("⚛️ Micro Alpha",  value=bool(micro_alpha),  key="decay_show_ma")
            with _dc2:
                _show_st  = st.checkbox("🏛️ Structural",   value=bool(structural),   key="decay_show_st")
            with _dc3:
                _show_gv  = st.checkbox("💀 Graveyard / Others", value=False,         key="decay_show_gv")

            _decay_feats = []
            if _show_ma:
                _decay_feats += micro_alpha
            if _show_st:
                _decay_feats += structural
            if _show_gv:
                _gv = router_result.graveyard if router_result else []
                _rt = router_result.retransform if router_result else []
                _decay_feats += _gv + _rt

            # De-duplicate preserving order
            _seen = set(); _decay_feats_uniq = []
            for _f in _decay_feats:
                if _f not in _seen:
                    _seen.add(_f); _decay_feats_uniq.append(_f)

            if _decay_feats_uniq:
                _ma_sc = scorecard[scorecard["feature"].isin(_decay_feats_uniq)]
                if not _ma_sc.empty:
                    _chart = _ma_sc[["feature"] + decay_cols].set_index("feature").T
                    _chart.index = [int(c.replace("ic_h","")) for c in _chart.index]
                    _chart.index.name = "horizon"
                    st.caption(
                        f"Menampilkan {len(_chart.columns)} fitur: "
                        + ("⚛️ micro_alpha " if _show_ma else "")
                        + ("🏛️ structural "   if _show_st else "")
                        + ("💀 others"          if _show_gv else "")
                    )
                    st.line_chart(_chart)
                else:
                    st.info("Tidak ada decay data untuk fitur yang dipilih.")
            else:
                st.info("Pilih minimal satu kategori fitur untuk ditampilkan.")

render_filter_results()

# =====================================================
# 8. RENDER MINING RESULTS (persistent)
# =====================================================

@st.fragment
def render_mining_results():
    if st.session_state.mining_results_df is None or not is_mining:
        return
    results_df     = st.session_state.mining_results_df
    artifacts_path = st.session_state.mining_artifacts_path
    pool           = st.session_state.mining_feature_pool or []
    snap           = st.session_state.mining_config_snapshot or {}

    from governance.manager import OutputManager as _OM
    _reader                = _OM.__new__(_OM)
    _reader.path           = artifacts_path
    _reader.save_artifacts = True
    _reader.logger         = _reader._setup_logger()

    with st.expander("📋 Feature Pool Used", expanded=False):
        src = "Alpha Filter" if snap.get("from_filter") else "Manual"
        st.write(f"Total: **{len(pool)}** features · Modes: **{snap.get('modes',[])}** · Source: **{src}**")
        cols = st.columns(3)
        for i,f in enumerate(pool): cols[i%3].write(f"• {f}")

    tp_display = [f"{v*100:.1f}%" for v in snap.get("tp_pcts",[0.02])]
    st.subheader("⚛️ Mining Results")
    st.caption(
        f"Feature pool: {len(pool)} features  |  Modes: {snap.get('modes',[])}  |  "
        f"Horizons: {snap.get('horizons',[])}  |  TP: {tp_display}  |  "
        f"A/B: tanpa regime  ·  C/D: dengan regime"
    )

    display_cols = ["mode","mode_description","horizon","tp_pct","features","alpha_type",
                    "sharpe_oos","sharpe_is","n_trades_oos",
                    "probabilistic_sharpe","deflated_sharpe","dsr_reliable",
                    "regime_mean","regime_std","worst_regime_sharpe","composite_score"]
    df_display = results_df[[c for c in display_cols if c in results_df.columns]].copy()
    if "tp_pct" in df_display.columns:
        df_display["tp_pct"] = df_display["tp_pct"].apply(lambda v: f"{v*100:.1f}%")
    for col in ["sharpe_oos","sharpe_is","regime_mean","regime_std","composite_score"]:
        if col in df_display.columns:
            df_display[col] = df_display[col].round(4)

    st.divider()
    st.markdown("#### 📋 Breakdown Detail")
    col_config = {
        "mode":                 st.column_config.TextColumn("Mode",       width="small"),
        "mode_description":     st.column_config.TextColumn("Deskripsi",  width="medium"),
        "horizon":              st.column_config.NumberColumn("Horizon",  width="small"),
        "tp_pct":               st.column_config.TextColumn("TP",         width="small"),
        "features":             st.column_config.TextColumn("Features",   width="large"),
        "alpha_type":           st.column_config.TextColumn("Alpha Type", width="medium"),
        "sharpe_oos":           st.column_config.NumberColumn("Sharpe OOS",  format="%.4f", width="small"),
        "sharpe_is":            st.column_config.NumberColumn("Sharpe IS",   format="%.4f", width="small"),
        "n_trades_oos":         st.column_config.NumberColumn("N Trades",    format="%d",   width="small"),
        "probabilistic_sharpe": st.column_config.NumberColumn("PSR",         format="%.3f", width="small"),
        "deflated_sharpe":      st.column_config.NumberColumn("DSR",         format="%.3f", width="small"),
        "dsr_reliable":         st.column_config.CheckboxColumn("DSR OK",                   width="small"),
        "regime_mean":          st.column_config.NumberColumn("Reg Mean",    format="%.4f", width="small"),
        "regime_std":           st.column_config.NumberColumn("Reg Std",     format="%.4f", width="small"),
        "worst_regime_sharpe":  st.column_config.NumberColumn("Worst Reg",   format="%.4f", width="small"),
        "composite_score":      st.column_config.NumberColumn("Score",       format="%.4f", width="small"),
    }
    selection = st.dataframe(
        df_display, width='stretch', hide_index=False,
        on_select="rerun", selection_mode="single-row",
        column_config=col_config, key="mining_results_table",
    )

    selected_rows = selection.selection.rows if selection and selection.selection else []

    if selected_rows:
        idx      = selected_rows[0]
        row      = results_df.iloc[idx]
        tp_str   = f"{row['tp_pct']*100:.1f}%" if "tp_pct" in row else "—"
        feat_str = ", ".join(row["features"]) if isinstance(row["features"],list) else str(row["features"])
        fhash    = row["feature_hash"]
        stats_df = _reader.load_stats_df(row["mode"],row["horizon"],fhash)
        dir_df   = _reader.load_direction_df(row["mode"],row["horizon"],fhash)

        psr      = row.get("probabilistic_sharpe", float("nan"))
        dsr      = row.get("deflated_sharpe",      float("nan"))
        dsr_ok   = row.get("dsr_reliable", False)
        worst    = row.get("worst_regime_sharpe",  float("nan"))
        overfit  = row["sharpe_oos"] / row["sharpe_is"] if row.get("sharpe_is", 0) > 0 else 0

        st.divider()

        # ── Header ───────────────────────────────────────────────
        st.markdown(
            f"#### #{idx+1} &nbsp;·&nbsp; Mode **{row['mode']}** — {row.get('mode_description','')} "
            f"&nbsp;·&nbsp; H: **{row['horizon']}** &nbsp;·&nbsp; TP: **{tp_str}** "
            f"&nbsp;·&nbsp; <span style='color:#2ecc71;font-weight:600'>{row['alpha_type']}</span>",
            unsafe_allow_html=True,
        )
        _thr_val = row.get("optimal_threshold", float("nan"))
        _thr_str = f"{_thr_val:.2f}" if _thr_val == _thr_val else "—"
        st.caption(f"Features: {feat_str}  |  Threshold final: {_thr_str}")

        # ── Metrics: 3 kelompok ───────────────────────────────────
        st.markdown("**Performance**")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Sharpe OOS",   f"{row['sharpe_oos']:.4f}")
        p2.metric("Sharpe IS",    f"{row['sharpe_is']:.4f}")
        p3.metric("N Trades OOS", f"{row.get('n_trades_oos', 0):,}")
        p4.metric("Overfit Ratio", f"{overfit:.2f}",
                  help="OOS/IS Sharpe. >0.5 = tidak overfit")

        st.markdown("**Quality & Confidence**")
        q1, q2, q3 = st.columns(3)
        q1.metric("Composite Score", f"{row['composite_score']:.4f}")
        q2.metric("PSR",
                  f"{psr:.3f}" if psr == psr else "—",
                  help="Probabilistic Sharpe Ratio. >0.95 = reliable")
        q3.metric("DSR",
                  f"{dsr:.3f} {'✓' if dsr_ok else '⚠️'}" if dsr == dsr else "—",
                  help="Deflated Sharpe Ratio — koreksi multiple testing. >0.90 = reliable")

        st.markdown("**Regime Stability**")
        r1, r2, r3 = st.columns(3)
        r1.metric("Regime Mean",  f"{row['regime_mean']:.4f}")
        r2.metric("Regime Std",   f"{row.get('regime_std', float('nan')):.4f}")
        r3.metric("Worst Regime", f"{worst:.4f}" if worst == worst else "—")

        # ── Breakdown Tabs ────────────────────────────────────────
        st.markdown("**Breakdown**")
        tab_reg, tab_dir = st.tabs(["Per Regime", "Per Direction"])
        with tab_reg:
            if stats_df is not None and not stats_df.empty:
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
            else:
                st.warning("Stats data tidak ditemukan.")
        with tab_dir:
            if dir_df is not None and not dir_df.empty:
                st.dataframe(dir_df, use_container_width=True, hide_index=True)
            else:
                st.info("Direction breakdown belum tersedia.")

        # ── Equity Curve ──────────────────────────────────────────
        returns_df = _reader.load_returns_df(row["mode"], row["horizon"], fhash)
        st.divider()
        st.markdown("**Equity Curve**")
        if returns_df is not None and not returns_df.empty:
            import plotly.graph_objects as go

            is_mask  = (returns_df["strategy_return_train"].notna() & (returns_df.get("position_seen",    0) != 0)) if "strategy_return_train" in returns_df.columns else pd.Series(False, index=returns_df.index)
            oos_mask = (returns_df["strategy_return_test"].notna()  & (returns_df.get("position_unseen",  0) != 0)) if "strategy_return_test"  in returns_df.columns else pd.Series(False, index=returns_df.index)

            is_rets  = returns_df.loc[is_mask,  "strategy_return_train"].fillna(0) if "strategy_return_train" in returns_df.columns else pd.Series(dtype=float)
            oos_rets = returns_df.loc[oos_mask, "strategy_return_test"].fillna(0)  if "strategy_return_test"  in returns_df.columns else pd.Series(dtype=float)

            is_equity  = (1 + is_rets).cumprod()  * 100
            oos_equity = (1 + oos_rets).cumprod() * 100

            # Metrik ringkas di atas chart
            eq_c1, eq_c2, eq_c3 = st.columns(3)
            if not is_equity.empty:
                is_final = float(is_equity.iloc[-1])
                eq_c1.metric("IS Final", f"{is_final:.1f}", f"{is_final - 100:+.1f}%")
            if not oos_equity.empty:
                oos_final = float(oos_equity.iloc[-1])
                eq_c2.metric("OOS Final", f"{oos_final:.1f}", f"{oos_final - 100:+.1f}%")
            if not is_equity.empty and not oos_equity.empty and abs(is_final - 100) > 0.01:
                eq_c3.metric("Equity Overfit", f"{(oos_final - 100) / (is_final - 100 + 1e-9):.2f}")

            fig = go.Figure()
            if not is_equity.empty:
                fig.add_trace(go.Scatter(
                    x=is_equity.index, y=is_equity.values,
                    mode="lines", name="IS",
                    line=dict(color="#3498db", width=1.5, dash="dot"),
                    opacity=0.6,
                ))
            if not oos_equity.empty:
                fig.add_trace(go.Scatter(
                    x=oos_equity.index, y=oos_equity.values,
                    mode="lines", name="OOS",
                    line=dict(color="#2ecc71", width=2),
                ))
            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis_title="Equity (base 100)",
                legend=dict(orientation="h", y=1.08, x=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.12)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.12)"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Equity chart belum tersedia — jalankan ulang mining untuk menghasilkan data returns.")

        # ── Export Model ──────────────────────────────────────────
        st.divider()
        st.markdown("#### 🤖 Export Model")
        st.caption(
            "Retrain model pada **seluruh data** menggunakan konfigurasi edge ini, "
            "lalu download sebagai file `.pkl`."
        )

        export_col1, export_col2 = st.columns([1, 3])
        export_key = f"export_{row['mode']}_{row['horizon']}_{fhash}"

        with export_col1:
            export_clicked = st.button(
                "🤖 Retrain & Export Model",
                key=f"btn_{export_key}",
                help="Train final model pada full dataset → download .pkl",
            )

        if export_clicked:
            row_features = row["features"] if isinstance(row["features"], list) else [row["features"]]
            with st.spinner(f"Training final model pada full data ({len(row_features)} features)..."):
                try:
                    import copy as _copy

                    # Bangun config dari row yang dipilih
                    export_cfg = _copy.deepcopy(config)
                    export_cfg["horizon"]  = int(row["horizon"])
                    export_cfg["max_hold"] = int(row["horizon"])
                    export_cfg["tp_pct"]   = float(row["tp_pct"])
                    export_cfg["logging"]["run_type"] = "atomic"
                    export_cfg["modeling"]["use_regime_conditioned"] = row["mode"] in ("C", "D")

                    export_engine = EdgeEngine(export_cfg, _reader)
                    export_engine.df       = st.session_state.df_built.copy()
                    export_engine.registry = _copy.deepcopy(st.session_state.registry_built)
                    export_engine.registry.set_feature_list(row_features)

                    export_engine.check_feature()
                    export_engine.build_target()
                    export_engine.train_last_wf()

                    # Simpan ke disk dan baca kembali sebagai bytes
                    # Simpan inner xgb.XGBClassifier (bukan InstitutionalModel wrapper)
                    # agar bisa di-load di environment lain tanpa butuh ml_models module
                    _is_regime = bool(
                        export_cfg["modeling"].get("use_regime_conditioned", False)
                        and getattr(export_engine, "regime_models", {})
                    )
                    if _is_regime:
                        # Mode C/D: dict per-regime + fallback global
                        model_data = {
                            "model":                 {r: m.model for r, m in export_engine.regime_models.items()},
                            "is_regime_conditioned": True,
                            "model_fallback":        export_engine.model.model,
                            "features":              row_features,
                            "label_map":             {-1: 0, 0: 1, 1: 2},
                        }
                    else:
                        # Mode A/B: single global model
                        model_data = {
                            "model":                 export_engine.model.model,
                            "is_regime_conditioned": False,
                            "features":              row_features,
                            "label_map":             {-1: 0, 0: 1, 1: 2},
                        }
                    _reader.save_model(
                        model_data, row["mode"], row["horizon"], fhash
                    )

                    # ── Export: supporting data CSV ────────────────────────
                    _reader.save_model_csv(
                        st.session_state.df_built,
                        row_features,
                        row["mode"], row["horizon"], fhash,
                    )

                    # ── Export: edge_metadata JSON ─────────────────────────
                    _edge_meta = {
                        "max_hold":           int(export_cfg.get("max_hold", row["horizon"])),
                        "tp_pct":             float(export_cfg.get("tp_pct", row["tp_pct"])),
                        "threshold":          float(getattr(export_engine, "optimal_threshold",
                                                  export_cfg.get("prob_threshold", 0.55))),
                        "regime_conditioned": bool(export_cfg["modeling"].get("use_regime_conditioned", False)),
                        "direction":          export_cfg.get("direction", "long"),
                        "features":           row_features,
                        "horizon":            int(row["horizon"]),
                    }
                    _reader.save_edge_metadata(
                        _edge_meta, row["mode"], row["horizon"], fhash
                    )

                    model_bytes, model_filename = _reader.load_model_bytes(
                        row["mode"], row["horizon"], fhash
                    )
                    csv_bytes, csv_filename = _reader.load_model_csv_bytes(
                        row["mode"], row["horizon"], fhash
                    )
                    meta_bytes, meta_filename = _reader.load_edge_metadata_bytes(
                        row["mode"], row["horizon"], fhash
                    )
                    st.session_state[f"model_bytes_{export_key}"]    = model_bytes
                    st.session_state[f"model_filename_{export_key}"] = model_filename
                    st.session_state[f"csv_bytes_{export_key}"]      = csv_bytes
                    st.session_state[f"csv_filename_{export_key}"]   = csv_filename
                    st.session_state[f"meta_bytes_{export_key}"]     = meta_bytes
                    st.session_state[f"meta_filename_{export_key}"]  = meta_filename
                    train_size = export_cfg.get("walkforward", {}).get("train_size", "?")
                    st.success(
                        f"Model berhasil ditraining! ({len(row_features)} features, last WF window ~{train_size} bars)\n\n"
                        f"📁 Disimpan di: `{_reader.path}`"
                    )

                except Exception as e:
                    st.error(f"Export error: {e}")

        # Tombol download muncul jika sudah ada model bytes di session state
        model_bytes    = st.session_state.get(f"model_bytes_{export_key}")
        model_filename = st.session_state.get(f"model_filename_{export_key}", "model.pkl")
        csv_bytes      = st.session_state.get(f"csv_bytes_{export_key}")
        csv_filename   = st.session_state.get(f"csv_filename_{export_key}", "data.csv")
        meta_bytes     = st.session_state.get(f"meta_bytes_{export_key}")
        meta_filename  = st.session_state.get(f"meta_filename_{export_key}", "edge_metadata.json")
        if model_bytes or csv_bytes or meta_bytes:
            with export_col2:
                dl_c1, dl_c2, dl_c3 = st.columns(3)
                if model_bytes:
                    dl_c1.download_button(
                        label=f"⬇️ {model_filename}",
                        data=model_bytes,
                        file_name=model_filename,
                        mime="application/octet-stream",
                        key=f"dl_model_{export_key}",
                    )
                if csv_bytes:
                    dl_c2.download_button(
                        label=f"⬇️ {csv_filename}",
                        data=csv_bytes,
                        file_name=csv_filename,
                        mime="text/csv",
                        key=f"dl_csv_{export_key}",
                    )
                if meta_bytes:
                    dl_c3.download_button(
                        label=f"⬇️ {meta_filename}",
                        data=meta_bytes,
                        file_name=meta_filename,
                        mime="application/json",
                        key=f"dl_meta_{export_key}",
                    )

render_mining_results()

# =====================================================
# 9. RENDER TRIAL SCORECARD (persistent)
# =====================================================

@st.fragment
def render_trial_scorecard():
    if st.session_state.get("mining_trial_scorecard") is None or not is_mining:
        return
    sc = st.session_state.mining_trial_scorecard

    st.divider()
    st.subheader("📊 Trial Scorecard — ALL")

    # ── Summary metrics ───────────────────────────────────────────
    total_tested  = len(sc)
    n_passed      = int(sc["gate_passed"].sum()) if "gate_passed" in sc.columns else 0
    n_gate_rej    = int((sc["gate_status"] == "gate_rejected").sum()) if "gate_status" in sc.columns else 0
    n_dsr_rej     = int((sc["gate_status"] == "dsr_rejected").sum()) if "gate_status" in sc.columns else 0
    n_error       = int(sc["gate_status"].isin(["engine_error","no_stats"]).sum()) if "gate_status" in sc.columns else 0
    pass_rate     = f"{n_passed/total_tested*100:.1f}%" if total_tested > 0 else "—"

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Total Diuji",   f"{total_tested:,}")
    c2.metric("✅ Final Valid",  f"{n_passed}")
    c3.metric("❌ Gate Rejected", f"{n_gate_rej}")
    c4.metric("⚠️ DSR Rejected", f"{n_dsr_rej}")
    c5.metric("💥 Error",        f"{n_error}")
    c6.metric("Pass Rate",      pass_rate)

    # ── Filter controls ───────────────────────────────────────────
    col_flt1, col_flt2, col_flt3 = st.columns(3)
    with col_flt1:
        status_opts = ["Semua"] + (list(sc["gate_status"].dropna().unique()) if "gate_status" in sc.columns else [])
        filter_status = st.selectbox("Filter Gate Status", options=status_opts, key="sc_status_filter")
    with col_flt2:
        mode_opts = ["Semua"] + (sorted(sc["mode"].dropna().unique().tolist()) if "mode" in sc.columns else [])
        filter_mode = st.selectbox("Filter Mode", options=mode_opts, key="sc_mode_filter")
    with col_flt3:
        rej_opts = ["Semua"] + (list(sc["rejection_reason"].dropna().unique()) if "rejection_reason" in sc.columns else [])
        filter_rej = st.selectbox("Filter Rejection Reason", options=rej_opts, key="sc_rej_filter")

    sc_display = sc.copy()
    if filter_status != "Semua" and "gate_status" in sc_display.columns:
        sc_display = sc_display[sc_display["gate_status"] == filter_status]
    if filter_mode != "Semua" and "mode" in sc_display.columns:
        sc_display = sc_display[sc_display["mode"] == filter_mode]
    if filter_rej != "Semua" and "rejection_reason" in sc_display.columns:
        sc_display = sc_display[sc_display["rejection_reason"] == filter_rej]

    # ── Display columns ───────────────────────────────────────────
    sc_cols = [c for c in [
        "mode", "horizon", "tp_pct", "features_str", "num_features",
        "optimal_threshold",
        "sharpe_oos", "sharpe_is", "n_trades_oos",
        "probabilistic_sharpe", "deflated_sharpe",
        "regime_mean", "regime_std", "worst_regime_sharpe",
        "regime_stability_score", "alpha_concentration_index",
        "composite_score", "alpha_type",
        "gate_status", "rejection_reason",
    ] if c in sc_display.columns]

    sc_col_config = {
        "mode":                        st.column_config.TextColumn("Mode",          width="small"),
        "horizon":                     st.column_config.NumberColumn("H",           width="small"),
        "tp_pct":                      st.column_config.NumberColumn("TP%",         format="%.2f", width="small"),
        "features_str":                st.column_config.TextColumn("Features",      width="large"),
        "num_features":                st.column_config.NumberColumn("#Feat",       format="%d",   width="small"),
        "optimal_threshold":           st.column_config.NumberColumn("Threshold",   format="%.2f", width="small"),
        "sharpe_oos":                  st.column_config.NumberColumn("Sharpe OOS",  format="%.4f", width="small"),
        "sharpe_is":                   st.column_config.NumberColumn("Sharpe IS",   format="%.4f", width="small"),
        "n_trades_oos":                st.column_config.NumberColumn("N Trades",    format="%d",   width="small"),
        "probabilistic_sharpe":        st.column_config.NumberColumn("PSR",         format="%.3f", width="small"),
        "deflated_sharpe":             st.column_config.NumberColumn("DSR",         format="%.3f", width="small"),
        "regime_mean":                 st.column_config.NumberColumn("Reg Mean",    format="%.4f", width="small"),
        "regime_std":                  st.column_config.NumberColumn("Reg Std",     format="%.4f", width="small"),
        "worst_regime_sharpe":         st.column_config.NumberColumn("Worst Reg",   format="%.4f", width="small"),
        "regime_stability_score":      st.column_config.NumberColumn("RSS",         format="%.4f", width="small"),
        "alpha_concentration_index":   st.column_config.NumberColumn("ACI",         format="%.4f", width="small"),
        "composite_score":             st.column_config.NumberColumn("Score",       format="%.4f", width="small"),
        "alpha_type":                  st.column_config.TextColumn("Alpha Type",    width="medium"),
        "gate_status":                 st.column_config.TextColumn("Gate Status",   width="medium"),
        "rejection_reason":            st.column_config.TextColumn("Reason",        width="large"),
    }

    _MAX_ROWS = 1000
    st.caption(f"Menampilkan {min(_MAX_ROWS, len(sc_display)):,} dari {total_tested:,} trial")
    st.dataframe(
        sc_display[sc_cols].head(_MAX_ROWS),
        width='stretch',
        hide_index=True,
        column_config=sc_col_config,
        key="trial_scorecard_table",
    )

render_trial_scorecard()