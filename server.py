"""
server.py — Flask UI pengganti Streamlit, 1:1 match dengan app.py
Taruh di folder yang sama dengan app.py (edge_v4/)
Jalankan: python server.py
Buka:     http://localhost:5000
"""

import os, sys, copy, json, threading, queue, traceback, time, io
import pandas as pd
import yaml
from flask import Flask, request, jsonify, render_template_string, Response, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

app = Flask(__name__)

# ── Global state (= st.session_state) ─────────────────────────────────────
STATE = {
    "df_built": None, "registry_built": None,
    "filter_feature_pool": None, "filter_structural": None,
    "filter_horizon_map": None, "filter_scorecard": None,
    "filter_router_result": None, "filter_passports": None,
    "filter_passport_df": None, "filter_model_map": None,
    "filter_transform_map": None,
    "mining_results_df": None, "mining_artifacts_path": None,
    "mining_feature_pool": None, "mining_config_snapshot": None,
    "mining_trial_scorecard": None,
    "export_store": {},
}

LOG_Q = queue.Queue()
JOB   = threading.Event()
HOURS_PER_DAY = 24

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    LOG_Q.put(line)
    print(line)

def load_cfg():
    p = os.path.join(BASE_DIR, "config.yaml")
    cfg = yaml.safe_load(open(p)) if os.path.exists(p) else {}
    for k in ["modeling","mining","feature_builder","walkforward","feature_pruning","logging"]:
        cfg.setdefault(k, {})
    cfg["logging"].setdefault("run_type", "atomic")
    return cfg

CONFIG = load_cfg()

def df_to_records(df, max_rows=2000):
    if df is None or (hasattr(df,'empty') and df.empty): return []
    return df.head(max_rows).fillna("").to_dict(orient="records")

def get_available_features():
    if STATE["registry_built"] is not None:
        return STATE["registry_built"].get_feature_list()
    return []

# ── API ────────────────────────────────────────────────────────────────────

@app.route("/api/logs")
def api_logs():
    msgs = []
    try:
        while True: msgs.append(LOG_Q.get_nowait())
    except queue.Empty: pass
    return jsonify({"logs": msgs})

@app.route("/api/state")
def api_state():
    rr = STATE.get("filter_router_result")
    router_summary = None
    if rr:
        try: router_summary = rr.summary
        except: pass
    return jsonify({
        "features_built":    STATE["df_built"] is not None,
        "n_features":        len(get_available_features()),
        "available_features":get_available_features(),
        "filter_done":       STATE["filter_scorecard"] is not None,
        "mining_done":       STATE["mining_results_df"] is not None,
        "job_running":       JOB.is_set(),
        "filter_horizon_map":STATE["filter_horizon_map"] or {},
        "filter_model_map":  STATE["filter_model_map"] or {},
        "filter_feature_pool": STATE["filter_feature_pool"] or [],
        "micro_alpha":  rr.micro_alpha  if rr else [],
        "structural":   rr.structural   if rr else [],
        "retransform":  rr.retransform  if rr else [],
        "graveyard":    rr.graveyard    if rr else [],
        "router_summary": router_summary,
        "scorecard":        df_to_records(STATE["filter_scorecard"]),
        "passport_df":      df_to_records(STATE["filter_passport_df"]),
        "mining_results":   df_to_records(STATE["mining_results_df"]),
        "trial_scorecard":  df_to_records(STATE["mining_trial_scorecard"], 5000),
        "mining_snapshot":  STATE["mining_config_snapshot"] or {},
        "mining_feature_pool": STATE["mining_feature_pool"] or [],
    })

@app.route("/api/build", methods=["POST"])
def api_build():
    if JOB.is_set(): return jsonify({"ok":False,"error":"Job sedang berjalan"}), 409
    d = request.json or {}
    cfg = load_cfg()
    fb  = cfg["feature_builder"]
    fb["technical"]           = d.get("technical", True)
    fb["onchain"]             = d.get("onchain", False)
    fb["enable_transform"]    = d.get("enable_transform", False)
    fb["enable_interaction"]  = d.get("enable_interaction", False)
    fb["dynamic_interactions"]= d.get("dynamic_interactions", False)
    fb["feature_windows"]     = [int(x) for x in str(d.get("feature_windows","5,10,20")).split(",") if x.strip().isdigit()]

    def _run():
        JOB.set()
        try:
            from governance.manager import OutputManager
            from engine import EdgeEngine
            log("🔨 Building features...")
            om = OutputManager(save_artifacts=False)
            engine = EdgeEngine(cfg, om)
            engine.load_data()
            df = engine.build_features(log_registry=False)
            STATE["df_built"]       = df
            STATE["registry_built"] = engine.registry
            n = len(engine.registry.get_feature_list())
            log(f"✅ Features built — {n} features, {len(df)} rows")
        except Exception:
            log("❌ Build error:\n" + traceback.format_exc())
        finally:
            JOB.clear()

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/filter", methods=["POST"])
def api_filter():
    if JOB.is_set(): return jsonify({"ok":False,"error":"Job sedang berjalan"}), 409
    if STATE["df_built"] is None: return jsonify({"ok":False,"error":"Build features dulu!"}), 400
    d = request.json or {}

    def _run():
        JOB.set()
        try:
            from signal_filter import FilterPipeline, FilterPipelineConfig
            from signal_filter.ic_filter import ICConfig
            from signal_filter.sign_consistency_filter import SignConsistencyConfig
            from signal_filter.decorrelation_filter import DecorrelationConfig
            from signal_filter.feature_router import RouterConfig
            from signal_filter.feature_passport import PassportConfig
            from governance.manager import OutputManager

            ic_window  = int(d.get("ic_window", 500))
            ic_step    = int(d.get("ic_step", 100))
            decay_h    = [int(x) for x in str(d.get("decay_horizons","1,5,10,24,48,72,100")).split(",") if x.strip().isdigit()]
            run_passport = bool(d.get("run_passport", True))

            _passport_cfg = PassportConfig(
                n_sub_periods=int(d.get("passport_n_periods",6)),
                min_stability=float(d.get("passport_min_stability",0.55)),
                regime_window=int(d.get("passport_regime_window",168)),
                tail_ratio_threshold=float(d.get("passport_tail_ratio",0.20)),
            ) if run_passport else PassportConfig()

            pipeline_config = FilterPipelineConfig(
                ic=ICConfig(
                    ic_window=ic_window, ic_step=ic_step,
                    decay_horizons=decay_h,
                    min_ic_mean=float(d.get("min_ic_mean",0.02)),
                    min_ic_ir=float(d.get("min_ic_ir",0.30)),
                    min_t_stat=float(d.get("min_t_stat",2.0)),
                    min_ic_positive_pct=float(d.get("min_ic_pos_pct",0.50)),
                ),
                sign_consistency=SignConsistencyConfig(
                    window=ic_window, step=ic_step,
                    min_monotonicity=float(d.get("min_monotonicity",0.60)),
                    min_spread_pct=float(d.get("min_spread_pct",0.55)),
                ),
                decorrelation=DecorrelationConfig(max_correlation=float(d.get("max_corr",0.80))),
                router=RouterConfig(
                    structural_min_peak_horizon=int(d.get("structural_horizon_threshold",72)),
                    structural_min_trend_ratio=float(d.get("structural_trend_ratio",1.5)),
                ),
                passport=_passport_cfg,
                run_sign_consistency=bool(d.get("run_sign",True)),
                run_decorrelation=bool(d.get("run_decorr",True)),
                run_passport=run_passport,
                close_col="close",
            )

            feature_pool = d.get("feature_pool") or get_available_features()
            log(f"🔬 Running filter on {len(feature_pool)} features...")

            om = OutputManager(save_artifacts=True, base_dir=os.path.join("experiments","alpha_statistical_filter"))
            pipeline = FilterPipeline(config=pipeline_config, logger=om.logger)
            router_result, scorecard_df = pipeline.run(STATE["df_built"], feature_pool)

            STATE["filter_feature_pool"]  = router_result.micro_alpha + router_result.structural
            STATE["filter_structural"]    = router_result.structural
            STATE["filter_horizon_map"]   = router_result.horizon_map
            STATE["filter_scorecard"]     = scorecard_df
            STATE["filter_router_result"] = router_result

            _pm, _tm = {}, {}
            if scorecard_df is not None and not scorecard_df.empty:
                if "model_type" in scorecard_df.columns:
                    _pm = dict(zip(scorecard_df["feature"], scorecard_df["model_type"].fillna("gradient_boosting")))
                if "transform" in scorecard_df.columns:
                    _tm = dict(zip(scorecard_df["feature"], scorecard_df["transform"].fillna("zscore")))
                _pdf = scorecard_df[scorecard_df["feature"].isin(router_result.micro_alpha)]
                STATE["filter_passport_df"] = _pdf
                sc_path = os.path.join(om.path, "filter_scorecard.csv")
                scorecard_df.to_csv(sc_path, index=False)
                log(f"💾 Scorecard saved: {sc_path}")

            STATE["filter_model_map"]     = _pm
            STATE["filter_transform_map"] = _tm

            s = router_result.summary
            log(f"✅ Filter done — micro_alpha:{s.get('micro_alpha')} structural:{s.get('structural')} graveyard:{s.get('graveyard')}")
        except Exception:
            log("❌ Filter error:\n" + traceback.format_exc())
        finally:
            JOB.clear()

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/mine", methods=["POST"])
def api_mine():
    if JOB.is_set(): return jsonify({"ok":False,"error":"Job sedang berjalan"}), 409
    if STATE["df_built"] is None: return jsonify({"ok":False,"error":"Build features dulu!"}), 400
    d = request.json or {}

    def _run():
        JOB.set()
        try:
            from governance.manager import OutputManager
            from next.edge_miner import EdgeMiner
            import numpy as np

            cfg = load_cfg()
            tp_pcts   = [float(x)/100 for x in str(d.get("tp_pcts","5")).split(",") if x.strip().replace(".","").isdigit()]
            horizons  = [int(x) for x in str(d.get("horizons","12,48,72")).split(",") if x.strip().isdigit()]
            modes     = d.get("modes", ["A"])
            max_komb  = int(d.get("max_kombinasi", 2))

            cfg["mining"]["tp_pcts"]      = tp_pcts
            cfg["mining"]["n_jobs"]       = int(d.get("n_jobs",-1))
            cfg["mining"]["batch_size"]   = int(d.get("batch_size",500))
            cfg["mining"]["max_kombinasi"]= max_komb
            cfg["mode"]      = d.get("mode","walkforward")
            cfg["direction"] = d.get("direction","both")
            wf = cfg.setdefault("walkforward",{})
            wf["train_size"] = int(d.get("train_days",360))*24
            wf["test_size"]  = int(d.get("test_days",30))*24
            wf["step_size"]  = int(d.get("step_days",30))*24
            cfg["tp_pct"]    = tp_pcts[0] if tp_pcts else 0.05
            cfg["max_hold"]  = int(d.get("max_hold",100))
            cfg["optimize_threshold"] = bool(d.get("optimize_threshold",True))
            cfg["prob_threshold"]     = float(d.get("prob_threshold",0.55))
            g_min  = float(d.get("threshold_g_min",0.30))
            g_max  = float(d.get("threshold_g_max",0.55))
            g_step = float(d.get("threshold_g_step",0.05))
            cfg["threshold_grid"] = [round(float(v),4) for v in np.arange(g_min, g_max+g_step*0.5, g_step)]
            cfg["threshold_precision_target"] = float(d.get("threshold_precision_target",0.55))
            cfg["threshold_min_trades"]       = int(d.get("threshold_min_trades",10))
            cfg["feature_pruning"] = {"enabled": bool(d.get("pruning_enabled",False)), "correlation_threshold": float(d.get("corr_threshold",0.95))}
            cfg["logging"]["run_type"] = "atomic"

            feature_pool = d.get("feature_pool") or STATE.get("filter_feature_pool") or get_available_features()
            if STATE["filter_horizon_map"]:
                horizons = sorted(set(STATE["filter_horizon_map"].values()))
                log(f"ℹ️ Horizon override dari filter: {horizons}")

            cfg["mining"]["feature_pool"] = feature_pool
            log(f"⚛️ Mining {len(feature_pool)} features × {len(horizons)} horizons × modes={modes}")

            om = OutputManager(save_artifacts=True, base_dir=os.path.join("experiments","atomic"))
            om.log_feature_built(STATE["registry_built"])
            miner = EdgeMiner(cfg, om)
            miner.df_built       = STATE["df_built"]
            miner.registry_built = copy.deepcopy(STATE["registry_built"])
            if STATE.get("filter_model_map"):
                miner.model_map     = STATE["filter_model_map"]
                miner.transform_map = STATE.get("filter_transform_map") or {}

            results_df = miner.mine(
                feature_pool  = feature_pool,
                modes         = tuple(modes),
                horizons      = horizons,
                tp_pcts       = tp_pcts,
                max_kombinasi = max_komb,
            )

            if miner.trial_scorecard is not None:
                STATE["mining_trial_scorecard"] = miner.trial_scorecard

            if results_df is not None and not results_df.empty:
                STATE["mining_results_df"]      = results_df.sort_values("composite_score",ascending=False).reset_index(drop=True)
                STATE["mining_artifacts_path"]  = om.path
                STATE["mining_feature_pool"]    = feature_pool
                STATE["mining_config_snapshot"] = {"horizons":horizons,"tp_pcts":tp_pcts,"modes":modes,"from_filter":STATE["filter_horizon_map"] is not None}
                om.save_atomic_summary(STATE["mining_results_df"])
                log(f"✅ Mining done — {len(results_df)} edges found")
            else:
                STATE["mining_results_df"] = None
                log("⚠️ Tidak ada alpha valid yang ditemukan.")
        except Exception:
            log("❌ Mining error:\n" + traceback.format_exc())
        finally:
            JOB.clear()

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/ensemble", methods=["POST"])
def api_ensemble():
    if JOB.is_set(): return jsonify({"ok":False,"error":"Job sedang berjalan"}), 409
    if STATE["df_built"] is None: return jsonify({"ok":False,"error":"Build features dulu!"}), 400
    d = request.json or {}

    def _run():
        JOB.set()
        try:
            from governance.manager import OutputManager
            from engine import EdgeEngine
            cfg = load_cfg()
            cfg["mode"]      = d.get("mode","walkforward")
            cfg["direction"] = "both"
            wf = cfg.setdefault("walkforward",{})
            wf["train_size"] = int(d.get("train_days",360))*24
            wf["test_size"]  = int(d.get("test_days",30))*24
            wf["step_size"]  = int(d.get("step_days",30))*24
            cfg["tp_pct"]    = float(d.get("tp_pct",0.02))
            cfg["max_hold"]  = int(d.get("max_hold",100))
            cfg["horizon"]   = cfg["max_hold"]
            cfg["optimize_threshold"] = bool(d.get("optimize_threshold",True))
            cfg["prob_threshold"]     = float(d.get("prob_threshold",0.55))
            cfg["modeling"]["use_regime_conditioned"] = bool(d.get("use_regime_conditioned",False))
            cfg["modeling"]["use_allowed_regimes"]    = bool(d.get("use_allowed_regimes",False))
            if d.get("allowed_regimes"):
                cfg["modeling"]["allowed_regimes"] = [int(x) for x in str(d["allowed_regimes"]).split(",") if x.strip().isdigit()]
            cfg["logging"]["run_type"] = "ensemble"
            cfg["mining"] = {"enabled": False}
            fp = d.get("feature_pool") or STATE.get("filter_feature_pool") or []
            cfg["modeling"]["feature_pool"] = fp

            log(f"🚀 Ensemble engine — {len(fp)} features, mode={cfg['mode']}")
            om = OutputManager(save_artifacts=True, base_dir=os.path.join("experiments","ensemble"))
            om.log_feature_built(STATE["registry_built"], feature_pool=fp or None)
            engine = EdgeEngine(cfg, om)
            engine.df       = STATE["df_built"]
            engine.registry = copy.deepcopy(STATE["registry_built"])
            if fp: engine.override_feature_list(fp)
            engine.run(use_prebuilt_data=True)

            STATE["_last_ensemble"] = {
                "sharpe_oos":   getattr(engine,"sharpe_oos",None),
                "sharpe_is":    getattr(engine,"sharpe_is",None),
                "sharpe_total": getattr(engine,"sharpe_total",None),
                "stats_df":     df_to_records(getattr(engine,"stats_df",None)),
                "direction_df": df_to_records(getattr(engine,"direction_df",None)),
                "feature_pool": fp,
            }
            log(f"✅ Ensemble done — OOS={getattr(engine,'sharpe_oos','?')}")
        except Exception:
            log("❌ Ensemble error:\n" + traceback.format_exc())
        finally:
            JOB.clear()

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/ensemble/result")
def api_ensemble_result():
    return jsonify(STATE.get("_last_ensemble") or {})

@app.route("/api/export_model", methods=["POST"])
def api_export_model():
    if JOB.is_set(): return jsonify({"ok":False,"error":"Job sedang berjalan"}), 409
    d = request.json or {}
    row_idx    = int(d.get("row_idx", 0))
    export_key = d.get("export_key","")

    def _run():
        JOB.set()
        try:
            from governance.manager import OutputManager
            from engine import EdgeEngine
            import copy as _copy

            results_df = STATE["mining_results_df"]
            if results_df is None or row_idx >= len(results_df):
                log("❌ Row tidak ditemukan"); return

            row   = results_df.iloc[row_idx]
            fhash = row["feature_hash"]
            cfg   = load_cfg()
            _reader = OutputManager.__new__(OutputManager)
            _reader.path = STATE["mining_artifacts_path"]
            _reader.save_artifacts = True
            _reader.logger = _reader._setup_logger()

            export_cfg = _copy.deepcopy(cfg)
            export_cfg["horizon"]  = int(row["horizon"])
            export_cfg["max_hold"] = int(row["horizon"])
            export_cfg["tp_pct"]   = float(row["tp_pct"])
            export_cfg["logging"]["run_type"] = "atomic"
            export_cfg["modeling"]["use_regime_conditioned"] = row["mode"] in ("C","D")

            row_features = row["features"] if isinstance(row["features"],list) else [row["features"]]
            log(f"🤖 Retraining — {len(row_features)} features...")

            export_engine = EdgeEngine(export_cfg, _reader)
            export_engine.df       = STATE["df_built"].copy()
            export_engine.registry = _copy.deepcopy(STATE["registry_built"])
            export_engine.registry.set_feature_list(row_features)
            export_engine.check_feature()
            export_engine.build_target()
            export_engine.train_last_wf()

            _is_regime = bool(export_cfg["modeling"].get("use_regime_conditioned",False) and getattr(export_engine,"regime_models",{}))
            if _is_regime:
                model_data = {"model":{r:m.model for r,m in export_engine.regime_models.items()},"is_regime_conditioned":True,"model_fallback":export_engine.model.model,"features":row_features,"label_map":{-1:0,0:1,1:2}}
            else:
                model_data = {"model":export_engine.model.model,"is_regime_conditioned":False,"features":row_features,"label_map":{-1:0,0:1,1:2}}

            _reader.save_model(model_data, row["mode"], row["horizon"], fhash)
            _reader.save_model_csv(STATE["df_built"], row_features, row["mode"], row["horizon"], fhash)
            _edge_meta = {"max_hold":int(export_cfg.get("max_hold",row["horizon"])),"tp_pct":float(export_cfg.get("tp_pct",row["tp_pct"])),"threshold":float(getattr(export_engine,"optimal_threshold",export_cfg.get("prob_threshold",0.55))),"regime_conditioned":bool(export_cfg["modeling"].get("use_regime_conditioned",False)),"direction":export_cfg.get("direction","long"),"features":row_features,"horizon":int(row["horizon"])}
            _reader.save_edge_metadata(_edge_meta, row["mode"], row["horizon"], fhash)

            mb, mfn   = _reader.load_model_bytes(row["mode"], row["horizon"], fhash)
            cb, cfn   = _reader.load_model_csv_bytes(row["mode"], row["horizon"], fhash)
            etb, etfn = _reader.load_edge_metadata_bytes(row["mode"], row["horizon"], fhash)

            STATE["export_store"][export_key] = {
                "model_bytes":mb,"model_filename":mfn,
                "csv_bytes":cb,"csv_filename":cfn,
                "meta_bytes":etb,"meta_filename":etfn,
            }
            log(f"✅ Model exported — {mfn}")
        except Exception:
            log("❌ Export error:\n" + traceback.format_exc())
        finally:
            JOB.clear()

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True})

@app.route("/api/download/model/<key>/<file_type>")
def api_dl_model(key, file_type):
    store = STATE["export_store"].get(key)
    if not store: return "Not found", 404
    if file_type == "model":
        return Response(store["model_bytes"], mimetype="application/octet-stream",
                        headers={"Content-Disposition":f"attachment;filename={store['model_filename']}"})
    elif file_type == "csv":
        return Response(store["csv_bytes"], mimetype="text/csv",
                        headers={"Content-Disposition":f"attachment;filename={store['csv_filename']}"})
    elif file_type == "meta":
        return Response(store["meta_bytes"], mimetype="application/json",
                        headers={"Content-Disposition":f"attachment;filename={store['meta_filename']}"})
    return "Unknown", 400

@app.route("/api/download/scorecard")
def api_dl_scorecard():
    if STATE["filter_scorecard"] is None: return "No data", 404
    buf = io.StringIO(); STATE["filter_scorecard"].to_csv(buf, index=False)
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition":"attachment;filename=filter_scorecard.csv"})

@app.route("/api/download/mining")
def api_dl_mining():
    if STATE["mining_results_df"] is None: return "No data", 404
    buf = io.StringIO(); STATE["mining_results_df"].to_csv(buf, index=False)
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition":"attachment;filename=mining_results.csv"})

@app.route("/api/reset/<key>", methods=["POST"])
def api_reset(key):
    if key == "filter":
        for k in ["filter_feature_pool","filter_structural","filter_horizon_map","filter_scorecard","filter_router_result","filter_passports","filter_passport_df","filter_model_map","filter_transform_map"]:
            STATE[k] = None
    elif key == "mining":
        for k in ["mining_results_df","mining_artifacts_path","mining_feature_pool","mining_config_snapshot","mining_trial_scorecard"]:
            STATE[k] = None
    return jsonify({"ok": True})

# ══════════════════════════════════════════════════════════════════════════
# Static frontend — Next.js build (frontend/out/) or legacy server_ui.html
# ══════════════════════════════════════════════════════════════════════════

REACT_BUILD = os.path.join(BASE_DIR, "frontend", "out")
_USE_REACT   = os.path.isdir(REACT_BUILD)

if not _USE_REACT:
    # Fallback: legacy single-file HTML
    _html_path = os.path.join(BASE_DIR, "server_ui.html")
    if not os.path.exists(_html_path):
        _html_path = os.path.join(BASE_DIR, "data", "server_ui.html")
    HTML = open(_html_path, encoding="utf-8").read()

from flask import send_from_directory as _sfd

@app.route("/")
def index():
    if _USE_REACT:
        return _sfd(REACT_BUILD, "index.html")
    return HTML

@app.route("/<path:path>")
def static_proxy(path):
    if path.startswith("api/"):
        return "Not found", 404
    if _USE_REACT:
        full = os.path.join(REACT_BUILD, path)
        if os.path.isfile(full):
            return _sfd(REACT_BUILD, path)
        # SPA fallback — try path/index.html (Next.js trailingSlash)
        slash = os.path.join(REACT_BUILD, path, "index.html")
        if os.path.isfile(slash):
            return _sfd(os.path.join(REACT_BUILD, path), "index.html")
        return _sfd(REACT_BUILD, "index.html")
    return "Not found", 404

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  SILENTWAVE — Flask UI (1:1 Streamlit)")
    print("  Buka browser:  http://localhost:5000")
    print("="*55 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)