import copy
import hashlib
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from engine import EdgeEngine
from engine import _ENGINE_COLS
from statistical_tests import (
    probabilistic_sharpe,
    deflated_sharpe_ratio,
    regime_stability_score,
    monte_carlo_equity_test,
)


class EdgeMiner:

    def __init__(self, config, output_manager):
        self.base_config     = config
        self.results         = []
        self.trial_scorecard = None
        self.modes           = config.get("mining", {}).get("modes", ["A", "B", "C", "D"])
        self.df_built        = None
        self.registry_built  = None
        self.output_manager  = output_manager
        self.logger          = self.output_manager.logger

        # Total tasks ditracking untuk deflated Sharpe correction
        self._total_tasks_run = 0

        # Passport maps — diisi dari app.py setelah Filter selesai
        # {feature_name: model_type}  → hint untuk _run_one (future use)
        # {feature_name: transform}   → hint untuk preprocessing (future use)
        # Saat ini: dilog dan disimpan di trial record untuk traceability
        self.model_map     = {}   # {feature: "linear" / "gradient_boosting" / ...}
        self.transform_map = {}   # {feature: "zscore" / "rank_transform" / ...}

    # ─────────────────────────────────────────────
    # CONFIG HELPERS
    # ─────────────────────────────────────────────
    def _prepare_config(self, mode_name):
        config   = copy.deepcopy(self.base_config)
        mode_map = {
            "A": {"feature_mode": "single", "regime_mode": False},
            "B": {"feature_mode": "multi",  "regime_mode": False},
            "C": {"feature_mode": "single", "regime_mode": True},
            "D": {"feature_mode": "multi",  "regime_mode": True},
        }
        if mode_name not in mode_map:
            raise ValueError(f"Unknown mining mode: {mode_name}")
        mode_cfg = mode_map[mode_name]
        config["modeling"]["use_regime_conditioned"] = mode_cfg["regime_mode"]
        config["logging"]["mining_mode_desc"] = f"{mode_name}: {self._describe_mode(mode_name)}"
        return config, mode_cfg["feature_mode"]

    def _describe_mode(self, mode_name):
        return {
            "A": "Single w/o regime",
            "B": "Multi w/o regime",
            "C": "Single with regime",
            "D": "Multi with regime",
        }.get(mode_name, "Unknown")

    def _feature_hash(self, features):
        s = "_".join(sorted(features))
        return hashlib.md5(s.encode()).hexdigest()[:12]

    # ─────────────────────────────────────────────
    # BUILD FEATURES
    # ─────────────────────────────────────────────
    def build_features(self, feature_override=None, log_registry=True):
        engine = EdgeEngine(self.base_config, self.output_manager)
        engine.load_data()
        df_built = engine.build_features(log_registry=log_registry)
        if feature_override is not None:
            engine.registry.set_feature_list(feature_override)
        self.df_built       = df_built
        self.registry_built = engine.registry
        return df_built, engine.registry

    # ─────────────────────────────────────────────
    # FEATURE PRUNING
    # ─────────────────────────────────────────────
    def _prune_feature_pool(self, feature_pool, df):
        pruning_cfg = self.base_config.get("feature_pruning", {})
        if not pruning_cfg.get("enabled", True):
            return feature_pool

        threshold = pruning_cfg.get("correlation_threshold", 0.95)
        valid_pool = [f for f in feature_pool if f in df.columns]
        if len(valid_pool) <= 1:
            return valid_pool

        corr_matrix = df[valid_pool].corr().abs()
        upper       = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = {col for col in upper.columns if (upper[col] > threshold).any()}
        pruned  = [f for f in valid_pool if f not in to_drop]

        n_dropped = len(valid_pool) - len(pruned)
        if n_dropped > 0:
            self.logger.info(
                f"Feature pruning: {len(valid_pool)} -> {len(pruned)} "
                f"({n_dropped} dropped, corr_threshold={threshold})"
            )
        return pruned

    # ─────────────────────────────────────────────
    # FEATURE COMBINATIONS
    # ─────────────────────────────────────────────
    def _generate_feature_combinations(self, feature_pool, max_kombinasi, is_regime_mode=False):
        from itertools import combinations
        combinations_list = [[f] for f in feature_pool]
        actual_max = min(2 if is_regime_mode else max_kombinasi, len(feature_pool))
        for combo_size in range(2, actual_max + 1):
            for combo in combinations(feature_pool, combo_size):
                combinations_list.append(list(combo))
        return combinations_list

    # ─────────────────────────────────────────────
    # WORKER — satu kombinasi, return hasil + logs
    # ─────────────────────────────────────────────
    @staticmethod
    def _run_one(config_copy, df_built, registry_built, output_manager,
                 subset, mode_name, horizon, tp_pct=None):
        """
        Worker function untuk satu kombinasi.
        Return: (trial_record, log_lines)
          trial_record SELALU berisi dict lengkap dengan field:
            gate_passed      : bool
            gate_status      : "passed" | "gate_rejected" | "engine_error" | "no_stats"
            rejection_reason : string alasan penolakan (kosong jika passed)
          Heavy fields (_stats_df, _oos_returns) hanya ada jika gate_passed=True.
        """
        logs         = []
        feature_hash = hashlib.md5("_".join(sorted(subset)).encode()).hexdigest()[:12]
        tp_val       = tp_pct if tp_pct is not None else config_copy.get("tp_pct", 0.02)
        tp_str       = f" TP:{tp_val*100:.1f}%"

        # ── Skeleton trial record (selalu dikembalikan) ────────────
        trial = {
            "mode":                       mode_name,
            "horizon":                    horizon,
            "features":                   subset,
            "features_str":               "|".join(sorted(subset)),
            "num_features":               len(subset),
            "feature_hash":               feature_hash,
            "tp_pct":                     tp_val,
            # Passport info (diisi dari model_map jika tersedia, else "default")
            "model_types":                "|".join(sorted(set(
                config_copy.get("_model_map", {}).get(f, "default") for f in subset
            ))),
            "transforms":                 "|".join(sorted(set(
                config_copy.get("_transform_map", {}).get(f, "none") for f in subset
            ))),
            # metrik — akan diisi setelah engine run
            "sharpe_oos":                 float("nan"),
            "sharpe_is":                  float("nan"),
            "n_trades_oos":               0,
            "n_trades_is":                0,
            "regime_mean":                float("nan"),
            "regime_std":                 float("nan"),
            "max_regime_sharpe":          float("nan"),
            "worst_regime_sharpe":        float("nan"),
            "regime_stability_score":     float("nan"),
            "alpha_concentration_index":  float("nan"),
            "probabilistic_sharpe":       float("nan"),
            "composite_score":            float("nan"),
            "alpha_type":                 None,
            "optimal_threshold":          float("nan"),   # threshold final yg dipakai
            # status
            "gate_passed":                False,
            "gate_status":                "engine_error",
            "rejection_reason":           "engine_error",
        }

        from governance.manager import SilentOutputManager
        worker_manager = SilentOutputManager()

        try:
            import copy as _copy
            engine          = EdgeEngine(config_copy, worker_manager)
            engine.df       = _selective_copy(df_built)
            engine.registry = _copy.deepcopy(registry_built)
            engine.run(feature_override=subset, use_prebuilt_data=True)

            if not hasattr(engine, "stats_df") or engine.stats_df is None or engine.stats_df.empty:
                trial["gate_status"]      = "no_stats"
                trial["rejection_reason"] = "no_stats"
                logs.append(f"  [{mode_name} H:{horizon}{tp_str}] {subset} -> No stats")
                return trial, logs

            # ── Kumpulkan metrik dasar ─────────────────────────────
            regime_sharpes        = engine.stats_df["Sharpe"]
            regime_positive_count = int((regime_sharpes > 0).sum())
            regime_mean           = float(regime_sharpes.mean())
            regime_std            = float(regime_sharpes.std())
            worst_sharpe          = float(regime_sharpes.min())

            n_oos = getattr(engine, "n_oos", 0)
            n_is  = getattr(engine, "n_is",  0)

            rss = regime_stability_score(regime_sharpes.values)
            aci = 1.0 / regime_positive_count if regime_positive_count > 0 else 1.0

            oos_mask    = engine.df["position_unseen"] != 0
            oos_returns = engine.df.loc[oos_mask, "strategy_return_test"].dropna()
            psr         = probabilistic_sharpe(oos_returns.values)

            # ── Isi metrik ke trial record ─────────────────────────
            trial.update({
                "sharpe_oos":                engine.sharpe_oos,
                "sharpe_is":                 engine.sharpe_is,
                "n_trades_oos":              n_oos,
                "n_trades_is":               n_is,
                "optimal_threshold":         getattr(engine, "optimal_threshold", float("nan")),
                "regime_mean":               regime_mean,
                "regime_std":                regime_std,
                "max_regime_sharpe":         float(regime_sharpes.max()),
                "worst_regime_sharpe":       worst_sharpe,
                "regime_stability_score":    rss,
                "alpha_concentration_index": aci,
                "probabilistic_sharpe":      psr,
            })

            metric_filter = config_copy.get("metric_filter", {})

            # ── Quality gate ───────────────────────────────────────
            alpha_type, rej_reason = EdgeMiner._quality_gate_static(
                sharpe_oos             = engine.sharpe_oos,
                sharpe_is              = engine.sharpe_is,
                regime_std             = regime_std,
                max_regime_sharpe      = float(regime_sharpes.max()),
                regime_mean            = regime_mean,
                regime_positive_count  = regime_positive_count,
                regime_stability_score = rss,
                alpha_concentration_index = aci,
                worst_regime_sharpe    = worst_sharpe,
                n_trades_oos           = n_oos,
                psr                    = psr,
                n_trials               = 1,
                metric_filter          = metric_filter,
                return_reason          = True,
            )

            trial["alpha_type"]      = alpha_type
            trial["rejection_reason"] = rej_reason

            if alpha_type is None:
                trial["gate_status"] = "gate_rejected"
                logs.append(f"  [{mode_name} H:{horizon}{tp_str}] {subset} -> Rejected ({rej_reason})")
                return trial, logs

            # ── Passed gate ────────────────────────────────────────
            composite = EdgeMiner._compute_composite(
                sharpe_oos  = engine.sharpe_oos,
                sharpe_is   = engine.sharpe_is,
                regime_mean = regime_mean,
                regime_std  = regime_std,
                n_trades    = n_oos,
                psr         = psr,
            )

            # Compact returns DataFrame untuk equity chart
            _ret_cols = [c for c in ["is_unseen","strategy_return_test","strategy_return_train",
                                     "position_unseen","position_seen"] if c in engine.df.columns]
            _returns_df = engine.df[_ret_cols].copy()

            trial.update({
                "composite_score":  composite,
                "gate_passed":      True,
                "gate_status":      "passed",
                "rejection_reason": "passed",
                # Heavy fields — hanya untuk passed (dipakai DSR + artifact saving)
                "_stats_df":        engine.stats_df,
                "_direction_df":    getattr(engine, "direction_df", None),
                "_oos_returns":     oos_returns.values,
                "_returns_df":      _returns_df,
            })

            logs.append(
                f"  [{mode_name} H:{horizon}{tp_str}] {subset} -> {alpha_type}  "
                f"Sharpe OOS={engine.sharpe_oos:.4f}  PSR={psr:.3f}  n={n_oos}"
            )

        except Exception as e:
            trial["gate_status"]      = "engine_error"
            trial["rejection_reason"] = f"error:{str(e)[:60]}"
            logs.append(f"  [{mode_name} H:{horizon}] {subset} -> ERROR: {e}")

        return trial, logs

    # ─────────────────────────────────────────────
    # COMPOSITE SCORE v4
    # Menambahkan n_trades confidence weighting via PSR
    # ─────────────────────────────────────────────
    @staticmethod
    def _compute_composite(sharpe_oos, sharpe_is, regime_mean, regime_std,
                           n_trades, psr):
        """
        Composite score v4:
          0.40 × sharpe_oos
          0.25 × regime_mean
          0.10 × stability (1/(1+regime_std))
          0.10 × overfit_penalty (OOS/IS ratio)
          0.15 × psr_weight (confidence dari ukuran sampel)

        PSR weight: penalti berat kalau n_trades kecil atau PSR rendah.
        """
        stability = 1.0 / (1.0 + regime_std)

        if sharpe_is is not None and sharpe_is > 0:
            overfit_penalty = min(sharpe_oos / sharpe_is, 1.0)
        elif sharpe_is is not None and sharpe_is < 0:
            # IS negatif tapi OOS positif — suspicious, penalti
            overfit_penalty = 0.0
        else:
            overfit_penalty = 0.0

        # PSR weight: 0 kalau PSR < 0.5, naik ke 1 kalau PSR → 1
        # Ini secara otomatis mendiskriminasi n_trades kecil
        psr_weight = max(0.0, (psr - 0.5) * 2.0)   # scale 0.5-1.0 → 0-1

        composite = (
            0.40 * sharpe_oos
            + 0.25 * regime_mean
            + 0.10 * stability
            + 0.10 * overfit_penalty
            + 0.15 * psr_weight
        )
        return float(composite)

    # ─────────────────────────────────────────────
    # MINE — parallel grid search dengan batching
    # ─────────────────────────────────────────────
    def mine(self, feature_pool, modes=("A", "B", "C", "D"),
             horizons=None, tp_pcts=None, max_kombinasi=2, top_k=None):
        """
        Mine edges dengan parallel grid search over modes × horizons × feature combinations.
        v4: tambah multiple testing correction (DSR) post-hoc setelah semua task selesai.
        """
        import platform
        self.results = []

        if self.df_built is None or self.registry_built is None:
            self.build_features(log_registry=False)

        if horizons is None:
            horizons = [self.base_config.get("horizon", 5)]
        if tp_pcts is None:
            tp_pcts = self.base_config.get("mining", {}).get(
                "tp_pcts", [self.base_config.get("tp_pct", 0.02)]
            )

        n_jobs     = self.base_config.get("mining", {}).get("n_jobs", -1)
        batch_size = self.base_config.get("mining", {}).get("batch_size", 500)

        is_windows = platform.system() == "Windows"
        if is_windows and n_jobs == -1:
            import os
            n_jobs = min(os.cpu_count() or 4, 8)
            self.logger.info(f"  [Windows] n_jobs capped to {n_jobs}")

        self.logger.info("Starting parallel grid search:")
        self.logger.info(f"  Modes        : {list(modes)}")
        self.logger.info(f"  Horizons     : {horizons}")
        self.logger.info(f"  Max Kombinasi: {max_kombinasi}")
        self.logger.info(f"  n_jobs       : {n_jobs}")
        self.logger.info(f"  Batch size   : {batch_size}")
        tp_display_log = [f"{v*100:.1f}%" for v in tp_pcts]
        self.logger.info(f"  TP grid      : {tp_display_log}")
        self.logger.info(f"  Feature pool (before pruning): {len(feature_pool)}")

        pruned_pool = self._prune_feature_pool(feature_pool, self.df_built)
        self.logger.info(f"  Feature pool (after pruning) : {len(pruned_pool)}")
        self.logger.info(f"  Features: {pruned_pool}")

        # Log passport model_map jika tersedia
        if self.model_map:
            self.logger.info(f"  Passport model_map ({len(self.model_map)} features):")
            for feat in pruned_pool:
                mt = self.model_map.get(feat, "default_xgboost")
                tr = self.transform_map.get(feat, "none")
                self.logger.info(f"    {feat:<35} model={mt:<22} transform={tr}")
        self.logger.info("")

        # ── Bangun task list ──────────────────────────────────────
        tasks = []
        for mode_name in modes:
            config_mode, feature_mode = self._prepare_config(mode_name)
            config_mode["direction"]  = "both"
            is_regime_mode = config_mode.get("modeling", {}).get("use_regime_conditioned", False)
            feature_sets   = self._generate_feature_combinations(
                pruned_pool, max_kombinasi, is_regime_mode=is_regime_mode
            )
            for horizon in horizons:
                for tp_pct in tp_pcts:
                    for subset in feature_sets:
                        if feature_mode == "single" and len(subset) > 1:
                            continue
                        if feature_mode == "multi"  and len(subset) == 1:
                            continue
                        tasks.append((config_mode, subset, mode_name, horizon, tp_pct))

        total_tasks = len(tasks)
        self._total_tasks_run = total_tasks

        self.logger.info(f"Total tasks to run: {total_tasks}")
        self.logger.info(f"Running in batches of {batch_size} ...")

        # ── Jalankan dalam batch ──────────────────────────────────
        raw_valid   = []   # semua hasil lolos quality gate (sebelum DSR)
        all_trials  = []   # SEMUA trial (termasuk rejected) — untuk scorecard
        n_batches   = (total_tasks + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            batch = tasks[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            self.logger.info(f"  Batch {batch_idx + 1}/{n_batches} — {len(batch)} tasks ...")

            batch_prepared = []
            for config_mode, subset, mode_name, horizon, tp_pct in batch:
                cfg = copy.deepcopy(config_mode)
                cfg["horizon"]  = horizon
                cfg["max_hold"] = horizon
                cfg["tp_pct"]   = tp_pct
                batch_prepared.append((cfg, subset, mode_name, horizon, tp_pct))

            raw_results = Parallel(n_jobs=n_jobs, backend="threading")(
                delayed(EdgeMiner._run_one)(
                    cfg, self.df_built, self.registry_built,
                    self.output_manager, subset, mode_name, horizon, tp_pct
                )
                for cfg, subset, mode_name, horizon, tp_pct in batch_prepared
            )

            for trial, logs in raw_results:
                for msg in logs:
                    self.logger.info(msg)
                all_trials.append(trial)
                if trial.get("gate_passed"):
                    raw_valid.append(trial)

            self.logger.info(f"  Batch {batch_idx + 1} done. Passed gate so far: {len(raw_valid)}")

        if not raw_valid:
            self.logger.info("\nNo valid alpha found.")
            # Tetap simpan trial scorecard meski tidak ada yang lolos
            _SC_COLS = [
                "mode", "horizon", "tp_pct", "features_str", "num_features", "feature_hash",
                "sharpe_oos", "sharpe_is", "n_trades_oos",
                "regime_mean", "regime_std", "max_regime_sharpe", "worst_regime_sharpe",
                "regime_stability_score", "alpha_concentration_index",
                "probabilistic_sharpe", "composite_score", "alpha_type",
                "optimal_threshold",
                "gate_passed", "gate_status", "rejection_reason",
            ]
            trial_records = [{c: t.get(c) for c in _SC_COLS} for t in all_trials]
            self.trial_scorecard = pd.DataFrame(trial_records)
            self.output_manager.save_trial_scorecard(self.trial_scorecard)
            return pd.DataFrame()

        # ─────────────────────────────────────────────────────────
        # v4: POST-HOC MULTIPLE TESTING CORRECTION
        # Setelah semua task selesai, koreksi dengan DSR menggunakan
        # total_tasks sebagai n_trials yang sebenarnya.
        # Edge yang lolos gate tapi DSR rendah di-filter atau di-flag.
        # ─────────────────────────────────────────────────────────
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info(f"v4 Post-hoc DSR correction (n_trials={total_tasks})")

        final_valid = []
        n_dsr_reject = 0

        for result in raw_valid:
            oos_rets     = result.pop("_oos_returns", np.array([]))
            stats_df     = result.pop("_stats_df")
            direction_df = result.pop("_direction_df")
            returns_df   = result.pop("_returns_df", None)

            # Hitung DSR dengan total_tasks sebagai n_trials
            dsr = deflated_sharpe_ratio(oos_rets, n_trials=total_tasks)
            result["deflated_sharpe"]  = float(dsr)
            result["dsr_reliable"]     = dsr >= 0.90   # threshold DSR

            # Re-koreksi composite score dengan DSR weight
            result["composite_score"] = self._compute_composite(
                sharpe_oos  = result["sharpe_oos"],
                sharpe_is   = result["sharpe_is"],
                regime_mean = result["regime_mean"],
                regime_std  = result["regime_std"],
                n_trades    = result["n_trades_oos"],
                psr         = result["probabilistic_sharpe"],
            )

            # Optional: hard reject kalau DSR sangat rendah
            _dsr_cfg = self.base_config.get("metric_filter", {}).get("dsr_min", 0.60)
            dsr_min  = None if _dsr_cfg is False else float(_dsr_cfg)
            if dsr_min is not None and dsr < dsr_min:
                n_dsr_reject += 1
                result["gate_status"]      = "dsr_rejected"
                result["rejection_reason"] = f"dsr={dsr:.3f}<{dsr_min:.2f}"
                self.logger.info(
                    f"  DSR reject: {result['features']} "
                    f"DSR={dsr:.3f} < {dsr_min:.2f}"
                )
                continue

            # Simpan artifacts
            self.output_manager.save_stats_df(
                stats_df, result["mode"], result["horizon"], result["feature_hash"]
            )
            if direction_df is not None:
                self.output_manager.save_direction_df(
                    direction_df, result["mode"], result["horizon"], result["feature_hash"]
                )
            if returns_df is not None:
                self.output_manager.save_returns_df(
                    returns_df, result["mode"], result["horizon"], result["feature_hash"]
                )

            final_valid.append(result)

        self.logger.info(f"  DSR filtered out : {n_dsr_reject}")
        self.logger.info(f"  Final valid alphas: {len(final_valid)}")
        self.logger.info("=" * 60 + "\n")

        # ── Build & save trial scorecard (SEMUA kombinasi yang diuji) ──
        _SCORECARD_COLS = [
            "mode", "horizon", "tp_pct", "features_str", "num_features", "feature_hash",
            "sharpe_oos", "sharpe_is", "n_trades_oos", "n_trades_is",
            "regime_mean", "regime_std", "max_regime_sharpe", "worst_regime_sharpe",
            "regime_stability_score", "alpha_concentration_index",
            "probabilistic_sharpe", "deflated_sharpe", "dsr_reliable",
            "composite_score", "alpha_type",
            "optimal_threshold",
            "gate_passed", "gate_status", "rejection_reason",
        ]
        trial_records = []
        for t in all_trials:
            row = {c: t.get(c, float("nan") if c not in ("gate_passed","gate_status","rejection_reason","alpha_type","features_str","feature_hash","mode") else None)
                   for c in _SCORECARD_COLS}
            trial_records.append(row)
        self.trial_scorecard = pd.DataFrame(trial_records)
        self.output_manager.save_trial_scorecard(self.trial_scorecard)
        self.logger.info(f"  Trial scorecard saved ({len(self.trial_scorecard)} rows)")

        if not final_valid:
            self.logger.info("No alpha survived DSR correction.")
            return pd.DataFrame()

        df_results = pd.DataFrame(final_valid)
        df_results = df_results.sort_values("composite_score", ascending=False).reset_index(drop=True)

        mode_descriptions = {
            "A": "Single w/o regime",
            "B": "Multi w/o regime",
            "C": "Single with regime",
            "D": "Multi with regime",
        }
        df_results["mode_description"] = df_results["mode"].map(mode_descriptions)

        if top_k:
            df_results = df_results.head(top_k)

        self.results = df_results.to_dict("records")

        self.logger.info("=" * 60)
        self.logger.info("Atomic Mining Grid Search completed.")
        self.logger.info(f"Total tasks executed : {total_tasks}")
        self.logger.info(f"Passed quality gate  : {len(raw_valid)}")
        self.logger.info(f"Survived DSR         : {len(df_results)}")
        self.logger.info("=" * 60 + "\n")

        return df_results

    # ─────────────────────────────────────────────
    # QUALITY GATE v4
    # Perbaikan:
    #   1. IS negatif → penalti, bukan lolos otomatis
    #   2. n_trades_oos minimum check
    #   3. PSR minimum check
    #   4. worst_regime_sharpe default diperketat ke -0.3
    # ─────────────────────────────────────────────
    @staticmethod
    def _quality_gate_static(
        sharpe_oos, sharpe_is, regime_std, max_regime_sharpe,
        regime_mean=None, regime_positive_count=None,
        regime_stability_score=None, alpha_concentration_index=None,
        worst_regime_sharpe=None,
        n_trades_oos=None, psr=None, n_trials=1,
        metric_filter=None,
        return_reason=False,
    ):
        if metric_filter is None:
            metric_filter = {}

        def _threshold(key, default):
            """
            Ambil threshold dari config.
            False di config = disabled → return None (check dilewati).
            """
            val = metric_filter.get(key, default)
            if val is False:
                return None
            return float(val) if val is not None else None

        SHARPE_MIN  = _threshold("sharpe_oos_min",    0.1)
        OVERFIT_MIN = _threshold("overfit_ratio_min", 0.1)
        PSR_MIN     = _threshold("psr_min",           0.3)

        def _ret(alpha_type, reason):
            return (alpha_type, reason) if return_reason else alpha_type

        # ── Sharpe OOS non-positif (selalu aktif) ────────────────
        if sharpe_oos <= 0:
            return _ret(None, "sharpe_oos<=0")

        # ── Sharpe OOS minimum ────────────────────────────────────
        if SHARPE_MIN is not None and sharpe_oos < SHARPE_MIN:
            return _ret(None, f"sharpe_oos<{SHARPE_MIN:.2f}")

        # ── Overfit ratio (OOS/IS) ────────────────────────────────
        if OVERFIT_MIN is not None and sharpe_is is not None:
            if sharpe_is > 0:
                if (sharpe_oos / sharpe_is) < OVERFIT_MIN:
                    return _ret(None, f"overfit_ratio<{OVERFIT_MIN:.2f}")
            elif sharpe_is < 0:
                return _ret(None, "sharpe_is_negative")

        # ── PSR ───────────────────────────────────────────────────
        if PSR_MIN is not None and psr is not None and psr < PSR_MIN:
            return _ret(None, f"psr<{PSR_MIN:.2f}")

        return _ret("global", "passed")

    # Backward-compat instance method
    def _quality_gate(self, *args, **kwargs):
        return EdgeMiner._quality_gate_static(*args, **kwargs)


# ─────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────

def _selective_copy(df_built):
    """
    Buat salinan df_built yang drop kolom _ENGINE_COLS jika ada.
    Lebih cepat dari df.copy() penuh.
    """
    cols_to_drop = [c for c in _ENGINE_COLS if c in df_built.columns]
    if cols_to_drop:
        return df_built.drop(columns=cols_to_drop).copy()
    return df_built.copy()