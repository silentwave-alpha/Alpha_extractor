import os
import json
import logging
from datetime import datetime
import sys

# Maps derived feature name → raw source columns needed to reconstruct it
_FEATURE_SOURCE_MAP = {
    # ── Technical ──
    "log_ret":               ["close"],
    "ema_ratio_20_50":       ["close"],
    "ema_ratio_50_200":      ["close"],
    "range_rank_20":         ["high", "low", "close"],
    "body":                  ["open", "close"],
    "wick_up":               ["high", "open", "close"],
    "wick_dn":               ["low",  "open", "close"],
    # ── Basis ──
    "basis_spread":          ["close_basis", "open_basis"],
    "basis_intrabar_change": ["close_change", "open_change"],
    "basis_level":           ["close_basis"],
    # ── Flow ──
    "taker_vol_delta":       ["taker_buy_vol", "taker_sell_vol"],
    "taker_vol_ratio":       ["taker_buy_vol", "taker_sell_vol"],
    "taker_usd_delta":       ["taker_buy_volume_usd", "taker_sell_volume_usd"],
    "taker_usd_ratio":       ["taker_buy_volume_usd", "taker_sell_volume_usd"],
    "agg_flow_delta":        ["agg_taker_buy_vol", "agg_taker_sell_vol"],
    "cvd_level":             ["cvd"],
    "cvd_change":            ["cvd"],
    # ── Open Interest ──
    "oi_level":              ["open_interest"],
    "oi_change":             ["open_interest"],
    "oi_margin_ratio":       ["oi_stablecoin_margin", "oi_aggregated_history"],
    # ── Position Flow ──
    "net_position_flow":     ["net_long_change", "net_short_change"],
    "net_position_change":   ["net_position_change_cum"],
    "net_long_change_raw":   ["net_long_change"],
    "net_short_change_raw":  ["net_short_change"],
    # ── Liquidation ──
    "liq_delta":             ["long_liquidation_usd", "short_liquidation_usd"],
    "liq_ratio":             ["long_liquidation_usd", "short_liquidation_usd"],
    "liq_total":             ["long_liquidation_usd", "short_liquidation_usd"],
    # ── Sentiment ──
    "global_long_bias":      ["global_account_long_percent"],
    "global_ratio_level":    ["global_account_long_short_ratio"],
    "top_long_bias":         ["top_account_long_percent"],
    "top_ratio_level":       ["top_account_long_short_ratio"],
    "position_ratio_level":  ["top_position_long_short_ratio"],
    "whale_vs_crowd":        ["top_account_long_percent", "global_account_long_percent"],
    # ── Orderbook ──
    "orderbook_usd_delta":   ["bids_usd", "asks_usd"],
    "orderbook_qty_delta":   ["bids_quantity", "asks_quantity"],
    # ── Funding & Whale ──
    "funding_level":         ["funding_rate"],
    "funding_change":        ["funding_rate"],
    "whale_index_level":     ["whale_index_value"],
    "whale_index_change":    ["whale_index_value"],
}


class OutputManager:
    def __init__(self, save_artifacts=False, base_dir="experiments"):
        self.save_artifacts = save_artifacts
        self.base_dir = base_dir
        self.path = None
        if self.save_artifacts:
            self._create_folder()
        self.logger = self._setup_logger()

    def _create_folder(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.path = os.path.join(self.base_dir, timestamp)
        os.makedirs(self.path, exist_ok=True)

    def _setup_logger(self):
        logger = logging.getLogger("silentwave_" + str(id(self)))
        logger.setLevel(logging.INFO)
        logger.handlers = []
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(ch)
        if self.save_artifacts:
            log_path = os.path.join(self.path, "log.txt")
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(fh)
        return logger

    def save_results(self, results_df):
        if not self.save_artifacts:
            return
        results_df.to_csv(os.path.join(self.path, "results.csv"), index=False)
        self.logger.info("Results saved")

    def save_config(self, config_dict):
        if not self.save_artifacts:
            return
        with open(os.path.join(self.path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4)
        self.logger.info("Config saved")

    def save_config_yaml(self, config_dict):
        if not self.save_artifacts:
            return
        import yaml
        with open(os.path.join(self.path, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def log_feature_built(self, registry, feature_pool=None):
        if not self.save_artifacts:
            return
        log_path = os.path.join(self.path, "Feature_list.txt")
        if feature_pool is not None:
            with open(log_path, "w", encoding="utf-8") as f:
                for feat in feature_pool:
                    f.write(feat + "\n")
        elif hasattr(registry, "feature_stats_df"):
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(registry.feature_stats_df.to_string())

    def save_object(self, obj, filename):
        if not self.save_artifacts:
            return
        import pickle
        with open(os.path.join(self.path, filename), "wb") as f:
            pickle.dump(obj, f)
        self.logger.info(filename + " saved")

    def save_stats_df(self, stats_df, mode, horizon, feature_hash):
        if not self.save_artifacts:
            return
        filename = "stats_" + mode + "_h" + str(horizon) + "_" + feature_hash + ".csv"
        stats_df.to_csv(os.path.join(self.path, filename), index=False)

    def load_stats_df(self, mode, horizon, feature_hash):
        import pandas as pd
        filename = "stats_" + mode + "_h" + str(horizon) + "_" + feature_hash + ".csv"
        file_path = os.path.join(self.path, filename)
        return pd.read_csv(file_path) if os.path.exists(file_path) else None

    def save_direction_df(self, direction_df, mode, horizon, feature_hash):
        if not self.save_artifacts:
            return
        filename = "direction_" + mode + "_h" + str(horizon) + "_" + feature_hash + ".csv"
        direction_df.to_csv(os.path.join(self.path, filename), index=False)

    def load_direction_df(self, mode, horizon, feature_hash):
        import pandas as pd
        filename = "direction_" + mode + "_h" + str(horizon) + "_" + feature_hash + ".csv"
        file_path = os.path.join(self.path, filename)
        return pd.read_csv(file_path) if os.path.exists(file_path) else None

    def save_returns_df(self, returns_df, mode, horizon, feature_hash):
        """Simpan per-bar IS/OOS strategy returns untuk equity chart."""
        if not self.save_artifacts:
            return
        filename = "returns_" + mode + "_h" + str(horizon) + "_" + feature_hash + ".csv"
        returns_df.to_csv(os.path.join(self.path, filename))

    def load_returns_df(self, mode, horizon, feature_hash):
        import pandas as pd
        filename = "returns_" + mode + "_h" + str(horizon) + "_" + feature_hash + ".csv"
        file_path = os.path.join(self.path, filename)
        return pd.read_csv(file_path, index_col=0, parse_dates=True) if os.path.exists(file_path) else None

    def save_trial_scorecard(self, scorecard_df):
        """Simpan trial scorecard (semua kombinasi yang diuji) ke CSV."""
        if not self.save_artifacts:
            return
        scorecard_df.to_csv(os.path.join(self.path, "trial_scorecard.csv"), index=False)
        self.logger.info(f"Trial scorecard saved ({len(scorecard_df)} rows)")

    def load_trial_scorecard(self):
        import pandas as pd
        file_path = os.path.join(self.path, "trial_scorecard.csv")
        return pd.read_csv(file_path) if os.path.exists(file_path) else None

    def save_model(self, model, mode, horizon, feature_hash):
        """Simpan final model dengan joblib (kompatibel dengan joblib.load di QuantConnect)."""
        if not self.save_artifacts:
            return None
        import joblib
        filename  = f"model_{mode}_h{horizon}_{feature_hash}.pkl"
        file_path = os.path.join(self.path, filename)
        joblib.dump(model, file_path)
        self.logger.info(f"Model saved: {filename}")
        return file_path

    def load_model_bytes(self, mode, horizon, feature_hash):
        """Return bytes dari model pickle untuk st.download_button."""
        filename  = f"model_{mode}_h{horizon}_{feature_hash}.pkl"
        file_path = os.path.join(self.path, filename)
        if not os.path.exists(file_path):
            return None, filename
        with open(file_path, "rb") as f:
            return f.read(), filename

    def save_model_csv(self, df, features, mode, horizon, feature_hash):
        """Simpan CSV berisi OHLC + source columns yang dibutuhkan features + derived features.

        Kolom OHLC (open/high/low/close) selalu ikut.
        Untuk setiap feature yang dikenal di _FEATURE_SOURCE_MAP, source columns-nya ikut.
        Untuk feature yang tidak ada di map (mis. mom_5 dari feature_builder lain),
        hanya feature itu sendiri yang diikutkan jika ada di df.
        """
        if not self.save_artifacts:
            return None
        import pandas as pd

        base_cols    = ["open", "high", "low", "close"]
        source_cols  = set()
        for feat in features:
            for src in _FEATURE_SOURCE_MAP.get(feat, []):
                source_cols.add(src)

        # Susun kolom: OHLC → source → features (deduplicated, hanya yang ada di df)
        all_cols = []
        for c in base_cols + sorted(source_cols) + list(features):
            if c not in all_cols and c in df.columns:
                all_cols.append(c)

        filename  = f"data_{mode}_h{horizon}_{feature_hash}.csv"
        file_path = os.path.join(self.path, filename)
        df[all_cols].to_csv(file_path)
        self.logger.info(f"Model CSV saved: {filename} ({len(all_cols)} columns)")
        return file_path

    def load_model_csv_bytes(self, mode, horizon, feature_hash):
        """Return bytes dari data CSV untuk st.download_button."""
        filename  = f"data_{mode}_h{horizon}_{feature_hash}.csv"
        file_path = os.path.join(self.path, filename)
        if not os.path.exists(file_path):
            return None, filename
        with open(file_path, "rb") as f:
            return f.read(), filename

    def save_edge_metadata(self, metadata_dict, mode, horizon, feature_hash):
        """Simpan edge_metadata sebagai JSON."""
        if not self.save_artifacts:
            return None
        filename  = f"edge_metadata_{mode}_h{horizon}_{feature_hash}.json"
        file_path = os.path.join(self.path, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, indent=4)
        self.logger.info(f"Edge metadata saved: {filename}")
        return file_path

    def load_edge_metadata_bytes(self, mode, horizon, feature_hash):
        """Return bytes dari edge_metadata JSON untuk st.download_button."""
        filename  = f"edge_metadata_{mode}_h{horizon}_{feature_hash}.json"
        file_path = os.path.join(self.path, filename)
        if not os.path.exists(file_path):
            return None, filename
        with open(file_path, "rb") as f:
            return f.read(), filename

    def save_atomic_summary(self, results_df):
        """Simpan summary.txt untuk Atomic Edge Discovery."""
        if not self.save_artifacts:
            return

        summary_path = os.path.join(self.path, "summary.txt")
        SEP = "=" * 60
        lines = [
            "SILENTWAVE - ATOMIC EDGE DISCOVERY SUMMARY",
            "Generated  : " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Total alpha: " + str(len(results_df)),
            "",
        ]

        for rank, (_, row) in enumerate(results_df.iterrows(), start=1):
            tp_val    = row.get("tp_pct")
            tp_str    = (str(round(tp_val * 100, 1)) + "%") if tp_val is not None else "-"
            features  = row["features"]
            feat_str  = ", ".join(features) if isinstance(features, list) else str(features)
            mode_desc = row.get("mode_description", "")
            rss       = row.get("regime_stability_score", float("nan"))
            aci       = row.get("alpha_concentration_index", float("nan"))
            worst     = row.get("worst_regime_sharpe", float("nan"))
            psr       = row.get("probabilistic_sharpe", float("nan"))
            dsr       = row.get("deflated_sharpe", float("nan"))
            dsr_ok    = row.get("dsr_reliable", False)
            n_trades  = row.get("n_trades_oos", 0)

            lines.append(SEP)
            lines.append(
                "#" + str(rank)
                + "  Mode: " + str(row["mode"]) + " (" + mode_desc + ")"
                + "  Horizon: " + str(row["horizon"])
                + "  TP: " + tp_str
                + "  Alpha: " + str(row["alpha_type"])
            )
            lines.append("Features       : " + feat_str)
            lines.append("Composite Score: " + str(round(row["composite_score"], 4)))
            lines.append(
                "Sharpe OOS : " + str(round(row["sharpe_oos"], 4))
                + "   Sharpe IS  : " + str(round(row["sharpe_is"], 4))
                + "   N Trades   : " + str(n_trades)
            )
            lines.append(
                "PSR        : " + str(round(psr, 4))
                + "   DSR        : " + str(round(dsr, 4))
                + " " + ("✓" if dsr_ok else "⚠")
            )
            lines.append(
                "Regime Mean: " + str(round(row["regime_mean"], 4))
                + "   Regime Std : " + str(round(row["regime_std"], 4))
                + "   Worst      : " + str(round(worst, 4))
            )
            lines.append(
                "RSS        : " + str(round(rss, 4))
                + "   ACI        : " + str(round(aci, 4))
            )
            lines.append("")
            lines.append("Per Regime:")
            fhash    = row["feature_hash"]
            stats_df = self.load_stats_df(row["mode"], row["horizon"], fhash)
            if stats_df is not None and not stats_df.empty:
                lines.append(stats_df.to_string(index=False))
            else:
                lines.append("  (tidak tersedia)")
            lines.append("")
            lines.append("Per Direction:")
            dir_df = self.load_direction_df(row["mode"], row["horizon"], fhash)
            if dir_df is not None and not dir_df.empty:
                lines.append(dir_df.to_string(index=False))
            else:
                lines.append("  (tidak tersedia)")
            lines.append("")

        lines.append(SEP)
        lines.append("END OF SUMMARY")

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        self.logger.info("Summary saved to: " + summary_path)


class SilentOutputManager:
    """Output manager no-op untuk worker thread — agar log tidak acak-aduk."""
    def __init__(self):
        self.save_artifacts = False
        self.path           = None
        self.logger         = self._setup_silent_logger()

    def _setup_silent_logger(self):
        logger = logging.getLogger("silent_" + str(id(self)))
        logger.handlers = []
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        return logger

    def save_results(self, *a, **kw):           pass
    def save_config(self, *a, **kw):            pass
    def save_config_yaml(self, *a, **kw):       pass
    def log_feature_built(self, *a, **kw):      pass
    def save_object(self, *a, **kw):            pass
    def save_stats_df(self, *a, **kw):          pass
    def save_direction_df(self, *a, **kw):      pass
    def save_returns_df(self, *a, **kw):        pass
    def save_atomic_summary(self, *a, **kw):    pass
    def save_trial_scorecard(self, *a, **kw):   pass
    def load_trial_scorecard(self, *a, **kw):   return None
    def save_model(self, *a, **kw):             return None
    def load_model_bytes(self, *a, **kw):       return None, ""
    def save_model_csv(self, *a, **kw):         return None
    def load_model_csv_bytes(self, *a, **kw):   return None, ""
    def save_edge_metadata(self, *a, **kw):     return None
    def load_edge_metadata_bytes(self, *a, **kw): return None, ""
    def load_stats_df(self, *a, **kw):          return None
    def load_direction_df(self, *a, **kw):      return None
    def load_returns_df(self, *a, **kw):        return None
