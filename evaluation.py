from logging import config
import pandas as pd
import numpy as np
from scipy import stats
from joblib import Parallel, delayed

class StrategyEvaluator:
    """
    Mengenkapsulasi logika untuk evaluasi strategi trading, termasuk
    walk-forward training (paralel) dan perhitungan metrik performa.
    """
    def __init__(self, config, model_class):
        self.config = config
        self.model_class = model_class
        self.logger = None

    # -----------------------
    # METODE UTILITY (STATIC)
    # -----------------------
    @staticmethod
    def sharpe_ratio(returns):
        returns = returns.dropna()
        if len(returns) == 0:
            return np.nan
        return returns.mean() / (returns.std() + 1e-9)

    @staticmethod
    def t_stat(returns):
        returns = returns.dropna()
        if len(returns) == 0:
            return np.nan
        mean = returns.mean()
        std  = returns.std()
        n    = len(returns)
        return mean / (std / np.sqrt(n) + 1e-9)

    @staticmethod
    def probabilistic_sharpe(returns, benchmark_sr=0):
        sr = StrategyEvaluator.sharpe_ratio(returns)
        n  = len(returns)
        if n < 2:
            return np.nan
        return stats.norm.cdf((sr - benchmark_sr) * np.sqrt(n - 1))

    @staticmethod
    def hit_rate(returns):
        returns = returns.dropna()
        if len(returns) == 0:
            return np.nan
        return np.mean(returns > 0) * 100

    @staticmethod
    def max_drawdown(returns):
        returns = returns.dropna().reset_index(drop=True)
        if len(returns) == 0:
            return np.nan
        equity   = (1 + returns).cumprod()
        peak     = equity.cummax()
        drawdown = (equity - peak) / peak
        return abs(drawdown.min()) * 100

    @staticmethod
    def expected_value(returns):
        returns = returns.dropna()
        if len(returns) == 0:
            return np.nan
        p_win    = np.mean(returns > 0)
        avg_win  = returns[returns >  0].mean() if p_win > 0 else 0
        avg_loss = returns[returns <= 0].mean() if p_win < 1 else 0
        return p_win * avg_win + (1 - p_win) * avg_loss

    @staticmethod
    def max_consecutive_loss(returns):
        returns   = returns.dropna().reset_index(drop=True)
        max_streak = streak = 0
        for r in returns:
            if r < 0:
                streak    += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        return max_streak

    @staticmethod
    def profit_factor(returns):
        returns      = returns.dropna()
        gross_profit = returns[returns > 0].sum()
        gross_loss   = abs(returns[returns < 0].sum())
        if gross_loss == 0:
            return np.inf
        return round(gross_profit / gross_loss, 3)

    def _compute_metrics(self, active_returns, sample_size, label):
        active_returns = active_returns.dropna()
        sharpe = self.sharpe_ratio(active_returns)
        dd     = self.max_drawdown(active_returns)
        calmar = round(sharpe / (dd / 100 + 1e-9), 3) if not np.isnan(dd) else np.nan
        return {
            "Label":         label,
            "Sample":        sample_size,
            "Trades":        len(active_returns),
            "Sharpe":        round(sharpe, 3),
            "Hitrate":       f"{self.hit_rate(active_returns):.1f}%",
            "Max_Streak":    self.max_consecutive_loss(active_returns),
            "Profit_Factor": self.profit_factor(active_returns),
            "Calmar":        calmar,
            "EV":            round(self.expected_value(active_returns), 6),
        }

    def performance_by_regime(self, df, regime_col="regime"):
        results = []
        for r in sorted(df[regime_col].dropna().unique()):
            sub = df[df[regime_col] == r].copy()
            if len(sub) < 10:
                continue
            # Hanya hitung bar OOS (is_unseen=True) yang punya posisi aktif
            if "is_unseen" in sub.columns and "position_unseen" in sub.columns:
                active = sub[(sub["is_unseen"] == True) & (sub["position_unseen"] != 0)]
            elif "position_unseen" in sub.columns:
                active = sub[sub["position_unseen"] != 0]
            else:
                active = sub
            active_returns = active["strategy_return_test"].dropna()
            if len(active_returns) == 0:
                continue
            row = self._compute_metrics(active_returns, len(sub), label=r)
            row["Regime"] = row.pop("Label")
            results.append(row)
        df_out = pd.DataFrame(results)
        if df_out.empty or "Sharpe" not in df_out.columns:
            return df_out
        return df_out.sort_values("Sharpe", ascending=False)

    def performance_by_direction(self, df):
        results = []
        if "position_unseen" not in df.columns:
            return pd.DataFrame()

        # Hanya pertimbangkan bar OOS untuk direction breakdown
        if "is_unseen" in df.columns:
            oos_df = df[df["is_unseen"] == True].copy()
        else:
            oos_df = df.copy()

        direction_map = {
            "long":    oos_df["position_unseen"] == 1,
            "short":   oos_df["position_unseen"] == -1,
            "neutral": oos_df["position_unseen"] == 0,
        }
        total_bars = len(oos_df)
        for dir_name, mask in direction_map.items():
            sub            = oos_df[mask].copy()
            active_returns = sub["strategy_return_test"].dropna()
            if dir_name == "neutral":
                results.append({
                    "Direction": dir_name, "Sample": total_bars,
                    "Trades": int(mask.sum()),
                    "Sharpe": np.nan, "Hitrate": np.nan,
                    "Max_Streak": np.nan, "Profit_Factor": np.nan,
                    "Calmar": np.nan, "EV": np.nan,
                })
                continue
            if len(active_returns) == 0:
                continue
            row = self._compute_metrics(active_returns, total_bars, label=dir_name)
            row["Direction"] = row.pop("Label")
            results.append(row)
        order  = {"long": 0, "short": 1, "neutral": 2}
        df_out = pd.DataFrame(results)
        if not df_out.empty:
            df_out = df_out.sort_values(
                "Direction",
                key=lambda s: s.map(order).fillna(99)
            ).reset_index(drop=True)
        return df_out

    # -----------------------
    # WALKFORWARD PARALEL
    # -----------------------
    def walk_forward_training(self, df, feature_cols):
        """
        Walk-forward training — PARALEL per window.

        Setiap window dijalankan secara independen di worker terpisah.
        Hasil dikumpulkan di main thread lalu di-assign ke df.
        GAP = max_hold untuk hindari label leakage.
        """
        df = df.copy()

        use_regime      = self.config["modeling"]["use_regime_conditioned"]
        regime_col      = self.config["data"]["regime_col"]
        use_allowed     = self.config["modeling"].get("use_allowed_regimes", False)
        allowed_regimes = self.config["modeling"].get("allowed_regimes", [])
        min_samples     = self.config["modeling"].get("min_samples_per_regime", 10)

        train_size = self.config["walkforward"]["train_size"]
        test_size  = self.config["walkforward"]["test_size"]
        step_size  = self.config["walkforward"]["step_size"]
        gap        = self.config.get("max_hold", 100)

        # Hitung semua window terlebih dahulu
        windows = []
        start = 0
        end   = train_size
        while True:
            test_start = end + gap
            test_end   = test_start + test_size
            if test_end > len(df):
                break
            windows.append((start, end, test_start, test_end))
            start += step_size
            end   += step_size

        n_jobs = self.config.get("walkforward", {}).get("n_jobs", -1)

        # ── DEBUG: tanggal data & window ────────────────────────────────────
        min_required = train_size + gap + test_size
        W = 68
        self.logger.info("=" * W)
        self.logger.info("  WALKFORWARD WINDOWS")
        self.logger.info(f"  Data   : {df.index[0].date()}  →  {df.index[-1].date()}  ({len(df)} bars)")
        self.logger.info(f"  Setup  : train={train_size}  gap={gap}  test={test_size}  step={step_size}")

        if not windows:
            self.logger.info(f"  [WARNING] Tidak ada window! Butuh min {min_required} bars, tersedia {len(df)}")
            self.logger.info("=" * W)
            return df, []

        self.logger.info(f"  Windows: {len(windows)}")
        self.logger.info("-" * W)
        self.logger.info(f"  {'#':<4}  {'TRAIN':^23}  {'GAP':^5}  {'OOS':^23}")
        self.logger.info(f"  {'─'*4}  {'─'*23}  {'─'*5}  {'─'*23}")

        sample_idxs = sorted(set([0, 1, len(windows) - 1]))
        for i in sample_idxs:
            s, e, ts, te = windows[i]
            self.logger.info(
                f"  {i+1:<4}"
                f"  {str(df.index[s].date())} → {str(df.index[e-1].date())}"
                f"  {ts-e:^5}"
                f"  {str(df.index[ts].date())} → {str(df.index[te-1].date())}"
            )
        if len(windows) > 3:
            self.logger.info(f"  {'...':<4}  {'(' + str(len(windows)-3) + ' windows lainnya)':^23}")

        self.logger.info("-" * W)
        first_oos_bar = windows[0][2]
        last_oos_bar  = windows[-1][3] - 1
        self.logger.info(f"  Blank  : {df.index[0].date()} → {df.index[first_oos_bar-1].date()}  ({first_oos_bar} bars, IS only)")
        self.logger.info(f"  OOS    : {df.index[first_oos_bar].date()} → {df.index[last_oos_bar].date()}  ({last_oos_bar - first_oos_bar + 1} bars)")
        self.logger.info("=" * W)
        # ────────────────────────────────────────────────────────────────────

        if not use_regime:
            self.logger.info(f"Walkforward global mode — {len(windows)} windows, parallel n_jobs={n_jobs}")
        else:
            self.logger.info(f"Walkforward regime-conditioned mode — {len(windows)} windows, parallel n_jobs={n_jobs}")
            if use_allowed:
                self.logger.info(f"Allowed regimes = {allowed_regimes}")

        # Worker function — tidak akses logger (thread-safe)
        def _run_window(win_idx, start, end, test_start, test_end):
            train_df = df.iloc[start:end]
            test_df  = df.iloc[test_start:test_end]
            logs     = []
            preds    = []   # list of (index_array, col_dict)

            if not use_regime:
                X_train = train_df[feature_cols]
                y_train = train_df["target"]
                X_test  = test_df[feature_cols]

                # Skip window jika target tidak punya cukup class
                n_classes = y_train.nunique()
                if n_classes < 3:
                    logs.append(
                        f"  [WF win={win_idx}] SKIP — target hanya {n_classes} class "
                        f"(perlu 3). Coba turunkan TP% atau naikkan horizon."
                    )
                    return preds, {"train_start": train_df.index[0], "train_end": train_df.index[-1],
                                   "gap": gap, "test_start": test_df.index[0],
                                   "test_end": test_df.index[-1], "mode": "global"}, logs

                from ml_models import InstitutionalModel as _Model
                model = _Model(self.config)
                model.fit(X_train, y_train)

                p_test  = model.predict_proba(X_test)
                p_train = model.predict_proba(X_train)

                preds.append(("oos", X_test.index,  p_test))
                preds.append(("is",  X_train.index, p_train))

            else:
                test_regimes = test_df[regime_col].unique()
                from ml_models import InstitutionalModel as _Model

                for regime in test_regimes:
                    if use_allowed and regime not in allowed_regimes:
                        continue
                    train_reg = train_df[train_df[regime_col] == regime]
                    if len(train_reg) < min_samples:
                        logs.append(
                            f"  [WF win={win_idx}] Regime {regime} skip — "
                            f"train samples {len(train_reg)} < {min_samples}"
                        )
                        continue
                    test_reg  = test_df[test_df[regime_col] == regime]
                    X_train_r = train_reg[feature_cols]
                    y_train_r = train_reg["target"]
                    X_test_r  = test_reg[feature_cols]

                    # Skip jika target tidak punya cukup class
                    if y_train_r.nunique() < 3:
                        logs.append(
                            f"  [WF win={win_idx}] Regime {regime} skip — "
                            f"target hanya {y_train_r.nunique()} class (perlu 3)."
                        )
                        continue

                    model_r = _Model(self.config)
                    model_r.fit(X_train_r, y_train_r)

                    p_test_r  = model_r.predict_proba(X_test_r)
                    p_train_r = model_r.predict_proba(X_train_r)

                    preds.append(("oos", X_test_r.index,  p_test_r))
                    preds.append(("is",  X_train_r.index, p_train_r))

            meta = {
                "train_start": train_df.index[0],
                "train_end":   train_df.index[-1],
                "gap":         gap,
                "test_start":  test_df.index[0],
                "test_end":    test_df.index[-1],
                "mode":        "regime" if use_regime else "global",
            }
            return preds, meta, logs

        # Initialize output columns
        for col in ["prob_short_oos", "prob_neutral_oos", "prob_long_oos",
                    "prob_short_is",  "prob_neutral_is",  "prob_long_is",
                    "score_oos", "score_is"]:
            df[col] = np.nan
        df["is_unseen"] = False

        # Jalankan semua window secara paralel
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_run_window)(i, s, e, ts, te)
            for i, (s, e, ts, te) in enumerate(windows)
        )

        # Kumpulkan hasil di main thread (sequential, tidak ada race condition)
        wf_meta = []
        for preds, meta, logs in results:
            # Tulis log dari worker
            for msg in logs:
                self.logger.info(msg)

            for split_type, idx, proba in preds:
                if split_type == "oos":
                    df.loc[idx, "prob_short_oos"]   = proba[:, 0]
                    df.loc[idx, "prob_neutral_oos"]  = proba[:, 1]
                    df.loc[idx, "prob_long_oos"]     = proba[:, 2]
                    df.loc[idx, "score_oos"]         = proba[:, 2]
                    df.loc[idx, "is_unseen"]         = True
                else:
                    df.loc[idx, "prob_short_is"]    = proba[:, 0]
                    df.loc[idx, "prob_neutral_is"]  = proba[:, 1]
                    df.loc[idx, "prob_long_is"]     = proba[:, 2]
                    df.loc[idx, "score_is"]         = proba[:, 2]

            wf_meta.append(meta)

        return df, wf_meta