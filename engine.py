import numpy as np

from data_loader import DataLoader
from governance.feature_registry import FeatureRegistry
from governance.leakage_guard import check_future_leakage
from ml_models import InstitutionalModel
from evaluation import StrategyEvaluator
from feature_builder.base_builder import BaseBuilder
from feature_builder.transform_builder import TransformBuilder
from feature_builder.interaction_builder import InteractionBuilder
from governance.manager import OutputManager

# Kolom yang dimodifikasi engine — dipakai untuk selective copy di mining
_ENGINE_COLS = [
    "label", "target", "future_return",
    "prob_short_is", "prob_neutral_is", "prob_long_is", "score_is",
    "prob_short_oos", "prob_neutral_oos", "prob_long_oos", "score_oos",
    "is_unseen",
    "position_seen", "position_unseen",
    "strategy_return_train", "strategy_return_test", "strategy_return",
]


class EdgeEngine:
    def __init__(self, config, output_manager):
        self.config = config
        self.registry = None
        self.model = None
        self.df = None

        self.output_manager = output_manager
        self.logger = self.output_manager.logger

        self.mode = self.config.get("mode", "walkforward")
        self.direction = self.config.get("direction", "long")
        self.run_type = config["logging"]["run_type"]

        self.evaluator = StrategyEvaluator(config, InstitutionalModel)
        self.evaluator.logger = self.logger

    def load_data(self):
        loader = DataLoader(self.config)
        self.df = loader.load()

    def build_features(self, log_registry=False):
        self.registry = FeatureRegistry()
        self.registry.logger = self.logger

        # 1 Base Features
        base_builder = BaseBuilder(self.config, self.registry)
        self.df = base_builder.build(self.df)

        # 2 Transform Layer (optional)
        if self.config.get("feature_builder", {}).get("enable_transform", True):
            transformer = TransformBuilder(self.config, self.registry)
            self.df = transformer.build(self.df)

        # 3 Interaction Layer (optional)
        if self.config.get("feature_builder", {}).get("enable_interaction", False):
            interactor = InteractionBuilder(self.config, self.registry)
            dynamic_mode = self.config.get("feature_builder", {}).get("dynamic_interactions", False)
            self.df = interactor.build(self.df, dynamic_mode=dynamic_mode)

        self.registry.calculate_stats(self.df)

        if log_registry:
            self.registry.log(self.logger)

        return self.df

    def override_feature_list(self, feature_list):
        self.registry.set_feature_list(feature_list)

    def check_feature(self):
        feature_list = self.registry.get_feature_list()
        if not feature_list:
            raise ValueError("No features registered.")

        missing = [f for f in feature_list if f not in self.df.columns]
        if missing:
            raise ValueError(f"Missing features in df: {missing}")

        for f in feature_list:
            if self.df[f].isna().all():
                raise ValueError(f"Feature {f} is all NaN.")
            if self.df[f].nunique() <= 1:
                raise ValueError(f"Feature {f} is constant.")

    def build_target(self):
        """
        Triple Barrier Labeling — VECTORIZED (NumPy).

          - TP_PCT  : take-profit threshold (symmetric long/short)
          - MAX_HOLD: max bars sebelum label neutral (0)
          - label   : -1 = short wins, 0 = neutral, 1 = long wins
          - target  : encoded untuk XGBoost: 0=short, 1=neutral, 2=long

          future_return = realized P&L dari actual exit:
            - Hit TP_LONG  -> return = +tp_pct
            - Hit TP_SHORT -> return = -tp_pct
            - Expired      -> return = (close[i+MAX_HOLD] - close[i]) / close[i]
        """
        tp_pct   = self.config.get("tp_pct",  0.02)
        max_hold = self.config.get("max_hold", 100)

        close = self.df["close"].values.astype(np.float64)
        high  = self.df["high"].values.astype(np.float64)
        low   = self.df["low"].values.astype(np.float64)
        n     = len(close)
        valid = n - max_hold

        # Sliding window matrices: [i, j] = bar i+j+1
        row_idx     = np.arange(valid)[:, None]
        col_idx     = np.arange(1, max_hold + 1)[None, :]
        high_window = high[row_idx + col_idx]   # (valid, max_hold) — long TP detection
        low_window  = low[row_idx + col_idx]    # (valid, max_hold) — short TP detection

        entry    = close[:valid, None]
        tp_long  = entry * (1 + tp_pct)
        tp_short = entry * (1 - tp_pct)

        # high untuk long TP (intrabar touch), low untuk short TP
        hit_long  = high_window >= tp_long
        hit_short = low_window  <= tp_short

        any_long  = hit_long.any(axis=1)
        any_short = hit_short.any(axis=1)

        first_long  = np.where(any_long,  hit_long.argmax(axis=1),  max_hold)
        first_short = np.where(any_short, hit_short.argmax(axis=1), max_hold)

        # Label assignment
        labels = np.zeros(valid, dtype=np.float64)
        labels[any_long  & ~any_short]                                       =  1
        labels[any_short & ~any_long]                                        = -1
        labels[any_long  &  any_short & (first_long < first_short)]          =  1
        labels[any_long  &  any_short & (first_long > first_short)]          = -1
        # tie -> neutral (0)

        # Exit bar offset untuk dynamic cooldown (bars dari entry ke exit: 1..max_hold)
        i_range      = np.arange(valid)
        hold_offsets = np.where(
            labels ==  1, (first_long  + 1).astype(np.float64),
            np.where(
                labels == -1, (first_short + 1).astype(np.float64),
                float(max_hold)   # expired
            )
        )

        # future_return: ±tp_pct untuk TP hit, actual price change untuk expired
        expired_ret    = (close[i_range + max_hold] - close[:valid]) / close[:valid]
        future_returns = np.where(
            labels ==  1,  tp_pct,
            np.where(
                labels == -1, -tp_pct,
                expired_ret
            )
        )

        # Append NaN tail untuk bar yang tidak punya label valid
        nan_tail = np.full(max_hold, np.nan)
        self.df["label"]           = np.concatenate([labels,         nan_tail])
        self.df["future_return"]   = np.concatenate([future_returns, nan_tail])
        self.df["exit_bar_offset"] = np.concatenate([hold_offsets,   nan_tail])

        self.df = self.df.dropna(subset=["label"]).copy()

        from ml_models import LABEL_MAP
        self.df.loc[:, "target"] = self.df["label"].map(LABEL_MAP).astype(int)

        self.logger.info(f"Triple Barrier labeling done (vectorized). TP={tp_pct:.1%}, MAX_HOLD={max_hold}")
        dist = self.df["label"].value_counts(normalize=True).sort_index()
        self.logger.info(f"Label dist: short={dist.get(-1,0):.2%}  neutral={dist.get(0,0):.2%}  long={dist.get(1,0):.2%}")

    def train_simple(self):
        feature_cols = self.registry.get_feature_list()
        check_future_leakage(self.df, feature_cols, target_col="target", logger=self.logger)
        self.logger.info("Simple mode running.")

        split_ratio = self.config.get("split_ratio")
        split_index = int(len(self.df) * split_ratio)

        train_df = self.df.iloc[:split_index]
        test_df  = self.df.iloc[split_index:]

        X_train = train_df[feature_cols]
        y_train = train_df["target"]
        X_test  = test_df[feature_cols]

        self.model = InstitutionalModel(self.config)
        self.model.fit(X_train, y_train)

        proba_train = self.model.predict_proba(X_train)
        self.df.loc[train_df.index, "prob_short_is"]   = proba_train[:, 0]
        self.df.loc[train_df.index, "prob_neutral_is"] = proba_train[:, 1]
        self.df.loc[train_df.index, "prob_long_is"]    = proba_train[:, 2]
        self.df.loc[train_df.index, "score_is"]        = proba_train[:, 2]

        proba_test = self.model.predict_proba(X_test)
        self.df.loc[test_df.index, "prob_short_oos"]   = proba_test[:, 0]
        self.df.loc[test_df.index, "prob_neutral_oos"] = proba_test[:, 1]
        self.df.loc[test_df.index, "prob_long_oos"]    = proba_test[:, 2]
        self.df.loc[test_df.index, "score_oos"]        = proba_test[:, 2]
        self.df.loc[test_df.index, "is_unseen"]        = True

    def train_walkforward(self):
        feature_cols = self.registry.get_feature_list()
        check_future_leakage(self.df, feature_cols, target_col="target", logger=self.logger)
        self.df, wf_meta = self.evaluator.walk_forward_training(self.df, feature_cols)
        # Simpan wf_meta sebagai attribute — dibutuhkan generate_strategy_returns
        # untuk cooldown per fold (reset cooldown di awal setiap OOS window)
        self.wf_meta = wf_meta

    def train_final(self):
        """
        Train FINAL model pada seluruh data (tanpa split).
        Digunakan untuk export model setelah edge ditemukan.

        Mode global (A/B): satu model dari semua data.
        Mode regime (C/D): model terpisah per regime + fallback global.
        Hasilnya disimpan di self.model (global) dan self.regime_models (dict).
        """
        feature_cols = self.registry.get_feature_list()
        check_future_leakage(self.df, feature_cols, target_col="target", logger=self.logger)

        use_regime  = self.config["modeling"].get("use_regime_conditioned", False)
        regime_col  = self.config["data"]["regime_col"]
        min_samples = self.config["modeling"].get("min_samples_per_regime", 50)

        # ── Global model (selalu dilatih sebagai fallback) ──────────────────
        X_all = self.df[feature_cols]
        y_all = self.df["target"]
        n_classes = y_all.nunique()
        if n_classes < 3:
            raise ValueError(
                f"Target hanya memiliki {n_classes} class. "
                "Turunkan TP% atau naikkan horizon agar semua class muncul."
            )
        self.model = InstitutionalModel(self.config)
        self.model.fit(X_all, y_all)
        self.logger.info(
            f"Final global model trained on {len(X_all)} bars, "
            f"{len(feature_cols)} features: {feature_cols}"
        )

        # ── Per-regime models (hanya untuk Mode C/D) ───────────────────────
        self.regime_models = {}
        if use_regime and regime_col in self.df.columns:
            for regime in sorted(self.df[regime_col].dropna().unique()):
                reg_df = self.df[self.df[regime_col] == regime]
                X_r    = reg_df[feature_cols]
                y_r    = reg_df["target"]
                if len(X_r) < min_samples or y_r.nunique() < 3:
                    self.logger.info(
                        f"Final model regime {regime}: skip "
                        f"({len(X_r)} samples, {y_r.nunique()} classes)"
                    )
                    continue
                m = InstitutionalModel(self.config)
                m.fit(X_r, y_r)
                self.regime_models[int(regime)] = m
                self.logger.info(
                    f"Final model regime {int(regime)} trained on {len(X_r)} bars"
                )
            self.logger.info(f"Per-regime models trained: {list(self.regime_models.keys())}")

    def train_last_wf(self):
        """
        Train model pada window walk-forward TERAKHIR.
        Identik dengan fold terakhir yang menghasilkan OOS predictions,
        sehingga distribusi probabilitasnya sesuai dengan yang dilihat saat mining.

        Mode global (A/B): satu model dari train window terakhir.
        Mode regime (C/D): model terpisah per regime + fallback global.
        Hasilnya disimpan di self.model dan self.regime_models.
        """
        feature_cols = self.registry.get_feature_list()
        check_future_leakage(self.df, feature_cols, target_col="target", logger=self.logger)

        train_size  = self.config["walkforward"]["train_size"]
        test_size   = self.config["walkforward"]["test_size"]
        step_size   = self.config["walkforward"]["step_size"]
        gap         = self.config.get("max_hold", 100)
        use_regime  = self.config["modeling"].get("use_regime_conditioned", False)
        regime_col  = self.config["data"]["regime_col"]
        min_samples = self.config["modeling"].get("min_samples_per_regime", 50)

        # Hitung semua window — identik dengan evaluator.walk_forward_training
        windows = []
        start = 0
        end   = train_size
        while True:
            test_start = end + gap
            test_end   = test_start + test_size
            if test_end > len(self.df):
                break
            windows.append((start, end, test_start, test_end))
            start += step_size
            end   += step_size

        if not windows:
            raise ValueError(
                f"Tidak ada window WF yang valid. Data: {len(self.df)} bars, "
                f"butuh minimal train({train_size})+gap({gap})+test({test_size})={train_size+gap+test_size}."
            )

        last_start, last_end, _, _ = windows[-1]
        train_df = self.df.iloc[last_start:last_end]
        self.logger.info(
            f"Last WF window: train rows [{last_start}:{last_end}] "
            f"({len(train_df)} bars) — window {len(windows)}/{len(windows)}"
        )
        self.logger.info(
            f"  Train period : {train_df.index[0]} → {train_df.index[-1]}"
        )

        # ── Global model (selalu dilatih sebagai fallback) ──────────────────
        X_all = train_df[feature_cols]
        y_all = train_df["target"]
        n_classes = y_all.nunique()
        if n_classes < 3:
            raise ValueError(
                f"Last WF train window hanya {n_classes} class. "
                "Coba turunkan TP% atau naikkan horizon."
            )
        self.model = InstitutionalModel(self.config)
        self.model.fit(X_all, y_all)
        self.logger.info(
            f"Last WF global model trained on {len(X_all)} bars, "
            f"{len(feature_cols)} features: {feature_cols}"
        )

        # ── Per-regime models (hanya untuk Mode C/D) ───────────────────────
        self.regime_models = {}
        if use_regime and regime_col in train_df.columns:
            for regime in sorted(train_df[regime_col].dropna().unique()):
                reg_df = train_df[train_df[regime_col] == regime]
                X_r    = reg_df[feature_cols]
                y_r    = reg_df["target"]
                if len(X_r) < min_samples or y_r.nunique() < 3:
                    self.logger.info(
                        f"Last WF model regime {regime}: skip "
                        f"({len(X_r)} samples, {y_r.nunique()} classes)"
                    )
                    continue
                m = InstitutionalModel(self.config)
                m.fit(X_r, y_r)
                self.regime_models[int(regime)] = m
                self.logger.info(
                    f"Last WF model regime {int(regime)} trained on {len(X_r)} bars"
                )
            self.logger.info(f"Per-regime models trained: {list(self.regime_models.keys())}")

    def generate_strategy_returns(self):
        """
        Bangun strategy_return dari posisi dan future_return.

        FIX v4.1:
          1. Precision check pakai self.df["label"] (-1/0/+1), bukan "target" (0/1/2)
             — sebelumnya bug: target==1 berarti neutral, bukan long
          2. Cooldown per fold — reset cooldown_end di awal setiap OOS/IS window
             — sebelumnya: cooldown jalan di full df, fold N bisa blokir fold N+1
          3. IS cooldown juga per fold (masing-masing IS window independen)

        Threshold: dicari optimal di seluruh IS bars (precision-based), apply ke OOS.
        Cooldown : dynamic per actual exit (exit_bar_offset), reset per fold.
        """
        default_threshold = self.config.get("prob_threshold", 0.55)
        max_hold          = self.config.get("max_hold", 100)
        optimize          = self.config.get("optimize_threshold", True)
        thr_grid          = self.config.get("threshold_grid", [0.50, 0.52, 0.55, 0.58, 0.60, 0.63, 0.65, 0.70])
        min_trades_thr    = self.config.get("threshold_min_trades", 10)
        precision_target  = self.config.get("threshold_precision_target", 0.55)

        if "prob_long_oos" not in self.df.columns:
            raise ValueError("prob_long_oos tidak ditemukan — jalankan training dulu.")
        if "prob_long_is" not in self.df.columns:
            raise ValueError("prob_long_is tidak ditemukan — jalankan training dulu.")

        # ── Helper: build position untuk satu window (cooldown fresh per call) ──
        def _build_position_window(p_long, p_short, exit_offs, thr):
            """
            Bangun posisi untuk satu window (OOS fold atau IS fold).
            Cooldown selalu mulai fresh dari 0 — tidak ada kontaminasi antar fold.

            Args:
                p_long     : array prob_long window ini (sudah fillna 0)
                p_short    : array prob_short window ini (sudah fillna 0)
                exit_offs  : array exit_bar_offset window ini
                thr        : threshold

            Returns:
                pos: array posisi (1=long, -1=short, 0=neutral)
            """
            n            = len(p_long)
            pos          = np.zeros(n, dtype=np.float64)
            cooldown_end = 0   # ← fresh untuk setiap window

            for i in range(n):
                if i < cooldown_end:
                    continue
                if p_long[i] > thr:
                    pos[i]       = 1.0
                    cooldown_end = i + max(1, int(exit_offs[i]))
                elif p_short[i] > thr:
                    pos[i]       = -1.0
                    cooldown_end = i + max(1, int(exit_offs[i]))
            return pos

        # ── Ambil wf_meta untuk tahu batas tiap fold ─────────────────────────
        # wf_meta disimpan oleh train_walkforward() sebagai self.wf_meta
        # Untuk simple mode (bukan walkforward), wf_meta kosong → treated as 1 fold
        wf_meta = getattr(self, "wf_meta", [])

        # Bangun fold index maps dari wf_meta
        # oos_fold_indices : list of integer index arrays, satu per OOS fold
        # is_fold_indices  : list of integer index arrays, satu per IS fold
        df_idx = self.df.index  # DatetimeIndex

        if wf_meta:
            oos_fold_indices = []
            is_fold_indices  = []
            for meta in wf_meta:
                # OOS fold: bar antara test_start dan test_end yang is_unseen=True
                oos_mask_fold = (
                    (df_idx >= meta["test_start"]) &
                    (df_idx <= meta["test_end"]) &
                    self.df["is_unseen"]
                )
                oos_iloc = np.where(oos_mask_fold.values)[0]
                if len(oos_iloc) > 0:
                    oos_fold_indices.append(oos_iloc)

                # IS fold: bar antara train_start dan train_end yang NOT is_unseen
                is_mask_fold = (
                    (df_idx >= meta["train_start"]) &
                    (df_idx <= meta["train_end"]) &
                    self.df["prob_long_is"].notna() &
                    ~self.df["is_unseen"]
                )
                is_iloc = np.where(is_mask_fold.values)[0]
                if len(is_iloc) > 0:
                    is_fold_indices.append(is_iloc)
        else:
            # Simple mode atau wf_meta tidak tersedia — treat sebagai 1 fold penuh
            oos_all = np.where(self.df["is_unseen"].values)[0]
            is_all  = np.where(
                self.df["prob_long_is"].notna().values & ~self.df["is_unseen"].values
            )[0]
            oos_fold_indices = [oos_all] if len(oos_all) > 0 else []
            is_fold_indices  = [is_all]  if len(is_all)  > 0 else []

        # ── Cari threshold optimal di IS — trade-level precision dengan cooldown ──
        #
        # v4.2: Trade-level precision (bukan bar-level)
        #   Bar-level: hitung precision semua bar IS dimana prob > thr
        #              → overestimate karena bar berturutan (cluster) masuk semua
        #              → bar yang diblokir cooldown ikut dihitung padahal tidak jadi trade
        #
        #   Trade-level: simulasi positioning dengan cooldown di tiap IS fold dulu
        #                → hanya bar yang benar-benar jadi entry yang dihitung
        #                → identik dengan cara OOS positioning nanti
        #
        # Weighted precision per direction (bukan simple average):
        #   direction=long  → hanya long_prec yang dihitung
        #   direction=short → hanya short_prec yang dihitung
        #   direction=both  → weighted: (long_prec×n_long + short_prec×n_short) / n_total
        #
        # Pemilihan: threshold dengan weighted_precision TERTINGGI yang >= precision_target
        #            Kalau tidak ada yang memenuhi → fallback ke precision tertinggi
        # ──────────────────────────────────────────────────────────────────────────────
        direction = self.config.get("direction", "long")

        best_threshold = default_threshold
        if optimize:
            labels_arr     = self.df["label"].values          # -1/0/+1
            exit_offs_thr  = self.df["exit_bar_offset"].fillna(max_hold).values
            p_long_is_arr  = self.df["prob_long_is"].fillna(0).values
            p_short_is_arr = self.df["prob_short_is"].fillna(0).values

            # ── Per threshold: simulasi di semua IS fold, kumpulkan trade aktual ──
            thr_scores = []   # list of (thr, weighted_prec, n_trades)

            for thr in thr_grid:
                # Kumpulkan semua trade yang terjadi di seluruh IS fold
                # dengan cooldown fresh per fold — sama persis dengan cara OOS nanti
                all_predicted = []   # posisi trade: +1 atau -1
                all_actual    = []   # label asli bar tersebut: -1/0/+1

                for fold_iloc in is_fold_indices:
                    pos_sim = _build_position_window(
                        p_long_is_arr[fold_iloc],
                        p_short_is_arr[fold_iloc],
                        exit_offs_thr[fold_iloc],
                        thr,
                    )
                    trade_mask = pos_sim != 0
                    if trade_mask.sum() == 0:
                        continue
                    all_predicted.extend(pos_sim[trade_mask].tolist())
                    all_actual.extend(labels_arr[fold_iloc][trade_mask].tolist())

                if len(all_predicted) < min_trades_thr:
                    continue   # tidak cukup trade untuk threshold ini

                pred_arr   = np.array(all_predicted)
                actual_arr = np.array(all_actual)

                long_mask_t  = pred_arr ==  1
                short_mask_t = pred_arr == -1
                n_long  = long_mask_t.sum()
                n_short = short_mask_t.sum()
                n_total = n_long + n_short

                # Precision per direction: berapa % trade yang benar (TP hit)
                long_prec  = float((actual_arr[long_mask_t]  ==  1).mean()) if n_long  > 0 else np.nan
                short_prec = float((actual_arr[short_mask_t] == -1).mean()) if n_short > 0 else np.nan

                # Weighted precision sesuai direction config
                if direction == "long":
                    if np.isnan(long_prec) or n_long < min_trades_thr:
                        continue
                    weighted_prec = long_prec
                elif direction == "short":
                    if np.isnan(short_prec) or n_short < min_trades_thr:
                        continue
                    weighted_prec = short_prec
                else:  # both
                    parts = []
                    if not np.isnan(long_prec)  and n_long  > 0: parts.append((long_prec,  n_long))
                    if not np.isnan(short_prec) and n_short > 0: parts.append((short_prec, n_short))
                    if not parts:
                        continue
                    total_w = sum(n for _, n in parts)
                    weighted_prec = sum(p * n for p, n in parts) / total_w

                thr_scores.append({
                    "thr":            thr,
                    "weighted_prec":  weighted_prec,
                    "long_prec":      long_prec,
                    "short_prec":     short_prec,
                    "n_long":         int(n_long),
                    "n_short":        int(n_short),
                    "n_trades":       int(n_total),
                })

            if not thr_scores:
                # Tidak ada threshold yang punya cukup trade — pakai default
                self.logger.info(
                    f"  Threshold search: tidak ada threshold valid "
                    f"(min_trades={min_trades_thr}), pakai default={best_threshold:.2f}"
                )
            else:
                # Kandidat: threshold yang memenuhi precision_target
                # - direction=long/short : weighted_prec >= precision_target
                # - direction=both       : SALAH SATU long_prec ATAU short_prec >= precision_target
                #   (agar alpha long yg kuat tidak tenggelam oleh short yg jelek)
                def _is_candidate(s):
                    if direction == "both":
                        long_ok  = (not np.isnan(s["long_prec"]))  and s["long_prec"]  >= precision_target
                        short_ok = (not np.isnan(s["short_prec"])) and s["short_prec"] >= precision_target
                        return long_ok or short_ok
                    return s["weighted_prec"] >= precision_target

                candidates = [s for s in thr_scores if _is_candidate(s)]

                if candidates:
                    # Pilih yang weighted_prec TERTINGGI di antara kandidat
                    best = max(candidates, key=lambda s: s["weighted_prec"])
                else:
                    # Fallback: tidak ada yang memenuhi target → pilih precision tertinggi
                    best = max(thr_scores, key=lambda s: s["weighted_prec"])
                    self.logger.info(
                        f"  Threshold search: tidak ada yang memenuhi target={precision_target:.2f}, "
                        f"fallback ke precision tertinggi"
                    )

                best_threshold = best["thr"]

                # Log semua kandidat untuk audit
                self.logger.info(f"  Threshold search (trade-level, direction={direction}):")
                for s in thr_scores:
                    marker = " ← DIPILIH" if s["thr"] == best_threshold else ""
                    long_prec_str  = 'n/a' if np.isnan(s['long_prec'])  else f"{s['long_prec']:.3f}"
                    short_prec_str = 'n/a' if np.isnan(s['short_prec']) else f"{s['short_prec']:.3f}"
                    self.logger.info(
                        f"    thr={s['thr']:.2f}  "
                        f"weighted_prec={s['weighted_prec']:.3f}  "
                        f"long_prec={long_prec_str}  "
                        f"short_prec={short_prec_str}  "
                        f"n_trades={s['n_trades']}{marker}"
                    )
        else:
            self.logger.info(f"  Threshold fixed: {best_threshold:.2f}")

        self.optimal_threshold = best_threshold

        # ── FIX 2: Build OOS positions — cooldown fresh per fold ─────────────
        # BUG LAMA: _build_position jalan di full df → cooldown fold N bocor ke fold N+1
        # FIX: per fold, cooldown selalu mulai dari 0
        exit_offs_arr   = self.df["exit_bar_offset"].fillna(max_hold).values
        p_long_oos_arr  = self.df["prob_long_oos"].fillna(0).values
        p_short_oos_arr = self.df["prob_short_oos"].fillna(0).values
        p_long_is_arr   = self.df["prob_long_is"].fillna(0).values
        p_short_is_arr  = self.df["prob_short_is"].fillna(0).values

        pos_oos = np.zeros(len(self.df), dtype=np.float64)
        for fold_iloc in oos_fold_indices:
            fold_pos = _build_position_window(
                p_long_oos_arr[fold_iloc],
                p_short_oos_arr[fold_iloc],
                exit_offs_arr[fold_iloc],
                best_threshold,
            )
            pos_oos[fold_iloc] = fold_pos

        self.df["position_unseen"]      = pos_oos
        self.df["strategy_return_test"] = pos_oos * self.df["future_return"]

        # ── FIX 3: Build IS positions — cooldown fresh per fold ──────────────
        pos_is = np.zeros(len(self.df), dtype=np.float64)
        for fold_iloc in is_fold_indices:
            fold_pos = _build_position_window(
                p_long_is_arr[fold_iloc],
                p_short_is_arr[fold_iloc],
                exit_offs_arr[fold_iloc],
                best_threshold,
            )
            pos_is[fold_iloc] = fold_pos

        self.df["position_seen"]         = pos_is
        self.df["strategy_return_train"] = pos_is * self.df["future_return"]

        # ── strategy_return gabungan: OOS prioritas, fallback IS ─────────────
        self.df["strategy_return"] = (
            self.df["strategy_return_test"]
            .combine_first(self.df["strategy_return_train"])
        )

        n_oos_trades = int((pos_oos != 0).sum())
        n_is_trades  = int((pos_is  != 0).sum())
        self.logger.info(
            f"  Positions built: OOS={n_oos_trades} trades "
            f"across {len(oos_fold_indices)} folds | "
            f"IS={n_is_trades} trades across {len(is_fold_indices)} folds"
        )

    def evaluate(self):
        if "strategy_return" not in self.df.columns:
            raise ValueError("strategy_return belum dibuat.")

        # OOS: bar yang masuk test window, posisi aktif (bukan 0)
        oos_mask   = self.df["position_unseen"] != 0
        oos_active = self.df.loc[oos_mask, "strategy_return_test"].dropna()

        # IS: bar yang masuk train window, posisi aktif, DAN bukan bar OOS
        #     Penting: exclude overlap agar IS tidak tercampur signal OOS
        is_mask    = (self.df["position_seen"] != 0) & ~self.df.get("is_unseen", False)
        is_active  = self.df.loc[is_mask, "strategy_return_train"].dropna()

        mask_active = oos_mask | (self.df["position_seen"] != 0)
        both_active = self.df.loc[mask_active, "strategy_return"].dropna()

        self.sharpe_oos   = self.evaluator.sharpe_ratio(oos_active)
        self.sharpe_is    = self.evaluator.sharpe_ratio(is_active)
        self.sharpe_total = self.evaluator.sharpe_ratio(both_active)

        # Expose sebagai attribute — dipakai quality gate & composite score
        self.n_oos = len(oos_active)
        self.n_is  = len(is_active)

        self.logger.info("────────────")
        self.logger.info("Evaluation result (active trades only):")
        self.logger.info(f"Sharpe OOS   : {self.sharpe_oos:.4f}  (n={self.n_oos})")
        self.logger.info(f"Sharpe IS    : {self.sharpe_is:.4f}  (n={self.n_is})")
        self.logger.info(f"Sharpe Total : {self.sharpe_total:.4f}")

        regime_col = self.config["data"]["regime_col"]
        self.stats_df = self.evaluator.performance_by_regime(self.df, regime_col=regime_col)

        self.logger.info("\nBreakdown per Regime:")
        self.logger.info(self.stats_df.to_string(index=False))

        self.direction_df = self.evaluator.performance_by_direction(self.df)

        self.logger.info("\nBreakdown per Direction:")
        self.logger.info(self.direction_df.to_string(index=False))
        self.logger.info("────────────")

    def run(self, feature_override=None, use_prebuilt_data=False):
        """
        Menjalankan pipeline engine.

        Args:
            feature_override  : list fitur untuk menimpa registry (opsional)
            use_prebuilt_data : jika True, skip load_data dan build_features
        """
        self.logger.info("═════════════════════════════════════════════════════")
        self.logger.info(f"EDGE ENGINE 2.0 STARTING [{self.run_type.upper()}]")

        mining_mode_desc = self.config.get("logging", {}).get("mining_mode_desc")
        if mining_mode_desc:
            self.logger.info(f"Mining Mode: {mining_mode_desc}")

        self.logger.info(f"Training Mode: {self.mode}")
        features_to_log = "ALL" if feature_override is None else feature_override
        self.logger.info(f"Features: {features_to_log}")
        self.logger.info(f"Direction: {self.direction}")
        self.logger.info(f"Horizon: {self.config.get('horizon')}")
        self.logger.info("═════════════════════════════════════════════════════")

        if not use_prebuilt_data:
            self.load_data()
            self.build_features()

        if feature_override is not None:
            self.registry.set_feature_list(feature_override)

        self.check_feature()
        self.build_target()

        if self.mode == "walkforward":
            self.train_walkforward()
        else:
            self.train_simple()

        self.generate_strategy_returns()
        self.evaluate()

        return self.df