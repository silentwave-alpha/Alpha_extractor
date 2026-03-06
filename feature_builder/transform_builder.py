import numpy as np

class TransformBuilder:
    """
    Layer 2 — Normalisasi & Contextualization.

    FILOSOFI:
      BaseBuilder menghasilkan dua jenis kolom:
        (A) _raw columns   : absolut/non-stationary (cvd_level_raw, liq_total_raw, dsb)
        (B) registered feat: sudah stationary tapi masih single-bar, belum punya konteks temporal

      TransformBuilder bertugas mengubah keduanya menjadi:
        - Stationary  : rolling z-score, percentile rank (0-1), vol-adjusted
        - Contextual  : posisi relatif vs historis, vs regime
        - Bounded     : clip outlier supaya model tidak didominasi spike ekstrem

      Output yang di-register ke mining pool adalah hasil transform,
      BUKAN primitif mentahnya.

    SKEMA TRANSFORM PER KOLOM:
      _raw columns     → rank_{W}, z_{W}, rz (regime-zscore)
      registered feats → rank_{W}, z_{W}, voladj_{W}, rz
    """

    def __init__(self, config, registry):
        self.config   = config
        self.registry = registry

    def build(self, df):
        z_windows   = self.config["feature_transform"].get("zscore_windows",    [20, 48])
        pct_windows = self.config["feature_transform"].get("percentile_windows", [20, 48])
        vol_windows = self.config["feature_transform"].get("volscale_windows",   [20])
        regime_col  = self.config["data"].get("regime_col")

        # ── A. Normalisasi kolom _raw ────────────────────────────────────────
        # Kolom _raw absolut/non-stationary: rank & zscore masuk registry
        # Primitive raw-nya sendiri TIDAK di-register
        raw_cols = [c for c in df.columns if c.endswith("_raw")]
        for col in raw_cols:
            base = col[:-4]   # strip "_raw"
            self._add_rank_and_z(df, col, base, pct_windows, z_windows, regime_col)

        # ── B. Transform kolom yang sudah di-register BaseBuilder ────────────
        # Sudah stationary, tapi perlu konteks temporal dan normalisasi regime
        base_features = self.registry.get_feature_list().copy()
        for feat in base_features:
            if feat not in df.columns:
                continue
            self._add_full_transform(
                df, feat, pct_windows, z_windows, vol_windows, regime_col
            )

        return df

    # ─── INTERNAL HELPERS ────────────────────────────────────────────────────

    def _add_rank_and_z(self, df, col, base_name, pct_windows, z_windows, regime_col):
        """Untuk _raw columns: rank + zscore + regime-zscore."""
        for w in pct_windows:
            name = f"{base_name}_rank{w}"
            df[name] = self._rolling_percentile(df[col], w)
            self.registry.register(name, category=f"transform_raw_{base_name}")

        for w in z_windows:
            name = f"{base_name}_z{w}"
            df[name] = self._rolling_zscore(df[col], w)
            self.registry.register(name, category=f"transform_raw_{base_name}")

        if regime_col and regime_col in df.columns:
            name = f"{base_name}_rz"
            df[name] = self._regime_zscore(df, col, regime_col)
            self.registry.register(name, category=f"transform_raw_{base_name}")

    def _add_full_transform(self, df, feat, pct_windows, z_windows, vol_windows, regime_col):
        """Untuk registered base features: rank + zscore + voladj + regime-zscore."""
        for w in pct_windows:
            name = f"{feat}_rank{w}"
            df[name] = self._rolling_percentile(df[feat], w)
            self.registry.register(name, category=f"transform_{feat}")

        for w in z_windows:
            name = f"{feat}_z{w}"
            df[name] = self._rolling_zscore(df[feat], w)
            self.registry.register(name, category=f"transform_{feat}")

        for w in vol_windows:
            name = f"{feat}_voladj{w}"
            df[name] = self._volatility_scale(df[feat], w)
            self.registry.register(name, category=f"transform_{feat}")

        if regime_col and regime_col in df.columns:
            name = f"{feat}_rz"
            df[name] = self._regime_zscore(df, feat, regime_col)
            self.registry.register(name, category=f"transform_{feat}")

    # ─── TRANSFORM FUNCTIONS ─────────────────────────────────────────────────

    @staticmethod
    def _rolling_zscore(series, window):
        mean = series.rolling(window, min_periods=window // 2).mean()
        std  = series.rolling(window, min_periods=window // 2).std()
        return (series - mean) / (std + 1e-8)

    @staticmethod
    def _rolling_percentile(series, window):
        return series.rolling(window, min_periods=window // 2).rank(pct=True)

    @staticmethod
    def _volatility_scale(series, window):
        vol = series.rolling(window, min_periods=window // 2).std()
        return series / (vol + 1e-8)

    @staticmethod
    def _regime_zscore(df, feature, regime_col):
        """
        Z-score expanding per-regime.
        Model tahu apakah nilai ini tinggi/rendah DALAM regime ini,
        bukan dibanding seluruh history campuran regime.
        """
        result = df[feature].copy() * np.nan
        for regime in df[regime_col].dropna().unique():
            mask = df[regime_col] == regime
            s    = df.loc[mask, feature]
            mean = s.expanding(min_periods=5).mean()
            std  = s.expanding(min_periods=5).std()
            result.loc[mask] = (s - mean) / (std + 1e-8)
        return result