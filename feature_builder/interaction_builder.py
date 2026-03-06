import numpy as np
import pandas as pd
from itertools import combinations

class InteractionBuilder:
    """
    Layer 3 — Composite Signal Assembly.

    FILOSOFI:
      TransformBuilder menghasilkan fitur yang stationary & contextual
      (cvd_level_rank20, liq_total_z48, funding_level_rz, dsb).

      InteractionBuilder merakit fitur-fitur tersebut menjadi
      COMPOSITE SIGNALS yang punya makna domain:

        Confluence  : dua sinyal searah memperkuat satu sama lain
                      → flow_bullish_confluence = cvd_rank × taker_ratio_rank
        Divergence  : dua sinyal yang seharusnya searah tapi berlawanan
                      → oi_price_div = oi_change_rank - price_rank
        Pressure    : gabungan beberapa sinyal tekanan satu arah
                      → liq_oi_pressure = liq_rank + oi_change_rank
        Regime bias : sinyal relatif terhadap kondisi regime saat ini
                      → funding_regime_extreme = funding_rz × whale_rz

      Dynamic pairwise (opsional): eksplorasi otomatis semua kombinasi
      dari transformed features — cocok untuk mining phase.
    """

    # Nama suffix transform yang dianggap "sudah layak" untuk interaction
    _TRANSFORM_SUFFIXES = ("_rank", "_z", "_rz", "_voladj")

    def __init__(self, config, registry):
        self.config   = config
        self.registry = registry

    def build(self, df, dynamic_mode=False):
        df = self._build_fixed_interactions(df)
        if dynamic_mode:
            df = self._build_dynamic_interactions(df)
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # FIXED INTERACTIONS — domain-driven composite signals
    # ─────────────────────────────────────────────────────────────────────────

    def _build_fixed_interactions(self, df):
        """
        Rakit composite signals berdasarkan logika domain onchain + technical.
        Setiap group hanya dibuat jika semua kolom yang dibutuhkan tersedia.
        """
        cat = "interaction_fixed"

        # ── Pilih window terbaik yang tersedia ───────────────────────────────
        # Ambil window pertama dari config sebagai "primary window"
        pct_windows = self.config["feature_transform"].get("percentile_windows", [20, 48])
        z_windows   = self.config["feature_transform"].get("zscore_windows",     [20, 48])
        pw  = pct_windows[0]   # e.g. 20
        pw2 = pct_windows[-1]  # e.g. 48 (longer context)
        zw  = z_windows[0]

        # ── 1. FLOW CONFLUENCE ───────────────────────────────────────────────
        # Apakah taker flow, CVD, dan agg flow semuanya searah?
        # Score tinggi = strong directional flow pressure
        cols_flow = [f"taker_vol_ratio_rank{pw}", f"cvd_level_rank{pw}", f"agg_flow_delta_rank{pw}"]
        if self._available(df, cols_flow):
            # Semua di atas 0.5 = bullish confluence, di bawah = bearish
            df["flow_bull_confluence"] = (
                df[f"taker_vol_ratio_rank{pw}"] *
                df[f"cvd_level_rank{pw}"] *
                df[f"agg_flow_delta_rank{pw}"]
            ) ** (1/3)   # geometric mean supaya tidak ada satu fitur yang dominasi
            self.registry.register("flow_bull_confluence", category=cat)

            # Bearish version: invert ranks
            df["flow_bear_confluence"] = (
                (1 - df[f"taker_vol_ratio_rank{pw}"]) *
                (1 - df[f"cvd_level_rank{pw}"]) *
                (1 - df[f"agg_flow_delta_rank{pw}"])
            ) ** (1/3)
            self.registry.register("flow_bear_confluence", category=cat)

        # ── 2. OI + FLOW DIVERGENCE ──────────────────────────────────────────
        # OI naik tapi flow bearish = potential squeeze setup
        # OI turun tapi flow bullish = deleveraging into strength
        c1, c2 = f"oi_change_rank{pw}", f"taker_vol_ratio_rank{pw}"
        if self._available(df, [c1, c2]):
            df["oi_flow_divergence"] = df[c1] - df[c2]
            self.registry.register("oi_flow_divergence", category=cat)

        # ── 3. LIQUIDATION PRESSURE ──────────────────────────────────────────
        # Gabungan liq spike + directional bias → cascade signal
        liq_r  = f"liq_total_rank{pw}"
        liq_dr = f"liq_delta_rank{pw}"
        if self._available(df, [liq_r, liq_dr]):
            # Magnitude × direction
            df["liq_cascade_signal"] = (
                df[liq_r] * (df[liq_dr] - 0.5) * 2
            )   # range approx -1 to +1, positif = long liq dominan
            self.registry.register("liq_cascade_signal", category=cat)

        # ── 4. SENTIMENT DIVERGENCE (whale vs crowd) ─────────────────────────
        # Whale long tapi crowd short = contrarian bull signal
        wc_z  = f"whale_vs_crowd_z{zw}"
        glb_z = f"global_long_bias_z{zw}"
        if self._available(df, [wc_z, glb_z]):
            # Positif = whale lebih bullish dari crowd (smart money leads)
            df["smart_money_divergence"] = df[wc_z] - df[glb_z]
            self.registry.register("smart_money_divergence", category=cat)

        # Regime-aware version jika tersedia
        wc_rz  = "whale_vs_crowd_rz"
        glb_rz = "global_long_bias_rz"
        if self._available(df, [wc_rz, glb_rz]):
            df["smart_money_divergence_rz"] = df[wc_rz] - df[glb_rz]
            self.registry.register("smart_money_divergence_rz", category=cat)

        # ── 5. FUNDING EXTREME × SENTIMENT ──────────────────────────────────
        # Funding ekstrem + sentiment satu arah = overextension signal
        fund_rz = "funding_level_rz"
        top_rz  = "top_long_bias_rz"
        if self._available(df, [fund_rz, top_rz]):
            # Keduanya tinggi = overextended long (potential reversal)
            df["funding_sentiment_extreme"] = df[fund_rz] * df[top_rz]
            self.registry.register("funding_sentiment_extreme", category=cat)

        fund_abs_r = f"funding_abs_rank{pw}"
        glb_bias_r = f"global_long_bias_rank{pw}"
        if self._available(df, [fund_abs_r, glb_bias_r]):
            df["funding_crowd_alignment"] = df[fund_abs_r] * df[glb_bias_r]
            self.registry.register("funding_crowd_alignment", category=cat)

        # ── 6. OI PRICE DIVERGENCE (multi-timeframe) ─────────────────────────
        # OI naik saat harga turun (atau sebaliknya) = structural imbalance
        oi_div_r1 = f"oi_price_divergence_rank{pw}"
        oi_div_r2 = f"oi_price_divergence_rank{pw2}"
        if self._available(df, [oi_div_r1, oi_div_r2]):
            # Konsisten di dua timeframe = sinyal lebih kuat
            df["oi_price_div_mtf"] = (df[oi_div_r1] + df[oi_div_r2]) / 2
            self.registry.register("oi_price_div_mtf", category=cat)

        # ── 7. ORDERBOOK × FLOW CONFIRMATION ────────────────────────────────
        # OB imbalance + taker flow searah = order flow confirmation
        ob_r    = f"ob_usd_imbalance_rank{pw}"
        flow_r  = f"taker_usd_ratio_rank{pw}"
        if self._available(df, [ob_r, flow_r]):
            # Geometric mean, centered di 0.5
            df["ob_flow_confirmation"] = (df[ob_r] * df[flow_r]) ** 0.5
            self.registry.register("ob_flow_confirmation", category=cat)

            # Divergence: OB says buy tapi taker says sell = potential reversal
            df["ob_flow_divergence"] = df[ob_r] - df[flow_r]
            self.registry.register("ob_flow_divergence", category=cat)

        # ── 8. MULTI-SIGNAL PRESSURE INDEX ──────────────────────────────────
        # Agregasi semua sinyal tekanan jadi satu index
        # Berguna sebagai single feature untuk mode A mining
        pressure_cols = [
            f"taker_vol_ratio_rank{pw}",
            f"ob_usd_imbalance_rank{pw}",
            f"cvd_level_rank{pw}",
            f"oi_change_rank{pw}",
        ]
        available_pressure = [c for c in pressure_cols if c in df.columns]
        if len(available_pressure) >= 2:
            df["market_pressure_index"] = df[available_pressure].mean(axis=1)
            self.registry.register("market_pressure_index", category=cat)

            # Extreme version: seberapa jauh dari 0.5 (netral)
            df["market_pressure_extreme"] = (df["market_pressure_index"] - 0.5).abs() * 2
            self.registry.register("market_pressure_extreme", category=cat)

        # ── 9. TECHNICAL × ONCHAIN CONFLUENCE ───────────────────────────────
        # EMA trend + flow confirmation = trend + flow aligned
        ema_r  = "ema_ratio_50_200"   # sudah stationary (ratio)
        flow_c = "flow_bull_confluence"
        if self._available(df, [ema_r, flow_c]):
            # EMA ratio di atas 1 = uptrend, dikali flow confluence
            ema_norm = (df[ema_r] - 1).clip(-0.1, 0.1) / 0.1   # normalize ke -1..+1
            df["trend_flow_alignment"] = ema_norm * (df[flow_c] - 0.5) * 2
            self.registry.register("trend_flow_alignment", category=cat)

        # ── 10. WHALE ACTIVITY × LIQUIDATION ────────────────────────────────
        # Whale aktif saat liq spike = potential manipulation / cascade
        whale_r = f"whale_index_rank{pw}" if f"whale_index_rank{pw}" in df.columns else None
        liq_r2  = f"liq_total_rank{pw}"
        if whale_r and self._available(df, [whale_r, liq_r2]):
            df["whale_liq_activity"] = df[whale_r] * df[liq_r2]
            self.registry.register("whale_liq_activity", category=cat)

        return df

    # ─────────────────────────────────────────────────────────────────────────
    # DYNAMIC INTERACTIONS — eksplorasi otomatis (opsional, untuk mining)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_dynamic_interactions(self, df):
        """
        Pairwise interactions hanya antar TRANSFORMED features
        (suffix _rank, _z, _rz, _voladj) — bukan raw primitives.

        Operasi: product dan diff saja (sum & ratio kurang informatif
        untuk fitur yang sudah dinormalisasi).
        """
        # Filter hanya transformed features yang layak
        exclude = {"target", "label", "future_return", "strategy_return",
                   "position_unseen", "position_seen", "is_unseen"}
        cols = [
            c for c in df.columns
            if any(sfx in c for sfx in self._TRANSFORM_SUFFIXES)
            and c not in exclude
            and df[c].dtype in [np.float64, np.float32]
        ]

        count = 0
        for a, b in combinations(cols, 2):
            # Product: kedua sinyal harus searah dan kuat untuk score tinggi
            name = f"ix_{a}_x_{b}"
            if len(name) < 120:   # hindari nama kolom terlalu panjang
                df[name] = df[a] * df[b]
                self.registry.register(name, category="interaction_dynamic")
                count += 1

            # Difference: divergence antar dua sinyal
            name = f"ix_{a}_d_{b}"
            if len(name) < 120:
                df[name] = df[a] - df[b]
                self.registry.register(name, category="interaction_dynamic")
                count += 1

        if count > 0:
            print(f"✓ Generated {count} dynamic interactions from {len(cols)} transformed features")

        return df

    # ─────────────────────────────────────────────────────────────────────────
    # HELPER
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _available(df, cols):
        """Return True jika semua kolom ada di df dan tidak all-NaN."""
        return all(c in df.columns and not df[c].isna().all() for c in cols)