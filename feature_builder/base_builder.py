# feature_builder/base_builder.py

import pandas as pd
import numpy as np

class BaseBuilder:
    def __init__(self, config, registry):
        self.config = config
        self.registry = registry
        self.technical = self.config["feature_builder"]["technical"]
        self.onchain = self.config["feature_builder"]["onchain"]

    def build(self, df):
        if self.technical:
            df = self._build_technical(df)
        if self.onchain:
            df = self._build_onchain(df)
            
        df = df.dropna()
        return df

    def _build_technical(self, df):
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        self.registry.register('log_ret', category='price_momentum')

        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['ema_200'] = df['close'].ewm(span=200).mean()

        df['ema_ratio_20_50'] = df['ema_20'] / df['ema_50']
        self.registry.register('ema_ratio_20_50', category='price_trend')
        df['ema_ratio_50_200'] = df['ema_50'] / df['ema_200']
        self.registry.register('ema_ratio_50_200', category='price_trend')

        df['range'] = (df['high'] - df['low']) / df['close']
        df['range_rank_20'] = df['range'].rolling(20).rank(pct=True)
        self.registry.register('range_rank_20', category='price_range')

        df['body'] = (df['close'] - df['open']) / df['close']
        self.registry.register('body', category='candle_structure')
        df['wick_up'] = (df['high'] - df[['open','close']].max(axis=1)) / df['close']
        self.registry.register('wick_up', category='candle_structure')
        df['wick_dn'] = (df[['open','close']].min(axis=1) - df['low']) / df['close']
        self.registry.register('wick_dn', category='candle_structure')
        return df


    def _build_onchain(self, df):

        required = [
            "open_basis","close_basis","open_change","close_change",
            "taker_buy_vol","taker_sell_vol",
            "cum_vol_delta","agg_taker_buy_vol","agg_taker_sell_vol",
            "cvd","funding_rate",
            "global_account_long_percent","global_account_short_percent",
            "global_account_long_short_ratio",
            "long_liquidation_usd","short_liquidation_usd",
            "net_long_change","net_short_change",
            "net_long_change_cum","net_short_change_cum",
            "net_position_change_cum",
            "oi_aggregated_history","oi_stablecoin_margin",
            "open_interest",
            "bids_usd","bids_quantity",
            "asks_usd","asks_quantity",
            "taker_buy_volume_usd","taker_sell_volume_usd",
            "top_account_long_percent","top_account_short_percent",
            "top_account_long_short_ratio",
            "top_position_long_percent","top_position_short_percent",
            "top_position_long_short_ratio",
            "whale_index_value"
        ]

        category = "onchain"

        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Onchain features require columns: {missing}")

        # =====================================================
        # 1️⃣ BASIS STRUCTURE
        # =====================================================
        name = "basis_spread"
        df[name] = df["close_basis"] - df["open_basis"]
        self.registry.register(name, category=category)

        name = "basis_intrabar_change"
        df[name] = df["close_change"] - df["open_change"]
        self.registry.register(name, category=category)

        name = "basis_level"
        df[name] = df["close_basis"]
        self.registry.register(name, category=category)


        # =====================================================
        # 2️⃣ FLOW PRIMITIVE
        # =====================================================
        total_vol = df["taker_buy_vol"] + df["taker_sell_vol"]

        name = "taker_vol_delta"
        df[name] = df["taker_buy_vol"] - df["taker_sell_vol"]
        self.registry.register(name, category=category)

        name = "taker_vol_ratio"
        df[name] = df["taker_buy_vol"] / (total_vol + 1e-9)
        self.registry.register(name, category=category)

        total_usd = df["taker_buy_volume_usd"] + df["taker_sell_volume_usd"]

        name = "taker_usd_delta"
        df[name] = df["taker_buy_volume_usd"] - df["taker_sell_volume_usd"]
        self.registry.register(name, category=category)

        name = "taker_usd_ratio"
        df[name] = df["taker_buy_volume_usd"] / (total_usd + 1e-9)
        self.registry.register(name, category=category)

        name = "agg_flow_delta"
        df[name] = df["agg_taker_buy_vol"] - df["agg_taker_sell_vol"]
        self.registry.register(name, category=category)

        name = "cvd_level"
        df[name] = df["cvd"]
        self.registry.register(name, category=category)

        name = "cvd_change"
        df[name] = df["cvd"].diff()
        self.registry.register(name, category=category)


        # =====================================================
        # 3️⃣ OPEN INTEREST PRIMITIVE
        # =====================================================
        name = "oi_level"
        df[name] = df["open_interest"]
        self.registry.register(name, category=category)

        name = "oi_change"
        df[name] = df["open_interest"].pct_change()
        self.registry.register(name, category=category)

        name = "oi_margin_ratio"
        df[name] = df["oi_stablecoin_margin"] / (df["oi_aggregated_history"] + 1e-9)
        self.registry.register(name, category=category)


        # =====================================================
        # 4️⃣ POSITION FLOW PRIMITIVE
        # =====================================================
        name = "net_position_flow"
        df[name] = df["net_long_change"] - df["net_short_change"]
        self.registry.register(name, category=category)

        name = "net_position_change"
        df[name] = df["net_position_change_cum"].diff()
        self.registry.register(name, category=category)

        name = "net_long_change_raw"
        df[name] = df["net_long_change"]
        self.registry.register(name, category=category)

        name = "net_short_change_raw"
        df[name] = df["net_short_change"]
        self.registry.register(name, category=category)


        # =====================================================
        # 5️⃣ LIQUIDATION PRIMITIVE
        # =====================================================
        total_liq = df["long_liquidation_usd"] + df["short_liquidation_usd"]

        name = "liq_delta"
        df[name] = df["long_liquidation_usd"] - df["short_liquidation_usd"]
        self.registry.register(name, category=category)

        name = "liq_ratio"
        df[name] = df["long_liquidation_usd"] / (total_liq + 1e-9)
        self.registry.register(name, category=category)

        name = "liq_total"
        df[name] = total_liq
        self.registry.register(name, category=category)


        # =====================================================
        # 6️⃣ SENTIMENT PRIMITIVE
        # =====================================================
        name = "global_long_bias"
        df[name] = df["global_account_long_percent"] - 50
        self.registry.register(name, category=category)

        name = "global_ratio_level"
        df[name] = df["global_account_long_short_ratio"]
        self.registry.register(name, category=category)

        name = "top_long_bias"
        df[name] = df["top_account_long_percent"] - 50
        self.registry.register(name, category=category)

        name = "top_ratio_level"
        df[name] = df["top_account_long_short_ratio"]
        self.registry.register(name, category=category)

        name = "position_ratio_level"
        df[name] = df["top_position_long_short_ratio"]
        self.registry.register(name, category=category)

        name = "whale_vs_crowd"
        df[name] = df["top_account_long_percent"] - df["global_account_long_percent"]
        self.registry.register(name, category=category)


        # =====================================================
        # 7️⃣ ORDERBOOK PRIMITIVE
        # =====================================================
        name = "orderbook_usd_delta"
        df[name] = df["bids_usd"] - df["asks_usd"]
        self.registry.register(name, category=category)

        name = "orderbook_qty_delta"
        df[name] = df["bids_quantity"] - df["asks_quantity"]
        self.registry.register(name, category=category)


        # =====================================================
        # 8️⃣ FUNDING & WHALE
        # =====================================================
        name = "funding_level"
        df[name] = df["funding_rate"]
        self.registry.register(name, category=category)

        name = "funding_change"
        df[name] = df["funding_rate"].diff()
        self.registry.register(name, category=category)

        name = "whale_index_level"
        df[name] = df["whale_index_value"]
        self.registry.register(name, category=category)

        name = "whale_index_change"
        df[name] = df["whale_index_value"].diff()
        self.registry.register(name, category=category)

        return df
