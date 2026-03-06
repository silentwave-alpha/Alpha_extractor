# governance/feature_registry.py

import pandas as pd

class FeatureRegistry:
    def __init__(self):
        self.features = []
        self.categories = {}
        self.feature_stats_df = None

    def register(self, name, category="general"):
        if name not in self.features:
            self.features.append(name)
            self.categories[name] = category

    def set_feature_list(self, feature_list):
        self.features = feature_list.copy()

    def get_feature_list(self):
        return self.features.copy()

    def get_by_category(self, category):
        return [
            f for f, c in self.categories.items()
            if c == category
        ]

    def summary(self):
        if self.feature_stats_df is not None:
            print("Total Features:", len(self.features))
            print("Categories:")
            for cat in set(self.categories.values()):
                print(" - {}: {}".format(cat, len(self.get_by_category(cat))))

    def calculate_stats(self, df):
        """
        Hanya menghitung statistik fitur dan menyimpannya ke self.feature_stats_df.
        Tidak mencetak log apa pun.
        """
        if df is None:
            return

        stats_list = []
        for f in self.features:
            if f in df.columns:
                stats_list.append({
                    "feature": f,
                    "mean": round(df[f].mean(), 6),
                    "std": round(df[f].std(), 6),
                    "NaN%": round(df[f].isna().mean() * 100, 2)
                })
        
        if stats_list:
            self.feature_stats_df = pd.DataFrame(stats_list)

    def log(self, logger):
        """
        Hanya mencetak log dari statistik yang sudah dihitung.
        """
        if self.feature_stats_df is None:
            # Tidak ada yang perlu di-log jika statistik belum dihitung
            return

        logger.info("=================================")
        logger.info("Feature registry successfully.")
        logger.info("Total features: %d", len(self.features))
        logger.info(self.feature_stats_df)
        logger.info("=================================")
