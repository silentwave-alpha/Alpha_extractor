import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, config):
        self.config = config

    def load(self):
        path = self.config["data"]["path"]
        date_col = self.config["data"].get("date_column")

        df = pd.read_csv(path)

        # -----------------------------
        # 1️⃣ Standardize column names
        # -----------------------------
        df.columns = [c.lower().strip() for c in df.columns]

        # -----------------------------
        # 2️⃣ Parse datetime
        # -----------------------------
        if date_col not in df.columns:
            raise ValueError(f"{date_col} not found in data.")

        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        df = df.set_index(date_col)

        # -----------------------------
        # 3️⃣ Remove duplicates
        # -----------------------------
        df = df[~df.index.duplicated(keep="first")]

        # Get regime column name early for validation
        regime_col = self.config["data"]["regime_col"]

        # Missing value handling
        if self.config["data"].get("forward_fill", True):
            df = df.ffill()

        # Drop rows only if REQUIRED columns have NaN
        # (don't drop entire dataframe just because optional columns are NaN)
        required_cols = ["open", "high", "low", "close"]
        if regime_col in df.columns:
            required_cols.append(regime_col)
        
        df = df.dropna(subset=required_cols)

        # Validate required columns
        for col in ["open", "high", "low", "close"]: 
            if col not in df.columns:
                raise ValueError(f"Required column missing: {col}")

        # Regime validation
        if regime_col in df.columns:
            if df[regime_col].isna().any():
                raise ValueError(f"Regime column '{regime_col}' contains NaN values.")

            # Ensure regime is integer
            df[regime_col] = df[regime_col].astype(int)

        # -----------------------------
        # 7️⃣ Freeze dataset (copy)
        # -----------------------------
        df = df.copy()

        print("Data loaded successfully.")
        print(f"Columns: {list(df.columns)}")

        return df

if __name__ == "__main__":

    import yaml

    # Load config.yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize loader
    loader = DataLoader(config)

    # Load data
    df = loader.load()

    print(df.head())
