# governance/leakage_guard.py

import pandas as pd

def check_future_leakage(df, feature_cols, target_col, logger):

    if target_col not in df.columns:
        raise ValueError("Target column missing.")

    for col in feature_cols:
        if col == target_col:
            continue

        # Simple heuristic: feature shouldn't equal future target
        if df[col].shift(-1).equals(df[target_col]):
            raise ValueError(f"Potential leakage detected in {col}")

    logger.info("Leakage check passed.")
