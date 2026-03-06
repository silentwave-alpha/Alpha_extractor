from sklearn.ensemble import GradientBoostingClassifier

def get_ml_model(df, cfg):
    params = cfg['params']
    model  = GradientBoostingClassifier(**params)
    features = [c for c in df.columns if c not in ['target','trade_signal','score','EV']]
    X = df[features]
    y = df['target']
    model.fit(X, y)
    return model

def predict_score(model, df):
    features = [c for c in df.columns if c not in ['target','trade_signal','score','EV']]
    return model.predict_proba(df[features])[:,1]

import numpy as np
import xgboost as xgb

# Label mapping: -1 (short) -> 0, 0 (neutral) -> 1, 1 (long) -> 2
LABEL_MAP     = {-1: 0, 0: 1, 1: 2}
LABEL_MAP_INV = {0: -1, 1: 0, 2: 1}


class InstitutionalModel:
    """
    XGBoost multiclass classifier (3 classes: short / neutral / long).
    predict_proba() returns shape (N, 3): [prob_short, prob_neutral, prob_long]

    nthread dikontrol via config ml.nthread:
      - Set ke 1 saat parallel mining untuk hindari CPU oversubscription
      - Set ke -1 (default XGBoost) untuk single-run mode
    """

    def __init__(self, config):
        self.config      = config
        self.model       = None
        self.random_state = config.get("random_state", 42)

    def initialize_model(self):
        ml_cfg  = self.config.get("ml", {})
        nthread = ml_cfg.get("nthread", -1)   # -1 = XGBoost default (semua core)

        self.model = xgb.XGBClassifier(
            objective        = "multi:softprob",
            num_class        = 3,
            n_estimators     = ml_cfg.get("n_estimators",     300),
            max_depth        = ml_cfg.get("max_depth",         4),
            learning_rate    = ml_cfg.get("learning_rate",    0.03),
            subsample        = ml_cfg.get("subsample",        0.9),
            colsample_bytree = ml_cfg.get("colsample_bytree", 0.9),
            min_child_weight = ml_cfg.get("min_child_weight",  3),
            gamma            = ml_cfg.get("gamma",             1),
            nthread          = nthread,
            tree_method      = "hist",
            eval_metric      = "mlogloss",
            random_state     = self.random_state,
        )

    def fit(self, X_train, y_train):
        """y_train must already be encoded: 0=short, 1=neutral, 2=long"""
        self.initialize_model()
        self.model.fit(X_train, y_train)

    def predict_proba(self, X):
        """Returns shape (N, 3): [prob_short, prob_neutral, prob_long]"""
        return self.model.predict_proba(X)

    def feature_importance(self, feature_names):
        if hasattr(self.model, "feature_importances_"):
            return dict(zip(feature_names, self.model.feature_importances_))
        return None
