import numpy as np
import pandas as pd


class FeatureCleaner:
    @staticmethod
    def clean_features_hybrid(X, fill_qol_value=-1, fill_other_value=0):
        X_clean = X.copy()

        # 1. Handle categoricals
        for col in X_clean.select_dtypes(include=["object", "category"]).columns:
            X_clean[col] = X_clean[col].astype("category").cat.codes

        # 2. Hybrid QoL imputation
        qol_cols = [
            c
            for c in X_clean.columns
            if "qol" in c.lower() or "pretreatment" in c.lower()
        ]
        if qol_cols:
            X_clean[qol_cols] = X_clean[qol_cols].fillna(fill_qol_value)

        # 3. Fill remaining
        X_clean = X_clean.fillna(fill_other_value)

        # 4. Replace inf
        X_clean = X_clean.replace([np.inf, -np.inf], 0)

        return X_clean
