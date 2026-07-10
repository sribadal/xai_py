"""
Sklearn-compatible data cleaning transformer for survival analysis.

Wraps imputation (median, KNN, constant, passthrough) and categorical
encoding into a single ColumnTransformer pipeline, with optional
high-missingness column dropping. Designed for use inside cross-validated
survival model comparisons.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

class SurvivalDataCleaner(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        strategy="simple",
        missing_threshold=0.0,
        fill_value_num=-999,
        fill_value_cat="Missing",
    ):
        self.strategy = strategy
        self.missing_threshold = missing_threshold
        self.fill_value_num = fill_value_num
        self.fill_value_cat = fill_value_cat

        self.pipeline_ = None
        self.cols_to_drop_ = []

    def fit(self, X, y=None):
        X = X.copy()

        if self.strategy == "drop_vars":
            miss_frac = X.isnull().mean()
            self.cols_to_drop_ = miss_frac[
                miss_frac > self.missing_threshold
            ].index.tolist()

            if self.cols_to_drop_:
                print(
                    f"[SurvivalDataCleaner] Dropping {len(self.cols_to_drop_)} "
                    f"columns: {self.cols_to_drop_}"
                )

        X_kept = X.drop(columns=self.cols_to_drop_, errors="ignore")

        cat_cols = X_kept.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = X_kept.select_dtypes(include=["number"]).columns.tolist()

        for col in cat_cols:
            X_kept[col] = X_kept[col].astype(str).replace("nan", np.nan)

        transformers = []

        if self.strategy == "passthrough":
            num_pipe = "passthrough"

        elif self.strategy == "knn":
            num_pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("imputer", KNNImputer(n_neighbors=5)),
                ]
            )

        elif self.strategy == "constant":
            num_pipe = SimpleImputer(
                strategy="constant", fill_value=self.fill_value_num
            )

        else:
            num_pipe = SimpleImputer(strategy="median")

        transformers.append(("num", num_pipe, num_cols))

        if self.strategy == "passthrough":
            cat_pipe = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=np.nan,
            )

        elif self.strategy == "constant":
            cat_pipe = Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="constant",
                            fill_value=self.fill_value_cat,
                        ),
                    ),
                    (
                        "encoder",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                    ),
                ]
            )

        else:
            cat_pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                    ),
                ]
            )

        transformers.append(("cat", cat_pipe, cat_cols))

        self.pipeline_ = ColumnTransformer(
            transformers=transformers,
            verbose_feature_names_out=False,
        )

        self.pipeline_.fit(X_kept, y)

        return self

    def transform(self, X):
        X_kept = X.drop(columns=self.cols_to_drop_, errors="ignore")

        cat_cols = X_kept.select_dtypes(include=["object", "category"]).columns.tolist()

        for col in cat_cols:
            X_kept[col] = X_kept[col].astype(str).replace("nan", np.nan)

        return self.pipeline_.transform(X_kept)

    def get_feature_names_out(self):
        if hasattr(self.pipeline_, "get_feature_names_out"):
            return self.pipeline_.get_feature_names_out()
        return None
