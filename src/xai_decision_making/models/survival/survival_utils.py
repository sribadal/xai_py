"""
Survival model utilities: data preparation, model wrapper, and artifact loading.

Provides SurvivalModelWrapper (wrapping masurvival's RandomSurvivalForest
with configurable imputation strategies and optional ConDOR RDM-based QoL
imputation), prepare_survival_data() for building Surv targets, and
helpers for loading/finding saved .pkl model artifacts.
"""
from __future__ import annotations

import glob
import os
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from xai_decision_making.preprocessing.cleaning.survival_data_cleaner import SurvivalDataCleaner
from xai_decision_making.models.survival import config_survival as config

try:
    from masurvival.ensemble import RandomSurvivalForest as RSF_MA
    from masurvival.util import Surv
    HAS_MASURVIVAL = True
except ImportError:
    HAS_MASURVIVAL = False
    RSF_MA = None
    Surv = None


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def find_artifact(model_dir, pattern):
    matches = sorted(glob.glob(os.path.join(model_dir, pattern)))
    if not matches:
        raise FileNotFoundError(
            f"No artifacts found in '{model_dir}' matching '{pattern}'."
        )
    return matches[0]


try:
    from xai_decision_making.models.synthesis.condor_core import ConDORConfig, RDM
    HAS_CONDOR = True
except ImportError:
    HAS_CONDOR = False
    ConDORConfig = None
    RDM = None


class RDMImputer:

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.device = torch.device("cpu")
        self.encoders = None
        self.success = False

    def fit(self, X, y=None):
        if not HAS_CONDOR:
            self.success = False
            return self

        if not self.model_path or not os.path.exists(self.model_path):
            self.success = False
            return self

        ckpt = torch.load(self.model_path, map_location=self.device)
        args = ckpt.get("config_args", {})

        conf = ConDORConfig(static_dim=args.get("static_dim", 8), use_static_rdm=True)
        conf.device = self.device

        self.model = RDM(conf).to(self.device)
        self.model.load_state_dict(ckpt["rdm"])
        self.model.eval()

        self.encoders = ckpt.get("encoders")
        self.success = True
        return self

    def transform(self, X):
        if not self.success or self.model is None:
            return X

        X_out = X.copy()
        qol_col = "qol_summary_pretreatment"

        if qol_col not in X_out.columns:
            return X_out

        missing_mask = X_out[qol_col].isna() | (X_out[qol_col] == -1)
        if not missing_mask.any():
            return X_out

        X_missing = X_out.loc[missing_mask]

        num_keys = ["cci", "ca199", "bilirubine", "albumine", "ldh", "crp"]
        cat_keys = ["leeft_cat", "gesl"]

        avail_nums = [c for c in num_keys if c in X_missing.columns]
        avail_cats = [c for c in cat_keys if c in X_missing.columns]

        nums = X_missing[avail_nums].fillna(X_missing[avail_nums].median()).values
        if self.encoders and "num_scaler" in self.encoders:
            nums = self.encoders["num_scaler"].transform(nums)

        cats = X_missing[avail_cats].astype(str).values
        if self.encoders and "cat_encoder" in self.encoders:
            cats = self.encoders["cat_encoder"].transform(cats)
        else:
            cats = np.zeros((len(X_missing), 0), dtype=float)

        static_vec = np.hstack([nums, cats])
        static_tensor = torch.FloatTensor(static_vec).to(self.device)
        batch_size = static_tensor.shape[0]

        x_t = torch.randn(batch_size, 1).to(self.device)
        t_0 = torch.zeros(batch_size).to(self.device)
        y_0 = torch.zeros(batch_size).to(self.device)

        with torch.no_grad():
            steps = self.model.config.rdm_steps
            for i in reversed(range(steps)):
                t_step = torch.full((batch_size,), i, device=self.device).float()
                pred = self.model(x_t, t_step, y_0, t_0, static=static_tensor)
                x_t = x_t - pred * (1.0 / steps)

        x_final = torch.clamp(x_t, 0, 1).cpu().numpy().flatten()
        X_out.loc[missing_mask, qol_col] = x_final * 100.0
        return X_out


def prepare_survival_data(df, features):
    if not HAS_MASURVIVAL:
        raise ImportError(
            "masurvival is required. Install: pip install masurvival==0.1.2"
        )

    valid_features = [f for f in features if f in df.columns]
    X = df[valid_features].copy()

    event = df[config.EVENT_COL].astype(bool)
    time = df[config.TIME_COL].astype(float)
    y = Surv.from_arrays(event=event, time=time)

    qol_col = "qol_summary_pretreatment"
    if qol_col in X.columns:
        X[qol_col] = pd.to_numeric(X[qol_col], errors="coerce")

    return X, y


class SurvivalModelWrapper:

    def __init__(self, imputation_strategy="simple", rdm_path=None, alpha=0.0, **params):
        if not HAS_MASURVIVAL:
            raise ImportError(
                "masurvival is required. Install: pip install masurvival==0.1.2"
            )
        self.imputation_strategy = imputation_strategy
        self.rdm_path = rdm_path
        self.alpha = alpha
        self.params = params

        self.model = RSF_MA(alpha=self.alpha, **self.params)
        self.pipeline = None

        self.feature_names_ = []
        self.label_encoders = {}

    def _encode_categorical_features(self, X, fit=True):
        X_encoded = X.copy()

        for col in X_encoded.columns:
            if (
                X_encoded[col].dtype == "object"
                or X_encoded[col].dtype.name == "category"
            ):
                if fit:
                    le = LabelEncoder()
                    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        X_encoded[col] = X_encoded[col].astype(str).apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else 0
                        )
                    else:
                        X_encoded[col] = pd.Categorical(X_encoded[col].astype(str)).codes

        return X_encoded

    def fit(self, X, y):
        self.feature_names_ = X.columns.tolist()

        if self.imputation_strategy == "passthrough":
            X_encoded = self._encode_categorical_features(X, fit=True)
            X_numeric = X_encoded.values.astype(np.float64)
            self.model.fit(X_numeric, y)
            return

        steps = []
        if self.rdm_path:
            steps.append(("qol_genai", RDMImputer(self.rdm_path)))

        cleaner = SurvivalDataCleaner(
            strategy=self.imputation_strategy,
            missing_threshold=0.0 if self.imputation_strategy == "drop_vars" else 0.4,
            fill_value_num=-999,
        )
        steps.append(("cleaner", cleaner))

        self.pipeline = Pipeline(steps)
        X_clean = self.pipeline.fit_transform(X)

        if hasattr(cleaner, "get_feature_names_out"):
            names = cleaner.get_feature_names_out()
            if names is not None:
                self.feature_names_ = list(names)

        X_final = X_clean
        if hasattr(X_clean, "to_numpy"):
            X_final = X_clean.to_numpy(dtype=np.float64)

        self.model.fit(X_final, y)

    def _prepare_input(self, X):
        if self.imputation_strategy == "passthrough":
            X_aligned = X.reindex(columns=self.feature_names_)
            X_encoded = self._encode_categorical_features(X_aligned, fit=False)
            return X_encoded.values.astype(np.float64)

        if self.pipeline is None:
            raise RuntimeError("Model pipeline is not fitted.")

        X_clean = self.pipeline.transform(X)
        if hasattr(X_clean, "to_numpy"):
            return X_clean.to_numpy(dtype=np.float64)

        return X_clean

    def predict(self, X):
        X_final = self._prepare_input(X)
        return self.model.predict(X_final)

    def predict_survival_function(self, X):
        X_final = self._prepare_input(X)
        return self.model.predict_survival_function(X_final)


@dataclass(frozen=True)
class LoadedArtifact:
    model: object
    path: str


def load_trained_survival_model(pattern):
    path = find_artifact(config.SURVIVAL_MODELS_DIR, pattern)
    return LoadedArtifact(model=load_pickle(path), path=path)