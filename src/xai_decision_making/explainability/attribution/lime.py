"""
Local surrogate (LIME) explanations for survival models.

Handles one-hot encoded features by detecting dummy groups and
decoding to integer-coded categoricals before perturbation, then
re-encoding for model scoring. Reports local R-squared as a
confidence metric for the linear surrogate fit.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from xai_decision_making.explainability.base import (
    AggregatedSanityResult,
    AttributionExplainer,
    Explanation,
    ExplainerModality,
    ExplainerRegistry,
    ModelTarget,
    SurvivalModelAdapter,
)
from xai_decision_making.explainability.sanity_checks import DeletionFaithfulnessCheck
from xai_decision_making.explainability.attribution._base import (
    _risk_predict_fn,
    _run_sanity_checks,
)

registry = ExplainerRegistry()

_DUMMY_RE = re.compile(r"^(?P<base>.+?)_(?P<idx>\d+)$")

def _detect_dummy_groups(feature_names, *, exclude_bases=()):
    raw = {}
    for col_idx, name in enumerate(feature_names):
        m = _DUMMY_RE.match(name)
        if m is None:
            continue
        base = m.group("base")
        if base in exclude_bases:
            continue
        raw.setdefault(base, []).append((int(m.group("idx")), col_idx))

    groups = {}
    for base, pairs in raw.items():
        if len(pairs) < 2:
            continue
        pairs.sort(key=lambda p: p[0])
        groups[base] = [col_idx for _, col_idx in pairs]
    return groups

@registry.register
class SurvLIMEExplainer(AttributionExplainer):
    _modality = ExplainerModality.ATTRIBUTION
    _target = ModelTarget.SURVIVAL

    def __init__(
        self,
        model_adapter,
        *,
        n_samples=2000,
        kernel_width=None,
        feature_selection="auto",
        time_horizon=None,
        random_state=None,
        exclude_dummy_bases=(),
    ):
        super().__init__(model_adapter)
        self._n_samples = n_samples
        self._kernel_width = kernel_width
        self._feature_selection = feature_selection
        self._time_horizon = time_horizon
        self._random_state = random_state
        self._exclude_dummy_bases = tuple(exclude_dummy_bases)
        self._lime_explainer = None
        self._last_local_r2 = None

        self._encoded_feature_names = None
        self._decoded_feature_names = None
        self._dummy_groups = None
        self._continuous_cols = None
        self._n_continuous = None

    def _build_lime_explainer(self):
        x_train = self._adapter.training_data()
        feat_names = self._adapter.feature_names()
        self._encoded_feature_names = list(feat_names)

        self._dummy_groups = _detect_dummy_groups(
            self._encoded_feature_names,
            exclude_bases=self._exclude_dummy_bases,
        )
        dummy_col_set = set()
        for cols in self._dummy_groups.values():
            dummy_col_set.update(cols)

        self._continuous_cols = [
            i for i in range(len(self._encoded_feature_names))
            if i not in dummy_col_set
        ]
        self._n_continuous = len(self._continuous_cols)
        cat_bases = list(self._dummy_groups.keys())
        self._decoded_feature_names = (
            [self._encoded_feature_names[i] for i in self._continuous_cols]
            + cat_bases
        )

        cat_indices = list(
            range(self._n_continuous, self._n_continuous + len(cat_bases))
        )
        categorical_names = {}
        for cat_pos, base in enumerate(cat_bases):
            cols = self._dummy_groups[base]
            suffixes = [
                _DUMMY_RE.match(self._encoded_feature_names[c]).group("idx")
                for c in cols
            ]
            categorical_names[self._n_continuous + cat_pos] = suffixes

        encoded_train = x_train[self._encoded_feature_names].fillna(0).values
        decoded_train = self._encode_to_decoded(encoded_train)

        self._lime_explainer = LimeTabularExplainer(
            training_data=decoded_train,
            feature_names=self._decoded_feature_names,
            categorical_features=cat_indices,
            categorical_names=categorical_names,
            mode="regression",
            kernel_width=self._kernel_width
                or 0.75 * np.sqrt(len(self._decoded_feature_names)),
            feature_selection=self._feature_selection,
            verbose=False,
            random_state=self._random_state,
            discretize_continuous=True,
        )

        encoded_predict_fn = _risk_predict_fn(
            self._adapter, self._time_horizon
        )

        def _decoded_predict_fn(decoded_batch):
            encoded_batch = self._decoded_to_encoded(decoded_batch)
            return encoded_predict_fn(encoded_batch)

        self._predict_fn = _decoded_predict_fn

    def _encode_to_decoded(self, encoded):
        encoded = np.asarray(encoded)
        if encoded.ndim == 1:
            encoded = encoded.reshape(1, -1)
        n_rows = encoded.shape[0]
        decoded = np.zeros(
            (n_rows, self._n_continuous + len(self._dummy_groups)),
            dtype=np.float64,
        )
        for new_idx, orig_idx in enumerate(self._continuous_cols):
            decoded[:, new_idx] = encoded[:, orig_idx]
        for cat_pos, cols in enumerate(self._dummy_groups.values()):
            decoded[:, self._n_continuous + cat_pos] = (
                encoded[:, cols].argmax(axis=1)
            )
        return decoded

    def _decoded_to_encoded(self, decoded):
        decoded = np.asarray(decoded)
        if decoded.ndim == 1:
            decoded = decoded.reshape(1, -1)
        n_rows = decoded.shape[0]
        encoded = np.zeros(
            (n_rows, len(self._encoded_feature_names)), dtype=np.float64,
        )
        for dec_idx, orig_idx in enumerate(self._continuous_cols):
            encoded[:, orig_idx] = decoded[:, dec_idx]
        for cat_pos, cols in enumerate(self._dummy_groups.values()):
            levels = decoded[:, self._n_continuous + cat_pos].astype(int)
            levels = np.clip(levels, 0, len(cols) - 1)
            for row_idx, lvl in enumerate(levels):
                encoded[row_idx, cols[lvl]] = 1.0
        return encoded

    @property
    def modality(self):
        return self._modality

    def feature_importances(self, patient_features):
        if self._lime_explainer is None:
            self._build_lime_explainer()

        encoded = (
            patient_features[self._encoded_feature_names].fillna(0).values
        )
        decoded = self._encode_to_decoded(encoded)[0]

        exp = self._lime_explainer.explain_instance(
            decoded,
            self._predict_fn,
            num_features=len(self._decoded_feature_names),
            num_samples=self._n_samples,
        )
        try:
            self._last_local_r2 = float(exp.score)
        except (AttributeError, TypeError):
            self._last_local_r2 = None
        return dict(exp.as_list())

    def explain(self, patient_features, patient_id=None):
        importances = self.feature_importances(patient_features)
        confidence = {
            "local_r2": self._last_local_r2,
            "n_neighborhood_samples": self._n_samples,
            "source": "tier_c_native",
        }
        return self._make_explanation(
            patient_id,
            content=importances,
            method="LIME",
            confidence_metrics=confidence,
        )

    def sanity_check(self, held_out_df):
        return _run_sanity_checks(
            self,
            held_out_df,
            [DeletionFaithfulnessCheck()],
        )