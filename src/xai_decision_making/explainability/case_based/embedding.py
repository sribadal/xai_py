"""
Embedding-based similar patient retrieval.

Retrieves similar patients in a learned embedding space rather than
raw feature space. Embedding strategies are selected automatically:
RSF leaf-node proximity, ConDOR RDM hidden activations, or PCA
fallback. Distance is measured with cosine similarity.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from xai_decision_making.explainability.base import (
    AggregatedSanityResult,
    CaseBasedExplainer,
    Explanation,
    ExplainerModality,
    ExplainerRegistry,
    ModelTarget,
    SurvivalModelAdapter,
)
from xai_decision_making.explainability.sanity_checks import CASE_BASED_CHECKS
from xai_decision_making.explainability.case_based._base import (
    _cosine_distance_row,
    _run_sanity_checks,
    compute_case_confidence,
    enrich_with_context,
)
from xai_decision_making.explainability.case_based.knn import (
    _default_pretreatment_features,
    _compute_trajectories,
)

registry = ExplainerRegistry()

def _rsf_leaf_embedding(rsf_model, X):
    leaves = rsf_model.apply(X)
    return leaves.astype(np.float64)

def _condor_embedding(diff_model, X_tensor):
    import torch

    activations = []

    def _hook(module, inp, out):
        activations.append(out.detach().cpu())

    rdm = diff_model.rdm
    handle = None
    linear_layers = [m for m in rdm.modules() if isinstance(m, torch.nn.Linear)]
    if len(linear_layers) >= 3:
        handle = linear_layers[-2].register_forward_hook(_hook)
    elif len(linear_layers) >= 1:
        handle = linear_layers[0].register_forward_hook(_hook)

    if handle is None:
        raise RuntimeError("Could not find a Linear layer in the RDM to hook into.")

    try:
        with torch.no_grad():
            diff_model(X_tensor)
    finally:
        handle.remove()

    if not activations:
        raise RuntimeError("Forward hook did not capture any activations.")

    return activations[0].numpy()

def _pca_embedding(X, n_components=16):
    n_components = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)

@registry.register
class EmbeddingRetriever(CaseBasedExplainer):
    _modality = ExplainerModality.CASE_BASED
    _target = ModelTarget.SURVIVAL

    def __init__(
        self,
        model_adapter,
        *,
        time_horizon=None,
        time_horizons=None,
        horizon_labels=None,
        n_components=16,
        pretreatment_cols=None,
    ):
        super().__init__(model_adapter)
        self._time_horizon = time_horizon
        self._time_horizons = time_horizons
        self._horizon_labels = horizon_labels or {}
        self._n_components = n_components

        all_feats = self._adapter.feature_names()
        self._pretreatment_cols = (
            pretreatment_cols if pretreatment_cols is not None
            else _default_pretreatment_features(all_feats)
        )

        self._strategy = "unknown"
        self._train_embeddings = None
        self._pca = None
        self._build_embeddings()

    def _build_embeddings(self):
        x_train = self._adapter.training_data()
        X_np = x_train.fillna(0).values.astype(np.float64)

        raw_model = self._adapter.raw_model

        if hasattr(raw_model, "apply") and hasattr(raw_model, "estimators_"):
            try:
                self._train_embeddings = _rsf_leaf_embedding(raw_model, X_np)
                self._strategy = "rsf_leaf"
                return
            except Exception:
                pass

        try:
            import torch
            if isinstance(raw_model, torch.nn.Module):
                X_t = torch.FloatTensor(X_np.astype(np.float32))
                self._train_embeddings = _condor_embedding(raw_model, X_t)
                self._strategy = "condor_hidden"
                return
        except Exception:
            pass

        n_comp = min(self._n_components, X_np.shape[0], X_np.shape[1])
        self._pca = PCA(n_components=n_comp)
        self._train_embeddings = self._pca.fit_transform(X_np)
        self._strategy = "pca"

    def _embed_patient(self, patient_features):
        feats = self._adapter.feature_names()
        x = patient_features.reindex(columns=feats).fillna(0).values.astype(np.float64)

        if self._strategy == "rsf_leaf":
            return _rsf_leaf_embedding(self._adapter.raw_model, x)

        if self._strategy == "condor_hidden":
            import torch
            x_t = torch.FloatTensor(x.astype(np.float32))
            return _condor_embedding(self._adapter.raw_model, x_t)

        return self._pca.transform(x)

    @property
    def modality(self):
        return self._modality

    def pretreatment_features(self):
        return list(self._pretreatment_cols)

    def retrieve_similar(self, patient_features, k=3):
        patient_emb = self._embed_patient(patient_features).ravel()

        distances = np.array([
            _cosine_distance_row(patient_emb, self._train_embeddings[i])
            for i in range(len(self._train_embeddings))
        ])

        order = np.argsort(distances)[:k]
        x_train = self._adapter.training_data()
        pre_cols = [c for c in self._pretreatment_cols if c in x_train.columns]
        result = x_train.iloc[order][pre_cols].copy().reset_index(drop=True)
        result["_distance"] = distances[order]

        if hasattr(self._adapter, "inverse_transform_numeric"):
            result = self._adapter.inverse_transform_numeric(result)

        result = enrich_with_context(result, self._adapter, order.tolist())

        trajectories = _compute_trajectories(
            self._adapter, x_train, order,
            self._time_horizons, self._horizon_labels,
        )
        if trajectories is not None:
            result["_trajectory"] = trajectories

        return result

    def explain(self, patient_features, patient_id=None):
        similar = self.retrieve_similar(patient_features)
        confidence_metrics = compute_case_confidence(similar)

        return self._make_explanation(
            patient_id,
            content=similar,
            method="EmbeddingRetrieval",
            strategy=self._strategy,
            confidence_metrics=confidence_metrics,
        )

    def sanity_check(self, held_out_df):
        return _run_sanity_checks(self, held_out_df, CASE_BASED_CHECKS)