"""
CounterNet counterfactual explainer.

Trains a neural CF generator that maps patients to nearby counterfactuals
via a validity + proximity + sparsity loss against an anchor pool of
good-outcome training patients.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from xai_decision_making.explainability.base import (
    AggregatedSanityResult, CounterfactualExplainer, ExplainerModality,
    ExplainerRegistry, Explanation, ModelTarget,
)
from xai_decision_making.explainability.counterfactual._base import (
    _run_sanity_checks, apply_direction_priors, clip_to_ranges, compute_content_deltas,
    enforce_immutables, extract_scalar_prediction, resolve_direction_priors,
    risk_predict_fn, snap_categoricals,
)
from xai_decision_making.explainability.counterfactual._plausibility import (
    build_plausibility_config, snap_with_constraints,
)

registry = ExplainerRegistry()

class _CFGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim), nn.Tanh())

    def forward(self, x_norm):
        return x_norm + self.net(x_norm)

@registry.register
class CounterNetExplainer(CounterfactualExplainer):
    _modality = ExplainerModality.COUNTERFACTUAL
    _target = ModelTarget.SURVIVAL

    def __init__(self, model_adapter, *, hidden_dim=128, epochs=200, batch_size=128,
                 lr=1e-3, validity_weight=1.0, proximity_weight=0.5, sparsity_weight=0.1,
                 target_iqr_fraction=0.5, max_train_samples=5000, immutable=None,
                 direction_priors=None, time_horizon=None):
        super().__init__(model_adapter)
        self._hidden_dim = hidden_dim
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = lr
        self._validity_weight = validity_weight
        self._proximity_weight = proximity_weight
        self._sparsity_weight = sparsity_weight
        self._max_train_samples = max_train_samples
        self._time_horizon = time_horizon
        self._predict_fn = risk_predict_fn(model_adapter, time_horizon)
        self._cfg = build_plausibility_config(model_adapter, extra_immutable=immutable)
        self._ranges = self._cfg.ranges
        self._immutable = self._cfg.immutable
        self._categorical_values = self._cfg.categorical_values
        self._direction_priors = resolve_direction_priors(model_adapter, override=direction_priors)
        self._higher_is_better = getattr(model_adapter, "higher_is_better", False)
        self._generator = None
        self._scaler = None
        self._device = torch.device("cpu")
        self._risk_min = self._risk_max = self._risk_q25 = self._risk_median = self._risk_q75 = None

    @property
    def modality(self):
        return self._modality

    def immutable_features(self):
        return list(self._immutable)

    def _build_feature_scaler(self):
        feat_names = self._adapter.feature_names()
        feat_min = np.zeros(len(feat_names), dtype=np.float64)
        feat_max = np.zeros(len(feat_names), dtype=np.float64)
        for i, feat in enumerate(feat_names):
            b = self._cfg.cf_bounds_for(feat)
            lo, hi = b if b is not None else (0.0, 1.0)
            feat_min[i], feat_max[i] = lo, hi
        span = np.where(feat_max - feat_min < 1e-9, 1.0, feat_max - feat_min)
        feat_max = feat_min + span
        scaler = MinMaxScaler()
        scaler.fit(np.stack([feat_min, feat_max]))
        return scaler, feat_min, feat_max

    def _train_generator(self):
        x_train = self._adapter.training_data()
        feat_names = self._adapter.feature_names()
        data_raw = x_train[feat_names].fillna(0).values.astype(np.float64)
        if len(data_raw) > self._max_train_samples:
            data_raw = data_raw[np.random.choice(len(data_raw), self._max_train_samples, replace=False)]

        self._scaler, feat_min, feat_max = self._build_feature_scaler()
        continuous_feats = [i for i, f in enumerate(feat_names) if self._cfg.constraints.get(f, {}).get("type") in ("continuous", "sentinel_continuous")]
        if continuous_feats:
            keep_mask = np.ones(len(data_raw), dtype=bool)
            for i in continuous_feats:
                keep_mask &= (data_raw[:, i] >= feat_min[i]) & (data_raw[:, i] <= feat_max[i] * 1.5)
            if keep_mask.sum() >= 100:
                data_raw = data_raw[keep_mask]
        for i in continuous_feats:
            data_raw[:, i] = np.clip(data_raw[:, i], feat_min[i], feat_max[i])

        data_norm = np.clip(self._scaler.transform(data_raw), 0.0, 1.0).astype(np.float32)
        risks = self._predict_fn(data_raw)
        self._risk_min, self._risk_max = float(risks.min()), float(risks.max())
        self._risk_q25, self._risk_median, self._risk_q75 = float(np.percentile(risks, 25)), float(np.median(risks)), float(np.percentile(risks, 75))
        iqr = self._risk_q75 - self._risk_q25

        if self._higher_is_better:
            anchor_low = self._risk_median + 0.25 * iqr
            anchor_high = min(self._risk_max - 0.1 * iqr, self._risk_median + 1.0 * iqr)
        else:
            anchor_high = self._risk_median - 0.25 * iqr
            anchor_low = max(self._risk_min + 0.1 * iqr, self._risk_median - 1.0 * iqr)
        anchor_mask = (risks >= anchor_low) & (risks <= anchor_high)
        anchor_pool_norm = data_norm[anchor_mask]
        if len(anchor_pool_norm) < 10:
            anchor_mask = risks >= self._risk_median if self._higher_is_better else risks <= self._risk_median
            anchor_pool_norm = data_norm[anchor_mask]
        anchor_pool_norm_t = torch.FloatTensor(anchor_pool_norm).to(self._device)

        immutable_idx = torch.LongTensor([i for i, f in enumerate(feat_names) if f in self._immutable]).to(self._device)
        mutable_mask = torch.ones(data_norm.shape[1], dtype=torch.bool)
        if len(immutable_idx) > 0:
            mutable_mask[immutable_idx] = False
        mutable_mask = mutable_mask.to(self._device)

        self._generator = _CFGenerator(data_norm.shape[1], self._hidden_dim).to(self._device)
        optimizer = optim.Adam(self._generator.parameters(), lr=self._lr)
        loader = DataLoader(TensorDataset(torch.FloatTensor(data_norm)), batch_size=self._batch_size, shuffle=True)

        for _ in range(self._epochs):
            for (batch,) in loader:
                batch = batch.to(self._device)
                cfs = self._generator(batch)
                if len(immutable_idx) > 0:
                    cfs = cfs.clone()
                    cfs[:, immutable_idx] = batch[:, immutable_idx]
                with torch.no_grad():
                    diff = (batch.unsqueeze(1) - anchor_pool_norm_t.unsqueeze(0))[:, :, mutable_mask]
                    nearest = diff.pow(2).sum(dim=-1).argmin(dim=-1)
                    anchors = anchor_pool_norm_t[nearest]
                    if len(immutable_idx) > 0:
                        anchors = anchors.clone()
                        anchors[:, immutable_idx] = batch[:, immutable_idx]
                loss = (self._validity_weight * ((cfs - anchors) ** 2).sum(dim=-1).mean()
                        + self._proximity_weight * ((cfs - batch) ** 2).sum(dim=-1).mean()
                        + self._sparsity_weight * (cfs - batch).abs().sum(dim=-1).mean())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self._generator.eval()

    def _snap_vector(self, x_raw, x_query):
        feat_names = self._adapter.feature_names()
        df = pd.DataFrame([x_raw], columns=feat_names)
        df = snap_categoricals(df, self._categorical_values)
        snapped, _, rejected = snap_with_constraints(df.values[0], x_query, feat_names, self._cfg.constraints)
        return snapped.astype(np.float64), rejected

    def generate_counterfactual(self, patient_features, target_outcome=None):
        if self._generator is None:
            self._train_generator()
        feat_names = self._adapter.feature_names()
        x_orig_raw = patient_features[feat_names].fillna(0).values[0].astype(np.float32)
        x_orig_f64 = x_orig_raw.astype(np.float64)
        current_risk = float(self._predict_fn(x_orig_f64.reshape(1, -1))[0])
        iqr = self._risk_q75 - self._risk_q25

        if self._higher_is_better and current_risk >= self._risk_max - 0.25 * iqr:
            return None
        if not self._higher_is_better and current_risk <= self._risk_min + 0.25 * iqr:
            return None

        x_orig_norm = self._scaler.transform(x_orig_raw.reshape(1, -1)).astype(np.float32)[0]
        with torch.no_grad():
            cf_norm = self._generator(torch.FloatTensor(x_orig_norm).unsqueeze(0).to(self._device)).squeeze(0).cpu().numpy()
        cf_raw = self._scaler.inverse_transform(cf_norm.reshape(1, -1)).astype(np.float64)[0]
        cf_raw, rejected = self._snap_vector(cf_raw, x_orig_f64)
        if rejected:
            return None

        cf_df = pd.DataFrame([cf_raw], columns=feat_names)
        cf_df = clip_to_ranges(cf_df, self._ranges)
        cf_df = enforce_immutables(patient_features[feat_names].head(1), cf_df, self._immutable)

        if self._direction_priors:
            cf_corrected, n_reset = apply_direction_priors(cf_df.values[0], x_orig_f64, feat_names, self._direction_priors)
            if n_reset > 0:
                if np.all(np.abs(cf_corrected - x_orig_f64) < 1e-8):
                    return None
                cf_df = pd.DataFrame([cf_corrected], columns=feat_names)

        cf_risk = float(self._predict_fn(cf_df.values.astype(np.float64))[0])
        improved = (self._higher_is_better and cf_risk > current_risk) or (not self._higher_is_better and cf_risk < current_risk)
        if not improved:
            return None
        return cf_df

    def explain(self, patient_features, patient_id=None):
        feat_names = self._adapter.feature_names()
        query = patient_features[feat_names].fillna(0).head(1)
        cf = self.generate_counterfactual(patient_features)
        if cf is not None:
            content, raw_query, raw_cf = compute_content_deltas(self._adapter, query, cf, feat_names)
        else:
            content, raw_query, raw_cf = {}, None, None
        orig_pred = extract_scalar_prediction(self._adapter.predict(query, time_horizon=self._time_horizon))
        cf_pred = extract_scalar_prediction(self._adapter.predict(cf.head(1), time_horizon=self._time_horizon)) if cf is not None else None
        return self._make_explanation(patient_id, content=content, method="CounterNet",
            counterfactual_df=cf, original_prediction=orig_pred, cf_prediction=cf_pred,
            raw_query=raw_query, raw_cf=raw_cf, higher_is_better=self._higher_is_better)

    def sanity_check(self, held_out_df):
        return _run_sanity_checks(self, held_out_df)