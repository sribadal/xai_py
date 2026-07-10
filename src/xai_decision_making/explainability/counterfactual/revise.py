"""
REVISE counterfactual explainer via VAE latent gradient descent.

Trains a tabular VAE on the training distribution, encodes the query
patient, and performs gradient descent in the latent space toward a
target risk value. Decoded candidates are snapped to plausibility
constraints with sentinel-aware empirical z-score bounds.
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
    _run_sanity_checks, apply_direction_priors, clip_to_ranges, compute_cf_confidence,
    compute_content_deltas, enforce_immutables, extract_scalar_prediction,
    resolve_direction_priors, risk_predict_fn, snap_categoricals,
)
from xai_decision_making.explainability.counterfactual._plausibility import (
    build_plausibility_config, condor_target_t_norm, snap_with_constraints,
)

registry = ExplainerRegistry()

class _TabularVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decode(z), mu, logvar

def _vae_loss(recon, x, mu, logvar, beta=1.0):
    return nn.functional.mse_loss(recon, x, reduction="sum") + beta * (-0.5) * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

class _DiffPredictWrapper(nn.Module):
    def __init__(self, rsf_predict_fn, vae, scaler_min, scaler_scale):
        super().__init__()
        self.vae = vae
        self.register_buffer("s_min", torch.FloatTensor(scaler_min))
        self.register_buffer("s_scale", torch.FloatTensor(scaler_scale))
        self._rsf_predict_fn = rsf_predict_fn
        self._last_predictions = None

    def forward(self, z):
        x_norm = self.vae.decode(z)
        x_raw = x_norm * self.s_scale + self.s_min
        with torch.no_grad():
            x_np = x_raw.detach().cpu().numpy().astype(np.float64)
            preds = self._rsf_predict_fn(x_np)
            self._last_predictions = preds
        preds_t = torch.FloatTensor(preds).to(z.device)
        return preds_t + (x_raw.sum(dim=-1) * 0.0)

@registry.register
class REVISEExplainer(CounterfactualExplainer):
    _modality = ExplainerModality.COUNTERFACTUAL
    _target = ModelTarget.SURVIVAL

    def __init__(self, model_adapter, *, latent_dim=8, hidden_dim=128, epochs=80,
                 batch_size=64, lr_vae=1e-3, beta=1.0, lr_cf=0.05, n_gradient_steps=200,
                 target_weight=1.0, proximity_weight=0.3, n_restarts=3, immutable=None,
                 direction_priors=None, time_horizon=None):
        super().__init__(model_adapter)
        self._latent_dim = latent_dim
        self._hidden_dim = hidden_dim
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr_vae = lr_vae
        self._beta = beta
        self._lr_cf = lr_cf
        self._n_gradient_steps = n_gradient_steps
        self._target_weight = target_weight
        self._proximity_weight = proximity_weight
        self._n_restarts = n_restarts
        self._time_horizon = time_horizon
        self._predict_fn = risk_predict_fn(model_adapter, time_horizon)
        self._cfg = build_plausibility_config(model_adapter, extra_immutable=immutable)
        self._ranges = self._cfg.ranges
        self._immutable = self._cfg.immutable
        self._categorical_values = self._cfg.categorical_values
        self._direction_priors = resolve_direction_priors(model_adapter, override=direction_priors)
        self._higher_is_better = getattr(model_adapter, "higher_is_better", False)
        self._vae = self._scaler = self._diff_wrapper = None
        self._risk_median = self._risk_q25 = self._risk_q75 = None
        self._device = torch.device("cpu")

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

    def _train_vae(self):
        x_train = self._adapter.training_data()
        feat_names = self._adapter.feature_names()
        data_raw = x_train[feat_names].fillna(0).values.astype(np.float64)
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
        self._risk_median = float(np.median(risks))
        self._risk_q25 = float(np.percentile(risks, 25))
        self._risk_q75 = float(np.percentile(risks, 75))

        self._vae = _TabularVAE(data_norm.shape[1], self._latent_dim, self._hidden_dim).to(self._device)
        loader = DataLoader(TensorDataset(torch.FloatTensor(data_norm)), batch_size=self._batch_size, shuffle=True)
        optimizer = optim.Adam(self._vae.parameters(), lr=self._lr_vae)
        self._vae.train()
        for _ in range(self._epochs):
            for (batch,) in loader:
                batch = batch.to(self._device)
                recon, mu, logvar = self._vae(batch)
                loss = _vae_loss(recon, batch, mu, logvar, self._beta)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self._vae.eval()

        s_min = self._scaler.data_min_.astype(np.float32)
        s_scale = self._scaler.data_range_.astype(np.float32)
        s_scale = np.where(s_scale < 1e-12, 1.0, s_scale)
        self._diff_wrapper = _DiffPredictWrapper(self._predict_fn, self._vae, s_min, s_scale).to(self._device)

    def _snap_vector(self, x_raw, x_query, target_t_norm=None):
        feat_names = self._adapter.feature_names()
        df = pd.DataFrame([x_raw], columns=feat_names)
        df = snap_categoricals(df, self._categorical_values)
        snapped, _, rejected = snap_with_constraints(df.values[0], x_query, feat_names, self._cfg.constraints, target_t_norm=target_t_norm)
        return snapped.astype(np.float64), rejected

    def _search_one_restart(self, z_init, x_orig_f64, feat_names, immutable_idx, threshold, go_lower, range_denom, target_t_norm):
        z = z_init.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([z], lr=self._lr_cf)
        target_t = torch.tensor(threshold, device=self._device)
        x_orig_t = torch.FloatTensor(self._scaler.transform(x_orig_f64.reshape(1, -1)).astype(np.float32)).to(self._device).squeeze(0)
        best_cf_raw, best_distance = None, float("inf")

        for _ in range(self._n_gradient_steps):
            optimizer.zero_grad()
            pred = self._diff_wrapper(z.unsqueeze(0)).squeeze()
            x_recon = self._vae.decode(z)
            proximity = ((x_recon - x_orig_t) ** 2).sum()
            if go_lower:
                target_loss = torch.clamp(pred - target_t, min=0.0) ** 2
            else:
                target_loss = torch.clamp(target_t - pred, min=0.0) ** 2
            loss = self._target_weight * target_loss + self._proximity_weight * proximity
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                x_norm = self._vae.decode(z).cpu().numpy()[0]
                x_raw = self._scaler.inverse_transform(x_norm.reshape(1, -1)).astype(np.float64)[0]
                if len(immutable_idx) > 0:
                    x_raw[immutable_idx] = x_orig_f64[immutable_idx]
                x_raw, rejected = self._snap_vector(x_raw, x_orig_f64, target_t_norm=target_t_norm)
                if rejected:
                    continue
                real_risk = float(self._predict_fn(x_raw.reshape(1, -1))[0])
                crossed = (go_lower and real_risk < threshold) or (not go_lower and real_risk >= threshold)
                if crossed:
                    dist = np.sum(((x_raw - x_orig_f64) / range_denom) ** 2)
                    if dist < best_distance:
                        best_distance = dist
                        best_cf_raw = x_raw.copy()
        return best_cf_raw

    def generate_counterfactual(self, patient_features, target_outcome=None):
        if self._vae is None:
            self._train_vae()
        feat_names = self._adapter.feature_names()
        target_t_norm = condor_target_t_norm(self._adapter, self._time_horizon)
        x_orig_raw = patient_features[feat_names].fillna(0).values[0].astype(np.float64)
        current_risk = float(self._predict_fn(x_orig_raw.reshape(1, -1))[0])
        threshold = float(target_outcome) if target_outcome is not None else self._risk_median
        go_lower = not self._higher_is_better
        immutable_idx = np.array([i for i, f in enumerate(feat_names) if f in self._immutable], dtype=int)
        range_denom = np.maximum(self._scaler.data_range_, 1e-12)

        x_norm = self._scaler.transform(x_orig_raw.reshape(1, -1)).astype(np.float32)
        x_t = torch.FloatTensor(x_norm).to(self._device)
        with torch.no_grad():
            mu0, logvar0 = self._vae.encode(x_t)

        best_cf, best_dist = None, float("inf")
        for r in range(self._n_restarts):
            if r == 0:
                z_init = mu0.squeeze(0).clone()
            else:
                z_init = mu0.squeeze(0) + torch.randn(self._latent_dim, device=self._device) * 0.5
            cf_candidate = self._search_one_restart(z_init, x_orig_raw, feat_names, immutable_idx, threshold, go_lower, range_denom, target_t_norm)
            if cf_candidate is not None:
                d = np.sum(((cf_candidate - x_orig_raw) / range_denom) ** 2)
                if d < best_dist:
                    best_dist = d
                    best_cf = cf_candidate

        if best_cf is None:
            return None

        cf_df = pd.DataFrame([best_cf], columns=feat_names)
        cf_df = clip_to_ranges(cf_df, self._ranges)
        cf_df = enforce_immutables(patient_features[feat_names].head(1), cf_df, self._immutable)

        if self._direction_priors:
            cf_corrected, n_reset = apply_direction_priors(cf_df.values[0], x_orig_raw, feat_names, self._direction_priors)
            if n_reset > 0:
                if np.all(np.abs(cf_corrected - x_orig_raw) < 1e-8):
                    return None
                cf_df = pd.DataFrame([cf_corrected], columns=feat_names)

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
        confidence_metrics = compute_cf_confidence(raw_query, raw_cf, orig_pred, cf_pred, higher_is_better=self._higher_is_better)
        return self._make_explanation(patient_id, content=content, method="REVISE",
            counterfactual_df=cf, original_prediction=orig_pred, cf_prediction=cf_pred,
            raw_query=raw_query, raw_cf=raw_cf, higher_is_better=self._higher_is_better, confidence_metrics=confidence_metrics)

    def sanity_check(self, held_out_df):
        return _run_sanity_checks(self, held_out_df)