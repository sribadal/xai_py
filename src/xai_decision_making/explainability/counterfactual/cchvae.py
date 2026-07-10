"""
C-CHVAE counterfactual explainer.

Generates counterfactuals via latent-space perturbation of a tabular
VAE trained on the model's feature space, with plausibility snapping,
immutability enforcement, and biological direction priors.
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
    AggregatedSanityResult,
    CounterfactualExplainer,
    ExplainerModality,
    ExplainerRegistry,
    Explanation,
    ModelTarget,
)
from xai_decision_making.explainability.counterfactual._base import (
    _run_sanity_checks,
    apply_direction_priors,
    clip_to_ranges,
    compute_cf_confidence,
    compute_content_deltas,
    enforce_immutables,
    extract_scalar_prediction,
    resolve_direction_priors,
    risk_predict_fn,
    snap_categoricals,
)
from xai_decision_making.explainability.counterfactual._plausibility import (
    build_plausibility_config,
    condor_target_t_norm,
    snap_with_constraints,
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
    recon_loss = nn.functional.mse_loss(recon, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld

@registry.register
class CCHVAEExplainer(CounterfactualExplainer):
    _modality = ExplainerModality.COUNTERFACTUAL
    _target = ModelTarget.SURVIVAL

    def __init__(self, model_adapter, *, latent_dim=8, hidden_dim=128, epochs=100,
                 batch_size=64, lr=1e-3, beta=1.0, n_search_steps=200, step_size=0.05,
                 immutable=None, direction_priors=None, time_horizon=None):
        super().__init__(model_adapter)
        self._latent_dim = latent_dim
        self._hidden_dim = hidden_dim
        self._epochs = epochs
        self._batch_size = batch_size
        self._lr = lr
        self._beta = beta
        self._n_search_steps = n_search_steps
        self._step_size = step_size
        self._time_horizon = time_horizon
        self._predict_fn = risk_predict_fn(model_adapter, time_horizon)
        self._cfg = build_plausibility_config(model_adapter, extra_immutable=immutable)
        self._ranges = self._cfg.ranges
        self._immutable = self._cfg.immutable
        self._categorical_values = self._cfg.categorical_values
        self._direction_priors = resolve_direction_priors(model_adapter, override=direction_priors)
        self._higher_is_better = getattr(model_adapter, "higher_is_better", False)
        self._vae = None
        self._scaler = None
        self._device = torch.device("cpu")
        self._risk_median = None

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
            feat_min[i] = lo
            feat_max[i] = hi
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

        continuous_feats = [
            i for i, f in enumerate(feat_names)
            if self._cfg.constraints.get(f, {}).get("type") in ("continuous", "sentinel_continuous")
        ]
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

        self._vae = _TabularVAE(data_norm.shape[1], self._latent_dim, self._hidden_dim).to(self._device)
        loader = DataLoader(TensorDataset(torch.FloatTensor(data_norm)), batch_size=self._batch_size, shuffle=True)
        optimizer = optim.Adam(self._vae.parameters(), lr=self._lr)

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

    def _snap_vector(self, x_raw, x_query, target_t_norm=None):
        feat_names = self._adapter.feature_names()
        df = pd.DataFrame([x_raw], columns=feat_names)
        df = snap_categoricals(df, self._categorical_values)
        snapped, _, rejected = snap_with_constraints(df.values[0], x_query, feat_names, self._cfg.constraints, target_t_norm=target_t_norm)
        return snapped.astype(np.float64), rejected

    def generate_counterfactual(self, patient_features, target_outcome=None):
        if self._vae is None:
            self._train_vae()

        feat_names = self._adapter.feature_names()
        target_t_norm = condor_target_t_norm(self._adapter, self._time_horizon)
        x_orig_raw = patient_features[feat_names].fillna(0).values[0].astype(np.float32)
        x_orig_norm = self._scaler.transform(x_orig_raw.reshape(1, -1)).astype(np.float32)[0]
        x_orig_f64 = x_orig_raw.astype(np.float64)
        current_risk = float(self._predict_fn(x_orig_f64.reshape(1, -1))[0])
        threshold = float(target_outcome) if target_outcome is not None else self._risk_median
        go_lower = not self._higher_is_better

        x_tensor = torch.FloatTensor(x_orig_norm).unsqueeze(0).to(self._device)
        with torch.no_grad():
            mu, _ = self._vae.encode(x_tensor)
        z = mu.clone()

        immutable_idx = np.array([i for i, f in enumerate(feat_names) if f in self._immutable], dtype=int)
        best_cf_raw = None
        best_distance = np.inf
        current_z_risk = current_risk
        range_denom = np.maximum(self._scaler.data_range_, 1e-12)

        for _ in range(self._n_search_steps):
            z_candidate = z + torch.randn_like(z) * self._step_size
            with torch.no_grad():
                x_cand_norm = self._vae.decode(z_candidate).cpu().numpy()[0]
            x_cand_pre = self._scaler.inverse_transform(x_cand_norm.reshape(1, -1)).astype(np.float64)[0]
            if len(immutable_idx) > 0:
                x_cand_pre[immutable_idx] = x_orig_f64[immutable_idx]
            x_cand_post, rejected = self._snap_vector(x_cand_pre, x_orig_f64, target_t_norm=target_t_norm)
            if rejected:
                continue

            cand_risk = float(self._predict_fn(x_cand_post.reshape(1, -1))[0])
            crossed = (go_lower and cand_risk < threshold) or (not go_lower and cand_risk >= threshold)
            if crossed:
                dist = np.sum(((x_cand_post - x_orig_f64) / range_denom) ** 2)
                if dist < best_distance:
                    best_distance = dist
                    best_cf_raw = x_cand_post.copy()

            moved_towards = (go_lower and cand_risk < current_z_risk) or (not go_lower and cand_risk > current_z_risk)
            if moved_towards:
                z = z_candidate.clone()
                current_z_risk = cand_risk

        if best_cf_raw is None:
            return None

        cf_df = pd.DataFrame([best_cf_raw], columns=feat_names)
        cf_df = clip_to_ranges(cf_df, self._ranges)
        cf_df = enforce_immutables(patient_features[feat_names].head(1), cf_df, self._immutable)

        if self._direction_priors:
            cf_corrected, n_reset = apply_direction_priors(cf_df.values[0], x_orig_f64, feat_names, self._direction_priors)
            if n_reset > 0:
                if np.all(np.abs(cf_corrected - x_orig_f64) < 1e-8):
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

        return self._make_explanation(
            patient_id, content=content, method="C-CHVAE",
            counterfactual_df=cf, original_prediction=orig_pred, cf_prediction=cf_pred,
            raw_query=raw_query, raw_cf=raw_cf, higher_is_better=self._higher_is_better,
            confidence_metrics=confidence_metrics,
        )

    def sanity_check(self, held_out_df):
        return _run_sanity_checks(self, held_out_df)