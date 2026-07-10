"""
REVISE counterfactual explainer via VAE latent gradient descent.

Trains a tabular VAE on the training distribution, encodes the query
patient, and performs zeroth-order probe-guided gradient descent in the
latent space toward a target risk value. Decoded candidates are snapped
to plausibility constraints with sentinel-aware empirical z-score bounds.
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
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

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
class REVISEExplainer(CounterfactualExplainer):
    _modality = ExplainerModality.COUNTERFACTUAL
    _target = ModelTarget.SURVIVAL

    def __init__(
        self,
        model_adapter,
        *,
        latent_dim=8,
        hidden_dim=128,
        epochs=100,
        batch_size=64,
        vae_lr=1e-3,
        beta=1.0,
        n_search_steps=300,
        search_lr=0.05,
        proximity_weight=0.5,
        target_iqr_fraction=0.5,
        immutable=None,
        direction_priors=None,
        time_horizon=None,
        horizon_saturation_threshold=99.0,
    ):
        super().__init__(model_adapter)
        self._latent_dim = latent_dim
        self._hidden_dim = hidden_dim
        self._epochs = epochs
        self._batch_size = batch_size
        self._vae_lr = vae_lr
        self._beta = beta
        self._n_search_steps = n_search_steps
        self._search_lr = search_lr
        self._proximity_weight = proximity_weight
        self._target_iqr_fraction = target_iqr_fraction
        self._time_horizon = time_horizon
        self._horizon_saturation_threshold = horizon_saturation_threshold

        bounds = getattr(model_adapter, "prediction_saturation_bounds", None)
        if bounds is not None:
            self._sat_floor, self._sat_ceiling = float(bounds[0]), float(bounds[1])
        else:
            t = float(horizon_saturation_threshold)
            self._sat_ceiling = t
            self._sat_floor = 100.0 - t

        self._predict_fn = risk_predict_fn(model_adapter, time_horizon)
        self._cfg = build_plausibility_config(
            model_adapter, extra_immutable=immutable,
        )
        self._ranges = self._cfg.ranges
        self._immutable = self._cfg.immutable
        self._categorical_values = self._cfg.categorical_values
        self._direction_priors = resolve_direction_priors(
            model_adapter, override=direction_priors,
        )
        self._higher_is_better = getattr(model_adapter, "higher_is_better", False)

        self._vae = None
        self._scaler = None
        self._device = torch.device("cpu")

        self._risk_min = None
        self._risk_max = None
        self._risk_q25 = None
        self._risk_median = None
        self._risk_q75 = None
        self._zscored_feats = set()

    @property
    def modality(self):
        return self._modality

    def immutable_features(self):
        return list(self._immutable)

    def _is_saturated_at_horizon(self, patient_features):
        if self._time_horizon is None:
            return False, None
        try:
            pred = extract_scalar_prediction(
                self._adapter.predict(
                    patient_features.head(1),
                    time_horizon=self._time_horizon,
                )
            )
        except Exception:
            return False, None
        if self._higher_is_better and pred >= self._sat_ceiling:
            return True, pred
        if (not self._higher_is_better) and pred <= self._sat_floor:
            return True, pred
        return False, pred

    def _build_feature_scaler(self):
        feat_names = self._adapter.feature_names()
        feat_min = np.zeros(len(feat_names))
        feat_max = np.zeros(len(feat_names))
        zscored = set()
        for i, feat in enumerate(feat_names):
            b = self._cfg.cf_bounds_for(feat)
            if b is not None:
                lo, hi = b
            else:
                lo, hi = 0.0, 1.0
            feat_min[i] = lo
            feat_max[i] = hi
            rule = self._cfg.constraints.get(feat)
            if rule and rule.get("zscore_range") is not None:
                zscored.add(feat)
        span = feat_max - feat_min
        span = np.where(span < 1e-9, 1.0, span)
        feat_max = feat_min + span

        scaler = MinMaxScaler()
        scaler.fit(np.stack([feat_min, feat_max]))
        self._zscored_feats = zscored
        return scaler, feat_min, feat_max

    def _train_vae(self):
        x_train = self._adapter.training_data()
        feat_names = self._adapter.feature_names()
        data_raw = x_train[feat_names].fillna(0).values

        self._scaler, feat_min, feat_max = self._build_feature_scaler()
        self._feat_min = feat_min
        self._feat_max = feat_max

        continuous_feats = [
            i for i, f in enumerate(feat_names)
            if self._cfg.constraints.get(f, {}).get("type") in (
                "continuous", "sentinel_continuous",
            )
        ]
        if continuous_feats:
            keep_mask = np.ones(len(data_raw), dtype=bool)
            for i in continuous_feats:
                lo, hi = feat_min[i], feat_max[i]
                keep_mask &= (data_raw[:, i] >= lo) & (data_raw[:, i] <= hi * 1.5)
            if keep_mask.sum() >= 100:
                data_raw = data_raw[keep_mask]

        for i in continuous_feats:
            lo, hi = feat_min[i], feat_max[i]
            data_raw[:, i] = np.clip(data_raw[:, i], lo, hi)

        data_norm = np.clip(
            self._scaler.transform(data_raw), 0.0, 1.0,
        ).astype(np.float32)
        input_dim = data_norm.shape[1]

        risks = self._predict_fn(data_raw)
        self._risk_min = float(risks.min())
        self._risk_max = float(risks.max())
        self._risk_q25 = float(np.percentile(risks, 25))
        self._risk_median = float(np.median(risks))
        self._risk_q75 = float(np.percentile(risks, 75))

        self._vae = _TabularVAE(
            input_dim, self._latent_dim, self._hidden_dim,
        ).to(self._device)

        loader = DataLoader(
            TensorDataset(torch.FloatTensor(data_norm)),
            batch_size=self._batch_size,
            shuffle=True,
        )
        optimizer = optim.Adam(self._vae.parameters(), lr=self._vae_lr)

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

    def _get_target_value(self, current_risk):
        iqr = self._risk_q75 - self._risk_q25
        delta = self._target_iqr_fraction * iqr

        if self._higher_is_better:
            if current_risk >= self._risk_max - 0.25 * iqr:
                return None
            target = current_risk + delta
            ceiling_bound = self._risk_max - 0.1 * iqr
            target = min(target, ceiling_bound)
            target = max(target, current_risk + 1e-3)
            return float(target)
        else:
            if current_risk <= self._risk_min + 0.25 * iqr:
                return None
            target = current_risk - delta
            floor_bound = self._risk_min + 0.1 * iqr
            target = max(target, floor_bound)
            target = min(target, current_risk - 1e-3)
            return float(target)

    def _snap_vector(self, x_raw, x_query, target_t_norm=None):
        feat_names = self._adapter.feature_names()
        df = pd.DataFrame([x_raw], columns=feat_names)
        df = snap_categoricals(df, self._categorical_values)
        snapped, _, rejected = snap_with_constraints(
            df.values[0], x_query, feat_names, self._cfg.constraints,
            target_t_norm=target_t_norm,
        )
        return snapped, rejected

    def generate_counterfactual(self, patient_features, target_outcome=None):
        if self._vae is None:
            self._train_vae()

        feat_names = self._adapter.feature_names()

        saturated, horizon_pred = self._is_saturated_at_horizon(patient_features)
        if saturated:
            return None

        target_t_norm = condor_target_t_norm(self._adapter, self._time_horizon)

        x_orig_raw = patient_features[feat_names].fillna(0).values[0]
        x_orig_f64 = x_orig_raw.astype(np.float64)
        current_risk = float(self._predict_fn(x_orig_f64.reshape(1, -1))[0])

        if target_outcome is not None:
            y_target = float(target_outcome)
        else:
            y_target = self._get_target_value(current_risk)
            if y_target is None:
                return None

        x_orig_norm = self._scaler.transform(
            x_orig_raw.reshape(1, -1)
        ).astype(np.float32)[0]
        x_orig_norm_t = torch.FloatTensor(x_orig_norm).unsqueeze(0).to(self._device)
        with torch.no_grad():
            mu, _ = self._vae.encode(x_orig_norm_t)

        z_prime = mu.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([z_prime], lr=self._search_lr)

        immutable_idx = np.array(
            [i for i, f in enumerate(feat_names) if f in self._immutable],
            dtype=int,
        )
        x_orig_norm_full = torch.FloatTensor(x_orig_norm).to(self._device)

        best_cf_raw = None
        best_cf_pre = None
        best_score = np.inf
        n_rejected = 0

        for step in range(self._n_search_steps):
            x_norm = self._vae.decode(z_prime).squeeze(0)

            recon_loss = self._proximity_weight * torch.sum(
                (x_norm - x_orig_norm_full) ** 2
            )

            with torch.no_grad():
                x_norm_clipped = np.clip(
                    x_norm.detach().cpu().numpy(), 0.0, 1.0,
                )
                x_raw_pre = self._scaler.inverse_transform(
                    x_norm_clipped.reshape(1, -1)
                ).astype(np.float64)[0]
                if len(immutable_idx) > 0:
                    x_raw_pre[immutable_idx] = x_orig_f64[immutable_idx]
                x_raw_snap, rejected = self._snap_vector(
                    x_raw_pre, x_orig_f64, target_t_norm=target_t_norm,
                )

                if rejected:
                    n_rejected += 1
                    optimizer.zero_grad()
                    recon_loss.backward()
                    optimizer.step()
                    continue

                pred_now = float(self._predict_fn(x_raw_snap.reshape(1, -1))[0])

                score = abs(pred_now - y_target)
                if score < best_score:
                    best_score = score
                    best_cf_raw = x_raw_snap.copy()
                    best_cf_pre = x_raw_pre.copy()

                probe_count = 3
                probe_eps = 0.1
                best_probe_dz = None
                best_probe_score = score
                for _ in range(probe_count):
                    dz = torch.randn_like(z_prime) * probe_eps
                    z_probe = (z_prime + dz).detach()
                    x_probe_norm = self._vae.decode(z_probe).squeeze(0).cpu().numpy()
                    x_probe_norm = np.clip(x_probe_norm, 0.0, 1.0)
                    x_probe_raw = self._scaler.inverse_transform(
                        x_probe_norm.reshape(1, -1)
                    ).astype(np.float64)[0]
                    if len(immutable_idx) > 0:
                        x_probe_raw[immutable_idx] = x_orig_f64[immutable_idx]
                    x_probe_snap, probe_rejected = self._snap_vector(
                        x_probe_raw, x_orig_f64, target_t_norm=target_t_norm,
                    )
                    if probe_rejected:
                        continue
                    pred_probe = float(self._predict_fn(x_probe_snap.reshape(1, -1))[0])
                    probe_score = abs(pred_probe - y_target)
                    if probe_score < best_probe_score:
                        best_probe_score = probe_score
                        best_probe_dz = dz

            optimizer.zero_grad()
            recon_loss.backward()
            if best_probe_dz is not None:
                with torch.no_grad():
                    z_prime.grad = z_prime.grad - best_probe_dz / probe_eps
            optimizer.step()

        if best_cf_raw is None:
            return None

        cf_df = pd.DataFrame([best_cf_raw], columns=feat_names)
        cf_df = clip_to_ranges(cf_df, self._ranges)
        cf_df = enforce_immutables(
            patient_features[feat_names].head(1),
            cf_df,
            self._immutable,
        )

        if self._direction_priors:
            cf_corrected, n_reset = apply_direction_priors(
                cf_df.values[0],
                x_orig_f64,
                feat_names,
                self._direction_priors,
            )
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
            content, raw_query, raw_cf = compute_content_deltas(
                self._adapter, query, cf, feat_names
            )
        else:
            content, raw_query, raw_cf = {}, None, None

        orig_pred = extract_scalar_prediction(
            self._adapter.predict(query, time_horizon=self._time_horizon)
        )
        cf_pred = (
            extract_scalar_prediction(
                self._adapter.predict(cf.head(1), time_horizon=self._time_horizon)
            )
            if cf is not None
            else None
        )

        confidence_metrics = compute_cf_confidence(
            raw_query, raw_cf, orig_pred, cf_pred,
            higher_is_better=self._higher_is_better,
        )

        return self._make_explanation(
            patient_id,
            content=content,
            method="REVISE",
            counterfactual_df=cf,
            original_prediction=orig_pred,
            cf_prediction=cf_pred,
            raw_query=raw_query,
            raw_cf=raw_cf,
            higher_is_better=self._higher_is_better,
            confidence_metrics=confidence_metrics,
        )

    def sanity_check(self, held_out_df):
        return _run_sanity_checks(self, held_out_df)