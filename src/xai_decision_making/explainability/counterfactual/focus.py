"""
FOCUS counterfactual explainer via differentiable soft trees.

Builds a soft-tree approximation of the Random Survival Forest and
optimises counterfactuals through gradient descent on the differentiable
proxy, then verifies against the real model with plausibility snapping.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

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
    clip_to_ranges,
    compute_content_deltas,
    enforce_immutables,
    extract_scalar_prediction,
    risk_predict_fn,
)
from xai_decision_making.explainability.counterfactual._plausibility import (
    build_plausibility_config,
    snap_and_verify,
)

registry = ExplainerRegistry()


def _extract_trees(rsf_model):
    trees = []
    for est in rsf_model.estimators_:
        t = est.tree_
        n_nodes = t.value.shape[0]
        leaf_risks = t.value.reshape(n_nodes, -1).sum(axis=1)
        trees.append({
            "feature": t.feature.copy(),
            "threshold": t.threshold.copy(),
            "children_left": t.children_left.copy(),
            "children_right": t.children_right.copy(),
            "value": leaf_risks,
        })
    return trees


class _SoftTree(nn.Module):
    def __init__(self, tree_dict, sigma=1.0):
        super().__init__()
        self.sigma = sigma

        feat = tree_dict["feature"]
        thresh = tree_dict["threshold"]
        left = tree_dict["children_left"]
        right = tree_dict["children_right"]
        values = tree_dict["value"].ravel()

        self.n_nodes = len(feat)
        self.register_buffer("feature_idx", torch.LongTensor(feat))
        self.register_buffer("threshold", torch.FloatTensor(thresh))
        self.register_buffer("children_left", torch.LongTensor(left))
        self.register_buffer("children_right", torch.LongTensor(right))
        self.register_buffer("leaf_values", torch.FloatTensor(values))
        self.register_buffer("is_leaf", torch.BoolTensor(left == -1))

    def forward(self, x):
        batch = x.shape[0]
        device = x.device

        node_probs = [None] * self.n_nodes
        node_probs[0] = torch.ones(batch, device=device)

        for nid in range(self.n_nodes):
            if self.is_leaf[nid]:
                continue
            parent_prob = node_probs[nid]
            if parent_prob is None:
                continue

            feat_idx = self.feature_idx[nid].item()
            left_child = self.children_left[nid].item()
            right_child = self.children_right[nid].item()

            p_left = torch.sigmoid(
                (self.threshold[nid] - x[:, feat_idx]) / self.sigma
            )
            left_contrib = parent_prob * p_left
            right_contrib = parent_prob * (1.0 - p_left)

            if node_probs[left_child] is None:
                node_probs[left_child] = left_contrib
            else:
                node_probs[left_child] = node_probs[left_child] + left_contrib

            if node_probs[right_child] is None:
                node_probs[right_child] = right_contrib
            else:
                node_probs[right_child] = node_probs[right_child] + right_contrib

        pred = torch.zeros(batch, device=device)
        for nid in range(self.n_nodes):
            if self.is_leaf[nid] and node_probs[nid] is not None:
                pred = pred + node_probs[nid] * self.leaf_values[nid]
        return pred


class _SoftForest(nn.Module):
    def __init__(self, tree_dicts, sigma=1.0):
        super().__init__()
        self.trees = nn.ModuleList(
            [_SoftTree(td, sigma=sigma) for td in tree_dicts]
        )

    def forward(self, x):
        preds = torch.stack([tree(x) for tree in self.trees], dim=1)
        return preds.mean(dim=1)


@registry.register
class FOCUSExplainer(CounterfactualExplainer):
    _modality = ExplainerModality.COUNTERFACTUAL
    _target = ModelTarget.SURVIVAL

    def __init__(
        self,
        model_adapter,
        *,
        sigma=1.0,
        lr=0.01,
        n_iter=500,
        distance_weight=1.0,
        sparsity_weight=0.05,
        immutable=None,
        time_horizon=None,
    ):
        super().__init__(model_adapter)
        self._sigma = sigma
        self._lr = lr
        self._n_iter = n_iter
        self._distance_weight = distance_weight
        self._sparsity_weight = sparsity_weight
        self._time_horizon = time_horizon
        self._predict_fn = risk_predict_fn(model_adapter, time_horizon)
        self._cfg = build_plausibility_config(
            model_adapter, extra_immutable=immutable,
        )
        self._ranges = self._cfg.ranges
        self._immutable = self._cfg.immutable
        self._categorical_values = self._cfg.categorical_values
        self._higher_is_better = getattr(model_adapter, "higher_is_better", False)
        self._soft_forest = None
        self._scaler = None
        self._risk_median = None
        self._device = torch.device("cpu")

    @property
    def modality(self):
        return self._modality

    def immutable_features(self):
        return list(self._immutable)

    def _build_soft_forest(self):
        raw_model = self._adapter.raw_model
        tree_dicts = _extract_trees(raw_model)

        self._soft_forest = _SoftForest(tree_dicts, sigma=self._sigma).to(self._device)
        self._soft_forest.eval()

        x_train = self._adapter.training_data()
        feat_names = self._adapter.feature_names()
        data = x_train[feat_names].fillna(0).values

        self._scaler = MinMaxScaler()
        self._scaler.fit(data)

        risks = self._predict_fn(data)
        self._risk_median = float(np.median(risks))

    def generate_counterfactual(self, patient_features, target_outcome=None):
        if self._soft_forest is None:
            self._build_soft_forest()

        feat_names = self._adapter.feature_names()
        x_orig = patient_features[feat_names].fillna(0).values[0].astype(np.float32)
        x_orig_f64 = x_orig.astype(np.float64)
        current_risk = float(self._predict_fn(x_orig.reshape(1, -1).astype(np.float64))[0])

        threshold = float(target_outcome) if target_outcome is not None else self._risk_median
        go_lower = not self._higher_is_better

        if target_outcome is not None:
            y_target = float(target_outcome)
        elif go_lower:
            y_target = threshold * 0.8
        else:
            y_target = threshold * 1.2

        mutable_mask = torch.BoolTensor([f not in self._immutable for f in feat_names])
        mutable_idx = torch.where(mutable_mask)[0]
        if len(mutable_idx) == 0:
            return None

        range_scales = np.ones(len(feat_names), dtype=np.float32)
        for i, feat in enumerate(feat_names):
            if feat in self._ranges:
                lo, hi = self._ranges[feat]
                span = hi - lo
                if span > 0:
                    range_scales[i] = span
        range_scales_t = torch.FloatTensor(range_scales).to(self._device)

        x_orig_t = torch.FloatTensor(x_orig).to(self._device)
        x_cf = x_orig_t.clone().detach().requires_grad_(True)

        bounds_lo = torch.FloatTensor(
            [self._ranges.get(f, (-1e6, 1e6))[0] for f in feat_names]
        ).to(self._device)
        bounds_hi = torch.FloatTensor(
            [self._ranges.get(f, (-1e6, 1e6))[1] for f in feat_names]
        ).to(self._device)

        optimizer = torch.optim.Adam([x_cf], lr=self._lr)
        target_t = torch.tensor(y_target, device=self._device)

        best_cf = None
        best_distance = float("inf")

        for step in range(self._n_iter):
            optimizer.zero_grad()

            pred = self._soft_forest(x_cf.unsqueeze(0)).squeeze()
            pred_loss = (pred - target_t) ** 2

            delta = (x_cf - x_orig_t) / range_scales_t
            proximity_loss = self._distance_weight * (delta ** 2).sum()
            sparsity_loss = self._sparsity_weight * delta.abs().sum()

            loss = pred_loss + proximity_loss + sparsity_loss
            loss.backward()

            if x_cf.grad is not None:
                x_cf.grad.data[~mutable_mask] = 0.0

            optimizer.step()

            with torch.no_grad():
                x_cf.data = torch.clamp(x_cf.data, bounds_lo, bounds_hi)
                x_cf.data[~mutable_mask] = x_orig_t[~mutable_mask]

            if (step + 1) % 50 == 0 or step == self._n_iter - 1:
                with torch.no_grad():
                    cf_np = x_cf.detach().cpu().numpy().astype(np.float64)
                    real_risk = float(self._predict_fn(cf_np.reshape(1, -1))[0])
                    crossed = (
                        (go_lower and real_risk < threshold)
                        or (not go_lower and real_risk >= threshold)
                    )
                    if crossed:
                        dist = float(((x_cf - x_orig_t) / range_scales_t).pow(2).sum())
                        if dist < best_distance:
                            best_distance = dist
                            best_cf = cf_np.copy()

        if best_cf is None:
            cf_np = x_cf.detach().cpu().numpy().astype(np.float64)
            real_risk = float(self._predict_fn(cf_np.reshape(1, -1))[0])
            crossed = (
                (go_lower and real_risk < threshold)
                or (not go_lower and real_risk >= threshold)
            )
            if crossed:
                best_cf = cf_np.copy()

        if best_cf is None:
            return None

        range_scales_f64 = range_scales.astype(np.float64)
        tiny = np.abs(best_cf - x_orig_f64) / range_scales_f64 < 1e-4
        best_cf[tiny] = x_orig_f64[tiny]

        cf_df = pd.DataFrame([best_cf], columns=feat_names)
        cf_df = clip_to_ranges(cf_df, self._ranges)

        cf_df, report = snap_and_verify(
            cf_df.values[0],
            feat_names,
            self._categorical_values,
            self._predict_fn,
            target=threshold,
            higher_is_better=self._higher_is_better,
        )
        cf_df = enforce_immutables(
            patient_features[feat_names].head(1),
            cf_df,
            self._immutable,
        )

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

        return self._make_explanation(
            patient_id,
            content=content,
            method="FOCUS",
            counterfactual_df=cf,
            original_prediction=orig_pred,
            cf_prediction=cf_pred,
            raw_query=raw_query,
            raw_cf=raw_cf,
            higher_is_better=self._higher_is_better,
        )

    def sanity_check(self, held_out_df):
        return _run_sanity_checks(self, held_out_df)