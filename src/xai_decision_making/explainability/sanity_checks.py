"""
Sanity check suite for validating explainer outputs. Provides per-modality
checks — top-k stability and deletion faithfulness for attribution; success
rate, proximity, and constraint adherence for counterfactual; pretreatment
leakage and distance for case-based; fidelity, rule length, and anchor
precision for rule-based — plus a QoL bounds check, all runnable against
a held-out patient set.
"""
from __future__ import annotations

import abc
import warnings
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from xai_decision_making.explainability.base import (
    AttributionExplainer,
    CaseBasedExplainer,
    RuleBasedExplainer,
    CounterfactualExplainer,
    Explainer,
    ExplainerModality,
    SanityCheckResult,
)


class SanityCheck(abc.ABC):

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    def run(self, explainer: Explainer, held_out_df: pd.DataFrame) -> SanityCheckResult: ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def _top_k_features(attrs: dict, k: int) -> set:
    return set(sorted(attrs, key=lambda f: abs(attrs[f]), reverse=True)[:k])


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _scalar_prediction(adapter, patient: pd.DataFrame):
    pred = adapter.predict(patient)
    if isinstance(pred, dict):
        if "risk_score" in pred:
            return pred["risk_score"]
        for v in pred.values():
            return np.asarray(v).ravel()[-1]
    return np.asarray(pred).ravel()[-1]


def _prediction_change_per_fraction(patient, feature_order, adapter, fill_values, fractions, base_pred):
    n = len(feature_order)
    changes = []
    for frac in fractions:
        n_mask = max(1, int(frac * n))
        masked = patient.copy()
        for feat in feature_order[:n_mask]:
            if feat in masked.columns:
                masked[feat] = fill_values.get(feat, 0.0)
        try:
            changes.append(abs(base_pred - _scalar_prediction(adapter, masked)))
        except Exception:
            pass
    return np.mean(changes) if changes else 0.0


def _as_dataframe(obj) -> pd.DataFrame:
    return obj if isinstance(obj, pd.DataFrame) else pd.DataFrame([obj])


def _iter_patients(held_out_df: pd.DataFrame, max_patients: int, desc: str = ""):
    rows = list(held_out_df.head(max_patients).iterrows())
    for _, row in tqdm(rows, desc=desc, leave=False):
        yield row.to_frame().T.reset_index(drop=True)


def _normalised_l1(a: pd.DataFrame, b: pd.DataFrame, col_ranges: dict):
    shared = [c for c in col_ranges if c in a.columns and c in b.columns]
    return sum(abs(a[c].iloc[0] - b[c].iloc[0]) / col_ranges[c] for c in shared)


@dataclass
class TopKStabilityCheck(SanityCheck):
    k: int = 5
    n_perturbations: int = 10
    noise_std: float = 0.01
    threshold: float = 0.6
    max_patients: int = 50

    @property
    def name(self) -> str:
        return "TopKStability"

    def run(self, explainer, held_out_df: pd.DataFrame) -> SanityCheckResult:
        if not isinstance(explainer, AttributionExplainer):
            return SanityCheckResult(name=self.name, passed=False,
                                     notes="Not an AttributionExplainer — skipped.")

        numeric_cols = held_out_df.select_dtypes(include=[np.number]).columns.tolist()
        jaccard_scores = []

        for patient in _iter_patients(held_out_df, self.max_patients, desc=self.name):
            try:
                base_attrs = explainer.feature_importances(patient)
            except Exception as exc:
                warnings.warn(f"[TopKStability] feature_importances failed: {exc}")
                continue

            base_top = _top_k_features(base_attrs, self.k)
            noise_matrix = np.random.normal(0, self.noise_std, size=(self.n_perturbations, len(numeric_cols)))

            row_jaccards = []
            for noise_row in noise_matrix:
                perturbed = patient.copy()
                perturbed[numeric_cols] = perturbed[numeric_cols].values + noise_row
                try:
                    pert_attrs = explainer.feature_importances(perturbed)
                    row_jaccards.append(_jaccard(base_top, _top_k_features(pert_attrs, self.k)))
                except Exception:
                    continue

            if row_jaccards:
                jaccard_scores.append(np.mean(row_jaccards))

        if not jaccard_scores:
            return SanityCheckResult(name=self.name, passed=False, notes="No Jaccard scores computed.")

        mean_j = np.mean(jaccard_scores)
        return SanityCheckResult(
            name=self.name,
            passed=mean_j >= self.threshold,
            metrics={"mean_jaccard": round(mean_j, 4), "n_patients": len(jaccard_scores)},
            notes=f"k={self.k}, noise_std={self.noise_std}, threshold={self.threshold}",
        )


@dataclass
class DeletionFaithfulnessCheck(SanityCheck):
    deletion_fractions: Sequence[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    n_random_repeats: int = 5
    threshold: float = 0.0
    max_patients: int = 30

    @property
    def name(self) -> str:
        return "DeletionFaithfulness"

    def run(self, explainer, held_out_df: pd.DataFrame) -> SanityCheckResult:
        if not isinstance(explainer, AttributionExplainer):
            return SanityCheckResult(name=self.name, passed=False,
                                     notes="Not an AttributionExplainer — skipped.")

        adapter = explainer._adapter
        fill_values = adapter.training_data().select_dtypes(include=[np.number]).mean().to_dict()
        attr_scores, rand_scores = [], []

        for patient in _iter_patients(held_out_df, self.max_patients, desc=self.name):
            try:
                base_pred = _scalar_prediction(adapter, patient)
                attrs = explainer.feature_importances(patient)
            except Exception as exc:
                warnings.warn(f"[DeletionFaithfulness] skipping patient: {exc}")
                continue

            features = list(attrs.keys())
            ranked = sorted(features, key=lambda f: abs(attrs[f]), reverse=True)

            attr_scores.append(_prediction_change_per_fraction(
                patient, ranked, adapter, fill_values, self.deletion_fractions, base_pred))

            random_repeat_scores = []
            for _ in range(self.n_random_repeats):
                shuffled = features.copy()
                np.random.shuffle(shuffled)
                random_repeat_scores.append(_prediction_change_per_fraction(
                    patient, shuffled, adapter, fill_values, self.deletion_fractions, base_pred))
            rand_scores.append(np.mean(random_repeat_scores))

        if not attr_scores:
            return SanityCheckResult(name=self.name, passed=False, notes="No deletion curves computed.")

        mean_attr = np.mean(attr_scores)
        mean_rand = np.mean(rand_scores)
        delta = mean_attr - mean_rand

        return SanityCheckResult(
            name=self.name,
            passed=delta >= self.threshold,
            metrics={"attr_mean_change": round(mean_attr, 4),
                     "rand_mean_change": round(mean_rand, 4),
                     "delta": round(delta, 4)},
            notes=f"threshold(delta)={self.threshold}",
        )


@dataclass
class CounterfactualSuccessRateCheck(SanityCheck):
    threshold: float = 0.7
    max_patients: int = 50

    @property
    def name(self) -> str:
        return "CFSuccessRate"

    def run(self, explainer, held_out_df: pd.DataFrame) -> SanityCheckResult:
        if not isinstance(explainer, CounterfactualExplainer):
            return SanityCheckResult(name=self.name, passed=False, notes="Not a CounterfactualExplainer.")

        successes = total = 0
        for patient in _iter_patients(held_out_df, self.max_patients, desc=self.name):
            total += 1
            try:
                cf = explainer.generate_counterfactual(patient)
                if cf is not None:
                    successes += 1
            except Exception as exc:
                warnings.warn(f"[CFSuccessRate] generation failed: {exc}")

        rate = successes / total if total else 0.0
        return SanityCheckResult(
            name=self.name,
            passed=rate >= self.threshold,
            metrics={"success_rate": round(rate, 4), "n_patients": total},
            notes=f"threshold={self.threshold}",
        )


@dataclass
class CounterfactualProximityCheck(SanityCheck):
    threshold: float = 1.0
    max_patients: int = 50

    @property
    def name(self) -> str:
        return "CFProximity"

    def run(self, explainer, held_out_df: pd.DataFrame) -> SanityCheckResult:
        if not isinstance(explainer, CounterfactualExplainer):
            return SanityCheckResult(name=self.name, passed=False, notes="Not a CounterfactualExplainer.")

        x_train = explainer._adapter.training_data()
        numeric_cols = x_train.select_dtypes(include=[np.number]).columns.tolist()
        col_ranges = {c: (x_train[c].max() - x_train[c].min() or 1.0) for c in numeric_cols}

        distances = []
        for patient in _iter_patients(held_out_df, self.max_patients, desc=self.name):
            try:
                cf = explainer.generate_counterfactual(patient)
            except Exception:
                continue
            if cf is not None:
                distances.append(_normalised_l1(patient, _as_dataframe(cf), col_ranges))

        if not distances:
            return SanityCheckResult(name=self.name, passed=False, notes="No CFs to measure.")

        median_dist = np.median(distances)
        return SanityCheckResult(
            name=self.name,
            passed=median_dist <= self.threshold,
            metrics={"median_l1": round(median_dist, 4), "n_cfs": len(distances)},
            notes=f"threshold={self.threshold}",
        )


@dataclass
class CounterfactualConstraintCheck(SanityCheck):
    threshold_violation_rate: float = 0.05
    max_patients: int = 50

    @property
    def name(self) -> str:
        return "CFConstraints"

    def run(self, explainer, held_out_df: pd.DataFrame) -> SanityCheckResult:
        if not isinstance(explainer, CounterfactualExplainer):
            return SanityCheckResult(name=self.name, passed=False, notes="Not a CounterfactualExplainer.")

        immutables = set(explainer.immutable_features())
        x_train = explainer._adapter.training_data()
        numeric_cols = x_train.select_dtypes(include=[np.number]).columns.tolist()
        col_min = x_train[numeric_cols].min()
        col_max = x_train[numeric_cols].max()
        total_checks = violations = 0

        for patient in _iter_patients(held_out_df, self.max_patients, desc=self.name):
            try:
                cf = explainer.generate_counterfactual(patient)
            except Exception:
                continue
            if cf is None:
                continue
            cf_df = _as_dataframe(cf)

            for feat in immutables:
                if feat in patient.columns and feat in cf_df.columns:
                    total_checks += 1
                    if patient[feat].iloc[0] != cf_df[feat].iloc[0]:
                        violations += 1

            for col in numeric_cols:
                if col in cf_df.columns:
                    total_checks += 1
                    val = cf_df[col].iloc[0]
                    if val < col_min[col] or val > col_max[col]:
                        violations += 1

        if total_checks == 0:
            return SanityCheckResult(name=self.name, passed=False, notes="No constraint checks performed.")

        rate = violations / total_checks
        return SanityCheckResult(
            name=self.name,
            passed=rate <= self.threshold_violation_rate,
            metrics={"violation_rate": round(rate, 4), "n_checks": total_checks},
            notes=f"threshold={self.threshold_violation_rate}",
        )


@dataclass
class CaseBasedPretreatmentCheck(SanityCheck):
    max_patients: int = 30

    @property
    def name(self) -> str:
        return "CaseBasedPretreatment"

    def run(self, explainer, held_out_df: pd.DataFrame) -> SanityCheckResult:
        if not isinstance(explainer, CaseBasedExplainer):
            return SanityCheckResult(name=self.name, passed=False, notes="Not a CaseBasedExplainer.")

        pretreatment_cols = set(explainer.pretreatment_features())
        post_treatment_cols = set(held_out_df.columns) - pretreatment_cols

        if not post_treatment_cols:
            return SanityCheckResult(name=self.name, passed=True,
                                     notes="No post-treatment columns detected.",
                                     metrics={"n_post_cols": 0})

        total_checks = violations = 0
        for patient in _iter_patients(held_out_df, self.max_patients, desc=self.name):
            try:
                cases = explainer.retrieve_similar(patient)
            except Exception as exc:
                warnings.warn(f"[CaseBasedPretreatment] retrieve_similar failed: {exc}")
                continue
            if cases is None or len(cases) == 0:
                continue

            cases_df = _as_dataframe(cases)
            for col in post_treatment_cols:
                if col in cases_df.columns:
                    total_checks += len(cases_df)
                    violations += cases_df[col].notna().sum()

        if total_checks == 0:
            return SanityCheckResult(name=self.name, passed=True,
                                     notes="Post-treatment columns absent from retrieved cases.",
                                     metrics={"violations": 0, "total_checks": 0})

        return SanityCheckResult(
            name=self.name,
            passed=violations == 0,
            metrics={"violations": violations, "total_checks": total_checks},
            notes="Post-treatment columns must be absent or NaN in retrieved cases.",
        )


@dataclass
class CaseBasedDistanceCheck(SanityCheck):
    k: int = 3
    threshold: float = 2.0
    max_patients: int = 30

    @property
    def name(self) -> str:
        return "CaseBasedDistance"

    def run(self, explainer, held_out_df: pd.DataFrame) -> SanityCheckResult:
        if not isinstance(explainer, CaseBasedExplainer):
            return SanityCheckResult(name=self.name, passed=False, notes="Not a CaseBasedExplainer.")

        x_train = explainer._adapter.training_data()
        pre_cols = [c for c in explainer.pretreatment_features() if c in x_train.columns]
        numeric_cols = x_train[pre_cols].select_dtypes(include=[np.number]).columns.tolist()
        col_ranges = {c: (x_train[c].max() - x_train[c].min() or 1.0) for c in numeric_cols}

        patient_medians = []
        for patient in _iter_patients(held_out_df, self.max_patients, desc=self.name):
            try:
                cases = explainer.retrieve_similar(patient, k=self.k)
            except Exception:
                continue
            if cases is None or len(cases) == 0:
                continue

            cases_df = _as_dataframe(cases)
            case_distances = [
                _normalised_l1(patient, case_row.to_frame().T, col_ranges)
                for _, case_row in cases_df.iterrows()
            ]
            if case_distances:
                patient_medians.append(np.median(case_distances))

        if not patient_medians:
            return SanityCheckResult(name=self.name, passed=False, notes="No distances computed.")

        overall_median = np.median(patient_medians)
        return SanityCheckResult(
            name=self.name,
            passed=overall_median <= self.threshold,
            metrics={"median_l1": round(overall_median, 4), "n_patients": len(patient_medians)},
            notes=f"threshold={self.threshold}",
        )


@dataclass
class QoLBoundsCheck(SanityCheck):
    lo: float = 0.0
    hi: float = 100.0
    threshold_oor_rate: float = 0.0
    max_patients: int = 50

    @property
    def name(self) -> str:
        return "QoLBounds"

    def run(self, explainer, held_out_df: pd.DataFrame) -> SanityCheckResult:
        total_vals = out_of_range = 0

        for patient in _iter_patients(held_out_df, self.max_patients, desc=self.name):
            try:
                pred = explainer._adapter.predict(patient)
            except Exception as exc:
                warnings.warn(f"[QoLBounds] prediction failed: {exc}")
                continue

            trajectory = pred.get("trajectory")
            if trajectory is None:
                continue

            arr = np.asarray(trajectory).ravel()
            total_vals += len(arr)
            out_of_range += np.sum((arr < self.lo) | (arr > self.hi))

        if total_vals == 0:
            return SanityCheckResult(name=self.name, passed=False, notes="No trajectory values obtained.")

        oor_rate = out_of_range / total_vals
        return SanityCheckResult(
            name=self.name,
            passed=oor_rate <= self.threshold_oor_rate,
            metrics={"oor_rate": round(oor_rate, 6), "n_values": total_vals, "n_oor": out_of_range},
            notes=f"valid_range=[{self.lo},{self.hi}], threshold={self.threshold_oor_rate}",
        )

@dataclass
class RuleFidelityCheck(SanityCheck):
    threshold: float = 0.70
    max_patients: int = 30

    @property
    def name(self):
        return "RuleFidelity"

    def run(self, explainer, held_out_df):
        if not isinstance(explainer, RuleBasedExplainer):
            return SanityCheckResult(name=self.name, passed=False,
                                     notes="Not a RuleBasedExplainer.")

        try:
            fidelity = explainer.fidelity(held_out_df.head(self.max_patients))
        except Exception as exc:
            return SanityCheckResult(name=self.name, passed=False,
                                     notes=f"fidelity() failed: {exc}")

        return SanityCheckResult(
            name=self.name,
            passed=fidelity >= self.threshold,
            metrics={"fidelity": round(float(fidelity), 4)},
            notes=f"threshold={self.threshold}",
        )


@dataclass
class RuleLengthCheck(SanityCheck):
    min_conditions: int = 1
    max_conditions: int = 10
    max_patients: int = 30

    @property
    def name(self):
        return "RuleLength"

    def run(self, explainer, held_out_df):
        if not isinstance(explainer, RuleBasedExplainer):
            return SanityCheckResult(name=self.name, passed=False,
                                     notes="Not a RuleBasedExplainer.")

        feat_names = explainer._adapter.feature_names()
        lengths = []
        trivial = 0
        too_long = 0

        for patient in _iter_patients(held_out_df, self.max_patients, desc=self.name):
            try:
                rule = explainer.extract_rule(patient[feat_names])
            except Exception:
                continue
            n = len(rule.get("conditions", []))
            lengths.append(n)
            if n < self.min_conditions:
                trivial += 1
            if n > self.max_conditions:
                too_long += 1

        if not lengths:
            return SanityCheckResult(name=self.name, passed=False,
                                     notes="No rules extracted.")

        mean_len = float(np.mean(lengths))
        passed = trivial == 0 and too_long == 0

        return SanityCheckResult(
            name=self.name,
            passed=passed,
            metrics={
                "mean_length": round(mean_len, 2),
                "n_trivial": trivial,
                "n_too_long": too_long,
                "n_patients": len(lengths),
            },
            notes=f"min={self.min_conditions}, max={self.max_conditions}",
        )


@dataclass
class AnchorPrecisionCheck(SanityCheck):
    threshold: float = 0.90
    max_patients: int = 30

    @property
    def name(self):
        return "AnchorPrecision"

    def run(self, explainer, held_out_df):
        if not isinstance(explainer, RuleBasedExplainer):
            return SanityCheckResult(name=self.name, passed=False,
                                     notes="Not a RuleBasedExplainer.")

        feat_names = explainer._adapter.feature_names()
        precisions = []

        for patient in _iter_patients(held_out_df, self.max_patients, desc=self.name):
            try:
                rule = explainer.extract_rule(patient[feat_names])
            except Exception:
                continue
            p = rule.get("precision", None)
            if p is not None and not np.isnan(p):
                precisions.append(float(p))

        if not precisions:
            return SanityCheckResult(
                name=self.name, passed=True,
                notes="No precision reported (not an Anchors-style explainer).",
                metrics={"n_with_precision": 0},
            )

        mean_prec = float(np.mean(precisions))
        return SanityCheckResult(
            name=self.name,
            passed=mean_prec >= self.threshold,
            metrics={"mean_precision": round(mean_prec, 4),
                     "n_patients": len(precisions)},
            notes=f"threshold={self.threshold}",
        )

RULE_BASED_CHECKS: list = [
    RuleFidelityCheck(),
    RuleLengthCheck(),
]

RULE_BASED_CHECKS: list = [
    RuleFidelityCheck(),
    RuleLengthCheck(),
    AnchorPrecisionCheck(),
]

ATTRIBUTION_CHECKS: list[SanityCheck] = [
    TopKStabilityCheck(),
    DeletionFaithfulnessCheck(),
]

COUNTERFACTUAL_CHECKS: list[SanityCheck] = [
    CounterfactualSuccessRateCheck(),
    CounterfactualProximityCheck(),
    CounterfactualConstraintCheck(),
]

CASE_BASED_CHECKS: list[SanityCheck] = [
    CaseBasedPretreatmentCheck(),
    CaseBasedDistanceCheck(),
]

QOL_CHECKS: list[SanityCheck] = [
    QoLBoundsCheck(),
]

CHECKS_BY_MODALITY: dict[Any, list[SanityCheck]] = {
    ExplainerModality.ATTRIBUTION: ATTRIBUTION_CHECKS,
    ExplainerModality.COUNTERFACTUAL: COUNTERFACTUAL_CHECKS,
    ExplainerModality.CASE_BASED: CASE_BASED_CHECKS,
    ExplainerModality.RULE_BASED: RULE_BASED_CHECKS,
}