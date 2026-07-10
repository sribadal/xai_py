"""
Core abstractions for the explainability framework. Defines model adapter
interfaces (survival, QoL), explainer base classes per modality (attribution,
counterfactual, case-based, rule-based), explanation containers, sanity check
result types, and an explainer registry for modality/target dispatch.
"""
import abc
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

class ExplainerModality(Enum):
    ATTRIBUTION = auto()
    COUNTERFACTUAL = auto()
    CASE_BASED = auto()
    RULE_BASED = auto()

class ModelTarget(Enum):
    SURVIVAL = auto()
    QOL = auto()

@dataclass
class SanityCheckResult:
    name: str
    passed: bool
    metrics: dict = field(default_factory=dict)
    notes: str = ""
    modality: "ExplainerModality" = None
    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name:<35} {self.metrics}  {self.notes}"

@dataclass
class AggregatedSanityResult:
    modality: ExplainerModality
    target: ModelTarget
    explainer_name: str
    check_results: list = field(default_factory=list)

    @property
    def passed(self):
        return bool(self.check_results) and all(r.passed for r in self.check_results)

    @property
    def metrics(self):
        merged = {}
        for r in self.check_results:
            for k, v in r.metrics.items():
                merged[f"{r.name}/{k}"] = v
        return merged

    def print_summary(self):
        status = "PASS" if self.passed else "FAIL"
        n_pass = sum(1 for r in self.check_results if r.passed)
        n_fail = len(self.check_results) - n_pass
        print(f"\n{status}  {self.explainer_name} "
              f"({self.modality.name} / {self.target.name})  "
              f"[{n_pass}/{len(self.check_results)} checks passed]")
        for r in self.check_results:
            sym = "PASS" if r.passed else "FAIL"
            print(f"{sym}  {r}")

@dataclass
class Explanation:
    modality: ExplainerModality
    target: ModelTarget
    patient_id: Any
    content: Any
    metadata: dict = field(default_factory=dict)

    def summary(self):
        return (
            f"Explanation({self.modality.name},"
            f"target={self.target.name}, "
            f"patient={self.patient_id})"
        )

class ModelAdapter(abc.ABC):
    @property
    @abc.abstractmethod
    def target(self):
        pass

    @abc.abstractmethod
    def predict(self, patient_features):
        pass

    @abc.abstractmethod
    def feature_names(self):
        pass

    @abc.abstractmethod
    def training_data(self):
        pass

class SurvivalModelAdapter(ModelAdapter):
    def __init__(self, model, x_train, feature_names_list):
        self._model = model
        self._x_train = x_train
        self._feature_names_list = feature_names_list

    @property
    def target(self):
        return ModelTarget.SURVIVAL

    def predict(self, patient_features):
        surv = self._model.predict_survival_function(patient_features)
        risk = self._model.predict(patient_features)
        return {"survival_curve": surv, "risk_score": float(risk[0])}

    def feature_names(self):
        return list(self._feature_names_list)

    def training_data(self):
        return self._x_train

    @property
    def raw_model(self):
        return self._model

class QoLModelAdapter(ModelAdapter):
    def __init__(self, models, config, encoder_bundle, x_train):
        self._models = models
        self._config = config
        self._enc = encoder_bundle
        self._x_train = x_train
    @property
    def target(self):
        return ModelTarget.QOL

    def predict(self, patient_features):
        from xai_decision_making.experiments.evaluate_condor import run_inference_for_patient
        trajectory = run_inference_for_patient(
            self._models, self._config, self._enc, patient_features
        )
        return {"trajectory": trajectory}

    def feature_names(self):
        return list(self._x_train.columns)

    def training_data(self):
        return self._x_train

    @property
    def raw_models(self):
        return self._models

    @property
    def config(self):
        return self._config

class Explainer(abc.ABC):
    _modality: ExplainerModality = None
    _target: ModelTarget = None
    checks: list = []

    def __init__(self, model_adapter):
        if not isinstance(model_adapter, ModelAdapter):
            raise TypeError("your model adapter must be a ModelAdapter instance")
        self._adapter = model_adapter

    @property
    @abc.abstractmethod
    def modality(self):
        return self._modality

    @property
    def target(self):
        return self._target

    @abc.abstractmethod
    def explain(self, patient_features, patient_id=None):
        pass

    @abc.abstractmethod
    def sanity_check(self, held_out_df):
        results = []
        for check in self.checks:
            result = check.run(self, held_out_df)
            result.modality = self._modality
            results.append(result)
        return AggregatedSanityResult(
            modality=self._modality,
            target=self._adapter.target,
            explainer_name=self.__class__.__name__,
            check_results=results
        )

    def _make_explanation(self, patient_id, content, **metadata):
        return Explanation(
            modality=self._modality,
            target=self._adapter.target,
            patient_id=patient_id,
            content=content,
            metadata=metadata
        )

    def __repr__(self):
        m = self._modality.name if self._modality else "None"
        t = self._target.name   if self._target   else "None"
        return f"{self.__class__.__name__}(modality={m}, target={t})"

class AttributionExplainer(Explainer, abc.ABC):
    @abc.abstractmethod
    def feature_importances(self, patient_features):
        pass

class CounterfactualExplainer(Explainer, abc.ABC):
    @abc.abstractmethod
    def generate_counterfactual(self, patient_features, target_outcome=None):
        pass

    @abc.abstractmethod
    def immutable_features(self):
        pass

class CaseBasedExplainer(Explainer, abc.ABC):
    @abc.abstractmethod
    def retrieve_similar(self, patient_features, k=3):
        pass

    @abc.abstractmethod
    def pretreatment_features(self):
        pass

class RuleBasedExplainer(Explainer, abc.ABC):
    @abc.abstractmethod
    def extract_rule(self, patient_features):
        pass

    @abc.abstractmethod
    def global_rules(self):
        pass

    @abc.abstractmethod
    def fidelity(self, held_out_df):
        pass

class ExplainerRegistry:
    def __init__(self):
        self._registry = {}

    def register(self, explainer_cls):
        if not (explainer_cls._modality and explainer_cls._target):
            raise ValueError(
                f"{explainer_cls.__name__} must define _modality and _target "
                "as class attributes before registration."
            )
        key = (explainer_cls._modality, explainer_cls._target)
        self._registry[key] = explainer_cls
        return explainer_cls

    def build(self, modality, target, model_adapter, **kwargs):
        key = (modality, target)
        if key not in self._registry:
            raise KeyError(
                f"No explainer for {modality.name}/{target.name}. "
                f"Registered: {list(self._registry.keys())}"
            )
        return self._registry[key](model_adapter, **kwargs)

    def available(self):
        return list(self._registry.keys())