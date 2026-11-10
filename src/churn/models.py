"""Small, explicit model search spaces; all learned transforms stay in folds."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, SplineTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from .data import CATEGORICAL, NUMERIC


class BalancedGradientBoosting(GradientBoostingClassifier):
    """Compute class weights within each fit, including each CV training fold."""

    def fit(self, X, y, sample_weight=None, monitor=None):
        weights = compute_sample_weight("balanced", y) if sample_weight is None else sample_weight
        return super().fit(X, y, sample_weight=weights, monitor=monitor)


class AccountFeatures(BaseEstimator, TransformerMixin):
    """Row-local features available at scoring time; no target or fitted aggregates."""

    def __init__(self, enabled=False):
        self.enabled = enabled

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = X.copy()
        if self.enabled:
            out["log_tenure"] = np.log1p(out.tenure)
            average = out.TotalCharges / out.tenure.replace(0, np.nan)
            out["average_monthly_charge"] = average
            out["charge_change"] = out.MonthlyCharges - average
            out["support_services"] = out[["OnlineSecurity", "OnlineBackup", "TechSupport"]].eq("Yes").sum(axis=1)
        return out


def original_pipeline(kind):
    binary = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    multi = [c for c in CATEGORICAL if c not in binary]
    pre = ColumnTransformer([
        ("num", Pipeline([("fill", SimpleImputer(strategy="constant", fill_value=0)),
                          ("scale", StandardScaler())]), NUMERIC),
        ("binary", OrdinalEncoder(), binary),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), multi),
    ])
    models = {
        "original_logistic": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "original_tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, class_weight="balanced", random_state=40),
        "original_boosting": BalancedGradientBoosting(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=40),
    }
    return Pipeline([("preprocessor", pre), ("model", models[kind])])


def improved_pipeline(kind):
    steps = [("fill", SimpleImputer(strategy="median"))]
    if kind == "spline_logistic":
        steps += [("spline", SplineTransformer(n_knots=4, degree=3, include_bias=False))]
    steps += [("scale", StandardScaler())]
    pre = ColumnTransformer([
        ("num", Pipeline(steps), make_column_selector(dtype_include=np.number)),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), make_column_selector(dtype_exclude=np.number)),
    ])
    if kind == "hist_boosting":
        model = HistGradientBoostingClassifier(max_iter=180, early_stopping=False, random_state=2026)
    else:
        model = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=2026)
    return Pipeline([("features", AccountFeatures()), ("preprocessor", pre), ("model", model)])


def candidates():
    result = {"dummy": (DummyClassifier(strategy="prior"), {})}
    for name in ("original_logistic", "original_tree", "original_boosting"):
        result[name] = (original_pipeline(name), {})
    logistic_grid = {"model__C": [0.01, 0.1, 1.0, 10.0], "model__class_weight": [None, "balanced"]}
    result["tuned_logistic"] = (improved_pipeline("tuned_logistic"), logistic_grid)
    result["spline_logistic"] = (improved_pipeline("spline_logistic"), logistic_grid)
    # Eight candidates rather than an exhaustive search over many weakly motivated knobs.
    hist_grid = []
    for enabled in (False, True):
        for leaves, regularization, rate in [(7, 1.0, 0.05), (7, 10.0, 0.1), (15, 1.0, 0.05), (15, 10.0, 0.1)]:
            hist_grid.append({"features__enabled": [enabled], "model__max_leaf_nodes": [leaves],
                              "model__l2_regularization": [regularization], "model__learning_rate": [rate]})
    result["hist_boosting"] = (improved_pipeline("hist_boosting"), hist_grid)
    return result
