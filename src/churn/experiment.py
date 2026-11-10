"""Run development first, freeze choices, then evaluate the final holdout once."""
import argparse
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, train_test_split
from threadpoolctl import threadpool_limits

from .data import DATA_FILE, fingerprint, load_data, profile_groups, split_indices
from .models import candidates, original_pipeline

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n")


def metrics(y, p, threshold=0.5):
    pred = np.asarray(p) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": threshold, "roc_auc": roc_auc_score(y, p),
            "average_precision": average_precision_score(y, p),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred, zero_division=0),
            "accuracy": accuracy_score(y, pred), "brier": brier_score_loss(y, p),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def select_threshold(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    eligible = np.where((thresholds >= 0.05) & (thresholds <= 0.95))[0]
    return float(thresholds[eligible[np.argmax(scores[eligible])]])


def fit_candidate(model, grid, x, y, seed):
    if not grid:
        return clone(model).fit(x, y), {}, None
    search = GridSearchCV(clone(model), grid, scoring="average_precision",
                          cv=StratifiedGroupKFold(3, shuffle=True, random_state=seed),
                          n_jobs=1, error_score="raise", return_train_score=False)
    search.fit(x, y, groups=profile_groups(x))
    return search.best_estimator_, search.best_params_, search.cv_results_


def develop(data_path, output):
    x, y, raw = load_data(data_path)
    dev_idx, test_idx = split_indices(y, x)
    x, y = x.iloc[dev_idx].reset_index(drop=True), y.iloc[dev_idx].reset_index(drop=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "private").mkdir(exist_ok=True)
    rows, oof_rows, final_params = [], {}, {}
    all_candidates = candidates()
    outer = list(StratifiedGroupKFold(5, shuffle=True, random_state=2026).split(x, y, groups=profile_groups(x)))
    for name, (model, grid) in all_candidates.items():
        probabilities = np.zeros(len(y))
        for fold, (train, valid) in enumerate(outer):
            fitted, params, _ = fit_candidate(model, grid, x.iloc[train], y.iloc[train], 3100 + fold)
            p = fitted.predict_proba(x.iloc[valid])[:, 1]
            probabilities[valid] = p
            row = {"model": name, "fold": fold, **metrics(y.iloc[valid], p), "parameters": json.dumps(params)}
            rows.append(row)
            print(f"{name:20s} fold {fold + 1}/5 AP={row['average_precision']:.4f}", flush=True)
        oof_rows[name] = probabilities
        fitted, params, search_results = fit_candidate(model, grid, x, y, 4100)
        final_params[name] = params
        joblib.dump(fitted, output / "private" / f"{name}.joblib")
        if search_results is not None:
            search_table = pd.DataFrame(search_results)
            # Wall-clock timings are machine-dependent, not experimental results.
            search_table = search_table.drop(columns=[c for c in search_table if c.endswith("_time")])
            search_table.to_csv(output / f"search_{name}.csv", index=False)
    folds = pd.DataFrame(rows)
    folds.to_csv(output / "cv_folds.csv", index=False)
    summary = folds.groupby("model").agg(
        ap_mean=("average_precision", "mean"), ap_sd=("average_precision", "std"),
        auc_mean=("roc_auc", "mean"), auc_sd=("roc_auc", "std")).reset_index()
    summary.to_csv(output / "cv_summary.csv", index=False)
    improved = summary[summary.model.isin(["tuned_logistic", "spline_logistic", "hist_boosting"])]
    best_ap = improved.ap_mean.max()
    # Predeclared practical tie: <= 0.005 AP favors the simpler model family.
    eligible = set(improved.loc[improved.ap_mean >= best_ap - 0.005, "model"])
    champion = next(n for n in ["tuned_logistic", "spline_logistic", "hist_boosting"] if n in eligible)
    thresholds = {n: select_threshold(y, p) for n, p in oof_rows.items() if n != "dummy"}
    np.savez_compressed(output / "private" / "development_oof.npz", y=y, **oof_rows)
    pd.DataFrame([{"model": n, **metrics(y, p, thresholds.get(n, 0.5))}
                  for n, p in oof_rows.items()]).to_csv(output / "development_thresholds.csv", index=False)
    # Interpretation is computed on development data only, with each model fitted without its validation fold.
    importance_rows = []
    model, grid = all_candidates[champion]
    for fold, (train, valid) in enumerate(outer):
        fitted, _, _ = fit_candidate(model, grid, x.iloc[train], y.iloc[train], 3100 + fold)
        imp = permutation_importance(fitted, x.iloc[valid], y.iloc[valid], scoring="average_precision",
                                     n_repeats=5, random_state=5100 + fold, n_jobs=1)
        for feature, values in zip(x.columns, imp.importances):
            for repeat, value in enumerate(values):
                importance_rows.append({"feature": feature, "fold": fold, "repeat": repeat, "ap_drop": value})
    pd.DataFrame(importance_rows).to_csv(output / "permutation_importance.csv", index=False)
    # Training-only descriptive evidence; no raw examples or customer identifiers are exported.
    eda = x.assign(Churn=y)
    for feature in ["Contract", "InternetService", "PaymentMethod"]:
        eda.groupby(feature).Churn.agg(["count", "mean"]).to_csv(output / f"eda_{feature}.csv")
    manifest = {
        "data_sha256": fingerprint(data_path), "rows": len(raw), "development_rows": len(dev_idx),
        "test_rows": len(test_idx), "development_churn_rate": float(y.mean()),
        "missing_total_charges": int(pd.to_numeric(raw.TotalCharges, errors="coerce").isna().sum()),
        "duplicate_predictor_rows": int(raw.drop(columns=["customerID", "Churn"]).duplicated().sum()),
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "champion": champion, "selection_metric": "nested CV average precision; 0.005 simplicity tolerance",
        "final_parameters": final_params, "thresholds": thresholds,
        "threshold_objective": "maximum development nested-OOF F1; no business cost assumption",
        "holdout_evaluated": False,
    }
    write_json(output / "selection.json", manifest)
    print("FROZEN SELECTION:", champion, thresholds[champion], flush=True)


def bootstrap(y, probabilities, champion, baseline, n_boot=2000):
    """Paired, class-stratified intervals conditional on fitted models and split."""
    rng = np.random.default_rng(2026)
    groups = [np.where(np.asarray(y) == c)[0] for c in [0, 1]]
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        target = np.asarray(y)[idx]
        cp, bp = probabilities[champion][idx], probabilities[baseline][idx]
        draws.append([roc_auc_score(target, cp), average_precision_score(target, cp),
                      roc_auc_score(target, cp) - roc_auc_score(target, bp),
                      average_precision_score(target, cp) - average_precision_score(target, bp)])
    intervals = np.percentile(draws, [2.5, 97.5], axis=0).T
    return {key: {"lower": float(lo), "upper": float(hi)} for key, (lo, hi) in zip(
        ["champion_roc_auc", "champion_ap", "delta_roc_auc_vs_original_boosting", "delta_ap_vs_original_boosting"], intervals)}


def evaluate(data_path, output):
    manifest = json.loads((output / "selection.json").read_text())
    if manifest["holdout_evaluated"]:
        raise RuntimeError("Holdout already evaluated. Use a separate output directory for a full reproduction.")
    if fingerprint(data_path) != manifest["data_sha256"]:
        raise ValueError("Dataset differs from frozen development run.")
    x, y, _ = load_data(data_path)
    _, test_idx = split_indices(y, x)
    xt, yt = x.iloc[test_idx], y.iloc[test_idx]
    results, probabilities = [], {}
    for name in candidates():
        model = joblib.load(output / "private" / f"{name}.joblib")
        p = model.predict_proba(xt)[:, 1]
        probabilities[name] = p
        results.append({"model": name, "operating_point": "default", **metrics(yt, p)})
        if name not in ["dummy", "original_tree"]:
            results.append({"model": name, "operating_point": "oof_f1", **metrics(yt, p, manifest["thresholds"][name])})
    pd.DataFrame(results).to_csv(output / "test_metrics.csv", index=False)
    np.savez_compressed(output / "private" / "test_predictions.npz", y=yt.to_numpy(), **probabilities)
    intervals = bootstrap(yt, probabilities, manifest["champion"], "original_boosting")
    write_json(output / "bootstrap_intervals.json", intervals)
    # Historical reproduction is separate, never used for new model selection.
    train, test = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
    history = []
    for name in ["original_logistic", "original_tree", "original_boosting"]:
        model = original_pipeline(name).fit(x.iloc[train], y.iloc[train])
        history.append({"model": name, **metrics(y.iloc[test], model.predict_proba(x.iloc[test])[:, 1])})
    pd.DataFrame(history).to_csv(output / "historical_reproduction.csv", index=False)
    # Top 20% capacity is predeclared; ranking evaluation does not tune a threshold on test labels.
    p = probabilities[manifest["champion"]]
    k = int(np.ceil(0.2 * len(yt)))
    order = np.argsort(-p, kind="stable")[:k]
    write_json(output / "capacity.json", {"contacted": k, "churners_found": int(yt.iloc[order].sum()),
               "precision": float(yt.iloc[order].mean()), "recall": float(yt.iloc[order].sum() / yt.sum()),
               "lift": float(yt.iloc[order].mean() / yt.mean()), "test_churn_rate": float(yt.mean())})
    manifest["holdout_evaluated"] = True
    write_json(output / "selection.json", manifest)
    print(pd.DataFrame(results).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["develop", "evaluate"])
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw" / DATA_FILE)
    parser.add_argument("--output", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        {"develop": develop, "evaluate": evaluate}[args.stage](args.data, args.output)


if __name__ == "__main__":
    main()
