"""Generate every README metric and figure from saved experimental results."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from .experiment import ROOT

LABELS = {"dummy": "No-skill baseline", "original_logistic": "Original logistic", "original_tree": "Original decision tree",
          "original_boosting": "Original gradient boosting", "tuned_logistic": "Tuned logistic",
          "spline_logistic": "Spline logistic", "hist_boosting": "Tuned histogram boosting"}
COLORS = {"original_boosting": "#75859b", "tuned_logistic": "#d78a26", "spline_logistic": "#087e8b", "hist_boosting": "#805cb5"}


def table(frame, columns, labels=None):
    labels = labels or columns
    lines = ["| " + " | ".join(labels) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        vals = [f"{row[c]:.3f}" if isinstance(row[c], (float, np.floating)) else str(row[c]) for c in columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def make_report(output, project_root=ROOT):
    selection = json.loads((output / "selection.json").read_text())
    if not selection["holdout_evaluated"]:
        raise RuntimeError("Evaluate the frozen experiment before making the report.")
    metrics = pd.read_csv(output / "test_metrics.csv")
    cv = pd.read_csv(output / "cv_summary.csv")
    intervals = json.loads((output / "bootstrap_intervals.json").read_text())
    capacity = json.loads((output / "capacity.json").read_text())
    champion = selection["champion"]
    selected = metrics[(metrics.model == champion) & (metrics.operating_point == "oof_f1")].iloc[0]
    default = metrics[(metrics.model == champion) & (metrics.operating_point == "default")].iloc[0]
    baseline = metrics[(metrics.model == "original_boosting") & (metrics.operating_point == "default")].iloc[0]
    predictions = np.load(output / "private/test_predictions.npz")
    y = predictions["y"]
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 14,
                         "axes.labelcolor": "#24364b", "text.color": "#172b42", "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white", "savefig.facecolor": "white"})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for name in COLORS:
        p = predictions[name]
        row = metrics[(metrics.model == name) & (metrics.operating_point == "default")].iloc[0]
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        style = {"color": COLORS[name], "lw": 2.4 if name == champion else 1.6, "alpha": 1 if name == champion else 0.85}
        axes[0].plot(fpr, tpr, label=f"{LABELS[name]}  {row.roc_auc:.3f}", **style)
        axes[1].step(recall, precision, where="post", label=f"{LABELS[name]}  {row.average_precision:.3f}", **style)
    axes[0].plot([0, 1], [0, 1], "--", color="#b4bfcb", lw=1)
    axes[1].axhline(y.mean(), ls="--", color="#b4bfcb", lw=1, label=f"Prevalence  {y.mean():.3f}")
    axes[0].set(title="ROC · ranking all customers", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].set(title="Precision–recall · finding churners", xlabel="Recall", ylabel="Precision")
    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1.02))
        ax.grid(alpha=0.16)
        ax.legend(loc="lower right" if ax == axes[0] else "lower left", frameon=True, fontsize=8.5,
                  title="ROC-AUC" if ax == axes[0] else "Average precision", title_fontsize=9)
    fig.suptitle(f"Final holdout · {len(y):,} customers · selected before evaluation", fontsize=16, fontweight="bold")
    fig.savefig(figures / "ranking_curves.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), layout="constrained")
    for ax, row, title in zip(axes, [baseline, selected], ["Original gradient boosting · t = 0.500", f"Spline logistic · t = {selected.threshold:.3f}"]):
        cm = np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=int)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(baseline.tn, selected.tn))
        names = [["Correctly unflagged", "Unnecessary contacts"], ["Missed churners", "Churners found"]]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]:,}\n{names[i][j]}", ha="center", va="center", fontsize=11,
                        color="white" if cm[i,j] > 500 else "#172b42")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Stay", "Churn"], yticklabels=["Stay", "Churn"],
               xlabel="Predicted outcome", ylabel="Actual outcome", title=title)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Contact fewer non-churners at similar recall", fontweight="bold", fontsize=16)
    fig.savefig(figures / "confusion_matrices.png", dpi=180)
    plt.close(fig)

    importance = pd.read_csv(output / "permutation_importance.csv")
    grouped = importance.groupby("feature").ap_drop.agg(["mean", "std"]).sort_values("mean", ascending=False)
    top = grouped.head(8).sort_values("mean")
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    ax.barh([{"tenure": "Tenure", "InternetService": "Internet service", "MonthlyCharges": "Monthly charges",
              "PaymentMethod": "Payment method", "OnlineSecurity": "Online security", "TechSupport": "Tech support",
              "PaperlessBilling": "Paperless billing"}.get(c, c) for c in top.index], top["mean"], xerr=top["std"], color="#087e8b", ecolor="#62768a", capsize=3, height=0.65)
    ax.axvline(0, color="#a0aab8", lw=1)
    ax.set(xlabel="Decrease in average precision when a feature is shuffled", ylabel="",
           title="What the selected model relies on")
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    fig.suptitle("Development validation folds · 5 folds × 5 permutations", fontsize=11, color="#62768a")
    fig.savefig(figures / "feature_importance.png", dpi=180)
    plt.close(fig)

    display = metrics[metrics.operating_point.eq("default")].copy()
    display["Model / operating point"] = display.model.map(LABELS) + " · 0.500"
    baseline_tuned = metrics[(metrics.model == "original_boosting") & (metrics.operating_point == "oof_f1")].copy()
    baseline_tuned["Model / operating point"] = baseline_tuned.threshold.map(lambda t: f"Original boosting · {t:.3f} (OOF F1)")
    extra = selected.to_frame().T
    extra["Model / operating point"] = f"**Spline logistic · {selected.threshold:.3f}**"
    display = pd.concat([display, baseline_tuned, extra], ignore_index=True)
    cols = ["Model / operating point", "roc_auc", "average_precision", "precision", "recall", "f1"]
    comparison = table(display, cols, ["Model / threshold", "ROC-AUC", "PR-AUC (AP)", "Precision", "Recall", "F1"])
    delta_ap = intervals["delta_ap_vs_original_boosting"]
    delta_auc = intervals["delta_roc_auc_vs_original_boosting"]
    friendly = {"tenure": "Tenure", "Contract": "Contract", "InternetService": "Internet service",
                "MonthlyCharges": "Monthly charges", "PaymentMethod": "Payment method", "OnlineSecurity": "Online security",
                "TechSupport": "Tech support", "PaperlessBilling": "Paperless billing"}
    feature_names = ", ".join(friendly.get(c, c).lower() for c in grouped.head(3).index)
    readme = f"""# Customer Churn Prediction

**Prioritize retention outreach with a model that balances missed churners and unnecessary contacts.** This study compares transparent logistic models with gradient boosting on the IBM Telco Customer Churn sample, available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

The selected **spline logistic regression** reaches **{selected.roc_auc:.3f} ROC-AUC**, **{selected.average_precision:.3f} average precision** and **{selected.f1:.3f} F1** on a reserved 1,409-customer holdout. At a threshold chosen on development data, it flags **{int(selected.tp)} of {int(selected.tp + selected.fn)} churners**, with **{int(baseline.fp-selected.fp)} fewer false alarms** than the original boosting baseline, while finding {int(baseline.tp-selected.tp)} fewer churners.

## Results

All models use the **same development and holdout customers**. Original methods are refitted for a fair comparison; historical scores are [reported separately](reports/results.md).

{comparison}

Positive class: churn. PR-AUC means **average precision (AP)**. Threshold {selected.threshold:.3f} maximizes development out-of-fold F1, not profit.

![ROC and precision–recall curves on the final holdout](reports/figures/ranking_curves.png)

The AP gain over original boosting is **{selected.average_precision-baseline.average_precision:+.3f}**, with a paired 95% bootstrap interval of **[{delta_ap['lower']:+.3f}, {delta_ap['upper']:+.3f}]**. This supports a small ranking gain on this holdout; the interval excludes training and model-selection uncertainty.

![Original and selected model confusion matrices](reports/figures/confusion_matrices.png)

## What this means for retention

- **Work within capacity:** ranking the top 20% contacts {capacity['contacted']} customers, finds {capacity['churners_found']} churners and achieves {capacity['precision']:.1%} precision ({capacity['lift']:.2f}× the holdout churn rate). These are detected churners, not customers saved.
- **Investigate predictive signals:** {feature_names} are the largest permutation signals. Use them to investigate onboarding and service reviews; they do not establish causation.
- **Measure incremental value:** test outreach against a randomized control group. Contact costs, margins and treatment effects are needed to estimate ROI.

![Validation-fold permutation importance for the selected model](reports/figures/feature_importance.png)

## Method and trade-offs

The data has **7,043 customers, 19 predictors and 26.5% churn**. Pipelines handle missing billing totals, encode categories and model numeric effects. Five outer / three inner grouped, stratified CV folds select settings by AP. IDs are excluded; identical profiles stay together. All choices are frozen before holdout evaluation.

Spline effects helped more than linear tuning; engineered features and histogram boosting did not improve selection. The final fit uses no class weights. [Detailed experiments](reports/results.md) separate estimator gains from threshold changes.

**Limits:** a fictional static benchmark, previously explored. This holdout excludes repeated profiles and historical test customers; it is untouched by current selection, not external validation. Deployment needs verified pre-outcome features, temporal validation, calibration, subgroup checks and a cost-based policy.

## Reproduce

Use **Python 3.12**. [Download the original CSV](data/README.md) to `data/raw/`; raw data and customer-level outputs are excluded because Kaggle lists “Data files © Original Authors,” without a clear redistribution grant.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m unittest discover -s tests -v
python -m churn.experiment develop --output reports/reproduction
python -m churn.experiment evaluate --output reports/reproduction
python -m churn.verify --actual reports/reproduction --expected reports
jupyter lab
```

Open the executed notebooks for EDA and model analysis. After a default-output training/evaluation run, `python -m churn.report` regenerates the reference README and figures. Seeds, settings and the data checksum are in `reports/selection.json`. Reproduction outputs remain separate and ignored.

```text
├── README.md
├── data/README.md              # source, access and data terms
├── notebooks/                 # executed EDA and experiment analysis
├── src/churn/                 # pipelines, experiment, reports, verification
├── reports/                   # aggregate results, method, three figures
├── tests/                     # split, preprocessing and metric safeguards
├── requirements.txt
└── pyproject.toml
```

Data credit: **IBM sample data / BlastChar on Kaggle**. See [dataset terms and attribution](data/README.md). No data redistribution license is asserted.
"""
    (project_root / "README.md").write_text(readme)

    cv_display = cv.copy()
    cv_display["Model"] = cv_display.model.map(LABELS)
    cv_table = table(cv_display, ["Model", "ap_mean", "ap_sd", "auc_mean", "auc_sd"],
                     ["Model", "Outer AP mean", "Fold SD", "Outer ROC-AUC mean", "Fold SD"])
    historic = pd.read_csv(output / "historical_reproduction.csv")
    historic["Model"] = historic.model.map(LABELS)
    historical_table = table(historic, ["Model", "roc_auc", "average_precision", "precision", "recall", "f1"],
                             ["Model", "ROC-AUC", "AP", "Precision", "Recall", "F1"])
    report = f"""# Experimental results and methodology

## 1. Question and dataset

Can account and service attributes rank customers by churn risk, and support a useful retention contact policy? The target is `Churn=Yes`, departure within the last month. This is a retrospective classification benchmark, not a demonstrated forward-looking intervention system.

Source: [BlastChar / IBM Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), version 1. There are 7,043 rows, 19 predictors, an ID and a label; 1,869 churners (26.54%). Data SHA-256: `{selection['data_sha256']}`. Eleven missing `TotalCharges` entries all have zero tenure. There are {selection['duplicate_predictor_rows']} repeated predictor rows beyond the first occurrence; identical profiles include some conflicting labels. They are not assumed to be erroneous duplicates and are retained.

## 2. Audit of the original approach

The original clean implementation dropped customer ID, parsed blank billing totals to zero, encoded binary features ordinally, one-hot encoded multi-category features, standardized numeric inputs, and used a stratified 80/20 split (seed 42). It compared balanced logistic regression, a shallow balanced tree and sample-weighted gradient boosting. Its pipelines correctly fitted encoders and scalers on training data.

Weaknesses addressed:

- Repeated test inspection informed model choice. Recall-only searches favored excessive false alarms and did not define a contact cost or capacity constraint. No threshold policy was selected separately.
- PR performance and uncertainty were missing. Accuracy and recall alone obscure the cost of imbalance. There was no nested evaluation of hyperparameter search.
- An exploratory variant used different seeds for full-feature and contract-only comparisons, preventing a paired comparison. Broad imports, repeated plotting code, suppressed warnings and hidden notebook state reduced reproducibility.
- Gradient-boosting weights were calculated before CV rather than independently in each fit. Recomputed fold weights now avoid using validation class frequencies in training weights.
- Identical profiles could cross split boundaries. Importance scores and correlated logistic coefficients were sometimes read too directly as business mechanisms. Impurity importance does not establish direction or causation; collinearity makes individual coefficient signs fragile.

### Historical reproduction (seed 42; context only)

{historical_table}

These recreate the original model settings on its original split, including the original precision/recall operating points. They match recorded ROC-AUC, precision, recall and F1 to three decimals. AP is newly calculated. They must not be compared directly with the new holdout table because the samples differ. The large original recall-only tuning grids are not rerun: the original three model settings establish a sufficient baseline without repeating hundreds of unhelpful search candidates.

## 3. Evaluation protocol

1. Reserve 1,409 customers before any current model comparison. Eligibility excludes customers in the historical seed-40 and seed-42 test sets and any repeated predictor profile; select stratified with seed 2026. The remaining 5,634 rows form development. This avoids recycling historical evaluation rows and exact profiles across the final split, at the cost of a slightly restricted holdout population. Development contains all repeated profiles.
2. Five outer **StratifiedGroupKFold** folds (seed 2026) evaluate model families. Groups are hashes of all 19 predictor fields. Identical profiles never cross outer or inner boundaries. Each outer training fold uses a separate three-fold grouped, stratified hyperparameter search, seeds 3100–3104. All learned preprocessing, class weights and spline knots are fitted inside each training fold.
3. Search metric: average precision. Compare outer-fold mean AP. A predeclared tolerance of 0.005 favors tuned linear logistic, then spline logistic, then histogram boosting. This chooses **{LABELS[champion]}**. Outer CV evaluates each search procedure; choosing the family on these outer scores still creates selection optimism, so the final holdout remains the primary assessment.
4. Refit each family with a three-fold development-only search (seed 4100). Freeze family, final hyperparameters and thresholds in `selection.json`. Thresholds maximize pooled nested out-of-fold F1, restricted to [0.05, 0.95]. Those development F1 values are optimized diagnostics, not unbiased performance estimates; threshold selection is assessed on the holdout.
5. Evaluate all frozen models in one final holdout pass. Reporting alternatives is descriptive; no post-test changes are made. The selected model remains the development-selected choice even if another model has a favorable test metric.

Historical full-dataset EDA and original fits mean no row can honestly be called wholly unseen across the entire project history. The test is untouched by **current** tuning and threshold selection; it is not a new external sample. The grouped design protects exact profiles, not unknown households or correlated accounts.

### Nested CV results

{cv_table}

Fold SD describes variation across five folds; it is not a confidence interval and folds are not independent replications. Per-fold metrics and search choices are in `cv_folds.csv`. Final inner-search results in `search_*.csv` are tuning diagnostics, not unbiased comparisons.

### Compact search space

- Linear and spline logistic: C ∈ {{0.01, 0.1, 1, 10}}, class weights ∈ {{none, balanced}} (8 candidates per family). Median numeric imputation, standardization, unknown-safe full one-hot categories. Spline variant uses four knots, cubic basis, no bias term, applied to tenure, monthly charges and total charges before scaling.
- Histogram boosting: 180 iterations, no early-stopping validation; four combinations of leaf count / L2 / learning rate: (7, 1, .05), (7, 10, .1), (15, 1, .05), (15, 10, .1). Each is evaluated with and without row-local engineered features (8 candidates). The same dense one-hot preprocessing keeps dependencies small.
- Engineered candidates: log tenure, historical average monthly charge (total / tenure, missing for zero tenure), current minus historical average charge, and count of security/backup/support subscriptions. No target aggregates, ID-derived features or resampling.
- Fixed original model settings and a prior-only dummy baseline are evaluated on the same outer folds. Balanced gradient boosting recalculates training weights per fold.

Final model parameters:

```json
{json.dumps(selection['final_parameters'], indent=2)}
```

## 4. Final holdout

{comparison}

{len(y)} customers; {int(y.sum())} churners ({y.mean():.2%}). All metrics concern the churn class. AP is the recall-weighted precision summary implemented by `average_precision_score`, not trapezoidal PR area. Every confusion count, threshold, accuracy and Brier score is in `test_metrics.csv`.

### Changes that helped, and those that did not

- **Nonlinear numeric effects:** spline logistic improves nested mean AP over tuned linear logistic by {float(cv.set_index('model').loc['spline_logistic','ap_mean']-cv.set_index('model').loc['tuned_logistic','ap_mean']):.4f}. This controlled family comparison supports keeping spline terms; it is a small dataset result, not proof of a general advantage.
- **Regularization / class weighting:** the final spline search selects C={selection['final_parameters']['spline_logistic']['model__C']} and no class weights. Corrected linear preprocessing and C tuning alone change nested AP only slightly relative to the original logistic. Search results show the alternatives; no isolated benefit is claimed for imputation or one-hot changes.
- **Advanced boosting / engineered features:** histogram boosting has nested AP {float(cv.set_index('model').loc['hist_boosting','ap_mean']):.4f}, compared with {float(cv.set_index('model').loc['original_boosting','ap_mean']):.4f} for original boosting. Engineered features are disabled in its final best configuration. Added complexity did not establish useful improvement. This bundled search is not a full causal ablation of every new feature.
- **Threshold selection:** moving the selected model from 0.500 to {selected.threshold:.6f} changes test precision {default.precision:.3f} → {selected.precision:.3f}, recall {default.recall:.3f} → {selected.recall:.3f}, and F1 {default.f1:.3f} → {selected.f1:.3f}. ROC-AUC and AP are unchanged because scores and ranking are unchanged. The original boosting model also benefits from threshold tuning (see `test_metrics.csv`), so not all operating-point gains should be attributed to a better estimator.
- **Common-population comparison:** against original boosting at 0.5, the selected operating point has {int(baseline.fp-selected.fp)} fewer false alarms, {int(baseline.tp-selected.tp)} fewer detected churners and F1 {baseline.f1:.3f} → {selected.f1:.3f}. Against threshold-tuned original boosting, compare the explicit `oof_f1` rows; both policies were chosen on development data.
- **Probability quality:** selected-model Brier score is {selected.brier:.4f}, original weighted boosting {baseline.brier:.4f} (lower is better). Balanced fitting changes effective class priors; raw weighted scores should not be treated as calibrated churn probabilities. No separate calibration model was tuned or tested.

### Uncertainty

2,000 paired, class-stratified bootstrap resamples of the final holdout, seed 2026, give 95% percentile intervals. The same resampled customers are used for both fitted models.

- Selected ROC-AUC: [{intervals['champion_roc_auc']['lower']:.3f}, {intervals['champion_roc_auc']['upper']:.3f}].
- Selected AP: [{intervals['champion_ap']['lower']:.3f}, {intervals['champion_ap']['upper']:.3f}].
- ROC-AUC difference vs original boosting: {selected.roc_auc-baseline.roc_auc:+.3f}, interval [{delta_auc['lower']:+.3f}, {delta_auc['upper']:+.3f}].
- AP difference vs original boosting: {selected.average_precision-baseline.average_precision:+.3f}, interval [{delta_ap['lower']:+.3f}, {delta_ap['upper']:+.3f}].

Both difference intervals are above zero, supporting a small positive ranking difference conditional on these fitted models and this holdout. This is not evidence of a general advantage across datasets or future time periods. Intervals condition on the fitted models and this split; they omit training and model-selection uncertainty and do not establish significance for the threshold-policy difference. No paired t-test on five dependent folds or multiple-comparison significance claims are made.

## 5. Business interpretation

**Default retention policy example:** at t={selected.threshold:.3f}, contact {int(selected.tp+selected.fp)} customers ({(selected.tp+selected.fp)/len(y):.1%} of the holdout), including {int(selected.tp)} churners and {int(selected.fp)} non-churners; miss {int(selected.fn)} churners. This explains workload and error counts without assigning invented monetary costs.

**Limited contact capacity:** a predeclared top-20% ranking rule contacts {capacity['contacted']} customers (ceiling of 20% of 1,409), including {capacity['churners_found']} churners. Precision {capacity['precision']:.1%}, recall {capacity['recall']:.1%}, lift {capacity['lift']:.2f}× prevalence. This is a score-ranked capacity rule, not a test-label-optimized threshold. It measures identified historical churners, not preventable churn.

Development-only category summaries (`eda_*.csv`) show where to investigate risk. Five outer-validation folds × five permutations measure the selected model's reliance on original raw features, with refitting and tuning confined to the corresponding training folds. The largest signals are {feature_names}. Error bars are SD across 25 fold/permutation values, not confidence intervals. Correlated variables can share or hide importance; shuffling can create unrealistic combinations, and the metric gives neither causal direction nor intervention effect.

Operational next steps: test onboarding and service-review outreach within appropriate high-risk segments, randomize eligible customers into treatment/control, measure incremental retention and contribution margin, and select a contact policy using real cost, capacity and response data. Do not use demographic features directly as outreach criteria without a subgroup/fairness review. No uplift or ROI estimate is available from this dataset.

## 6. Remaining limits

- Fictional static sample; no timestamps, temporal drift, treatment response, service quality or support history. Current features may describe the same period as churn. Pre-outcome availability cannot be verified, so the workflow prevents procedural leakage but cannot certify absence of source-level timing leakage.
- Holdout eligibility deliberately restricts the evaluated population; historical dataset exposure cannot be undone. External/time-based validation is needed before deployment.
- Moderate minority prevalence; no need for synthetic oversampling. Class-weight search and threshold selection address different questions and should not be conflated with probability calibration.
- Small ranking gains and conditional intervals; no broad claim that this model will outperform boosting on new data. Hyperparameter search is intentionally compact and does not establish a global optimum.
- No production deployment, causal analysis, profitability estimate, formal subgroup audit or calibrated intervention policy. These require appropriate additional data.

## 7. Reproducibility and references

Run the commands in [README](../README.md). `develop` writes model selection before `evaluate` opens holdout features/labels for prediction and scoring. Split construction uses labels for stratification only. The data loader validates schema and missingness without learning statistics. Private fitted models and row-level predictions stay under ignored `reports/private/`; only aggregate tables, JSON settings and figures are publishable. A separate output directory reproduces a complete run without replacing the reference results. `churn.verify` compares every published numeric table and selection artifact.

Reference environment: Python {selection['python']}, scikit-learn {selection['sklearn']}; other explicit versions are pinned in `requirements.txt`. Floating point results may differ slightly across platforms. The supplied notebooks are executed from top to bottom and contain only aggregate displays and generated figures.

- [Kaggle dataset and displayed data terms](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [IBM sample context](https://www.ibm.com/docs/en/cognos-analytics/11.2.x?topic=samples-telco-customer-churn)
- [scikit-learn: average precision](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.average_precision_score.html)
- [scikit-learn: cross-validation](https://scikit-learn.org/1.5/modules/cross_validation.html)
- [scikit-learn: decision thresholds](https://scikit-learn.org/1.5/modules/classification_threshold.html)
"""
    (output / "results.md").write_text(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    args = parser.parse_args()
    make_report(args.output, args.project_root)


if __name__ == "__main__":
    main()
