# Experimental results and methodology

## 1. Question and dataset

Can account and service attributes rank customers by churn risk, and support a useful retention contact policy? The target is `Churn=Yes`, departure within the last month. This is a retrospective classification benchmark, not a demonstrated forward-looking intervention system.

Source: [BlastChar / IBM Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), version 1. There are 7,043 rows, 19 predictors, an ID and a label; 1,869 churners (26.54%). Data SHA-256: `88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a`. Eleven missing `TotalCharges` entries all have zero tenure. There are 40 repeated predictor rows beyond the first occurrence; identical profiles include some conflicting labels. They are not assumed to be erroneous duplicates and are retained.

## 2. Audit of the original approach

The original clean implementation dropped customer ID, parsed blank billing totals to zero, encoded binary features ordinally, one-hot encoded multi-category features, standardized numeric inputs, and used a stratified 80/20 split (seed 42). It compared balanced logistic regression, a shallow balanced tree and sample-weighted gradient boosting. Its pipelines correctly fitted encoders and scalers on training data.

Weaknesses addressed:

- Repeated test inspection informed model choice. Recall-only searches favored excessive false alarms and did not define a contact cost or capacity constraint. No threshold policy was selected separately.
- PR performance and uncertainty were missing. Accuracy and recall alone obscure the cost of imbalance. There was no nested evaluation of hyperparameter search.
- An exploratory variant used different seeds for full-feature and contract-only comparisons, preventing a paired comparison. Broad imports, repeated plotting code, suppressed warnings and hidden notebook state reduced reproducibility.
- Gradient-boosting weights were calculated before CV rather than independently in each fit. Recomputed fold weights now avoid using validation class frequencies in training weights.
- Identical profiles could cross split boundaries. Importance scores and correlated logistic coefficients were sometimes read too directly as business mechanisms. Impurity importance does not establish direction or causation; collinearity makes individual coefficient signs fragile.

### Historical reproduction (seed 42; context only)

| Model | ROC-AUC | AP | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| Original logistic | 0.842 | 0.633 | 0.504 | 0.783 | 0.614 |
| Original decision tree | 0.814 | 0.549 | 0.514 | 0.783 | 0.621 |
| Original gradient boosting | 0.845 | 0.658 | 0.513 | 0.810 | 0.628 |

These recreate the original model settings on its original split, including the original precision/recall operating points. They match recorded ROC-AUC, precision, recall and F1 to three decimals. AP is newly calculated. They must not be compared directly with the new holdout table because the samples differ. The large original recall-only tuning grids are not rerun: the original three model settings establish a sufficient baseline without repeating hundreds of unhelpful search candidates.

## 3. Evaluation protocol

1. Reserve 1,409 customers before any current model comparison. Eligibility excludes customers in the historical seed-40 and seed-42 test sets and any repeated predictor profile; select stratified with seed 2026. The remaining 5,634 rows form development. This avoids recycling historical evaluation rows and exact profiles across the final split, at the cost of a slightly restricted holdout population. Development contains all repeated profiles.
2. Five outer **StratifiedGroupKFold** folds (seed 2026) evaluate model families. Groups are hashes of all 19 predictor fields. Identical profiles never cross outer or inner boundaries. Each outer training fold uses a separate three-fold grouped, stratified hyperparameter search, seeds 3100–3104. All learned preprocessing, class weights and spline knots are fitted inside each training fold.
3. Search metric: average precision. Compare outer-fold mean AP. A predeclared tolerance of 0.005 favors tuned linear logistic, then spline logistic, then histogram boosting. This chooses **Spline logistic**. Outer CV evaluates each search procedure; choosing the family on these outer scores still creates selection optimism, so the final holdout remains the primary assessment.
4. Refit each family with a three-fold development-only search (seed 4100). Freeze family, final hyperparameters and thresholds in `selection.json`. Thresholds maximize pooled nested out-of-fold F1, restricted to [0.05, 0.95]. Those development F1 values are optimized diagnostics, not unbiased performance estimates; threshold selection is assessed on the holdout.
5. Evaluate all frozen models in one final holdout pass. Reporting alternatives is descriptive; no post-test changes are made. The selected model remains the development-selected choice even if another model has a favorable test metric.

Historical full-dataset EDA and original fits mean no row can honestly be called wholly unseen across the entire project history. The test is untouched by **current** tuning and threshold selection; it is not a new external sample. The grouped design protects exact profiles, not unknown households or correlated accounts.

### Nested CV results

| Model | Outer AP mean | Fold SD | Outer ROC-AUC mean | Fold SD |
| --- | --- | --- | --- | --- |
| No-skill baseline | 0.266 | 0.012 | 0.500 | 0.000 |
| Tuned histogram boosting | 0.654 | 0.031 | 0.843 | 0.012 |
| Original gradient boosting | 0.654 | 0.029 | 0.844 | 0.012 |
| Original logistic | 0.646 | 0.042 | 0.842 | 0.013 |
| Original decision tree | 0.552 | 0.030 | 0.816 | 0.013 |
| Spline logistic | 0.660 | 0.036 | 0.846 | 0.013 |
| Tuned logistic | 0.648 | 0.041 | 0.843 | 0.013 |

Fold SD describes variation across five folds; it is not a confidence interval and folds are not independent replications. Per-fold metrics and search choices are in `cv_folds.csv`. Final inner-search results in `search_*.csv` are tuning diagnostics, not unbiased comparisons.

### Compact search space

- Linear and spline logistic: C ∈ {0.01, 0.1, 1, 10}, class weights ∈ {none, balanced} (8 candidates per family). Median numeric imputation, standardization, unknown-safe full one-hot categories. Spline variant uses four knots, cubic basis, no bias term, applied to tenure, monthly charges and total charges before scaling.
- Histogram boosting: 180 iterations, no early-stopping validation; four combinations of leaf count / L2 / learning rate: (7, 1, .05), (7, 10, .1), (15, 1, .05), (15, 10, .1). Each is evaluated with and without row-local engineered features (8 candidates). The same dense one-hot preprocessing keeps dependencies small.
- Engineered candidates: log tenure, historical average monthly charge (total / tenure, missing for zero tenure), current minus historical average charge, and count of security/backup/support subscriptions. No target aggregates, ID-derived features or resampling.
- Fixed original model settings and a prior-only dummy baseline are evaluated on the same outer folds. Balanced gradient boosting recalculates training weights per fold.

Final model parameters:

```json
{
  "dummy": {},
  "original_logistic": {},
  "original_tree": {},
  "original_boosting": {},
  "tuned_logistic": {
    "model__C": 10.0,
    "model__class_weight": null
  },
  "spline_logistic": {
    "model__C": 0.1,
    "model__class_weight": null
  },
  "hist_boosting": {
    "features__enabled": false,
    "model__l2_regularization": 1.0,
    "model__learning_rate": 0.05,
    "model__max_leaf_nodes": 7
  }
}
```

## 4. Final holdout

| Model / threshold | ROC-AUC | PR-AUC (AP) | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| No-skill baseline · 0.500 | 0.500 | 0.263 | 0.000 | 0.000 | 0.000 |
| Original logistic · 0.500 | 0.856 | 0.680 | 0.521 | 0.814 | 0.635 |
| Original decision tree · 0.500 | 0.835 | 0.598 | 0.526 | 0.787 | 0.631 |
| Original gradient boosting · 0.500 | 0.857 | 0.694 | 0.524 | 0.795 | 0.632 |
| Tuned logistic · 0.500 | 0.855 | 0.683 | 0.692 | 0.593 | 0.639 |
| Spline logistic · 0.500 | 0.863 | 0.711 | 0.723 | 0.569 | 0.637 |
| Tuned histogram boosting · 0.500 | 0.861 | 0.698 | 0.710 | 0.580 | 0.638 |
| Original boosting · 0.538 (OOF F1) | 0.857 | 0.694 | 0.549 | 0.779 | 0.644 |
| **Spline logistic · 0.308** | 0.863 | 0.711 | 0.576 | 0.790 | 0.666 |

1409 customers; 371 churners (26.33%). All metrics concern the churn class. AP is the recall-weighted precision summary implemented by `average_precision_score`, not trapezoidal PR area. Every confusion count, threshold, accuracy and Brier score is in `test_metrics.csv`.

### Changes that helped, and those that did not

- **Nonlinear numeric effects:** spline logistic improves nested mean AP over tuned linear logistic by 0.0114. This controlled family comparison supports keeping spline terms; it is a small dataset result, not proof of a general advantage.
- **Regularization / class weighting:** the final spline search selects C=0.1 and no class weights. Corrected linear preprocessing and C tuning alone change nested AP only slightly relative to the original logistic. Search results show the alternatives; no isolated benefit is claimed for imputation or one-hot changes.
- **Advanced boosting / engineered features:** histogram boosting has nested AP 0.6540, compared with 0.6539 for original boosting. Engineered features are disabled in its final best configuration. Added complexity did not establish useful improvement. This bundled search is not a full causal ablation of every new feature.
- **Threshold selection:** moving the selected model from 0.500 to 0.308185 changes test precision 0.723 → 0.576, recall 0.569 → 0.790, and F1 0.637 → 0.666. ROC-AUC and AP are unchanged because scores and ranking are unchanged. The original boosting model also benefits from threshold tuning (see `test_metrics.csv`), so not all operating-point gains should be attributed to a better estimator.
- **Common-population comparison:** against original boosting at 0.5, the selected operating point has 52 fewer false alarms, 2 fewer detected churners and F1 0.632 → 0.666. Against threshold-tuned original boosting, compare the explicit `oof_f1` rows; both policies were chosen on development data.
- **Probability quality:** selected-model Brier score is 0.1251, original weighted boosting 0.1543 (lower is better). Balanced fitting changes effective class priors; raw weighted scores should not be treated as calibrated churn probabilities. No separate calibration model was tuned or tested.

### Uncertainty

2,000 paired, class-stratified bootstrap resamples of the final holdout, seed 2026, give 95% percentile intervals. The same resampled customers are used for both fitted models.

- Selected ROC-AUC: [0.841, 0.884].
- Selected AP: [0.669, 0.752].
- ROC-AUC difference vs original boosting: +0.006, interval [+0.000, +0.011].
- AP difference vs original boosting: +0.017, interval [+0.003, +0.032].

Both difference intervals are above zero, supporting a small positive ranking difference conditional on these fitted models and this holdout. This is not evidence of a general advantage across datasets or future time periods. Intervals condition on the fitted models and this split; they omit training and model-selection uncertainty and do not establish significance for the threshold-policy difference. No paired t-test on five dependent folds or multiple-comparison significance claims are made.

## 5. Business interpretation

**Default retention policy example:** at t=0.308, contact 509 customers (36.1% of the holdout), including 293 churners and 216 non-churners; miss 78 churners. This explains workload and error counts without assigning invented monetary costs.

**Limited contact capacity:** a predeclared top-20% ranking rule contacts 282 customers (ceiling of 20% of 1,409), including 205 churners. Precision 72.7%, recall 55.3%, lift 2.76× prevalence. This is a score-ranked capacity rule, not a test-label-optimized threshold. It measures identified historical churners, not preventable churn.

Development-only category summaries (`eda_*.csv`) show where to investigate risk. Five outer-validation folds × five permutations measure the selected model's reliance on original raw features, with refitting and tuning confined to the corresponding training folds. The largest signals are tenure, contract, internet service. Error bars are SD across 25 fold/permutation values, not confidence intervals. Correlated variables can share or hide importance; shuffling can create unrealistic combinations, and the metric gives neither causal direction nor intervention effect.

Operational next steps: test onboarding and service-review outreach within appropriate high-risk segments, randomize eligible customers into treatment/control, measure incremental retention and contribution margin, and select a contact policy using real cost, capacity and response data. Do not use demographic features directly as outreach criteria without a subgroup/fairness review. No uplift or ROI estimate is available from this dataset.

## 6. Remaining limits

- Fictional static sample; no timestamps, temporal drift, treatment response, service quality or support history. Current features may describe the same period as churn. Pre-outcome availability cannot be verified, so the workflow prevents procedural leakage but cannot certify absence of source-level timing leakage.
- Holdout eligibility deliberately restricts the evaluated population; historical dataset exposure cannot be undone. External/time-based validation is needed before deployment.
- Moderate minority prevalence; no need for synthetic oversampling. Class-weight search and threshold selection address different questions and should not be conflated with probability calibration.
- Small ranking gains and conditional intervals; no broad claim that this model will outperform boosting on new data. Hyperparameter search is intentionally compact and does not establish a global optimum.
- No production deployment, causal analysis, profitability estimate, formal subgroup audit or calibrated intervention policy. These require appropriate additional data.

## 7. Reproducibility and references

Run the commands in [README](../README.md). `develop` writes model selection before `evaluate` opens holdout features/labels for prediction and scoring. Split construction uses labels for stratification only. The data loader validates schema and missingness without learning statistics. Private fitted models and row-level predictions stay under ignored `reports/private/`; only aggregate tables, JSON settings and figures are publishable. A separate output directory reproduces a complete run without replacing the reference results. `churn.verify` compares every published numeric table and selection artifact.

Reference environment: Python 3.12.2, scikit-learn 1.5.0; other explicit versions are pinned in `requirements.txt`. Floating point results may differ slightly across platforms. The supplied notebooks are executed from top to bottom and contain only aggregate displays and generated figures.

- [Kaggle dataset and displayed data terms](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [IBM sample context](https://www.ibm.com/docs/en/cognos-analytics/11.2.x?topic=samples-telco-customer-churn)
- [scikit-learn: average precision](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.average_precision_score.html)
- [scikit-learn: cross-validation](https://scikit-learn.org/1.5/modules/cross_validation.html)
- [scikit-learn: decision thresholds](https://scikit-learn.org/1.5/modules/classification_threshold.html)
