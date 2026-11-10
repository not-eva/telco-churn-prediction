"""Guard the evaluation boundaries, not just individual implementation lines."""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from churn.data import DATA_FILE, load_data, profile_groups, split_indices
from churn.experiment import metrics, select_threshold
from churn.models import improved_pipeline

ROOT = Path(__file__).resolve().parents[1]


class EvaluationTests(unittest.TestCase):
    def test_threshold_and_confusion_counts(self):
        y, p = np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])
        threshold = select_threshold(y, p)
        result = metrics(y, p, threshold)
        self.assertAlmostEqual(result["f1"], 0.8)
        self.assertEqual((result["tn"], result["fp"], result["fn"], result["tp"]), (1, 1, 0, 2))

    def test_imputation_and_scaling_only_learn_training_rows(self):
        training = pd.DataFrame({"tenure": [1., 2., 3., 4.], "Contract": ["A", "B", "A", "B"]})
        pipe = improved_pipeline("tuned_logistic").fit(training, [0, 1, 0, 1])
        numeric = pipe.named_steps["preprocessor"].named_transformers_["num"]
        before = numeric.named_steps["scale"].mean_.copy()
        prediction = pipe.predict_proba(pd.DataFrame({"tenure": [1e9, np.nan], "Contract": ["new", "A"]}))
        np.testing.assert_array_equal(before, numeric.named_steps["scale"].mean_)
        np.testing.assert_allclose(numeric.named_steps["fill"].statistics_, [2.5])
        self.assertTrue(np.isfinite(prediction).all())

    @unittest.skipUnless((ROOT / "data/raw" / DATA_FILE).exists(), "Download Kaggle data for split checks")
    def test_holdout_and_nested_folds_do_not_share_profiles(self):
        x, y, _ = load_data(ROOT / "data/raw" / DATA_FILE)
        dev, test = split_indices(y, x)
        self.assertEqual(len(test), 1409)
        self.assertEqual(len(dev), 5634)
        self.assertNotIn("customerID", x)
        self.assertNotIn("Churn", x)
        self.assertFalse(set(dev) & set(test))
        self.assertEqual(set(dev) | set(test), set(range(len(y))))
        groups = profile_groups(x)
        self.assertFalse(set(groups[dev]) & set(groups[test]))
        for seed in (40, 42):
            _, historic = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed, stratify=y)
            self.assertFalse(set(test) & set(historic))
        xd, yd, gd = x.iloc[dev], y.iloc[dev], groups[dev]
        for train, valid in StratifiedGroupKFold(5, shuffle=True, random_state=2026).split(xd, yd, gd):
            self.assertFalse(set(gd[train]) & set(gd[valid]))
            for inner_train, inner_valid in StratifiedGroupKFold(3, shuffle=True, random_state=3100).split(xd.iloc[train], yd.iloc[train], gd[train]):
                self.assertFalse(set(gd[train][inner_train]) & set(gd[train][inner_valid]))


if __name__ == "__main__":
    unittest.main()
