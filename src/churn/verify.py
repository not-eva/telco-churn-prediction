"""Compare a complete reproduction against the checked-in reference experiment."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


def verify(actual, expected):
    csvs = sorted(expected.glob("*.csv"))
    if not csvs:
        raise AssertionError("No reference tables found.")
    for reference in csvs:
        left, right = pd.read_csv(actual / reference.name), pd.read_csv(reference)
        assert_frame_equal(left, right, check_exact=False, rtol=1e-8, atol=1e-10)
    for name in ["selection.json", "bootstrap_intervals.json", "capacity.json"]:
        left = json.loads((actual / name).read_text())
        right = json.loads((expected / name).read_text())
        if left != right:
            # Environment version strings are deliberately part of exact reproducibility.
            raise AssertionError(f"JSON result mismatch: {name}")
    historical = pd.read_csv(actual / "historical_reproduction.csv").set_index("model")
    recorded = {"original_logistic": [0.841732, 0.504303, 0.783422, 0.613613],
                "original_tree": [0.814112, 0.514035, 0.783422, 0.620763],
                "original_boosting": [0.845050, 0.512690, 0.810160, 0.627979]}
    for model, values in recorded.items():
        np.testing.assert_allclose(historical.loc[model, ["roc_auc", "precision", "recall", "f1"]].to_numpy(float), values, atol=5e-4, rtol=0)
    print(f"PASS: {len(csvs)} aggregate tables, 3 JSON artifacts, and original recorded metrics reproduced.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    args = parser.parse_args()
    verify(args.actual, args.expected)


if __name__ == "__main__":
    main()
