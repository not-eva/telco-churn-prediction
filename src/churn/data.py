"""Strict input validation and a reproducible experiment split."""
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]


def load_data(path):
    """Parse types without learning any statistics from the full dataset."""
    raw = pd.read_csv(path)
    expected = set(NUMERIC + CATEGORICAL + ["customerID", "Churn"])
    if set(raw.columns) != expected or len(raw) != 7043:
        raise ValueError("Expected the original 7,043-row, 21-column Kaggle CSV.")
    if raw.customerID.isna().any() or not raw.customerID.is_unique:
        raise ValueError("Customer identifiers must be nonmissing and unique.")
    if not raw.Churn.isin(["Yes", "No"]).all():
        raise ValueError("Unexpected target labels.")
    x = raw.drop(columns=["customerID", "Churn"]).copy()
    x["SeniorCitizen"] = x.SeniorCitizen.map({0: "No", 1: "Yes"})
    x["TotalCharges"] = pd.to_numeric(x.TotalCharges, errors="coerce")
    missing = x.TotalCharges.isna()
    if not x.loc[missing, "tenure"].eq(0).all():
        raise ValueError("Unexpected missing billing totals outside zero-tenure accounts.")
    return x, raw.Churn.map({"No": 0, "Yes": 1}), raw


def profile_groups(x):
    """Conservatively group identical predictor profiles, including missing values."""
    return pd.util.hash_pandas_object(x, index=False).to_numpy()


def split_indices(y, x):
    """Keep the new holdout out of both historically used evaluation sets.

    Historical analyses explored this dataset already. This is an untouched
    holdout for the current experiment, not independent external validation.
    """
    indices = np.arange(len(y))
    historical = set()
    for seed in (40, 42):
        _, test = train_test_split(indices, test_size=0.2, stratify=y, random_state=seed)
        historical.update(test)
    # Duplicate profiles may be different customers; keep them together anyway.
    # Retain all repeated profiles in development and group them within CV.
    groups = pd.Series(profile_groups(x))
    repeated = groups.duplicated(keep=False).to_numpy()
    eligible = np.array([i for i in indices if i not in historical and not repeated[i]])
    _, test = train_test_split(
        eligible, test_size=1409, stratify=y.iloc[eligible], random_state=2026
    )
    test = np.sort(test)
    development = np.setdiff1d(indices, test)
    assert not historical.intersection(test)
    assert not set(development).intersection(test)
    assert not set(groups.iloc[development]).intersection(groups.iloc[test])
    return development, test


def fingerprint(path):
    return sha256(Path(path).read_bytes()).hexdigest()
