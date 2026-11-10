# Dataset access and attribution

Download version 1 of [Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), published by **BlastChar** from the **IBM sample data**. Extract `WA_Fn-UseC_-Telco-Customer-Churn.csv` into `data/raw/`. A Kaggle account may be required. No credentials are needed by the experiment code.

The original file has 7,043 rows and 21 columns: an identifier, 19 predictors, and `Churn`. The label records departure within the last month. The predictors describe demographics, account tenure, services, contracts, payments and charges. `TotalCharges` contains 11 blank values, all at zero tenure.

Kaggle currently displays **“Data files © Original Authors”**, checked 10 November 2026. This is not a clear redistribution grant. The repository therefore contains no raw data, customer records, customer-level predictions, or fitted models. Obtain the file directly from the source and comply with its terms. No CC0 or other open data license is claimed here. Aggregate results and plots credit that source.

IBM describes its telco sample as a fictional provider: [IBM sample documentation](https://www.ibm.com/docs/en/cognos-analytics/11.2.x?topic=samples-telco-customer-churn). That documentation includes expanded variants; this project uses only the original 21-column Kaggle CSV, without churn scores, churn reasons or customer status fields.

The expected SHA-256 fingerprint is recorded in `reports/selection.json`. A different dataset version needs a separate experiment and must not be compared as an exact reproduction.
