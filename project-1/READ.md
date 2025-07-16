#  Anomaly Detection for Directory Traversal Attacks

This project implements an **automated anomaly detection pipeline** to identify **Directory Traversal attacks** using real-world log data from an **Elasticsearch** database. It combines traditional **rule-based detection** techniques with a **supervised machine learning model** (Random Forest Classifier) to enhance anomaly detection in URL patterns indicative of malicious behavior.

---

###  Data Collection

- Logs are ingested from a live **Elasticsearch** instance.
- Extracted fields:
  - Timestamps
  - URL paths
  - Client IPs
  - Hostnames
  - Country codes
- The time window is **dynamically configurable** to retrieve specific historical or real-time data.

###  Labeling & Preprocessing

- Logs labeled as `"DIRECTORY_TRAVERSAL_BEYOND_ROOT"` → **Positive samples (attacks)**
- Logs with `log_type = TR` → **Negative samples (benign traffic)**
- Preprocessing steps:
  - Flattening nested JSON
  - Consistent formatting across all features

###  Feature Engineering

Custom features are derived from URL paths to capture behavioral patterns:

- URL length
- Number of dots (`.`)
- Number of slashes (`/`)
- Number of traversal indicators (e.g., `../`, `%2e`, `%2f`)
- Binary flag for URLs ending with a slash (`/`)

###  Model Training

- **Random Forest Classifier** trained on engineered features
- **Hyperparameter tuning** via **Bayesian Optimization** using `BayesSearchCV`
- Evaluation metrics:
  - Accuracy
  - Confusion matrix
  - Classification report

###  Model Interpretation

Model insights were visualized using:

- Bar plots for **feature importance**
- **SHAP** (SHapley Additive exPlanations) summary plots for model explainability

###  Inference & Rule-based Augmentation

- Predicts recent log entries using the trained model
- A parallel **rule-based filter** (regex-based) augments detection of common traversal patterns  
  → Adds interpretability and acts as a safeguard

###  Versioning & Deployment

- Trained models are **timestamped and serialized** for version control
- The system supports automatic loading of the **latest model** for real-time predictions

---

## 🛠 Technologies Used

- `Python`, `pandas`, `scikit-learn`, `matplotlib`, `SHAP`
- `Elasticsearch` (via `elasticsearch-py`)
- `dotenv` for secure credential management
- `skopt` for Bayesian hyperparameter optimization

---

##  Summary

This project showcases how **traditional security signatures** and **machine learning** can be **combined** for robust anomaly detection. Its modular architecture allows easy extension to detect other attack types beyond directory traversal.
