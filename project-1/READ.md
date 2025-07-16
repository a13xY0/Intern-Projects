Anomaly Detection for Directory Traversal Attacks
This project implements an automated anomaly detection pipeline to identify Directory Traversal attacks using real-world log data sourced from an Elasticsearch database. It integrates traditional rule-based detection techniques with a supervised machine learning approach, specifically a Random Forest Classifier, to enhance the detection of anomalous URL patterns indicative of malicious activity.

Key Components
Data Collection:
Logs are ingested from a live Elasticsearch instance. The pipeline extracts relevant fields including timestamps, URL paths, client IPs, hostnames, and country codes. The time window can be dynamically configured to retrieve specific historical or real-time data.

Labeling and Preprocessing:
Logs labeled as "DIRECTORY_TRAVERSAL_BEYOND_ROOT" are treated as positive samples (attacks), while traffic logs (log_type = TR) are treated as negative samples. Preprocessing includes flattening nested JSON structures and applying consistent formatting across features.

Feature Engineering:
Custom features were derived from URL paths to capture behavioral patterns, including:

URL length

Number of dots (.)

Number of slashes (/)

Number of traversal indicators (e.g., ../, %2e, %2f)

Binary flag for URLs ending with a slash

Model Training:
A Random Forest classifier was trained using these features, with hyperparameters optimized via Bayesian Optimization using BayesSearchCV for efficient tuning. The classifier was validated using stratified train-test splitting and evaluated using metrics like accuracy, confusion matrix, and classification report.

Model Interpretation:
Feature importance was visualized using:

Standard bar plots

SHAP (SHapley Additive exPlanations) summary plots for a deeper understanding of model decisions.

Inference & Rule-based Augmentation:
For recent log entries, predictions are made using the trained model. Additionally, a parallel rule-based filter using regular expressions targets common traversal patterns as a safeguard and interpretability layer.

Versioning and Deployment:
Trained models are serialized with timestamps for version tracking. The system supports loading the most recent model automatically for real-time predictions.

Technologies Used
Python, pandas, scikit-learn, matplotlib, SHAP

Elasticsearch (via elasticsearch-py)

dotenv for secure credential management

skopt for Bayesian hyperparameter tuning

This project demonstrates how traditional security signatures and machine learning models can be combined for more robust anomaly detection in cybersecurity pipelines. The modular design allows for extension to other attack types beyond directory traversal.