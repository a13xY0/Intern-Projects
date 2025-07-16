# Intern-Projects
PROJECT 1:

Anomaly Detection for Directory Traversal Attacks
This project builds a pipeline for detecting Directory Traversal attacks using log data from Elasticsearch, combining rule-based filters with a Random Forest machine learning model.

Highlights
Data Ingestion: Logs are dynamically pulled from Elasticsearch, extracting key fields like url.path, ip, and timestamp.

Labeling: Logs labeled as DIRECTORY_TRAVERSAL_BEYOND_ROOT are treated as attacks; traffic logs (type TR) are used as benign samples.

Feature Engineering: URLs are transformed into numeric features such as length, number of dots/slashes, presence of traversal patterns, and trailing slash flags.

Modeling: A Random Forest Classifier is trained with Bayesian optimization (BayesSearchCV) for hyperparameter tuning. Evaluation includes accuracy, confusion matrix, and classification report.

Explainability: Model decisions are interpreted using feature importance plots and SHAP summaries.

Detection Logic: The system performs predictions on new logs and applies a regex-based rule engine for robust traversal pattern matching.

Model Management: Trained models are versioned by timestamp and automatically loaded for real-time use.

Stack
Languages & Libraries: Python, pandas, scikit-learn, matplotlib, SHAP, skopt

Data Layer: Elasticsearch (via elasticsearch-py)

Security: dotenv for credential handling

This hybrid approach enables real-time, interpretable attack detection, and the pipeline is designed to be extended for other types of threats.
