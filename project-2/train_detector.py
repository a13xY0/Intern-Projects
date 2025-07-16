import os
import json
import pandas as pd
from elasticsearch import Elasticsearch
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from pathlib import Path
import warnings
import copy
from utilities import active_ma_sites
from requests.packages.urllib3.exceptions import InsecureRequestWarning # type: ignore
# Suppress only the InsecureRequestWarning
warnings.simplefilter('ignore', InsecureRequestWarning)

AGG_TEMPLATE = {
    "size": 0,
    "_source": False,
    "query": {"range": {"@timestamp": {"gte": "", "lt": ""}}},
    "aggs": {
        "by_src_window": {
            "composite": {
                "size": 1000,
                "sources": [
                    {"client_ip": {"terms": {"field": "client.ip"}}},
                    {"host_name": {"terms": {"field": "host.name"}}},
                    {"window_start": {"date_histogram": {
                        "field": "@timestamp", "fixed_interval": "5m"}}}
                ]
            },
            "aggs": {
                "distinct_urls": {"cardinality": {"field": "url.path", "precision_threshold": 300}},
                "methods_variety": {"cardinality": {"field": "http.request.method", "precision_threshold": 40}},
                "unique_user_agents": {"cardinality": {"field": "user_agent.original", "precision_threshold": 400}},
                "bytes_avg_request": {"avg": {"field": "http.request.bytes"}},
                "bytes_avg_response": {"avg": {"field": "http.response.bytes"}},
                "avg_response_time": {"avg": {"field": "barracuda.waf.response_timetaken"}},
                "avg_server_time": {"avg": {"field": "barracuda.waf.server_time"}},
                "error_reqs": {"filter": {"range": {"http.response.status_code": {"gte": 400}}}},
                "error_ratio": {
                    "bucket_script": {
                        "buckets_path": {"err": "error_reqs>_count", "tot": "_count"},
                        "script": "params.tot > 0 ? params.err / params.tot : 0"
                    }
                }
            }
        }
    }
}

env_path = Path(__file__).resolve().parent / ".env"

load_dotenv(dotenv_path=env_path)
ELASTIC=os.getenv("ELASTIC")
USER_NAME=os.getenv("USER_NAME")
ELASTIC_PASSWORD=os.getenv("ELASTIC_PASSWORD")
INDEX_NAME=os.getenv("INDEX_NAME")
ALL_MANAGE_URL=os.getenv("ALL_MANAGE_URL")

print("Connecting to:", ELASTIC)

es = Elasticsearch(
        hosts=[ELASTIC],
        basic_auth=(USER_NAME,ELASTIC_PASSWORD),  
        verify_certs=False
    )
print("Connected")

now = datetime.now(timezone.utc)

end_time = datetime.now(timezone.utc) - timedelta(days=10)
start_time = end_time.replace(hour=0, minute=0, second=0, microsecond=0)
end_time = end_time.replace(hour=23, minute=59, second=59, microsecond=999999)

def fetch_recent_logs_aggregated(es, index_name, start_time, end_time):
    print(f"Aggregating logs from {start_time.isoformat()} to {end_time.isoformat()}")

    body = copy.deepcopy(AGG_TEMPLATE)
    body["query"]["range"]["@timestamp"]["gte"] = start_time.isoformat()
    body["query"]["range"]["@timestamp"]["lt"] = end_time.isoformat()

    all_data = []
    after_key = None

    while True:
        if after_key:
            body["aggs"]["by_src_window"]["composite"]["after"] = after_key

        response = es.search(index=index_name, body=body)
        buckets = response["aggregations"]["by_src_window"]["buckets"]

        for b in buckets:
            row = {
                "client_ip": b["key"]["client_ip"],
                "host_name": b["key"]["host_name"],
                "window_start": datetime.fromtimestamp(b["key"]["window_start"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "distinct_urls": b["distinct_urls"]["value"],
                "methods_variety": b["methods_variety"]["value"],
                "unique_user_agents": b["unique_user_agents"]["value"],
                "bytes_avg_request": b["bytes_avg_request"]["value"] or 0.0,
                "bytes_avg_response": b["bytes_avg_response"]["value"] or 0.0,
                "avg_response_time": b["avg_response_time"]["value"] or 0.0,
                "avg_server_time": b["avg_server_time"]["value"] or 0.0,
                "error_ratio": b["error_ratio"]["value"],
                "total_requests": b["doc_count"]
            }
            all_data.append(row)

        after_key = response["aggregations"]["by_src_window"].get("after_key")
        if not after_key:
            break

    return pd.DataFrame(all_data)


df_recent_logs_aggregated = fetch_recent_logs_aggregated(es, INDEX_NAME, start_time, end_time)
print(df_recent_logs_aggregated)


ma_links_pd = active_ma_sites(ALL_MANAGE_URL)
df_recent_logs_aggregated = df_recent_logs_aggregated.merge(ma_links_pd, how="inner", left_on="host_name", right_on="Domain")
df_recent_logs_aggregated = df_recent_logs_aggregated.drop(["MA_Links", "Domain"], axis=1)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from skorch import NeuralNetRegressor
from skorch.callbacks import EpochScoring
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from sklearn.preprocessing import MinMaxScaler

# Step 1: Prepare Data
feature_cols = [
    "distinct_urls", "methods_variety", "unique_user_agents",
    "bytes_avg_request", "bytes_avg_response",
    "avg_response_time", "avg_server_time",
    "error_ratio", "total_requests"
]

X_full = df_recent_logs_aggregated[feature_cols].fillna(0)

#10% of the data for tuning
X_sampled = X_full.sample(frac=0.1, random_state=42)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_sampled).astype(np.float32)

input_tensor = torch.tensor(X_scaled, dtype=torch.float32)
dataset = TensorDataset(input_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Step 2: Define dynamic Autoencoder

class DynamicAutoencoder(nn.Module):
    def __init__(self, input_dim=9, ld=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2 * ld),
            nn.ReLU(),
            nn.Linear(2 * ld, ld),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(ld, 2 * ld),
            nn.ReLU(),
            nn.Linear(2 * ld, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# Step 3: Prepare for BayesSearchCV

net = NeuralNetRegressor(
    module=DynamicAutoencoder,
    module__input_dim=X_scaled.shape[1],
    module__ld=4,
    max_epochs=20,
    lr=0.001,
    optimizer=torch.optim.Adam,
    criterion=nn.MSELoss,
    batch_size=64,
    train_split=None,
    verbose=0
)

search_space = {
    'lr': Real(1e-4, 1e-2, prior='log-uniform'),
    'module__ld': Integer(3, 5)
}


opt = BayesSearchCV(
    estimator=net,
    search_spaces=search_space,
    n_iter=15,
    scoring='neg_mean_squared_error',
    cv=3,
    random_state=42,
    n_jobs=-1
)

print("Tuning autoencoder with BayesSearchCV...")
X_scaled_float32 = X_scaled.astype(np.float32)
opt.fit(X_scaled_float32, X_scaled_float32)


print("Best hyperparameters:", opt.best_params_)
best_model = opt.best_estimator_
best_ld = opt.best_params_['module__ld']
best_lr = opt.best_params_['lr']

model_save_dir = "{directory path}"
os.makedirs(model_save_dir, exist_ok=True)
model_path = os.path.join(model_save_dir, "best_autoencoder.pt")
best_model.save_params(f_params=model_path)
print(f"Model saved to: {model_path}")

import joblib
scaler_path = os.path.join(model_save_dir, "scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to: {scaler_path}")
# Save best parameters to JSON
params = {
    "module__ld": best_ld,
    "lr": best_lr
}
params_path = os.path.join(model_save_dir, "best_params.json")
with open(params_path, "w") as f:
    json.dump(params, f)
print(f"Best parameters saved to: {params_path}")


df_recent_logs_aggregated["best_ld"] = best_ld
df_recent_logs_aggregated["best_lr"] = best_lr

# Step 4: Compute Anomaly Scores

X_full_scaled = scaler.transform(X_full).astype(np.float32)

with torch.no_grad():
    reconstructions_full = best_model.predict(X_full_scaled)


test_recon_error = np.mean((X_full_scaled - reconstructions_full) ** 2, axis=1)
threshold = np.percentile(test_recon_error, 99)

df_recent_logs_aggregated["anomaly_score"] = test_recon_error
df_recent_logs_aggregated["is_anomaly"] = df_recent_logs_aggregated["anomaly_score"] > threshold

# Step 5: Output Results
print("\nAnomaly detection complete.")
print(
    df_recent_logs_aggregated[
        ["client_ip", "window_start"] + feature_cols + ["anomaly_score", "is_anomaly"]
    ].sort_values("anomaly_score", ascending=False).head(10)
)

output_dir = "/home/administrator/behaviorate/scripts/intern_code/Alexandros Codes/project/anomalies/"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "anomaly_results.csv")
df_recent_logs_aggregated.to_csv(output_path, index=False)
print("\nThreshold is: ",threshold)
print(f"\nResults saved to: {output_path}")
