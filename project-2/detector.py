import os
import pandas as pd
from elasticsearch import Elasticsearch
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from pathlib import Path
import warnings
from utilities import active_ma_sites
from requests.packages.urllib3.exceptions import InsecureRequestWarning # type: ignore
# Suppress only the InsecureRequestWarning
warnings.simplefilter('ignore', InsecureRequestWarning)

import copy  # required for deepcopy

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

# Construct the path to the .env file (parent directory)
env_path = Path(__file__).resolve().parent / ".env"
# Load the .env file
load_dotenv(dotenv_path=env_path)
ELASTIC=os.getenv("ELASTIC")
USER_NAME=os.getenv("USER_NAME")
ELASTIC_PASSWORD=os.getenv("ELASTIC_PASSWORD")
INDEX_NAME=os.getenv("INDEX_NAME")
ALL_MANAGE_URL=os.getenv("ALL_MANAGE_URL")

print("Connecting to:", ELASTIC)

# Connect to Elasticsearch
es = Elasticsearch(
        hosts=[ELASTIC],
        basic_auth=(USER_NAME,ELASTIC_PASSWORD),  
        verify_certs=False
    )
print("Connected")

# Current time in UTC
now = datetime.now(timezone.utc)

# Fetch logs from exactly X time ago
end_time = datetime.now(timezone.utc) - timedelta(days=1)
start_time = end_time.replace(hour=0, minute=0, second=0, microsecond=0)
end_time = end_time.replace(hour=0, minute=4, second=59, microsecond=999999)

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
#Filter: Keep only MA active sites
ma_links_pd = active_ma_sites(ALL_MANAGE_URL)
df_recent_logs_aggregated = df_recent_logs_aggregated.merge(
    ma_links_pd,
    how="inner",
    left_on="host_name",
    right_on="Domain"
)
df_recent_logs_aggregated = df_recent_logs_aggregated.drop(["MA_Links", "Domain"], axis=1)

print(df_recent_logs_aggregated)

import torch
import torch.nn as nn
import numpy as np
from skorch import NeuralNetRegressor
import joblib

feature_cols = [
    "distinct_urls", "methods_variety", "unique_user_agents",
    "bytes_avg_request", "bytes_avg_response",
    "avg_response_time", "avg_server_time",
    "error_ratio", "total_requests"
]

X_full = df_recent_logs_aggregated[feature_cols].fillna(0)
# Load the trained scaler from disk
scaler_path = "/home/administrator/behaviorate/scripts/intern_code/Alexandros Codes/project/models/scaler.pkl"
scaler = joblib.load(scaler_path)

# Apply same transformation as used during training
X_full_scaled = scaler.transform(X_full).astype(np.float32)


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

from skorch import NeuralNetRegressor

import json
params_path = "/home/administrator/behaviorate/scripts/intern_code/Alexandros Codes/project/models/best_params.json"
with open(params_path, "r") as f:
    best_params = json.load(f)

best_ld = best_params["module__ld"]
best_lr = best_params["lr"]

model = NeuralNetRegressor(
    module=DynamicAutoencoder,
    module__input_dim=9,
    module__ld=best_ld,
    max_epochs=1,  # not used during inference
    lr=best_lr,
    train_split=None,
    verbose=0
)
model.initialize()
model_path = "/home/administrator/behaviorate/scripts/intern_code/Alexandros Codes/project/models/best_autoencoder.pt"
model.load_params(f_params=model_path)

with torch.no_grad():
    reconstructions = model.predict(X_full_scaled)

test_recon_error = np.mean((X_full_scaled - reconstructions) ** 2, axis=1)
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

output_path = os.path.join(output_dir, "detector_results.csv")
df_recent_logs_aggregated.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")
