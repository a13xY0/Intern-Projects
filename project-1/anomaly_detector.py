import os
import re
import pandas as pd
from elasticsearch import Elasticsearch
from datetime import datetime
import pandas as pd
from elasticsearch.helpers import scan
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from pathlib import Path
import warnings
from requests.packages.urllib3.exceptions import InsecureRequestWarning # type: ignore
# Suppress only the InsecureRequestWarning
warnings.simplefilter('ignore', InsecureRequestWarning)

TEMP = {
    "size": 0,
    "_source": False,
    "query": {
        "range": {
            "@timestamp": {
                "gte": "",
                "lt": ""
            }
        }
    },
    "aggs": {
        "by_src_window": {
            "composite": {
                "size": 1000,
                "sources": [
                    {"client_ip": {"terms": {"field": "client.ip"}}},
                    {"host_name": {"terms": {"field": "host.name"}}},
                    {"window_start": {"date_histogram": {
                        "field": "@timestamp", "fixed_interval": "5m"
                    }}}
                ]
            },
            "aggs": {
                "distinct_urls": {
                    "cardinality": {
                        "field": "url.path",
                        "precision_threshold": 300
                    }
                }
            }
        }
    }
}

#timestamp for model versioning
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

env_path = Path(__file__).resolve().parent / ".env"

load_dotenv(dotenv_path=env_path)
ELASTIC=os.getenv("ELASTIC")
USER_NAME=os.getenv("USER_NAME")
ELASTIC_PASSWORD=os.getenv("ELASTIC_PASSWORD")
INDEX_NAME=os.getenv("INDEX_NAME")

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


end_time = datetime.now(timezone.utc) - timedelta(days=1)
start_time = end_time.replace(hour=0, minute=0, second=0, microsecond=0)
end_time = end_time.replace(hour=9, minute=59, second=59, microsecond=999999)


def fetch_recent_logs(es, index_name, start_time, end_time, scroll_size=1000):
    print(f"Fetching logs from {start_time.isoformat()} to {end_time.isoformat()}")

    query = {
        "_source": ["@timestamp", "url.path", "client.geo.country_iso_code", "host.name", "client.ip"],
        "size": scroll_size,
        "query": {
            "bool": {
                "must": [
                    {
                        "range": {
                            "@timestamp": {
                                "gte": start_time.isoformat(),
                                "lte": end_time.isoformat()
                            }
                        }
                    }
                ]
            }
        }
        
    }

    try:
        results = scan(es, query=query, index=index_name)
        docs = [doc["_source"] for doc in results]

        for doc in docs:
            
            url = doc.get("url")
            if isinstance(url, dict):
                doc["url.path"] = url.get("path", "")
            elif isinstance(url, str):
                doc["url.path"] = url

            client = doc.get("client", {})
            if isinstance(client, dict):
                doc["client.ip"] = client.get("ip")
                geo = client.get("geo", {})
                if isinstance(geo, dict):
                    doc["client.geo.country_iso_code"] = geo.get("country_iso_code")

            host = doc.get("host", {})
            if isinstance(host, dict):
                doc["host.name"] = host.get("name")

        df = pd.DataFrame(docs)

        desired_columns = {
            "@timestamp": "timestamp",
            "url.path": "url_path",
            "client.geo.country_iso_code": "country",
            "host.name": "host",
            "client.ip": "ip"
        }

        for col in desired_columns.keys():
            if col not in df.columns:
                df[col] = None

        df = df[list(desired_columns.keys())].rename(columns=desired_columns)
        df = df.rename(columns={"url.path": "url_path"})

        print(f"\nRetrieved {len(df)} records in the last {end_time}.\n")

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)

        return df

    except Exception as e:
        print("Error fetching data:", e)
        return pd.DataFrame()

df_recent_logs = fetch_recent_logs(es, INDEX_NAME, start_time, end_time)
df_recent_logs["window_start"] = pd.to_datetime(df_recent_logs["timestamp"]).dt.floor("5min")


# Feature Engineering
df_recent_logs["url_length"] = df_recent_logs["url_path"].astype(str).apply(len)
df_recent_logs["num_dots"] = df_recent_logs["url_path"].astype(str).apply(lambda x: x.count('.'))
df_recent_logs["num_slashes"] = df_recent_logs["url_path"].astype(str).apply(lambda x: x.count('/'))
df_recent_logs["num_traversals"] = df_recent_logs["url_path"].astype(str).apply(
    lambda x: len(re.findall(r"(\.\./|%2e|%2f)", x, flags=re.IGNORECASE))
)
df_recent_logs["ends_with_slash"] = df_recent_logs["url_path"].astype(str).apply(lambda x: int(x.endswith('/')))

# Load latest trained model
import pickle
import glob

model_files = sorted(glob.glob("C:/Users/avryonides/Desktop/Python Codes/project/random_forest_*.pkl"), reverse=True)
latest_model_path = model_files[0]

with open(latest_model_path, 'rb') as f:
    model = pickle.load(f)

print(f"Loaded model from: {latest_model_path}")

# Rule-based Directory Traversal detection
traversal_pattern = re.compile(
    r"(%2e%2e%2f|%2e%2e/|\.\./|\.\.%2f|%2e%2e%5c|\.\.\\|%2e%2e\\|\.{2}%5c|%252e%252e%255c|\.{2}%255c)",
    flags=re.IGNORECASE
)
)
df_recent_logs["Dir.Trav.Prediction"] = df_recent_logs["url_path"].astype(str).apply(
    lambda x: 1 if traversal_pattern.search(x) else 0
)

print("\nPredictions on recent data:")
desired_output_columns = ["timestamp", "window_start", "url_path", "country", "host", "ip", "Dir.Trav.Prediction"]

existing_output_columns = [col for col in desired_output_columns if col in df_recent_logs.columns]

print(df_recent_logs[existing_output_columns])
