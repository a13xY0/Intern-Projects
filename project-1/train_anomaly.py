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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
import matplotlib.pyplot as plt
import shap
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from requests.packages.urllib3.exceptions import InsecureRequestWarning # type: ignore
# Suppress only the InsecureRequestWarning
warnings.simplefilter('ignore', InsecureRequestWarning)


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
env_path = Path(__file__).resolve().parent / ".env"

load_dotenv(dotenv_path=env_path)
ELASTIC=os.getenv("ELASTIC")
USER_NAME=os.getenv("USER_NAME")
ELASTIC_PASSWORD=os.getenv("ELASTIC_PASSWORD")
INDEX_NAME=os.getenv("INDEX_NAME")

print("Connecting to:", ELASTIC)

es = Elasticsearch(
        hosts=[ELASTIC],
        basic_auth=(USER_NAME,ELASTIC_PASSWORD),  
        verify_certs=False
    )
print("Connected")

# Current time in UTC
now = datetime.now(timezone.utc)

#TIME RANGE FOR TR LOGS
tr_start_time = (now - timedelta(hours=1))
tr_end_time = now

#TIME RANGE FOR WF LOGS
wf_start_time = (now - timedelta(days=5)).replace(hour=0, minute=0, second=0, microsecond=0)
wf_end_time = (now - timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=999999)

def fetch_directory_traversal_logs(es, index_name, start_time, end_time, scroll_size=1000):
    query = {
        "_source": ["@timestamp", "url.path"],
        "size": scroll_size,
        "query": {
            "bool": {
                "must": [
                    {
                        "term": {
                            "barracuda.waf.log_type": "WF"
                        }
                    },
                    {
                        "term": {
                            "barracuda.waf.attack_description": "DIRECTORY_TRAVERSAL_BEYOND_ROOT"
                        }
                    },
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
            if isinstance(doc.get("url"), dict):
                doc["url"] = doc["url"].get("path", "")
        df = pd.DataFrame(docs)
        print(f" Retrieved {len(df)} records.")
        return df

    except Exception as e:
        print(" Error fetching data:", e)
        return pd.DataFrame()

def fetch_all_unique_url_paths(es, index_name, start_time, end_time, batch_size=1000):
    unique_paths = []
    after_key = None

    while True:
        query = {
            "size": 0,
            "query": {
                "bool": {
                    "must": [
                        {"term": {"barracuda.waf.log_type": "WF"}},
                        {"term": {"barracuda.waf.attack_description": "DIRECTORY_TRAVERSAL_BEYOND_ROOT"}},
                        {"range": {
                            "@timestamp": {
                                "gte": start_time.isoformat(),
                                "lte": end_time.isoformat()
                            }
                        }}
                    ]
                }
            },
            "aggs": {
                "unique_paths": {
                    "composite": {
                        "size": batch_size,
                        "sources": [
                            {"url_path": {"terms": {"field": "url.path"}}}
                        ],
                        **({"after": after_key} if after_key else {})
                    }
                }
            }
        }

        try:
            response = es.search(index=index_name, body=query)
            buckets = response["aggregations"]["unique_paths"]["buckets"]
            unique_paths.extend([b["key"]["url_path"] for b in buckets])

            if "after_key" in response["aggregations"]["unique_paths"]:
                after_key = response["aggregations"]["unique_paths"]["after_key"]
            else:
                break  #All data fetched

        except Exception as e:
            print("Error fetching composite aggregation:", e)
            break

    print(f"Total unique url.path values: {len(unique_paths)}")
    return pd.DataFrame(unique_paths, columns=["url_path"])


def fetch_tr_logs(es, index_name, start_time, end_time, scroll_size=1000):
    query = {
        "_source": ["@timestamp", "url.path"],
        "size": scroll_size,
        "query": {
            "bool": {
                "must": [
                    {"term": {"barracuda.waf.log_type": "TR"}},
                    {"range": {
                        "@timestamp": {
                            "gte": start_time.isoformat(),
                            "lte": end_time.isoformat()
                        }
                    }}
                ]
            }
        }
    }

    try:
        results = scan(es, query=query, index=index_name)
        docs = [doc["_source"] for doc in results]
        for doc in docs:
            if isinstance(doc.get("url"), dict):
                doc["url"] = doc["url"].get("path", "")
        df = pd.DataFrame(docs)
        print(f"Retrieved {len(df)} TR records.")
        return df

    except Exception as e:
        print("Error fetching TR data:", e)
        return pd.DataFrame()

df_unique_paths = fetch_all_unique_url_paths(es, INDEX_NAME, wf_start_time, wf_end_time)
df_unique_paths.to_csv("unique_directory_traversal_paths.csv", index=False)

docs = fetch_directory_traversal_logs(es, INDEX_NAME, start_time=wf_start_time, end_time=wf_end_time)
docs["Dir.Trav"] = 1  # Label as attack
docs.to_csv("wf_directory_traversal_logs_labeled.csv", index=False)

print(f"Total directory traversal logs fetched: {len(docs)}")

docs_tr = fetch_tr_logs(es, INDEX_NAME, start_time=tr_start_time, end_time=tr_end_time)
docs_tr["Dir.Trav"] = 0  # Label as non-attack
docs_tr_unique = docs_tr.drop_duplicates(subset="url")
docs_tr_unique.to_csv("unique_tr_paths_labeled.csv", index=False)

merge = pd.concat([docs, docs_tr_unique], ignore_index=True)

# Feature Eng.

# Feature 1: URL Length
merge["url_length"] = merge["url"].astype(str).apply(len)

# Feature 2: Number of dots
merge["num_dots"] = merge["url"].astype(str).apply(lambda x: x.count('.'))

# Feature 3: Number of slashes
merge["num_slashes"] = merge["url"].astype(str).apply(lambda x: x.count('/'))

# Feature 4: Number of traversal patterns (../, %2e, %2f)
merge["num_traversals"] = merge["url"].astype(str).apply(
    lambda x: len(re.findall(r"(\.\./|%2e|%2f)", x, flags=re.IGNORECASE))
)

# Feature 5: Ends with slash
merge["ends_with_slash"] = merge["url"].astype(str).apply(lambda x: int(x.endswith('/')))

#features and target
features = ["url_length", "num_dots", "num_slashes", "num_traversals", "ends_with_slash"]
target = "Dir.Trav"

#Data Split
X_train, X_test, y_train, y_test = train_test_split(
    merge[features],
    merge[target],
    test_size=0.2,         
    random_state=42,       
    stratify=merge[target]
)

print("Data split complete.")
print(f"Training set: {X_train.shape[0]} rows")
print(f"Test set: {X_test.shape[0]} rows")

#Traing using hyperparameters
search_space = {
    'n_estimators': Integer(50, 300),
    'max_depth': Integer(3, 20),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 20),
    'max_features': Real(0.1, 1.0, prior='uniform')
}

opt = BayesSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    search_spaces=search_space,
    n_iter=32,                
    cv=3,                     
    scoring='accuracy',      
    random_state=42,
    n_jobs=-1                 
)

print("Starting Bayesian Optimization:")
opt.fit(X_train, y_train)
print("Best parameters found:", opt.best_params_)

opt_results = opt.cv_results_

#scores per iter.
scores = opt_results['mean_test_score']

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(scores) + 1), scores, marker='o')
plt.title("Bayesian Optimization Score per Iteration")
plt.xlabel("Iteration")
plt.ylabel("CV Mean Accuracy")
plt.grid(True)
plt.tight_layout()
plt.show()

# Use the best model
rf = opt.best_estimator_
model_name = f"random_forest_{timestamp}"

#Save and load a trained model 
#Save  trained model to file
import pickle
model_path = fr"C:{directory_path}"

with open(model_path, 'wb') as f:
    pickle.dump(rf, f)

print(f"Model saved to: {model_path}")
with open(model_path, 'rb') as f:
    loaded_rf = pickle.load(f)

#Load Model
y_pred_loaded = loaded_rf.predict(X_test)
#Predict on test set
y_pred = rf.predict(X_test)

print("\n Random Forest Classification Report: ")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

importances = rf.feature_importances_
feature_names = features

plt.figure(figsize=(8, 4))
plt.barh(feature_names, importances)
plt.xlabel("Feature Importance")
plt.title("Random Forest - Feature Importance")
plt.tight_layout()
plt.show()

explainer = shap.Explainer(rf, X_train) 
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, feature_names=features)

print(f"Total TR logs fetched: {len(docs_tr)}")
print(f"Unique TR url.path entries: {len(docs_tr_unique)}")

print(f"Total directory traversal logs fetched: {len(docs)}")
