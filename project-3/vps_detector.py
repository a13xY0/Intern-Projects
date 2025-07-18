import geoip2.database
import os
import pandas as pd
from elasticsearch import Elasticsearch
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from pathlib import Path
import warnings
import re
from utilities import active_ma_sites
from requests.packages.urllib3.exceptions import InsecureRequestWarning # type: ignore
# Suppress only the InsecureRequestWarning
warnings.simplefilter('ignore', InsecureRequestWarning)

env_path = Path(__file__).resolve().parent.parent / ".env"

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

db_path = '/home/administrator/behaviorate/scripts/intern_code/Alexandros Codes/project/PROJECT3/GeoLite2-ASN.mmdb'
reader = geoip2.database.Reader(db_path)

# Current time in UTC
now = datetime.now(timezone.utc)

end_time = datetime.now(timezone.utc)
start_time = end_time - timedelta(minutes=10)

print("Fetching logs from:", start_time.isoformat(), "to", end_time.isoformat())

def is_vps(ip):
    try:
        response = reader.asn(ip)
        org = response.autonomous_system_organization or ""
        known_vps_providers = [
            "Amazon Technologies Inc.", "Google LLC", "Microsoft Corporation",
            "DigitalOcean", "Linode", "Vultr", "Hetzner Online GmbH", "OVH SAS",
            "Alibaba Cloud", "Contabo GmbH"
        ]
        is_match = any(provider.lower() in org.lower() for provider in known_vps_providers)
        return is_match, org if is_match else None
    except Exception as e:
        return False, None


def extract_cid(referrer):
    if pd.isna(referrer):
        return None
    match = re.search(r'cid=([^&]+)', referrer)
    return match.group(1) if match else None

def fetch_logs():
    query = {
        "bool": {
            "filter": [
                {"range": {
                    "@timestamp": {
                        "gte": start_time.isoformat(),
                        "lte": end_time.isoformat()
                    }
                }},
                {"exists": {"field": "client.ip"}}
            ]
        }
    }

    fields = ["@timestamp", "host.name", "client.ip", "http.request.referrer"]

    scroll_time = "2m"
    batch_size = 1000

    response = es.search(
        index=INDEX_NAME,
        query=query,
        size=batch_size,
        _source=fields,
        scroll=scroll_time
    )

    scroll_id = response["_scroll_id"]
    all_hits = response["hits"]["hits"]

    while True:
        response = es.scroll(scroll_id=scroll_id, scroll=scroll_time)
        hits = response["hits"]["hits"]
        if not hits:
            break
        all_hits.extend(hits)

    logs = []
    seen_ips = set()

    for hit in all_hits:
        source = hit['_source']
        client_ip = source.get("client", {}).get("ip")
        if not client_ip or client_ip in seen_ips:
            continue

        seen_ips.add(client_ip)

        is_vps_flag, vps_provider = is_vps(client_ip)

        log_entry = {
            "timestamp": source.get("@timestamp"),
            "host.name": source.get("host", {}).get("name", [None])[0],
            "client.ip": client_ip,
            "user_id": extract_cid(source.get("http", {}).get("request", {}).get("referrer")),
            "is_VPS": is_vps_flag,
            "VPS_provider": vps_provider
        }

        logs.append(log_entry)

    df = pd.DataFrame(logs)

    if df.empty:
        print("No logs found with client.ip or they were all duplicates.")
        return df

    df["host.name"] = df["host.name"].astype(str)

    ma_links_pd = active_ma_sites(ALL_MANAGE_URL)
    ma_links_pd["Domain"] = ma_links_pd["Domain"].astype(str)

    df = df.merge(ma_links_pd, how="inner", left_on="host.name", right_on="Domain")

    df = df.drop(["MA_Links", "Domain"], axis=1)

    output_path = "/home/administrator/behaviorate/scripts/intern_code/Alexandros Codes/project/PROJECT3/vps_detection_logs.csv"
    df.to_csv(output_path)
    print(f"Total documents with distinct client.ip's: {len(df)}")
    return df

if __name__ == "__main__":
    fetch_logs()
