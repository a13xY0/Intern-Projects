from datetime import datetime, timedelta,timezone
import requests
import os
import secrets
import json
import pandas as pd
from bs4 import BeautifulSoup

def get_time_window(minutes):
    """
    Get the  time window based on the current UTC time.
    Ensures the time window is always the one that just finished.
    
    Returns:
        start_time (datetime): The start of the time window.
        end_time (datetime): The end of the time window.
        time_window_str (str): Formatted string (YYYYMMDD_HHMM) for filename usage.
    """
    now = datetime.now(timezone.utc)
    # Round `now` down to the nearest 15-minute mark
    rounded_now = now.replace(minute=(now.minute // minutes) * minutes, second=0, microsecond=0)
    # Move back one full 15-minute window
    start_time = rounded_now - timedelta(minutes=minutes)
    end_time = rounded_now  # The current rounded time is the end of the previous window
    # Format time window for filename
    time_window_str = start_time.strftime("%Y%m%d_%H%M")
    return start_time, end_time, time_window_str


def generate_secret_key(length: int = 32) -> str:
    """
    Generate a high-entropy secret key using Python's 'secrets' module.

    :param length: Number of random bytes to generate.
                   Each byte is represented as two hex characters,
                   so total hex string length = 2 * length.
    :return: A hex-encoded secret key string.
    """
    return secrets.token_hex(length)



def send_webhook(webhook_url,dataframe, type):
    headers = {"Content-Type": "application/json"}
    headers["signature"] = generate_secret_key()
    json_str= dataframe.to_json(orient="records",date_format="iso")
    # Parse that string into a Python list
    list_of_records = json.loads(json_str)
    payload = {}
    payload["anomaly"] = type
    payload["data"] = list_of_records
    response = requests.post(webhook_url, json=payload, headers=headers)
    print(response.status_code, response.text)


#Function for MA sites
def active_ma_sites(all_manage_url):
    # Fetch the page content
    # URL of the webpage
    url=all_manage_url

    # Headers to mimic a real browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        
    # Find all 'btn-group' rows
    rows = soup.find_all("tr")
    # Extract MA links where hidden <span class="hidden"> value is "1"
    # Use a set to store unique MA links
    unique_ma_links = set()

    for row in rows:
        hidden_span = row.find("i", class_="fa fa-circle blink")  # Find the <i> icon that contains hidden <span>

        # Ensure the hidden span exists inside <i> and its value is "1"
        if hidden_span and hidden_span.find("span", class_="hidden") and hidden_span.find("span", class_="hidden").text.strip() == "1":
            # Find the MA button within this row
            ma_button = row.find("button", class_="btn btn-default btn-xs", string=lambda text: text and text.strip() == "MA")

            if ma_button and "window.open" in ma_button.get("onclick", ""):
                # Extract URL inside window.open
                ma_link = ma_button["onclick"].split("window.open('")[1].split("')")[0]
                unique_ma_links.add(ma_link)
    # Convert set to DataFrame
    ma_links_pd= pd.DataFrame(list(unique_ma_links), columns=["MA_Links"])
    # Extract the text after "https://"
    ma_links_pd["Domain"] = ma_links_pd["MA_Links"].str.replace("https://", "", regex=True)
    return ma_links_pd


def active_crm_sites(all_manage_url):
    # Fetch the page content
    # URL of the webpage
    url=all_manage_url

    # Headers to mimic a real browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        
    # Find all 'btn-group' rows
    rows = soup.find_all("tr")
    # Extract CRM links where hidden <span class="hidden"> value is "1"
    # Use a set to store unique CRM links
    unique_crm_links = set()

    for row in rows:
        hidden_span = row.find("i", class_="fa fa-circle blink")  # Find the <i> icon that contains hidden <span>

        # Ensure the hidden span exists inside <i> and its value is "1"
        if hidden_span and hidden_span.find("span", class_="hidden") and hidden_span.find("span", class_="hidden").text.strip() == "1":
            # Find the CRM button within this row
            crm_button = row.find("button", class_="btn btn-default btn-xs", string=lambda text: text and text.strip() == "CRM")

            if crm_button and "window.open" in crm_button.get("onclick", ""):
                # Extract URL inside window.open
                crm_link = crm_button["onclick"].split("window.open('")[1].split("')")[0]
                unique_crm_links.add(crm_link)
    # Convert set to DataFrame
    crm_links_pd= pd.DataFrame(list(unique_crm_links), columns=["CRM_Links"])
    # Extract the text after "https://" and before /console with one or even more slashes
    crm_links_pd["Domain"] = crm_links_pd["CRM_Links"].str.replace(r"https?://", "", regex=True).str.replace(r"/+console$", "", regex=True)
    return crm_links_pd