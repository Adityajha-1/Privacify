import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
from pymongo import MongoClient
from requests.exceptions import RequestException
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import numpy as np
from flask import Flask, render_template, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Patterns for detecting emails and phone numbers
email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
phone_pattern = re.compile(r'(\+?91[-\s]?)?(\d{10})')

def fetch_website_content(url):
    """Fetches and returns the HTML content of a webpage."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except RequestException as e:
        print(f"Error accessing {url}: {e}")
        return None

def parse_content_for_privacy_data(content, user_email, user_phone):
    """Parses HTML content to check for user's email and phone number."""
    # Extract emails and phones from content
    found_emails = email_pattern.findall(content)
    found_phones = phone_pattern.findall(content)

    # Normalize user inputs
    user_email = user_email.strip().lower()
    user_phone = normalize_phone_number(user_phone)

    # Normalize and compare emails
    email_leak = any(email.strip().lower() == user_email for email in found_emails)

    # Normalize found phones and compare with user phone
    normalized_found_phones = [normalize_phone_number(phone[1]) for phone in found_phones if phone[1]]
    phone_leak = user_phone in normalized_found_phones

    return {
        "emails_found": list(set(found_emails)),
        "phones_found": list(set(normalized_found_phones)),
        "email_leak": email_leak,
        "phone_leak": phone_leak,
    }


def crawl_website(url, user_email, user_phone):
    """Crawls a given website and checks for user's sensitive data."""
    print(f"Scanning {url}...")
    content = fetch_website_content(url)
    if content:
        data = parse_content_for_privacy_data(content, user_email, user_phone)
        print(f"Scan results for {url}: {data}")
        return {
            "url": url,
            "emails_found": data["emails_found"],
            "phones_found": data["phones_found"],
            "email_leak": data["email_leak"],
            "phone_leak": data["phone_leak"],
        }
    else:
        print(f"Failed to fetch content from {url}")
        return {"url": url, "email_leak": False, "phone_leak": False}


def normalize_data(results):
    """Converts the crawl results into a structured DataFrame."""
    return pd.DataFrame(results)

def save_data_to_mongo(data, db_name="privacy_scan", collection_name="user_data"):
    """Saves data to a MongoDB collection."""
    client = MongoClient("mongodb://localhost:27017/")
    db = client[db_name]
    collection = db[collection_name]
    collection.insert_many(data)
    print(f"Data saved to MongoDB collection: {collection_name}")

def save_data_as_json(data, user_email, user_phone, filename="privacy_report.json"):
    """Saves URLs scanned, email, phone number, and leak status to a JSON file."""
    simplified_data = [
        {
            "url": item["url"],  
            "email_checked": user_email,  
            "phone_checked": user_phone,  
            "email_found": item["email_leak"],
            "phone_found": item["phone_leak"],
            "is_anomalous": item.get("is_anomalous", False)  
        } for item in data
    ]
    with open(filename, 'w') as file:
        json.dump(simplified_data, file, indent=4)
        print(f"Data saved to {filename}")

def crawl_with_rate_limiting(urls, user_email, user_phone, delay=2):
    """Crawls multiple websites with rate limiting."""
    results = []
    for url in urls:
        result = crawl_website(url, user_email, user_phone)
        if result:
            results.append(result)
        time.sleep(delay)  # Delay between requests
    return results

def normalize_phone_number(phone):
    """Normalizes phone numbers to a standard format (10 digits)."""
    # Remove all non-numeric characters
    digits = re.sub(r'\D', '', phone)

    # If the number starts with '91' or '+91', remove the country code
    if digits.startswith('91') and len(digits) > 10:
        digits = digits[2:]
    elif digits.startswith('+91') and len(digits) > 10:
        digits = digits[3:]

    # Return the last 10 digits (assuming it's a valid phone number)
    return digits[-10:]


# Feature Extraction for ML model
def extract_features(data):
    """Extracts features from the crawled data for anomaly detection."""
    features = []

    for item in data:
        email_mentions = int(item["email_leak"])
        phone_mentions = int(item["phone_leak"])
        platform_count = len(set([result['platform'] for result in data if 'platform' in result]))
        sentiment_score = 1 if "positive" in item.get("data", {}).get("text", "").lower() else -1
        
        features.append([email_mentions, phone_mentions, platform_count, sentiment_score])

    # Standardize features for model training
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    return np.array(scaled_features)

# ML Model Training
def train_anomaly_detection_model(features):
    """Trains an Isolation Forest model to detect anomalies."""
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(features)
    return model

@app.route('/')
def home():
    """Serve the HTML form for user input."""
    return render_template('index.html')  # Ensure the file index.html is in templates/

@app.route('/scan', methods=['POST'])
def scan():
    """Handle scan requests from the frontend."""
    data = request.get_json()
    user_email = data.get('email')
    user_phone = data.get('phone')
    urls = data.get('urls', [])

    # Validate inputs
    if not user_email or not email_pattern.match(user_email):
        return jsonify({"error": "Invalid email address!"}), 400
    if not user_phone or len(normalize_phone_number(user_phone)) != 10:
        return jsonify({"error": "Invalid phone number!"}), 400
    if not urls or not all(urls):
        return jsonify({"error": "Invalid or empty URL list!"}), 400

    # Scan websites
    web_results = [crawl_website(url.strip(), user_email, user_phone) for url in urls]

    return jsonify(web_results)


if __name__ == '__main__':
    app.run(debug=True)
