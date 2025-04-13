"""
import requests
import pymongo
import time
from datetime import datetime

# MongoDB Connection
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["AirQualityDB"]
collection = db["real_time_aqi"]

# AQICN API Token
API_TOKEN = "e5e3afafdb9a63b47110eebe74bce12c3eaf8dc6"

# List of 25+ cities
CITIES = [
    "New Delhi", "Mumbai", "Bangalore", "chennai", "Kolkata", "Hyderabad", "Ahmedabad",
    "Pune", "Surat", "Jaipur", "Lucknow", "Kanpur", "Nagpur", "Indore", "Thane",
    "Bhopal", "Visakhapatnam", "Patna", "Vadodara", "Ghaziabad", "Ludhiana", "Agra",
    "Nashik", "Meerut", "Rajkot"
]

def fetch_aqi(city):
    #Fetch AQI for a single city and store in MongoDB#
    url = f"https://api.waqi.info/feed/{city}/?token={API_TOKEN}"
    response = requests.get(url).json()
    print("API Response:", response)  # Debug to check if fresh data is coming


    if response["status"] == "ok":
        data = {
            "city": city,
            "aqi": response["data"]["aqi"],
            "pm25": response["data"]["iaqi"].get("pm25", {}).get("v", None),
            "pm10": response["data"]["iaqi"].get("pm10", {}).get("v", None),
            "no2": response["data"]["iaqi"].get("no2", {}).get("v", None),
            "co": response["data"]["iaqi"].get("co", {}).get("v", None),
            "o3": response["data"]["iaqi"].get("o3", {}).get("v", None),
            "so2": response["data"]["iaqi"].get("so2", {}).get("v", None),
            "timestamp": datetime.now()
        }
        collection.insert_one(data)
        print(f"✅ AQI data inserted for {city}")

def fetch_and_store_aqi():
    #Fetch AQI data in small batches
    for i in range(0, len(CITIES), 5):  # Process 5 cities at a time
        batch = CITIES[i:i+5]
        for city in batch:
            fetch_aqi(city)
        time.sleep(2)  # Pause between batches

    print("✅ All cities updated!")

if __name__ == "__main__":
    fetch_and_store_aqi()
"""



# fetch_aqi.py

import streamlit as st
import requests
import pymongo
from datetime import datetime
import time
import certifi
from pymongo import MongoClient
# MongoDB setup
#client = pymongo.MongoClient("mongodb://localhost:27017/")
#db = client["AirQualityDB"]
#collection = db["real_time_aqi"]


# Connect to MongoDB
client = pymongo.MongoClient(
    st.secrets["MONGO"]["URI"],
    tls=True,              # Enforce TLS
    tlsCAFile=certifi.where()  # if using self-signed certs or unverified env (optional)
)
db = client["AirQualityDB"]
collection = db["real_time_aqi"]

API_TOKEN = "e5e3afafdb9a63b47110eebe74bce12c3eaf8dc6"

CITIES = [
    "New Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Ahmedabad", "Pune",
    "Surat", "Jaipur", "Lucknow", "Kanpur", "Nagpur", "Indore", "Thane", "Bhopal", "Visakhapatnam",
    "Patna", "Vadodara", "Ghaziabad", "Ludhiana", "Agra", "Nashik", "Meerut", "Rajkot",
    "Amritsar", "Ranchi", "Coimbatore", "Gwalior", "Vijayawada", "Jodhpur", "Madurai", "Raipur",
    "Kota", "Chandigarh", "Guwahati", "Solapur", "Hubli", "Tiruchirappalli", "Bareilly", "Mysore",
    "Tiruppur", "Gurgaon", "Aligarh", "Jalandhar", "Moradabad", "Jamshedpur", "Bhilai", "Srinagar",
    "Noida", "Howrah"
]


def fetch_aqi(city):
    url = f"https://api.waqi.info/feed/{city}/?token={API_TOKEN}"
    response = requests.get(url).json()

    if response["status"] == "ok":
        data = {
            "city": city,
            "aqi": response["data"]["aqi"],
            "pm25": response["data"]["iaqi"].get("pm25", {}).get("v"),
            "pm10": response["data"]["iaqi"].get("pm10", {}).get("v"),
            "no2": response["data"]["iaqi"].get("no2", {}).get("v"),
            "co": response["data"]["iaqi"].get("co", {}).get("v"),
            "o3": response["data"]["iaqi"].get("o3", {}).get("v"),
            "so2": response["data"]["iaqi"].get("so2", {}).get("v"),
            "timestamp": datetime.now()
        }
        collection.insert_one(data)

def run_fetch_aqi():
    for i in range(0, len(CITIES), 5):
        for city in CITIES[i:i+5]:
            fetch_aqi(city)
        time.sleep(2)  # Rate limit
