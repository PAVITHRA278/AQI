import requests

# AQICN API token
API_TOKEN = "af959b3b498cdfab8a7c7d84436087701df68bd8"

# Get city name from user input
city = input("Enter city name: ")

# API URL
url = f"https://api.waqi.info/feed/{city}/?token={API_TOKEN}"

# Make the request
response = requests.get(url).json()

# Check response status
if response["status"] == "ok":
    print("\n✅ AQI Data Fetched Successfully!\n")
    print("City:", city)
    print("AQI:", response["data"]["aqi"])
    print("PM2.5:", response["data"]["iaqi"].get("pm25", {}).get("v", "N/A"))
    print("PM10:", response["data"]["iaqi"].get("pm10", {}).get("v", "N/A"))
    print("NO2:", response["data"]["iaqi"].get("no2", {}).get("v", "N/A"))
    print("CO:", response["data"]["iaqi"].get("co", {}).get("v", "N/A"))
    print("O3:", response["data"]["iaqi"].get("o3", {}).get("v", "N/A"))
    print("SO2:", response["data"]["iaqi"].get("so2", {}).get("v", "N/A"))
else:
    print(f"❌ Error fetching data for {city}. Status: {response['status']}")
