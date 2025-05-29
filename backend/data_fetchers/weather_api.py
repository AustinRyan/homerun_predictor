import requests

OPEN_METEO_API_URL = "https://api.open-meteo.com/v1/forecast"

def get_weather_forecast(latitude, longitude):
    """Fetches daily and hourly weather forecast for a given lat/lon."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation_probability,weather_code,wind_speed_10m,wind_direction_10m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,wind_speed_10m_max",
        "timezone": "auto",
        "forecast_days": 1 # For now, just current day, can be extended
    }
    try:
        response = requests.get(OPEN_METEO_API_URL, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

if __name__ == '__main__':
    # Example usage (e.g., for New York)
    lat_ny = 40.7128
    lon_ny = -74.0060
    weather_data = get_weather_forecast(lat_ny, lon_ny)
    if weather_data:
        print("Fetched weather data:")
        print(f"Latitude: {weather_data.get('latitude')}, Longitude: {weather_data.get('longitude')}")
        if 'daily' in weather_data and 'temperature_2m_max' in weather_data['daily']:
            print(f"Max Temp Today: {weather_data['daily']['temperature_2m_max'][0]}{weather_data['daily_units']['temperature_2m_max']}")
        if 'hourly' in weather_data and 'temperature_2m' in weather_data['hourly']:
            print(f"Current approx. hourly temp: {weather_data['hourly']['temperature_2m'][0]}{weather_data['hourly_units']['temperature_2m']}") # Example of accessing first hour data
    else:
        print("Failed to fetch weather data.")
