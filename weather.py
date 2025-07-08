from meteostat import Daily, Point
from datetime import datetime
import pandas as pd

def get_weather_data(lat, lon, date_str):
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d")
        location = Point(lat, lon)
        weather = Daily(location, date, date)
        data = weather.fetch()
        avg_temp = data['tavg'].iloc[0] if not data.empty else None
        avg_humidity = data['rhum'].iloc[0] if 'rhum' in data.columns and not data.empty else None
        return {"avg_temp": avg_temp, "avg_humidity": avg_humidity}
    except Exception as e:
        print(f"❌ Weather fetch failed for ({lat}, {lon}) on {date_str}: {e}")
        return {"avg_temp": None, "avg_humidity": None}

def add_weather_to_df(df, lat_lon_dict):
    df = df.copy()
    df["avg_temp"] = None
    df["avg_humidity"] = None

    for idx, row in df.iterrows():
        region = row["region"]
        date = row["date"]

        if region not in lat_lon_dict:
            print(f"⚠️ No coordinates for region: {region}")
            continue

        lat, lon = lat_lon_dict[region]
        weather = get_weather_data(lat, lon, date)
        df.at[idx, "avg_temp"] = weather["avg_temp"]
        df.at[idx, "avg_humidity"] = weather["avg_humidity"]

    return df
