import requests
import json

url = "https://historical-forecast-api.open-meteo.com/v1/forecast?latitude=40.677868&longitude=-105.413367&start_date=2022-01-01&end_date=2025-07-12&hourly=temperature_2m,precipitation,evapotranspiration,et0_fao_evapotranspiration,snowfall,snow_depth,soil_moisture_0_to_1cm,soil_moisture_1_to_3cm,soil_moisture_3_to_9cm,soil_moisture_27_to_81cm,soil_moisture_9_to_27cm,relative_humidity_2m,precipitation_probability"

max_retries = 5
for attempt in range(max_retries):
    try:
        response = requests.get(url)
        response.raise_for_status()
        break
    except Exception as e:
        if attempt == max_retries - 1:
            raise
        else:
            continue

with open("weather_data.json", "w") as f:
    json.dump(response.json(), f, indent=2)
