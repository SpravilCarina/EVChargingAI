import pandas as pd
from influxdb_client import InfluxDBClient, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

# Configurare conexiune InfluxDB
influx_url = "http://localhost:8086"
influx_token = "WDo6QMstKcq2TvrA355ST8B7wowGyDEafy2R_C1rZmd_WqB5nRrYYCsFLmDt-NnT0URHhGGx1mUcYJNvJDsgHQ=="
influx_org = "ev-charging-ai"
influx_bucket = "incarcare_ev"

# Citește fișierul CSV
df = pd.read_csv(r'D:\ev-charging-ai\notebooks\data_curate\date_curate.csv', parse_dates=['timestamp'])
df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True, errors='coerce')

# Forțează anul 2025 pentru toate timestamp-urile valide
df['timestamp'] = df['timestamp'].apply(lambda x: x.replace(year=2025) if pd.notnull(x) else x)

# Creează clientul InfluxDB
client = InfluxDBClient(
    url=influx_url,
    token=influx_token,
    org=influx_org
)

write_api = client.write_api(write_options=SYNCHRONOUS)

rows = []

for idx, r in df.iterrows():
    # Skip rândurile cu valori NaN critice
    if pd.isnull(r['timestamp']) or pd.isnull(r['voltage']) or pd.isnull(r['current']) or pd.isnull(r['temperature']) or pd.isnull(r['battery_state']):
        continue

    # Creează câte un punct separat pentru fiecare stație activă (valoare 1)
    for sid in range(1, 5):
        if r[f'station_id_{sid}'] == 1:
            point = {
                "measurement": "charging_data",
                "tags": {
                    "station_id": str(sid)
                },
                "fields": {
                    "voltage": float(r["voltage"]),
                    "current": float(r["current"]),
                    "temperature": float(r["temperature"]),
                   "battery_soc": float(r["battery_state"]),
                    "hour": int(r["hour"]),
                    "day_of_week": int(r["day_of_week"]),
                    "is_weekend": int(r["is_weekend"]),
                },
                "time": r["timestamp"]
            }
            rows.append(point)

# Scrie toate punctele în InfluxDB
write_api.write(
    bucket=influx_bucket,
    org=influx_org,
    record=rows,
    write_precision=WritePrecision.S
)

print("Datele au fost uploadate cu succes în InfluxDB OSS v2!")

client.close()
