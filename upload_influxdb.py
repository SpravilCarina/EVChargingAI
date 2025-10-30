from influxdb_client import InfluxDBClient, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import pandas as pd

# Setările tale
influx_url = "http://localhost:8086"
influx_token = "WDo6QMstKcq2TvrA355ST8B7wowGyDEafy2R_C1rZmd_WqB5nRrYYCsFLmDt-NnT0URHhGGx1mUcYJNvJDsgHQ=="
influx_org = "ev-charging-ai"
influx_bucket = "incarcare_ev"

df = pd.read_csv('D:/ev-charging-ai/data/date_simulate.csv')

client = InfluxDBClient(
    url=influx_url,
    token=influx_token,
    org=influx_org
)
write_api = client.write_api(write_options=SYNCHRONOUS)

rows = []
for idx, r in df.iterrows():
    point = {
        "measurement": "charging_data",
        "tags": {"station_id": r["station_id"]},
        "fields": {
            "voltage": r["voltage"],
            "current": r["current"],
            "temperature": r["temperature"],
            "battery_state": r["battery_state"]
        },
        "time": r["timestamp"]
    }
    rows.append(point)

write_api.write(
    bucket=influx_bucket,
    org=influx_org,
    record=rows,
    write_precision=WritePrecision.S
)
print("Datele au fost uploadate cu succes în InfluxDB OSS v2!")
