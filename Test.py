from influxdb_client import InfluxDBClient

url = "http://localhost:8086"
token = "WDo6QMstKcq2TvrA355ST8B7wowGyDEafy2R_C1rZmd_WqB5nRrYYCsFLmDt-NnT0URHhGGx1mUcYJNvJDsgHQ=="
org = "ev-charging-ai"
bucket = "incarcare_ev"

client = InfluxDBClient(url=url, token=token, org=org)
query_api = client.query_api()

flux_query = '''
import "influxdata/influxdb/schema"
schema.tagValues(
  bucket: "incarcare_ev",
  tag: "station_id"
)
'''

tables = query_api.query(flux_query, org=org)
print("Station_IDs found in bucket:")
for table in tables:
    for record in table.records:
        print("-", record.get_value())
client.close()
