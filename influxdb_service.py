import os
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime
from typing import Optional, List, Dict, Any

# =======================
# Configurare InfluxDB
# =======================

INFLUX_URL = os.getenv('INFLUX_URL', 'http://localhost:8086')
INFLUX_TOKEN = os.getenv('INFLUX_TOKEN', 'WDo6QMstKcq2TvrA355ST8B7wowGyDEafy2R_C1rZmd_WqB5nRrYYCsFLmDt-NnT0URHhGGx1mUcYJNvJDsgHQ==')
INFLUX_ORG = os.getenv('INFLUX_ORG', 'ev-charging-ai')
INFLUX_BUCKET = os.getenv('INFLUX_BUCKET', 'incarcare_ev')

def get_influx_client() -> InfluxDBClient:
    """
    Creează un client InfluxDB pe baza variabilelor de mediu/config.
    """
    return InfluxDBClient(
        url=INFLUX_URL,
        token=INFLUX_TOKEN,
        org=INFLUX_ORG
    )

# =======================
# Funcții utile pentru API
# =======================

def get_latest_station_metrics(station_id: str) -> Optional[Dict[str, Any]]:
    """
    Ia cele mai recente valori pentru o stație (ultimul timestamp).
    Returnează dict sau None dacă nu găsește date.
    """
    query = f'''
        from(bucket: "{INFLUX_BUCKET}")
        |> range(start: 0)  
        |> filter(fn: (r) => r._measurement == "charging_data")
        |> filter(fn: (r) => r.station_id == "{station_id}")
        |> sort(columns: ["_time"], desc: true)
        |> limit(n: 1)
    '''
    with get_influx_client() as client:
        tables = client.query_api().query(query, org=INFLUX_ORG)
        result = {}
        for table in tables:
            for record in table.records:
                # Adaugă fiecare field la dict
                result[record.get_field()] = record.get_value()
                result["timestamp"] = record.get_time()
        return result or None

def get_all_stations_latest() -> List[Dict[str, Any]]:
    """
    Returnează lista cu cele mai recente valori pentru toate stațiile distincte.
    """
    query = f'''
        import "influxdata/influxdb/schema"
        lastVals = from(bucket: "{INFLUX_BUCKET}")
          |> range(start: 0)  
          |> filter(fn: (r) => r._measurement == "charging_data")
        schema.tagValues(bucket: "{INFLUX_BUCKET}", tag: "station_id")
          |> map(fn: (r) => ({
                    station_id: r._value,
                    data: lastVals
                      |> filter(fn: (v) => v.station_id == r._value)
                      |> sort(columns: ["_time"], desc:true)
                      |> limit(n:1)
                  }))
    '''
    # Varianta simplă, citind pe rând fiecare stație (optimizat pentru puține stații):
    stations = get_distinct_station_ids()
    out = []
    for sid in stations:
        val = get_latest_station_metrics(sid)
        if val:
            val["station_id"] = sid
            out.append(val)
    return out

def get_distinct_station_ids() -> List[str]:
    """
    Obține toate valorile distincte pentru station_id (tag) din bucket (pentru afișare bulk).
    """
    query = f'''
      import "influxdata/influxdb/schema"
      schema.tagValues(
        bucket: "{INFLUX_BUCKET}",
        tag: "station_id"
      )
    '''
    with get_influx_client() as client:
        tables = client.query_api().query(query, org=INFLUX_ORG)
        ids = []
        for table in tables:
            for record in table.records:
                ids.append(record.get_value())
        return ids

def get_history_station(
    station_id: str, 
      start: str = "0",
    end: Optional[str]=None
) -> List[Dict[str, Any]]:
    """
    Lista cu istoric de înregistrări pentru o stație pe un interval dat (Flux time format: -1h, -7d, etc).
    """
    time_range = f'start: {start}'
    if end:
        time_range += f', stop: {end}'
    query = f'''
        from(bucket: "{INFLUX_BUCKET}")
        |> range({time_range})
        |> filter(fn: (r) => r._measurement == "charging_data")
        |> filter(fn: (r) => r.station_id == "{station_id}")
        |> sort(columns: ["_time"], desc: false)
    '''
    results = []
    with get_influx_client() as client:
        tables = client.query_api().query(query, org=INFLUX_ORG)
        for table in tables:
            for record in table.records:
                results.append({
                    "field": record.get_field(),
                    "value": record.get_value(),
                    "timestamp": record.get_time()
                })
    return results

def write_station_metrics(
    station_id: str,
    voltage: float,
    current: float,
    temperature: float,
    battery_soc: float,
    timestamp: Optional[datetime] = None
):
    """
    Scrie o nouă înregistrare pentru o stație.
    """
    point = Point("charging_data") \
        .tag("station_id", station_id) \
        .field("voltage", voltage) \
        .field("current", current) \
        .field("temperature", temperature) \
        .field("battery_soc", battery_soc)
    if timestamp:
        point = point.time(timestamp)
    with get_influx_client() as client:
        write_api = client.write_api(write_options=SYNCHRONOUS)
        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)

def get_station_stats(
    station_id: str,
    start: str = "0",
    window: str = "1d"
) -> List[Dict[str, Any]]:
    """
    Agregă datele pe ferestre de timp (ex: medie zilnică pe 7 zile).
    """
    query = f'''
        from(bucket: "{INFLUX_BUCKET}")
        |> range(start: {start})
        |> filter(fn: (r) => r._measurement == "charging_data")
        |> filter(fn: (r) => r.station_id == "{station_id}")
        |> aggregateWindow(every: {window}, fn: mean)
        |> yield(name: "mean")
    '''
    results = []
    with get_influx_client() as client:
        tables = client.query_api().query(query, org=INFLUX_ORG)
        for table in tables:
            for record in table.records:
                results.append({
                    "field": record.get_field(),
                    "mean_value": record.get_value(),
                    "window_time": record.get_time()
                })
    return results

def get_history_station_custom(
    station_id: str,
    start_time: str,
    stop_time: str
) -> List[Dict[str, Any]]:
    """
    Returnează istoric stație între două timestamp-uri ISO8601 sau RFC3339.
    Exemplu: '2024-07-15T10:00:00Z'
    """
    query = f'''
        from(bucket: "{INFLUX_BUCKET}")
        |> range(start: time(v: "{start_time}"), stop: time(v: "{stop_time}"))
        |> filter(fn: (r) => r._measurement == "charging_data")
        |> filter(fn: (r) => r.station_id == "{station_id}")
        |> sort(columns: ["_time"], desc: false)
    '''
    results = []
    with get_influx_client() as client:
        tables = client.query_api().query(query, org=INFLUX_ORG)
        for table in tables:
            for record in table.records:
                results.append({
                    "field": record.get_field(),
                    "value": record.get_value(),
                    "timestamp": record.get_time()
                })
    return results

def get_anomaly_alerts(
    field: str = "temperature",
    threshold: float = 50.0,
    since: str = "-7d"
) -> List[Dict[str, Any]]:
    """
    Returnează măsurătorile anormale pentru un anumit field și prag.
    """
    query = f'''
        from(bucket: "{INFLUX_BUCKET}")
        |> range(start: {since})
        |> filter(fn: (r) => r._measurement == "charging_data")
        |> filter(fn: (r) => r.{field} > {threshold})
    '''
    alerts = []
    with get_influx_client() as client:
        tables = client.query_api().query(query, org=INFLUX_ORG)
        for table in tables:
            for record in table.records:
                alerts.append({
                    "station_id": record.values.get("station_id"),
                    "field": field,
                    "value": record.get_value(),
                    "timestamp": record.get_time()
                })
    return alerts

import logging

logger = logging.getLogger(__name__)

def safe_query(query):
    try:
        with get_influx_client() as client:
            return client.query_api().query(query, org=INFLUX_ORG)
    except Exception as e:
        logger.error(f"Eroare InfluxDB: {e}")
        return []

def write_batch(station_data: List[Dict[str, Any]]):
    """
    Scrie o listă de dict-uri cu date pentru stații (ex: ingestie automată).
    """
    points = []
    for rec in station_data:
        point = Point("charging_data") \
            .tag("station_id", rec["station_id"]) \
            .field("voltage", rec["voltage"]) \
            .field("current", rec["current"]) \
            .field("temperature", rec["temperature"]) \
            .field("battery_soc", rec["battery_soc"])
        if "timestamp" in rec:
            point = point.time(rec["timestamp"])
        points.append(point)
    with get_influx_client() as client:
        client.write_api(write_options=SYNCHRONOUS).write(
            bucket=INFLUX_BUCKET,
            org=INFLUX_ORG,
            record=points
        )
