from fastapi import APIRouter, Body, HTTPException, Response
import numpy as np
from tensorflow.keras.models import load_model
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from typing import List
from datetime import datetime, timezone

# ------------------------------
# Configurare conexiune InfluxDB
# ------------------------------
INFLUXDB_URL = "http://localhost:8086"  # Modifică dacă e alt host
INFLUXDB_TOKEN = "WDo6QMstKcq2TvrA355ST8B7wowGyDEafy2R_C1rZmd_WqB5nRrYYCsFLmDt-NnT0URHhGGx1mUcYJNvJDsgHQ=="
INFLUXDB_ORG = "ev-charging-ai"
INFLUXDB_BUCKET = "incarcare_ev"

client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
query_api = client.query_api()

# ------------------------------
# Încarcă modelul AI LSTM
# ------------------------------
MODEL_PATH = r"D:\\ev-charging-ai\\backend\\app\\ev_lstm_tfmodel.h5"
model = load_model(MODEL_PATH, compile=False)
history_steps = 24
feature_count = 8  # 4 reale + 4 dummy zero pentru forma de input

router = APIRouter()

# ------------------------------
# Funcții pentru Interogări InfluxDB
# ------------------------------

def get_all_stations() -> List[str]:
    flux_q = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
    |> range(start: 0)
    |> filter(fn: (r) => r._measurement == "charging_data")
    |> keep(columns: ["station_id"])
    |> group(columns: ["station_id"])
    |> distinct(column: "station_id")
    '''
    tables = query_api.query(flux_q, org=INFLUXDB_ORG)
    station_ids = []
    for table in tables:
        for record in table.records:
            print("DEBUG station_id VALUE:", record.get_value())  # Pentru debugging în consolă
            station_ids.append(record.get_value())
    return list(set(station_ids))


def get_station_history(station_id: str) -> List[List[float]]:
    flux_query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
        |> range(start: -3650d)
        |> filter(fn: (r) => r["_measurement"] == "charging_data")
        |> filter(fn: (r) => r["station_id"] == "{station_id}")
        |> filter(fn: (r) =>
             r["_field"] == "voltage" or
             r["_field"] == "current" or
             r["_field"] == "temperature" or
             r["_field"] == "battery_soc"
        )
        |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")
        |> sort(columns: ["_time"], desc: true)
        |> limit(n: 24)
        |> sort(columns: ["_time"])
        |> keep(columns: ["voltage", "current", "temperature", "battery_soc"])
    '''

    try:
        tables = query_api.query(flux_query, org=INFLUXDB_ORG)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Eroare la interogarea InfluxDB: {str(e)}")

    rows = []
    for table in tables:
        for record in table.records:
            rows.append(record.values)

    if len(rows) < history_steps:
        diff = history_steps - len(rows)
        padding = [[0.0]*feature_count]*diff
    else:
        padding = []

    history = []
    for row in rows[-history_steps:]:
        history.append([
            float(row.get("voltage", 0.0)),
            float(row.get("current", 0.0)),
            float(row.get("temperature", 0.0)),
            float(row.get("battery_soc", 0.0)),
            0.0, 0.0, 0.0, 0.0
        ])

    return padding + history


def get_station_status(station_id: str) -> str:
    flux_query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
        |> range(start: -3650d)
        |> filter(fn: (r) => r["station_id"] == "{station_id}")
        |> last()
    '''
    try:
        tables = query_api.query(flux_query, org=INFLUXDB_ORG)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Eroare la interogarea statusului InfluxDB: {str(e)}")

    for table in tables:
        for _ in table.records:
            return "online"
    return "offline"


@router.get("/monitor")
async def monitor_stations():
    stations = get_all_stations()
    result = []
    for sid in stations:
        status = get_station_status(sid)
        result.append({"station_id": sid, "status": status})
    return result


@router.get("/full_status")
async def stations_full_status():
    stations = get_all_stations()
    result = []
    for sid in stations:
        try:
            status = get_station_status(sid)
            latest_metrics = get_station_history(sid)
            if latest_metrics:
                last_values = latest_metrics[-1]
                item = {
                    "station_id": sid,
                    "status": status,
                    "voltage": last_values[0],
                    "current": last_values[1],
                    "temperature": last_values[2],
                    "battery_soc": last_values[3],
                }
            else:
                item = {
                    "station_id": sid,
                    "status": status,
                    "voltage": None,
                    "current": None,
                    "temperature": None,
                    "battery_soc": None,
                }
            result.append(item)
        except Exception as e:
            result.append({
                "station_id": sid,
                "status": "unknown",
                "voltage": None,
                "current": None,
                "temperature": None,
                "battery_soc": None,
                "error": str(e)
            })
    return result


@router.get("/stats")
async def get_stats():
    try:
        stations = get_all_stations()
        total = len(stations)
        online = sum(1 for sid in stations if get_station_status(sid) == "online")
        offline = total - online
        return {"total": total, "online": online, "offline": offline}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Eroare la calcul stats: {str(e)}")


@router.get("/export_csv")
async def export_csv():
    stations = get_all_stations()
    csv = "ID,Tensiune,Curent,Temperatură,Baterie\n"
    for sid in stations:
        try:
            flux_query = f'''
                from(bucket: "{INFLUXDB_BUCKET}")
                |> range(start: -3650d)
                |> filter(fn: (r) => r["_measurement"] == "charging_data")
                |> filter(fn: (r) => r["station_id"] == "{sid}")
                |> filter(fn: (r) =>
                     r["_field"] == "voltage" or
                     r["_field"] == "current" or
                     r["_field"] == "temperature" or
                     r["_field"] == "battery_soc"
                )
                |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")
                |> sort(columns: ["_time"], desc: true)
                |> limit(n: 1)
            '''
            tables = query_api.query(flux_query, org=INFLUXDB_ORG)
            data = {}
            for table in tables:
                for record in table.records:
                    data = record.values
            voltage = data.get("voltage", 0)
            current = data.get("current", 0)
            temperature = data.get("temperature", 0)
            battery_soc = data.get("battery_soc", 0)
            csv += f'{sid},{voltage},{current},{temperature},{battery_soc}\n'
        except Exception:
            csv += f'{sid},0,0,0,0\n'
    return Response(content=csv, media_type="text/csv")


@router.post("/add_station")
async def add_station(data: dict = Body(...)):
    sid = data.get("station_id")
    if not sid:
        return {"status": "fail", "msg": "station_id required"}
    point = (
        Point("charging_data")
        .tag("station_id", sid)
        .field("voltage", 0.0)
        .field("current", 0.0)
        .field("temperature", 0.0)
        .field("battery_soc", 0.0)
        .time(datetime.utcnow())
    )
    try:
        write_api = client.write_api(write_options=SYNCHRONOUS)
        write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)
    except Exception as e:
        return {"status": "fail", "msg": f"Eroare la adăugare: {e}"}
    return {"status": "success", "msg": f"Stația {sid} a fost adăugată"}


@router.post("/delete_station")
async def delete_station(data: dict = Body(...)):
    sid = data.get("station_id")
    if not sid:
        return {"status": "fail", "msg": "station_id required"}
    try:
        # Șterge toate datele pentru stația specificată pe tot intervalul
        start = "1970-01-01T00:00:00Z"
        stop = "2100-01-01T00:00:00Z"
        predicate = f'_measurement="charging_data" AND station_id="{sid}"'
        client.delete_api().delete(start, stop, predicate, bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG)
        return {"status": "success", "msg": f"Stația {sid} ștearsă complet"}
    except Exception as e:
        return {"status": "fail", "msg": f"Eroare la ștergere fizică: {e}"}


@router.post("/reset_all")
async def reset_all(data: dict = Body(...)):
    status = data.get("status", "online")
    stations = get_all_stations()
    write_api = client.write_api(write_options=SYNCHRONOUS)
    for sid in stations:
        point = (
            Point("charging_data")
            .tag("station_id", sid)
            .field("voltage", 0.0 if status == "offline" else 230.0)
            .field("current", 0.0 if status == "offline" else 16.0)
            .field("temperature", 0.0 if status == "offline" else 25.0)
            .field("battery_soc", 0.0 if status == "offline" else 50.0)
            .field("reset_flag", True)
            .time(datetime.utcnow())
        )
        write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)
    return {"status": "success", "msg": f"Toate stațiile marcate ca {status}"}


@router.get("/maintenance/alerts")
async def get_alerts():
    return []


@router.post("/push_notification")
async def push_notification(data: dict):
    msg = data.get("msg", "Notificare goală")
    print(f"Notificare trimisă: {msg}")
    return {"msg": f"Notificare trimisă: {msg}"}


@router.get("/predict_all_lstm")
async def predict_all_lstm():
    results = {}
    try:
        station_ids = get_all_stations()
        for sid in station_ids:
            input_seq = get_station_history(sid)
            arr = np.array(input_seq).reshape(1, history_steps, feature_count)
            pred = model.predict(arr).flatten().tolist()
            results[sid] = pred
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Eroare la predicția globală: {str(e)}")


@router.post("/predict_lstm")
async def predict_lstm(input_seq: list = Body(...)):
    try:
        arr = np.array(input_seq).reshape(1, history_steps, feature_count)
        result = model.predict(arr)
        pred = result.flatten().tolist()
        return {"prediction": pred}
    except Exception as e:
        return {"status": "fail", "msg": f"Eroare la predicție: {str(e)}"}


@router.post("/dynamic_price")
async def dynamic_price(payload: list):
    return {"dynamic_price": 1.23}


@router.post("/recommend_smart_schedule")
async def recommend_smart_schedule(payload: dict):
    return {"suggestion": "Încărcarea optimă e la ora 23"}
