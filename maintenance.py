from fastapi import APIRouter, HTTPException
from influxdb_client import InfluxDBClient
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText

router = APIRouter()

# Configurația InfluxDB folosind datele tale reale
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = "WDo6QMstKcq2TvrA355ST8B7wowGyDEafy2R_C1rZmd_WqB5nRrYYCsFLmDt-NnT0URHhGGx1mUcYJNvJDsgHQ=="
INFLUXDB_ORG = "ev-charging-ai"
INFLUXDB_BUCKET = "incarcare_ev"

client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
query_api = client.query_api()

# Praguri simple pentru alerte critice
ALERT_THRESHOLDS = {
    "temperature": 50.0,  # °C (critică)
    "current": 100.0,     # amperi (ridicat)
    "battery_soc": 10.0,  # % (nivel scăzut baterie)
}

# Prag pentru scor anomalie AI
ANOMALY_SCORE_THRESHOLD = 0.8

# Configurații pentru notificări email (exemplu pentru SMTP)
SMTP_SERVER = "smtp.example.com"
SMTP_PORT = 587
SMTP_USER = "user@example.com"
SMTP_PASS = "yourpassword"
ALERT_RECEIVERS = ["operator@example.com"]  # listează email-urile destinatarilor


def send_email(subject: str, body: str, to: list):
    """Trimite o notificare simplă prin email."""
    msg = MIMEText(body)
    msg['From'] = SMTP_USER
    msg['To'] = ", ".join(to)
    msg['Subject'] = subject

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, to, msg.as_string())
    except Exception as e:
        print(f"EROARE la trimiterea email-ului: {e}")


@router.get("/alerts")
async def obtine_alerte():
    alerts = []

    # 1. Detectare alerte pe praguri simple
    query = f'''
    from(bucket:"{INFLUXDB_BUCKET}")
      |> range(start: -1h)
      |> filter(fn: (r) => r["_measurement"] == "charging_data")
      |> filter(fn: (r) => 
          (r["_field"] == "temperature" and r["_value"] > {ALERT_THRESHOLDS["temperature"]}) or
          (r["_field"] == "current" and r["_value"] > {ALERT_THRESHOLDS["current"]}) or
          (r["_field"] == "battery_soc" and r["_value"] < {ALERT_THRESHOLDS["battery_soc"]})
      )
      |> keep(columns: ["_time", "_field", "_value", "station_id"])
    '''
    try:
        tables = query_api.query(query)
        for table in tables:
            for record in table.records:
                msg = f"{record.get_field()} valoare {record.get_value()} depășește pragul"
                alert = {
                    "station_id": record.values.get("station_id"),
                    "alert": msg,
                    "severity": "critical",
                    "timestamp": record.get_time().isoformat()
                }
                alerts.append(alert)
                # Trimite notificare email
                send_email(f"Alerte Critice Stație {alert['station_id']}", msg, ALERT_RECEIVERS)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Eroare la interogare InfluxDB (praguri): {str(e)}")

    # 2. Detectare anomalii AI
    query_anomaly = f'''
    from(bucket:"{INFLUXDB_BUCKET}")
      |> range(start: -1h)
      |> filter(fn: (r) => r["_measurement"] == "anomaly_scores")
      |> filter(fn: (r) => r["_field"] == "score" and r._value > {ANOMALY_SCORE_THRESHOLD})
      |> keep(columns: ["_time", "_field", "_value", "station_id"])
    '''
    try:
        tables = query_api.query(query_anomaly)
        for table in tables:
            for record in table.records:
                msg = f"Anomalie detectată - scor {record.get_value():.2f}"
                alert = {
                    "station_id": record.values.get("station_id"),
                    "alert": msg,
                    "severity": "warning",
                    "timestamp": record.get_time().isoformat()
                }
                alerts.append(alert)
                # Notificare email alertă warning
                send_email(f"Alerte Warning Stație {alert['station_id']}", msg, ALERT_RECEIVERS)
    except Exception:
        pass

    # 3. Alertă predictivă - trend de creștere rapidă a temperaturii
    try:
        query_temp_trend = f'''
        from(bucket:"{INFLUXDB_BUCKET}")
          |> range(start: -2h)
          |> filter(fn: (r) => r["_measurement"] == "charging_data" and r["_field"] == "temperature")
          |> filter(fn: (r) => exists r.station_id)
          |> keep(columns: ["_time", "_value", "station_id"])
          |> sort(columns: ["_time"])
        '''
        tables = query_api.query(query_temp_trend)

        temp_by_station = {}
        for table in tables:
            for record in table.records:
                station = record.values.get("station_id")
                if station not in temp_by_station:
                    temp_by_station[station] = []
                temp_by_station[station].append((record.get_time(), record.get_value()))

        for station, vals in temp_by_station.items():
            if len(vals) < 2:
                continue
            last_temp = vals[-1][1]
            prev_temp = None
            for t, val in reversed(vals):
                if (vals[-1][0] - t).total_seconds() >= 3600:
                    prev_temp = val
                    break
            if prev_temp and last_temp > prev_temp + 5.0:
                msg = f"Temperatura crește rapid (+{last_temp - prev_temp:.1f}°C în ultima oră)"
                alert = {
                    "station_id": station,
                    "alert": msg,
                    "severity": "warning",
                    "timestamp": datetime.utcnow().isoformat()
                }
                alerts.append(alert)
                send_email(f"Alerte Warning Trend Stație {station}", msg, ALERT_RECEIVERS)
    except Exception:
        pass

    # 4. Dacă nu există alerte, mesaj informativ
    if not alerts:
        alerts.append({
            "station_id": "N/A",
            "alert": "Nicio alertă detectată în ultima oră",
            "severity": "info",
            "timestamp": datetime.utcnow().isoformat()
        })

    return alerts
