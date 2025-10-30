import pandas as pd
from influxdb_service import write_batch

# Calea către datele CSV (folosește calea corectă către fișierul tău)
CSV_FILE = r"D:\ev-charging-ai\notebooks\data_curate\date_curate.csv"
  # Schimbă dacă fișierul este în alt folder

def main():
    # Citește fișierul CSV cu coloana 'timestamp' interpretată ca dată
    df = pd.read_csv(CSV_FILE, parse_dates=["timestamp"])

    station_data = []

    # Construiește lista de dicționare cu date pentru fiecare stație activă
    for _, row in df.iterrows():
        for station in range(1, 5):
            if row[f"station_id_{station}"] == 1.0:
                record = {
                    "station_id": str(station),
                    "voltage": float(row["voltage"]),
                    "current": float(row["current"]),
                    "temperature": float(row["temperature"]),
                    "battery_soc": float(row["battery_state"]),  # Rename important
                    "timestamp": row["timestamp"]
                }
                station_data.append(record)
                break  # Doar o stație activă pe fiecare rând

    # Apeleză funcția care scrie batch în InfluxDB
    write_batch(station_data)

    print(f"Importul a fost finalizat. Au fost încărcate {len(station_data)} înregistrări.")

if __name__ == "__main__":
    main()
