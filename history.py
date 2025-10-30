from fastapi import APIRouter, HTTPException
from typing import List
import pandas as pd
import numpy as np

router = APIRouter()

DATA_FILE = r"D:\ev-charging-ai\notebooks\data_curate\date_curate.csv"

df_simulated = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])

# Lista fixă de features (7 coloane bază + 1 coloana one-hot pentru stația curentă)
BASE_FEATURES = [
    'voltage',
    'current',
    'temperature',
    'battery_state',
    'hour',
    'day_of_week',
    'is_weekend'
]

@router.get("/stations/{station_id}", response_model=List[List[float]])
async def get_station_history(station_id: str):
    # Găsește coloana one-hot corectă pentru stația cerută
    matching_col = None
    for col in df_simulated.columns:
        if col.startswith("station_id_") and (col == f"station_id_{station_id}" or col.endswith(f"_{station_id}")):
            matching_col = col
            break

    if not matching_col:
        raise HTTPException(status_code=404, detail=f"Nu există coloana pentru stația {station_id}")

    # Construiește lista completă de features pentru predicție (exact 8 coloane)
    FEATURES = BASE_FEATURES + [matching_col]

    # Filtrează rândurile corespunzătoare stației cerute (coloana one-hot == 1)
    df_station = df_simulated[df_simulated[matching_col] == 1]
    if df_station.empty:
        raise HTTPException(status_code=404, detail=f"Stația {station_id} nu are date disponibile.")

    df_station = df_station.sort_values('timestamp')

    # Ia ultimele 24 de înregistrări (timp istoric) și extrage coloanele relevante
    data_last_24 = df_station.tail(24)
    data_values = data_last_24[FEATURES].values

    # Dacă lipsesc înregistrări ca să faci 24 de timesteps, completează cu zerouri la început
    n_missing = 24 - data_values.shape[0]
    if n_missing > 0:
        padding = np.zeros((n_missing, len(FEATURES)))
        data_values = np.vstack([padding, data_values])

    # Returnează o listă de liste cu datele pregătite pentru input model
    return data_values.tolist()
