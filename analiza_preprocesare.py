# analiza_preprocesare.py

import pandas as pd
import numpy as np
import os

# Importuri ML utile
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import sklearn

# 1. Încarcă datele simulate din fișierul CSV
file_path = r'D:\ev-charging-ai\data\date_simulate.csv'

# Verifică dacă fișierul există
if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ Fișierul nu a fost găsit la: {file_path}")

# Citește fișierul .csv
df = pd.read_csv(file_path)

print(f"📁 scikit-learn version: {sklearn.__version__}")
print("📊 Forma inițială a datelor:", df.shape)
print(df.head())

# 2. Verifică valori lipsă
print("\n🔍 Valori lipsă pe coloană:\n", df.isnull().sum())
df = df.dropna()  # elimină rândurile cu valori lipsă

# 3. Curăță anomaliile
df = df[
    (df['voltage'] >= 350) & (df['voltage'] <= 450) &
    (df['current'] >= 0) & (df['current'] <= 40) &
    (df['temperature'] >= -20) & (df['temperature'] <= 60) &
    (df['battery_state'] >= 0) & (df['battery_state'] <= 100)
]
print("\n✅ Forma după filtrarea anomaliilor:", df.shape)

# 4. Convertiri de tip pentru siguranță
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['station_id'] = df['station_id'].astype(int)
df['voltage'] = df['voltage'].round(2)
df['current'] = df['current'].round(2)
df['temperature'] = df['temperature'].round(2)
df['battery_state'] = df['battery_state'].round(2)

# 6. One-Hot Encoding pentru station_id
try:
    # Pentru scikit-learn >= 1.2
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
except TypeError:
    # Pentru versiuni mai vechi
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

station_encoded = encoder.fit_transform(df[['station_id']])
station_columns = encoder.get_feature_names_out(['station_id'])
station_df = pd.DataFrame(station_encoded, columns=station_columns)

# Adaugă datele codificate la dataframe
df = pd.concat([df.reset_index(drop=True), station_df.reset_index(drop=True)], axis=1)
df = df.drop(columns=['station_id'])

# 7. Feature Engineering: ora, ziua săptămânii, weekend
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0 = luni, 6 = duminică
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# 8. Salvare fișier rezultat
output_folder = r'D:\ev-charging-ai\notebooks\data_curate'
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, 'date_curate.csv')
df.to_csv(output_path, index=False)

print(f"\n✅ Datele preprocesate au fost salvate în:\n{output_path}")
