import pandas as pd
import numpy as np

n = 1000  # numărul de înregistrări generate
df = pd.DataFrame({
    'timestamp': pd.date_range('2025-07-14', periods=n, freq='min'),
    'station_id': np.random.randint(1, 5, n),
    'voltage': np.random.normal(400, 10, n),
    'current': np.random.normal(32, 2, n),
    'temperature': np.random.normal(25, 3, n),
    'battery_state': np.random.uniform(20, 100, n)
})

df.to_csv('date_simulate.csv', index=False)
print("Datele simulate au fost salvate în date_simulate.csv")
