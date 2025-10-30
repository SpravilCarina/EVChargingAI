# statistici_descriptive.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Încarcă datele preprocesate
file_path = r'D:\ev-charging-ai\notebooks\data_curate\date_curate.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Fișierul nu a fost găsit la: {file_path}")

df = pd.read_csv(file_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])

print("📊 Dimensiunea setului de date:", df.shape)
print("Primele 5 rânduri:\n", df.head())

# 2. Statistici descriptive detaliate
print("\n🔢 Statistici descriptive generale:\n")
print(df.describe())

print("\n🔹 Mediana parametrilor numerici:\n")
print(df.median(numeric_only=True))

print("\n🔻 Valori minime:\n", df.min(numeric_only=True))
print("\n🔺 Valori maxime:\n", df.max(numeric_only=True))

percentile = df.quantile([0.1, 0.25, 0.5, 0.75, 0.9], numeric_only=True)
print("\n📈 Percentile selectate (10%, 25%, 50%, 75%, 90%):\n", percentile)

params = ['voltage', 'current', 'temperature', 'battery_state']
corelatie = df[params].corr()
print("\n🔗 Matrice de corelație (Pearson):\n", corelatie)

print("\nℹ️ Informații coloane și tipuri:\n")
print(df.info())

# 3. Vizualizări fundamentale

# Histogramă + KDE pentru fiecare parametru numeric
for col, color in zip(params, ['blue', 'green', 'red', 'purple']):
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, color=color)
    plt.title(f'Distribuția și KDE pentru {col}')
    plt.xlabel(col)
    plt.ylabel('Frecvență')
    plt.tight_layout()
    plt.show()

# Boxplot all-in-one pentru outlieri
plt.figure(figsize=(10, 5))
sns.boxplot(data=df[params])
plt.title('Boxplot pentru parametrii principali')
plt.tight_layout()
plt.show()

# Scatter plot: voltage vs current
plt.figure(figsize=(8, 6))
sns.scatterplot(x='voltage', y='current', data=df)
plt.title('Corelație între tensiune și curent')
plt.tight_layout()
plt.show()

# Heatmap corelație numerică
plt.figure(figsize=(6, 5))
sns.heatmap(corelatie, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matricea de corelație')
plt.tight_layout()
plt.show()

# 4. Vizualizări avansate

# Distribuții pe ore (pentru sezonalitate)
plt.figure(figsize=(12, 6))
sns.boxplot(x='hour', y='current', data=df, palette='Blues')
plt.title('Boxplot curent pe interval orar')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='hour', y='voltage', data=df, palette='Oranges')
plt.title('Boxplot tensiune pe interval orar')
plt.tight_layout()
plt.show()

# Distribuții pe zilele săptămânii
plt.figure(figsize=(10, 5))
sns.boxplot(x='day_of_week', y='current', data=df)
plt.title('Curent (A) - distribuție pe zilele săptămânii (0=Luni)')
plt.tight_layout()
plt.show()

# Analiza pe fiecare stație (dacă există coloane station_id_x)
station_id_cols = [col for col in df.columns if col.startswith('station_id_')]
if station_id_cols:
    for col in station_id_cols:
        val_sum = df[col].sum()
        print(f"\n✔️ Total sesiuni în {col}: {int(val_sum)}")

    # Exemplu: boxplot pentru curent per stație (doar pentru primele 4, dacă sunt multe)
    melted = df.melt(id_vars=['current'], value_vars=station_id_cols, var_name='station_id', value_name='is_this')
    melted = melted[melted['is_this'] == 1]
    plt.figure(figsize=(10,6))
    sns.boxplot(x='station_id', y='current', data=melted)
    plt.title('Distribuție curent pe fiecare stație (OneHot)')
    plt.tight_layout()
    plt.show()

# Trenduri temporale
# Medie mobilă pe 60 de minute pentru current
df = df.sort_values('timestamp')
df['current_rolling_mean_60'] = df['current'].rolling(window=60, min_periods=1).mean()

plt.figure(figsize=(16, 4))
plt.plot(df['timestamp'], df['current'], label='Current', alpha=0.3)
plt.plot(df['timestamp'], df['current_rolling_mean_60'], label='Medie mobilă (60 min)', linewidth=2, color='red')
plt.title('Evoluția curentului și media mobilă orară')
plt.xlabel('Timp')
plt.ylabel('Curent (standardizat)')
plt.legend()
plt.tight_layout()
plt.show()

# Dacă vrei să studiezi day/night pattern:
plt.figure(figsize=(12,4))
sns.lineplot(x='hour', y='current', data=df, ci=None, estimator=np.mean)
plt.title('Media curentului pe fiecare oră a zilei')
plt.tight_layout()
plt.show()

# Relatie battery_state - ora
plt.figure(figsize=(12, 4))
sns.boxplot(x='hour', y='battery_state', data=df)
plt.title('Distribuție battery_state pe interval orar')
plt.tight_layout()
plt.show()

# 5. Analize agregate per stație și zi (exemplu avansat)
if station_id_cols:
    df['station'] = df[station_id_cols].idxmax(axis=1).str.extract('(\d+)').astype(int)
    grouped = df.groupby(['station', 'day_of_week'])['current'].mean().unstack()
    plt.figure(figsize=(12,5))
    sns.heatmap(grouped, annot=True, fmt='.2f', cmap='crest')
    plt.title('Media curentului pe stație și zi a săptămânii')
    plt.xlabel('Zi a săptămânii (0=Luni)')
    plt.ylabel('ID stație')
    plt.tight_layout()
    plt.show()
