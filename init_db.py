import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text

DATABASE_URL = st.secrets["postgres"]["url"]
engine=create_engine(DATABASE_URL)

create_table_query="""
CREATE TABLE IF NOT EXISTS prediction_telemetry(
id SERIAL PRIMARY KEY,
timestamp TIMESTAMP,
predicted_crop VARCHAR(50),
latency_ms FLOAT,
user_status VARCHAR(50),
input_n FLOAT,
input_p FLOAT,
input_k FLOAT,
input_temp FLOAT,
input_humidity FLOAT,
input_ph FLOAT,
input_rainfall FLOAT
);
"""

print("Connecting to Neon PostgreSQL...")
with engine.connect() as conn:
    conn.execute(text(create_table_query))
    conn.commit()
    print("Table 'prediction_telemetry' created successfully in the cloud!")
    result = conn.execute(text("SELECT COUNT(*) FROM prediction_telemetry;"))
    row_count = result.scalar()

np.random.seed(42)
n_records = 500

crops = [
    'rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans',
    'mungbean', 'blackgram', 'lentil', 'pomegranate', 'banana', 'mango',
    'grapes', 'watermelon', 'muskmelon', 'apple', 'orange', 'papaya',
    'coconut', 'cotton', 'jute', 'coffee'
]
raw_weights = np.array([
    11, 8, 5, 4, 4, 3, 3, 3, 3, 4,
    6, 5, 4, 5, 4, 5, 4, 5, 4, 8, 3, 5
], dtype=float)
probabilities = raw_weights / raw_weights.sum()

start_date = datetime.now() - timedelta(days=30)
intervals = np.random.exponential(scale=85, size=n_records).cumsum()
timestamps = [start_date + timedelta(minutes=int(x)) for x in intervals]

data = {
    "timestamp": timestamps[:n_records],
    "predicted_crop": np.random.choice(crops, size=n_records, p=probabilities),
    "latency_ms": np.random.normal(loc=42.5, scale=11.2, size=n_records).clip(12.0, 115.0).round(2),
    "user_status": np.random.choice(["Success", "Validation Error"], size=n_records, p=[0.96, 0.04]),
    "input_n": np.random.uniform(0, 140, size=n_records).round(1),
    "input_p": np.random.uniform(5, 145, size=n_records).round(1),
    "input_k": np.random.uniform(5, 205, size=n_records).round(1),
    "input_temp": np.random.uniform(12, 42, size=n_records).round(2),
    "input_humidity": np.random.uniform(15, 99, size=n_records).round(2),
    "input_ph": np.random.uniform(3.5, 9.5, size=n_records).round(2),
    "input_rainfall": np.random.uniform(20, 300, size=n_records).round(2)
}
df_logs=pd.DataFrame(data)
if row_count == 0:
    print("Database is empty. Generating and uploading 500 baseline records...")
    df_logs.to_sql("prediction_telemetry", engine, if_exists="append", index=False)
    print(" 500 records successfully populated into Cloud PostgreSQL!")
else:
    print(f" Database already has {row_count} records. Skipping baseline seeding.")