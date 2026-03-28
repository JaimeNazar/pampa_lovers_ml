
from pydantic import BaseModel
from supabase import create_client, Client
import os
import numpy as np
import tensorflow as tf
import tempfile
import pandas as pd
import numpy as np
from enum import Enum
import asyncio

from pydantic import BaseModel
import numpy as np
from fastapi import FastAPI, Query


SUPABASE_MODEL_FILE = "model-1.keras"  # model filename

def load_global_model(supabase: Client):
    """
    Load the global model from Supabase Storage.
    Returns a Keras model or None if the file does not exist.
    """
    try:
        # Download the model file from Supabase
        res = supabase.storage.from_("models").download(SUPABASE_MODEL_FILE)

        # Save to a temporary file and load with TensorFlow
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
            tmp.write(res)
            tmp.flush()
            model = tf.keras.models.load_model(tmp.name)

        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

app = FastAPI()

# --- Supabase setup ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Request schema ---
class PlotQuery(BaseModel):
    plot_id: str

@app.get("/")
def root():
    return {"message": "API running"}

@app.get("/predict")
def predict_from_plot(plot_id: str = Query(..., description="ID of the plot to predict")):

    # Load the global model
    model = load_global_model(supabase)  # single model
    if model is None:
        return {"error": "Model not trained yet"}

    # Columns to fetch from the "plots" table
    columns = [
        "crop_type",
        "soil_moisture",
        "soil_ph",
        "temperature_c",
        "rainfall_mm",
        "humidity_percent",
        "irrigation_type",
        "fertilizer_type",
        "pesticide_usage_ml",
        "sowing_date",
        "harvest_date",
        "ndvi_index",
        "crop_disease_status",
        "sunlight_hours",
        "total_days"
    ]
    select_string = ",".join(columns)

    # Fetch the specific plot data
    response = (
        supabase.table("plots")
        .select(select_string)
        .eq("id", input.plot_id)
        .execute()
    )

    data = response.data
    if not data:
        return {"error": "No data found for this plot"}

    row = data[0]  # Only one plot

    # Prepare model input
    X = np.array([[
        row.get("crop_type") or 0,
        row.get("irrigation_type") or 0,
        row.get("fertilizer_type") or 0,
        row.get("crop_disease_status") or 0,
        row.get("soil_moisture") or 0,
        row.get("soil_ph") or 0,
        row.get("temperature_c") or 0,
        row.get("rainfall_mm") or 0,
        row.get("humidity_percent") or 0,
        row.get("sunlight_hours") or 0,
        row.get("pesticide_usage_ml") or 0,
        row.get("total_days") or 0,
        row.get("ndvi_index") or 0
    ]])

    # Run prediction
    predictions = model.predict(X).tolist()

    return {
        "plot_id": input.plot_id,
        "predictions": predictions
    }

@app.post("/train-model")
async def train_model():
    def fetch_logs():
        all_data = []
        limit = 1000
        offset = 0

        while True:
            resp = supabase.table("logs").select("*").range(offset, offset+limit-1).execute()
            batch = resp.data or []
            if not batch:
                break
            all_data.extend(batch)
            offset += limit
        return all_data

    logs_data = await asyncio.to_thread(fetch_logs)
    if not logs_data:
        return {"error": "No training data found."}

    X = []
    y = []

    for row in logs_data:
        X.append([
            row.get("crop_type") or 0,
            row.get("irrigation_type") or 0,
            row.get("fertilizer_type") or 0,
            row.get("crop_disease_status") or 0,
            row.get("soil_moisture") or 0,
            row.get("soil_ph") or 0,
            row.get("temperature") or 0,
            row.get("rainfall") or 0,
            row.get("humidity") or 0,
            row.get("sunlight_hours") or 0,
            row.get("pesticide_usage") or 0,
            row.get("total_days") or 0,
            row.get("ndvi_index") or 0
        ])
        y.append(row.get("yield_kg_per_hectare") or 0)

    X = np.array(X)
    y = np.array(y)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(X.shape[1],)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=10, verbose=0)

    # Save in .keras format
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        model.save(tmp.name)
        with open(tmp.name, "rb") as f:
            supabase.storage.from_("models").upload(
                "model-1.keras",
                f,
                {"upsert": "true"}  # <-- string, not bool
            )

    return {"message": "Model trained and saved successfully!", "samples": len(X)}