from fastapi import FastAPI
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

class CropType(Enum):
    WHEAT = 1
    RICE = 2
    MAIZE = 3
    COTTON = 4
    SOYBEAN = 5

class IrrigationType(Enum):
    DRIP = 1
    SPRINKLER = 2
    MANUAL = 3
    NONE = 4

class FertilizerType(Enum):
    ORGANIC = 1
    INORGANIC = 2
    MIXED = 3

class CropDiseaseStatus(Enum):
    NONE = 1
    MILD = 2
    MODERATE = 3
    SEVERE = 4
    
# --- enum conversion ---

def enum_to_int(enum_class, value):
    if not value:
        return 0
    try:
        return enum_class[value.upper()].value
    except KeyError:
        return 0  # unknown value fallback

def save_model_to_supabase(user_id, model):
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        model.save(tmp.name)

        with open(tmp.name, "rb") as f:
            supabase.storage.from_("models").upload(
                f"{user_id}/model.h5",
                f,
                {"upsert": True}
            )

def load_model_from_supabase(user_id):
    try:
        res = supabase.storage.from_("models").download(f"{user_id}/model.h5")

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp.write(res)

        model = tf.keras.models.load_model(tmp.name)
        return model

    except Exception:
        return None

app = FastAPI()

# --- Supabase setup ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Request schema ---
class PlotQuery(BaseModel):
    group: str
    farm_id: str


@app.get("/")
def root():
    return {"message": "API running"}


@app.get("/predict")
def predict_from_db(input: PlotQuery, user_id: str):

    model = load_model_from_supabase(user_id)

    if model is None:
        return {"error": "Model not trained yet"}

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
        "crop_disease_status"
    ]

    # Convert list → comma string for Supabase
    select_string = ",".join(columns)

    # Query Supabase
    response = (
        supabase.table("plots")
        .select(select_string)
        .eq("group", input.group)
        .eq("farm_id", input.field_id)
        .execute()
    )

    data = response.data

    if not data:
        return {"error": "No data found"}

    # Convert to model-friendly format
    processed = []
    for row in data:
        processed.append([

            # Enums to int
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

    X = np.array(processed)

    predictions = model.predict(X).tolist()

    return {
        "predictions": predictions
    }

@app.post("/train-model")
async def train_model():
    """
    Train a TensorFlow model for a specific user based on their 'logs' table,
    then save the trained model to Supabase Storage.
    """

    # Wrap all synchronous Supabase calls in asyncio.to_thread
    def fetch_logs():
        all_data = []
        limit = 1000
        offset = 0

        while True:
            resp = (
                supabase.table("logs")
                .select("*")
                .range(offset, offset + limit - 1)
                .execute()
            )

            batch = resp.data or []
            if not batch:
                break

            all_data.extend(batch)
            offset += limit

        return all_data

    logs_data = await asyncio.to_thread(fetch_logs)

    if not logs_data:
        return {"error": "No training data found."}

    # Prepare features (X) and target (y)
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

    # Build a simple TensorFlow model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation="relu", input_shape=(X.shape[1],)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    # Train the model
    model.fit(X, y, epochs=10, verbose=0)

    # Save model to Supabase Storage
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        model.save(tmp.name)
        with open(tmp.name, "rb") as f:
            supabase.storage.from_("models").upload(
                f"model-1.h5",
                f,
                {"upsert": True}
            )

    return {
        "message": "Model trained and saved successfully!",
        "samples": len(X)
    }