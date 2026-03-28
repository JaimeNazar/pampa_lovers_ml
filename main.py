from fastapi import FastAPI
from pydantic import BaseModel
import random

app = FastAPI()

# Request body schema
class InputData(BaseModel):
    numbers: list[float]

@app.get("/")
def root():
    return {"message": "API is working 🚀"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: InputData):
    # Fake "model" logic (just for testing)
    total = sum(data.numbers)
    avg = total / len(data.numbers) if data.numbers else 0
    
    return {
        "input": data.numbers,
        "sum": total,
        "average": avg,
        "random_score": random.random()
    }